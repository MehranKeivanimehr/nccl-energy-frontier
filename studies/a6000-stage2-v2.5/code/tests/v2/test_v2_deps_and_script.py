"""Dependency separation (analysis without PyTorch/NVML) and the campaign script's signal handling (CPU only)."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BLOCKER = r'''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "pynvml"):
            raise ImportError(f"blocked for the analysis-only test: {name}")
        return None
sys.meta_path.insert(0, Block())
'''


def test_analysis_modules_import_and_run_without_torch_or_nvml():
    code = BLOCKER + r'''
import numpy as np, pandas as pd
from ncclenergy.v2 import analysis, analyze_cli, phase89, recommender, plots_v2, profile_scope, numerics, reanalysis
from ncclenergy.v2 import measval, measval_posthoc, amendment_check, release_repro, release_tables
from ncclenergy.v2 import phase10   # the analyze command must not need NVML or torch
from ncclenergy.v2.analysis import paired_test, tost
r = paired_test(np.log(np.array([0.9, 0.92, 0.91, 0.93])), "less")
assert r["n"] == 4 and r["mean"] < 0
print("OK")
'''
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = str(ROOT / "src")
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip().endswith("OK"), p.stderr[-2000:]


def test_requirements_are_split_and_pinned():
    ana = (ROOT / "requirements-analysis.txt").read_text()
    gpu = (ROOT / "requirements-gpu.txt").read_text()
    pins = [l for l in ana.splitlines() if l and not l.startswith("#")]
    assert pins and all("==" in l for l in pins)
    assert not any(l.split("==")[0].lower() in ("torch", "nvidia-ml-py", "pynvml") for l in pins)
    assert "scipy==" in ana
    assert "-r requirements-analysis.txt" in gpu and "torch==" in gpu and "nvidia-ml-py==" in gpu


def test_campaign_script_forwards_signals_and_owned_tree_is_cleaned():
    """run_confirmatory_v2.sh with a stub runner (the CPU controller helper) and a stub nvidia-smi on PATH."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pidfile, report = tmp / "pids", tmp / "report.json"
            (tmp / "bin").mkdir()
            (tmp / "bin/nvidia-smi").write_text("#!/bin/sh\necho '4, 30'\necho '5, 30'\n")
            (tmp / "bin/nvidia-smi").chmod(0o755)
            stub = tmp / "stub_python"
            stub.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests/v2/helpers/controller.py'} "
                            f"{pidfile} sleep {report}\n")
            stub.chmod(0o755)
            camp = tmp / "camp.yaml"
            camp.write_text("max_blocks: 3\nallocation: [4, 5]\n")
            env = {k: v for k, v in os.environ.items()}
            env.update(PATH=f"{tmp / 'bin'}:{env['PATH']}", PYTHON=str(stub), CAMPAIGN=str(camp),
                       LOGDIR=str(tmp / "logs"))
            sh = subprocess.Popen(["bash", str(ROOT / "scripts/run_confirmatory_v2.sh")], env=env,
                                  start_new_session=True)
            t0 = time.time()
            while not pidfile.exists() and time.time() - t0 < 20:
                time.sleep(0.05)
            pids = [int(x) for x in pidfile.read_text().split()]
            time.sleep(0.8)
            os.kill(sh.pid, sig)
            rc = sh.wait(timeout=30)
            out = json.loads(report.read_text())
            assert rc == 130 and out["result"] == "ControllerSignal" and out["cleanup"]["ok"], (sig, rc, out)
            alive = [p for p in pids if os.path.exists(f"/proc/{p}") and
                     open(f"/proc/{p}/stat").read().split(")")[-1].split()[0] != "Z"]
            assert not alive, (sig, alive)
            prog = (tmp / "logs/confirmatory_progress.log").read_text()
            assert "block 1 exit" in prog and "block 2" not in prog      # campaign stopped, no further blocks


def _script_env(tmp, pidfile, report, extra):
    (tmp / "bin").mkdir(exist_ok=True)
    (tmp / "bin/nvidia-smi").write_text("#!/bin/sh\necho '4, 30'\necho '5, 30'\n")
    (tmp / "bin/nvidia-smi").chmod(0o755)
    stub = tmp / "stub_python"
    stub.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests/v2/helpers/controller.py'} {pidfile} sleep {report}\n")
    stub.chmod(0o755)
    camp = tmp / "camp.yaml"
    camp.write_text("max_blocks: 3\nallocation: [4, 5]\n")
    env = {k: v for k, v in os.environ.items()}
    env.update(PATH=f"{tmp / 'bin'}:{env['PATH']}", PYTHON=str(stub), CAMPAIGN=str(camp), LOGDIR=str(tmp / "logs"),
               **extra)
    return env


def test_campaign_script_wall_budget_stops_running_block_and_refuses_new_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pidfile, report = tmp / "pids", tmp / "report.json"
        env = _script_env(tmp, pidfile, report, {"WALL_BUDGET_S": "4", "BLOCK_EST_S": "0"})
        t0 = time.time()
        rc = subprocess.run(["bash", str(ROOT / "scripts/run_confirmatory_v2.sh")], env=env, timeout=90).returncode
        pids = [int(x) for x in pidfile.read_text().split()]
        out = json.loads(report.read_text())
        prog = (tmp / "logs/confirmatory_progress.log").read_text()
        assert rc == 130 and "wall budget 4s reached in block 1" in prog and "block 2" not in prog, prog
        assert out["result"] == "ControllerSignal" and out["cleanup"]["ok"] and time.time() - t0 < 30
        assert not [p for p in pids if os.path.exists(f"/proc/{p}") and
                    open(f"/proc/{p}/stat").read().split(")")[-1].split()[0] != "Z"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = _script_env(tmp, tmp / "pids", tmp / "report.json", {"WALL_BUDGET_S": "5", "BLOCK_EST_S": "100"})
        rc = subprocess.run(["bash", str(ROOT / "scripts/run_confirmatory_v2.sh")], env=env, timeout=60).returncode
        prog = (tmp / "logs/confirmatory_progress.log").read_text()
        assert rc == 3 and "does not allow block 1" in prog and not (tmp / "pids").exists()


def test_campaign_script_exit_status_reflects_the_blocks():
    """All blocks exit 0 -> script exits 0 (gate 3b found 1); a failing block -> its code, and no further blocks."""
    for runner_rc, want_rc, want_blocks in ((0, 0, 3), (2, 2, 1)):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = _script_env(tmp, tmp / "pids", tmp / "report.json", {})
            (tmp / "stub_python").write_text(f"#!/bin/sh\nexit {runner_rc}\n")
            rc = subprocess.run(["bash", str(ROOT / "scripts/run_confirmatory_v2.sh")], env=env, timeout=60).returncode
            prog = (tmp / "logs/confirmatory_progress.log").read_text()
            assert rc == want_rc and prog.count(" exit ") == want_blocks, (runner_rc, rc, prog)


def test_exit_status_fix_does_not_change_runner_invocations():
    """Amendment AM-2: the script used for gate 3b collection (configs/v2.2/frozen_code/run_confirmatory_v2_f4735b1.sh)
    and the fixed script start the runner identically (argv, working directory, environment, number of blocks);
    only the script's own exit status differs (1 vs 0 after a fully successful campaign)."""
    import shutil
    frozen = ROOT / "configs/v2.2/frozen_code/run_confirmatory_v2_f4735b1.sh"
    records = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/env.sh", tmp / "scripts/env.sh")
        (tmp / "bin").mkdir()
        (tmp / "bin/nvidia-smi").write_text("#!/bin/sh\necho '4, 30'\necho '5, 30'\n")
        (tmp / "bin/nvidia-smi").chmod(0o755)
        rec = tmp / "invocations.txt"
        stub = tmp / "stub_python"
        stub.write_text(f"#!/bin/sh\n{{ echo \"argv=$*\"; echo \"cwd=$(pwd)\"; env | grep -E '^(PYTHONPATH|CAMPAIGN|CUDA_HOME"
                        f"|NCCL[A-Z_]*|CUDA_VISIBLE_DEVICES)=' | sort; echo ---; }} >> {rec}\nexit 0\n")
        stub.chmod(0o755)
        camp = tmp / "camp.yaml"
        camp.write_text("max_blocks: 3\nallocation: [4, 5]\n")
        for label, src in (("frozen", frozen), ("fixed", ROOT / "scripts/run_confirmatory_v2.sh")):
            shutil.copy(src, tmp / "scripts/run.sh")
            rec.unlink(missing_ok=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL", "CUDA_VISIBLE"))}
            env.update(PATH=f"{tmp / 'bin'}:{env['PATH']}", PYTHON=str(stub), CAMPAIGN=str(camp),
                       LOGDIR=str(tmp / f"logs_{label}"))
            rc = subprocess.run(["bash", str(tmp / "scripts/run.sh")], env=env, timeout=60).returncode
            records[label] = (rc, rec.read_text())
    assert records["frozen"][1] == records["fixed"][1] and records["fixed"][1].count("argv=") == 3, records
    assert "argv=-m ncclenergy.v2.runner block --campaign" in records["fixed"][1]
    assert (records["frozen"][0], records["fixed"][0]) == (1, 0)
