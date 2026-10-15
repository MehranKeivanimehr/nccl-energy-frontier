"""CPU-only tests of the energy meters and the sampled-energy fallback. NVML is replaced by tests/v2/fake_nvml
(never the real library); subprocesses get it first on PYTHONPATH."""

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from ncclenergy.v2.aligned_meter import ALIGNED_METHOD, aligned_energy, period_from_edges

ROOT = Path(__file__).resolve().parents[2]
(Path(__file__).resolve().parents[2] / "logs").mkdir(exist_ok=True)   # git-ignored; absent in a fresh checkout
FAKE = ROOT / "tests/v2/fake_nvml"
PY = sys.executable


def _env(**kw):
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    e.update(PYTHONPATH=f"{FAKE}:{ROOT / 'src'}", **{k: str(v) for k, v in kw.items()})
    return e


def _legacy_v21(e0, t0, e1, t1, epost, commit, t_end, P):
    """v2.1 AlignedMeter.stop() accounting (commit 63c5332, src/ncclenergy/v2/aligned_meter.py)."""
    p_ctx = (epost - e1) / 1000 / ((commit + P - t1) / 1e9)
    tail_s = (commit - t_end) / 1e9
    aligned = (e1 - e0) / 1000
    return aligned - p_ctx * tail_s


def test_v22_aligned_accounting_is_identical_to_v21():
    rng = random.Random(3)
    P = 100_000_000
    for _ in range(200):
        t_end = 10**12 + rng.randint(0, 10**9)
        commit = t_end + rng.randint(2_000_000, 99_000_000)
        br0 = (t_end - rng.randint(2 * 10**9, 20 * 10**9), 0)
        br0 = (br0[0], br0[0] + rng.randint(500_000, 9_000_000))
        br1 = (commit + 8_000_000, commit + 8_000_000 + rng.randint(500_000, 9_000_000))
        brp = (commit + P + 8_000_000, commit + P + 9_000_000)
        e0 = rng.randint(10**6, 10**9)
        e1 = e0 + rng.randint(10**5, 10**7)
        ep = e1 + rng.randint(5_000, 20_000)
        raw = {"c0_mJ": e0, "rc0": 0, "br0": br0, "c1_mJ": e1, "rc1": 0, "br1": br1, "cpost_mJ": ep, "rcpost": 0,
               "brpost": brp, "phase_ns": commit % P, "bracket_ns": 2_000_000, "edges": [], "period_ns": P,
               "period_est_ns": None, "commit_ns": commit, "t_end_ns": t_end}
        new = aligned_energy(raw)["energy_J"]
        old = _legacy_v21(e0, (br0[0] + br0[1]) // 2, e1, (br1[0] + br1[1]) // 2, ep, commit, t_end, P)
        assert abs(new - old) <= 1e-9 * max(1.0, abs(old))


def test_aligned_meter_qc_flags():
    P = 100_000_000
    base = {"c0_mJ": 1000, "rc0": 0, "br0": (0, 1000), "c1_mJ": 500_000, "rc1": 0, "br1": (5 * 10**9 + 8 * 10**6,) * 2,
            "cpost_mJ": 509_000, "rcpost": 0, "brpost": (5 * 10**9 + 108 * 10**6,) * 2, "phase_ns": 0,
            "bracket_ns": 2_000_000, "edges": [], "period_ns": P, "period_est_ns": None, "commit_ns": 5 * 10**9,
            "t_end_ns": 5 * 10**9 - 30_000_000}
    assert aligned_energy(base)["flags"] == []
    assert aligned_energy({**base, "rc1": 3})["energy_J"] is None                       # NVML read error
    assert aligned_energy({**base, "phase_ns": None})["flags"] == ["PHASE_UNKNOWN"]
    assert "PHASE_IMPRECISE" in aligned_energy({**base, "bracket_ns": 30_000_000})["flags"]
    assert "PERIOD_MISMATCH" in aligned_energy({**base, "period_est_ns": 120_000_000})["flags"]
    assert "END_READ_BEFORE_COMMIT" in aligned_energy({**base, "br1": (base["commit_ns"] - 1, base["commit_ns"] + 1)})["flags"]
    assert "COUNTER_NOT_UPDATING" in aligned_energy({**base, "c1_mJ": 1000, "cpost_mJ": 10_000})["flags"]
    assert "TAIL_IMPLAUSIBLE" in aligned_energy({**base, "cpost_mJ": 500_100})["flags"]
    assert period_from_edges([(0, 2), (100_000_000, 100_000_002), (200_000_000, 200_000_004)]) == 100_000_001


def test_aligned_meter_logs_raw_values_and_brackets():
    code = ("import json,time\nfrom ncclenergy.v2.aligned_meter import AlignedMeter\n"
            "m=AlignedMeter([4,5]); p=m.probe(); m.start(); time.sleep(0.6); r=m.stop()\n"
            "print(json.dumps({'p':p,'r':r}, default=str))")
    out = subprocess.run([PY, "-c", code], env=_env(FAKE_NVML_COUNTER="commit", FAKE_NVML_POWER_W=100),
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    d = json.loads(out.stdout.strip().splitlines()[-1])
    r = d["r"]
    assert r["method"] == ALIGNED_METHOD and r["energy_J"] is not None and r["flags"] == []
    for g, v in r["per_gpu"].items():
        raw = v["raw"]
        for k in ("c0_mJ", "c1_mJ", "cpost_mJ", "rc0", "rc1", "rcpost", "br0", "br1", "brpost", "phase_ns",
                  "bracket_ns", "edges", "period_ns", "period_est_ns", "commit_ns", "t_end_ns"):
            assert k in raw, k
        assert abs(raw["period_est_ns"] - 100_000_000) < 2_000_000                   # period measured, not assumed
        assert aligned_energy(raw)["energy_J"] == v["energy_J"]                        # recomputable offline


def test_counter_unsupported_is_reported_not_crashed():
    code = ("import json,time\nfrom ncclenergy.v2.aligned_meter import AlignedMeter\n"
            "m=AlignedMeter([4]); p=m.probe(); m.start(); r=m.stop(); print(json.dumps(r, default=str))")
    out = subprocess.run([PY, "-c", code], env=_env(FAKE_NVML_COUNTER="error"), capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["energy_J"] is None and "READ_ERROR" in r["flags"]


def test_sampled_meter_never_calls_the_counter():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "counter.log"
        code = ("import json,time\nfrom ncclenergy.v2.aligned_meter import SampledMeter\n"
                "m=SampledMeter([4,5]); m.probe(); m.start(); time.sleep(1.0); r=m.stop()\n"
                "import pynvml; print(json.dumps({'E':r['energy_J'],'flags':r['flags'],'calls':pynvml.CALLS['counter']}))")
        out = subprocess.run([PY, "-c", code], env=_env(FAKE_NVML_COUNTER="forbid", FAKE_NVML_COUNTER_LOG=log,
                                                         FAKE_NVML_POWER_W=100), capture_output=True, text=True,
                             timeout=60)
        assert out.returncode == 0, out.stderr
        d = json.loads(out.stdout.strip().splitlines()[-1])
        assert d["calls"] == 0 and not log.exists() and d["flags"] == []
        assert abs(d["E"] - 200.0) < 12.0                                          # 2 GPUs x 100 W x ~1 s


def test_sampled_mode_end_to_end_without_counter_calls():
    with tempfile.TemporaryDirectory(dir=ROOT / "logs") as tmp:
        log = Path(tmp) / "counter.log"
        out = subprocess.run([PY, str(ROOT / "tests/v2/helpers/sampled_e2e.py"), tmp],
                             env=_env(FAKE_NVML_COUNTER="forbid", FAKE_NVML_COUNTER_LOG=log), capture_output=True,
                             text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        d = json.loads(out.stdout.strip().splitlines()[-1])
        assert d["counter_calls"] == 0 and not log.exists()
        assert all(v["counter_status"] == "NOT_READ_SAMPLED_MODE" for v in d["idle"]["per_gpu"].values())
        assert d["qc_status"] == "OK" and d["energy_source"] == "sampled" and d["evidence"] == "VERIFIED"
        assert d["cleanup_ok"] and d["telemetry_problems"] == []
        assert abs(d["sampled_energy_J"] - 2 * 100 * d["T_region_s"]) < 0.02 * d["sampled_energy_J"]


def test_counter_mode_idle_baseline_does_call_the_counter():
    """Negative control: the forbidding mock really detects counter use."""
    code = ("from types import SimpleNamespace\nfrom ncclenergy.v2 import runner\n"
            "ctx=SimpleNamespace(cfg={'energy_source':'counter','sampler_period_s':0.05},"
            "by_index={4:{'pci':'00000000:81:00.0','alias':'a'}})\n"
            "import tempfile,pathlib\nruns=runner.idle_baseline(ctx,[4],pathlib.Path(tempfile.mkdtemp()),0.2,0.0,1)")
    out = subprocess.run([PY, "-c", code], env=_env(FAKE_NVML_COUNTER="forbid"), capture_output=True, text=True,
                         timeout=60)
    assert out.returncode != 0 and "cumulative energy counter called" in out.stderr


def test_nvml_audit_shim_records_calls_without_changing_results():
    """The test-only audit shim (gate 3a) wraps pynvml calls and writes them at exit; values are unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        code = ("import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(4); "
                "print(pynvml.nvmlDeviceGetPowerUsage(h), pynvml.nvmlDeviceGetTotalEnergyConsumption(h) >= 0)")
        env = _env(FAKE_NVML_COUNTER="commit", FAKE_NVML_POWER_W=100, NCCLE_NVML_AUDIT_LOG=f"{tmp}/audit")
        env["PYTHONPATH"] = f"{ROOT / 'tests/v2/helpers/nvml_audit_shim'}:{env['PYTHONPATH']}"
        out = subprocess.run([PY, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and out.stdout.split() == ["100000", "True"], out.stderr
        logs = list(Path(tmp).glob("audit.*.jsonl"))
        assert len(logs) == 1
        fns = [json.loads(l)["fn"] for l in logs[0].read_text().splitlines()]
        assert fns == ["nvmlInit", "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetPowerUsage",
                       "nvmlDeviceGetTotalEnergyConsumption"]


def test_flagged_aligned_reading_has_no_energy():
    """v2.2 release audit: a flagged reading (e.g. implausible tail, reversal) yields energy None; the unvalidated
    value is kept only for diagnosis."""
    P = 100_000_000
    base = {"c0_mJ": 1000, "rc0": 0, "br0": (0, 1000), "c1_mJ": 500_000, "rc1": 0, "br1": (5 * 10**9 + 8 * 10**6,) * 2,
            "cpost_mJ": 509_000, "rcpost": 0, "brpost": (5 * 10**9 + 108 * 10**6,) * 2, "phase_ns": 0,
            "bracket_ns": 2_000_000, "edges": [], "period_ns": P, "period_est_ns": None, "commit_ns": 5 * 10**9,
            "t_end_ns": 5 * 10**9 - 30_000_000}
    assert aligned_energy(base)["energy_J"] is not None
    for bad in ({"cpost_mJ": 500_100}, {"c1_mJ": 900, "cpost_mJ": 10_000}):
        r = aligned_energy({**base, **bad})
        assert r["flags"] and r["energy_J"] is None and r["energy_J_unvalidated"] is not None


def test_failed_process_query_is_recorded_as_unknown():
    code = ("import time\nfrom ncclenergy.v2.telemetry import Sampler\n"
            "s=Sampler([4], period_s=0.05).start(); time.sleep(0.3); s.stop()\n"
            "print(sorted({repr(p) for (_, _, p) in s.proc_events}))")
    for err, want in (("1", "['None']"), ("0", "['[]']")):
        out = subprocess.run([PY, "-c", code], env=_env(FAKE_NVML_PROCS_ERROR=err), capture_output=True, text=True,
                             timeout=60)
        assert out.returncode == 0 and out.stdout.strip() == want, (err, out.stdout, out.stderr[-500:])


def test_runner_fails_closed_on_log_errors_corrupt_region_and_unknown_process_list():
    """v2.2 release audit, through the real run_unit (sampled mode, fake NVML, fake benchmark)."""
    cases = ({"FAKE_BENCH_ERRLINE": "NCCL failure common.cu:1 'unhandled cuda error'"},
             {"FAKE_BENCH_CORRUPT_REGION": "1"},
             {"FAKE_NVML_PROCS_ERROR": "1"})
    got = []
    for extra in cases:
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as tmp:
            out = subprocess.run([PY, str(ROOT / "tests/v2/helpers/sampled_e2e.py"), tmp],
                                 env=_env(FAKE_NVML_COUNTER="forbid", **extra), capture_output=True, text=True,
                                 timeout=180)
            assert out.returncode == 0, out.stderr[-2000:]
            d = json.loads(out.stdout.strip().splitlines()[-1])
            got.append((d["qc_status"], d["qc_reasons"]))
    assert got[0][0] == "EXECUTION_ERROR" and "ERROR_IN_LOGS" in got[0][1], got[0]
    assert got[1][0] == "MEASUREMENT_INVALID" and any(r.startswith("REGION_PARSE:") for r in got[1][1]), got[1]
    assert got[2][0] == "SAFETY_STOP" and "PROCESS_LIST_UNAVAILABLE" in str(got[2][1]), got[2]


def test_phase10_meter_qc_excludes_flagged_v22_runs_and_ignores_v21_records():
    import pandas as pd
    from ncclenergy.v2.phase10 import meter_qc
    with tempfile.TemporaryDirectory() as tmp:
        meters = {"v21": {"energy_J": 5.0, "per_gpu": {"4": {"energy_J": 5.0}}},
                  "ok": {"energy_J": 5.0, "flags": [], "per_gpu": {"4": {"energy_J": 5.0, "flags": []}}},
                  "bad": {"energy_J": None, "flags": [], "per_gpu": {"4": {"energy_J": None, "flags": ["TAIL_IMPLAUSIBLE"]}}}}
        rows = []
        for i, (name, m) in enumerate(meters.items()):
            d = Path(tmp) / name
            d.mkdir()
            (d / "result.json").write_text(json.dumps({"meter": m}))
            rows.append({"block": 1, "arm": name, "dir": str(d)})
        mq, v22 = meter_qc(pd.DataFrame(rows))
        assert v22 and mq == {"b1:bad": ["TAIL_IMPLAUSIBLE"]}
        assert meter_qc(pd.DataFrame(rows[:1])) == ({}, False)
