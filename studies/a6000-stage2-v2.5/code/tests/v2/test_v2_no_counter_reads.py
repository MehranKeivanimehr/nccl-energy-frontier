"""The controller (runner, Sampler, watchdog, owned-process control) adds no cumulative-energy-counter reads
inside measured regions. Static: call sites of nvmlDeviceGetTotalEnergyConsumption in the v2 package are limited
to an allow-list, none of them in the sampler, watchdog, process control, preflight, QC or workload paths.
Dynamic: every NVML call made while the real runner code runs a unit is timestamped (fake NVML, CPU only)."""

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
(Path(__file__).resolve().parents[2] / "logs").mkdir(exist_ok=True)   # git-ignored; absent in a fresh checkout
PKG = ROOT / "src/ncclenergy/v2"
COUNTER = "nvmlDeviceGetTotalEnergyConsumption"
# where the counter may be read, and when:
ALLOWED = {
    "aligned_meter.py": "AlignedMeter: workload rank 0, probe before the region, start/stop at its boundaries",
    "runner.py": "idle_baseline (counter mode): between units, GPUs idle, no experiment process",
    "capabilities.py": "--probe-counter: only on allowed GPUs without any process",
    "validate.py": "Phase 4 validation tool, incl. the deliberate dose experiment (not used by campaigns)",
    "stage2_runner.py": "Stage 2 session baseline only (_counter_read): GPUs idle, no experiment process, called only "
                        "after a live P8/ownership/health validation (corrective gate cg1)",
}


def _calls(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name == COUNTER:
                fn = None
                for p in ast.walk(tree):
                    if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and p.lineno <= node.lineno <= p.end_lineno:
                        fn = p.name
                out.append((node.lineno, fn))
    return out


def test_static_counter_call_sites_are_allow_listed():
    found = {p.name: _calls(p) for p in sorted(PKG.glob("*.py"))}
    offenders = {k: v for k, v in found.items() if v and k not in ALLOWED}
    assert not offenders, offenders
    for mod in ("telemetry.py", "watchdog.py", "procctl.py", "preflight.py", "qc.py", "workload.py", "phase10.py",
                "numerics.py", "profile_scope.py"):
        assert not found.get(mod), mod
    assert {fn for _, fn in found["runner.py"]} == {"idle_baseline"}
    assert {fn for _, fn in found["aligned_meter.py"]} == {"_read"}
    assert {fn for _, fn in found["stage2_runner.py"]} == {"_counter_read"}
    # the sampler reads only the instantaneous-power field id, never an energy field id
    import re
    fields = set(re.findall(r"NVML_FI_[A-Z0-9_]+", (PKG / "telemetry.py").read_text()))
    assert fields == {"NVML_FI_DEV_POWER_INSTANT"}, fields
    assert not re.findall(r"NVML_FI_[A-Z0-9_]*ENERGY", (PKG / "watchdog.py").read_text() + (PKG / "procctl.py").read_text())


def test_no_counter_reads_by_controller_during_a_unit():
    with tempfile.TemporaryDirectory(dir=ROOT / "logs") as tmp:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(PYTHONPATH=f"{ROOT / 'tests/v2/fake_nvml'}:{ROOT / 'src'}", FAKE_NVML_COUNTER="commit")
        p = subprocess.run([sys.executable, str(ROOT / "tests/v2/helpers/nvml_call_audit.py"), tmp], env=env,
                           capture_output=True, text=True, timeout=180)
        assert p.returncode == 0, p.stderr[-3000:]
        d = json.loads(p.stdout.strip().splitlines()[-1])
        assert d["qc_status"] == "OK" and d["cleanup_ok"]
        assert d["idle_baseline"]["counter_calls"] == 4            # 2 GPUs x (start, end): outside any region
        assert d["run_unit"]["counter_calls"] == 0                 # controller never reads it while a unit runs
        assert d["measured_region"]["counter_calls"] == 0
        assert d["measured_region"]["duration_s"] > 2.0
        # what the controller does read during the region: sampler telemetry only (no main/watchdog NVML calls)
        assert set(d["measured_region"]["by_thread"]) == {"nccle-sampler"}, d["measured_region"]["by_thread"]
