"""Telemetry schema v3 (opt-in) with the fake NVML, in subprocesses run with `python -W error` (CPU only)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _run(**fake) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("FAKE_")}
        env.update(PYTHONPATH=f"{ROOT / 'tests/v2/fake_nvml'}:{ROOT / 'src'}", **fake)
        p = subprocess.run([sys.executable, "-W", "error", str(ROOT / "tests/v2/helpers/stage2_telemetry.py"), tmp],
                           env=env, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr[-3000:]
        return json.loads(p.stdout.strip().splitlines()[-1])


@strict
def test_default_sampler_unchanged_and_v3_appends_columns():
    from ncclenergy.v2.telemetry import FIELDS, FIELDS_V3_EXTRA
    r = _run()
    d, v = r["default"], r["v3"]
    assert d["header"] == FIELDS and d["row_len"] == 14 and d["ids"] == [186] and d["array_cols"] == 14
    assert v["header"] == FIELDS + FIELDS_V3_EXTRA and v["row_len"] == 23 and v["array_cols"] == 14
    assert v["ids"] == [[186, 0]] or v["ids"] == [(186, 0)]          # POWER_INSTANT with scopeId 0 explicit
    assert v["order"] and v["field_ts_is_realtime_us"] and v["realtime_close"] and v["mono_close"]
    assert v["status"] == 0 and v["latency"] == 120 and v["errors"] == 0


@strict
def test_total_power_samples_recorded_or_failure_recorded_never_substituted():
    ok = _run()["v3_tps"]
    assert ok["tps_rows"] > 0 and ok["tps_errors"] == 0 and ok["tps_ts_increasing"]
    bad = _run(FAKE_NVML_SAMPLES="0", FAKE_NVML_INSTANT="0")["v3_tps"]
    assert bad["tps_rows"] == 0 and bad["tps_errors"] > 0
    assert bad["status"] == 3                                           # unsupported field: status kept, value NaN
