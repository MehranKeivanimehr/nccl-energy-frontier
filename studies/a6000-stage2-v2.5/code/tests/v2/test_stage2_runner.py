"""Stage 2 runner integration (CPU only): the real stage2_runner session code with the fake NVML, the fake v2.5 bench and
short test timing, in subprocesses run with `python -W error`. Every NVML call is audited: the controller reads the
cumulative counter only in validated session baselines (GPUs idle, no experiment process), never during a unit."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
(ROOT / "logs").mkdir(exist_ok=True)          # git-ignored
_CACHE = {}


def scenario(name: str) -> dict:
    if name not in _CACHE:
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as tmp:
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(("FAKE_", "NCCLE_"))}
            env["PYTHONPATH"] = f"{ROOT / 'tests/v2/fake_nvml'}:{ROOT / 'src'}"
            p = subprocess.run([sys.executable, "-W", "error", str(ROOT / "tests/v2/helpers/stage2_session.py"), name, tmp],
                               env=env, capture_output=True, text=True, timeout=300)
            assert p.returncode == 0, p.stderr[-4000:]
            _CACHE[name] = json.loads(p.stdout.strip().splitlines()[-1])
    return _CACHE[name]


@strict
def test_calibration_session_end_to_end_with_v25_dose_rule():
    d = scenario("calibration")
    assert d["status"] == "COMPLETE" and d["baseline_start"]["status"] == "OK" and d["baseline_end"]["status"] == "OK"
    assert [u["id"] for u in d["units"]] == ["S0a-01", "S0b-02", "C2r1-03"] and all(u["accepted"] for u in d["units"])
    s0a, s0b, c2 = d["units"]
    assert "NCCLE_DOSE_PERIOD_MS" not in s0a["nccle_env"] and "NCCLE_DOSE_PERIOD_MS" in s0b["nccle_env"]
    assert all(u["required_channels"] == [] and u["unsafe"] is False for u in d["units"])
    assert [x["eligible"] for x in s0b["dose_detail"]] == [24, 24]     # D100, 2.5 s: request 25 is not eligible
    assert [x["eligible"] for x in c2["dose_detail"]] == [4, 4]        # D500, 2.5 s: the boundary request is not eligible
    assert s0b["dose"] == [1.0, 1.0] and c2["dose"] == [1.0, 1.0]
    assert all(a == b for u in d["units"] for a, b in u["inventory"])
    assert all(n >= 3 for u in d["units"] for n in u["pp_brackets"]) and all(u["tps_file"] for u in d["units"])
    assert d["counter_calls_in_units"] == 0 and d["counter_calls_total"] == 8 and d["n_unit_windows"] == 3
    assert d["rerun_refused"] and "runs once" in d["rerun_refused"]
    assert d["postflight"]["owned_processes_left_on_gpus"] == [] and d["n_accepted"] == 3


@strict
def test_heldout_session_readiness_and_pre_region_checks():
    d = scenario("heldout")
    assert d["status"] == "COMPLETE" and all(u["accepted"] and u["readiness"] == "READY" and u["pre_region_ok"]
                                             for u in d["units"])
    assert all("NCCLE_DOSE_PERIOD_MS" not in u["nccle_env"] and "NCCLE_POST_PROBE" not in u["nccle_env"]
               for u in d["units"])
    assert d["counter_calls_in_units"] == 0


@strict
def test_readiness_never_launches_on_expiry():
    d = scenario("heldout_hot")
    assert d["status"] == "STOPPED" and d["stop_reason"] == "READINESS_NOT_ACHIEVED" and d["unit_dirs"] == []
    assert d["units"][0]["qc"] == "NOT_RUN_READINESS_NOT_ACHIEVED"


@strict
def test_budget_is_checked_before_the_start_baseline_and_before_each_unit():
    s = scenario("budget_start")
    assert s["stop_reason"] == "BUDGET_INSUFFICIENT_FOR_START_BASELINE" and s["counter_calls_total"] == 0
    b = scenario("budget")
    assert b["status"] == "STOPPED" and b["stop_reason"] == "UNIT_START_BUDGET_GATE" and b["unit_dirs"] == []
    assert b["incomplete"]["reason"] == "UNIT_START_BUDGET_GATE"


@strict
def test_baseline_counter_reads_only_after_validity():
    f = scenario("baseline_fail")                                       # never P8: no counter read at all
    assert f["stop_reason"].startswith("SESSION_START_BASELINE_FAILED: NO_VALID_P8_ENTRY_WITHIN_CAP")
    assert "NOT_P8_GPU4" in f["stop_reason"] and f["counter_calls_total"] == 0 and f["unit_dirs"] == []
    g = scenario("baseline_foreign")                                    # another process on GPU 4: no counter read
    assert "FOREIGN_OR_UNKNOWN_PROCESS_GPU4" in g["stop_reason"] and g["counter_calls_total"] == 0
    assert g["baseline_end"]["status"] == "NOT_DONE"


@strict
def test_scope_mismatch_stops_before_any_counter_read_or_launch():
    m = scenario("scope_mismatch")
    assert m["stop_reason"].startswith("SCOPE_MISMATCH") and "driver" in m["stop_reason"]
    assert m["unit_dirs"] == [] and m["counter_calls_total"] == 0


@strict
def test_no_end_baseline_after_contention_policy_change_or_interruption():
    f = scenario("foreign_in_unit")                                     # another user's process during the unit
    assert f["status"] == "STOPPED" and "FOREIGN_PROCESS" in f["stop_reason"] and f["units"][0]["unsafe"] is True
    assert f["baseline_end"]["status"] == "NOT_DONE" and "contention" in f["baseline_end"]["reason"]
    assert f["counter_calls_total"] == 4 and f["counter_calls_in_units"] == 0     # start baseline only
    pc = scenario("policy_change")
    assert pc["stop_reason"].startswith("power limit / clock policy changed") and pc["baseline_end"]["status"] == "NOT_DONE"
    it = scenario("interrupt_unit")                                     # interrupted while the unit runs
    assert it["status"] == "INTERRUPTED" and it["units"][0]["qc"] == "INTERRUPTED_UNIT"   # attempted unit kept
    assert it["partial_unit"] == {"unit_json": True, "telemetry": True}                    # sampler stopped, data kept
    assert it["baseline_end"]["status"] == "NOT_DONE" and it["counter_calls_total"] == 4


@strict
def test_unit_acceptance_failures_stop_calibration_after_s0a_without_retry():
    for name, why in (("s0a_fail", "PREP_DURATION_OFF_TARGET"),
                      ("inventory", "STAGE2_RECORD_FLAGS:COUNTER_CALL_INVENTORY_MISMATCH")):
        d = scenario(name)
        assert d["status"] == "STOPPED" and d["unit_dirs"] == ["S0a-01"] and len(d["units"]) == 1
        assert d["units"][0]["why"] == [why] and d["units"][0]["accepted"] is False
        assert d["n_accepted"] == 0 and d["not_accepted"] == {"S0a-01": [why]}


@strict
def test_heldout_schedule_with_dose_is_refused_before_anything_runs():
    d = scenario("heldout_dose_refused")
    assert d["status"] == "REFUSED" and "refused in held-out mode" in d["stop_reason"]
    assert d["unit_dirs"] == [] and d["counter_calls_total"] == 0
