"""Smoke-1 corrections (2026-10-13), CPU only.
A  S5 reconciled with the L3 dose rule: straddling/shutdown reads logged separately and never delivered, an in-flight
   read at stop is not a failure, work_end <= stop_request <= join_end for active pollers, E0 needs no stop/join times.
B  primary period guard: P_hat outside the hard feasible interval -> aligned_hat unavailable with a reason (no clip).
C  baseline raw evidence (e0/e1 values, statuses, call times, identity, clock, units), kept on later failure, no extra
   NVML call.
F  smoke deadline = min(reservation end, start + 30 min); LD_PRELOAD must be unset or empty."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

from ncclenergy.v2 import stage2_calibration as K  # noqa: E402
from ncclenergy.v2 import stage2_energy as en  # noqa: E402
from ncclenergy.v2 import stage2_phase as ph  # noqa: E402
from ncclenergy.v2 import stage2_runner as S  # noqa: E402
from ncclenergy.v2.common import REPO  # noqa: E402

P5, BASE = 500_000_000, 10_000_000_000
(REPO / "logs").mkdir(exist_ok=True)          # git-ignored


def _rd(k, late=1_400_000, dur=1_300_000, rc=0):
    return [k, BASE + k * P5, BASE + k * P5 + late, BASE + k * P5 + late + dur, 100 + k, rc]


def _dose(we, reads, stop=None, joined=None, inflight=0):
    stop = we + 20_000 if stop is None else stop
    joined = stop + 3_000_000 if joined is None else joined
    return {"base_ns": BASE, "stop_req_ns": stop, "joined_ns": joined, "n": len(reads), "skipped": 0, "overflow": 0,
            "start_failed": 0, "inflight_at_stop": inflight, "reads": reads}


def _session(name):
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as tmp:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(("FAKE_", "NCCLE_"))}
        env["PYTHONPATH"] = f"{REPO / 'tests/v2/fake_nvml'}:{REPO / 'src'}"
        p = subprocess.run([sys.executable, "-W", "error", str(REPO / "tests/v2/helpers/stage2_session.py"), name, tmp],
                           env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-4000:]
        return json.loads(p.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ A: S5 and the dose rule

@strict
def test_A_inflight_read_at_stop_is_logged_never_delivered_and_not_a_failure():
    we = BASE + 5 * P5 + 1_500_000                       # read 5 started 0.1 ms before the work end: straddles
    s = en.dose_summary(_dose(we, [_rd(k) for k in range(1, 6)], inflight=1), 500, BASE, we)
    assert (s["eligible"], s["eligible_completed"], s["delivered"], s["delivered_frac"]) == (4, 4, 4, 1.0)
    assert s["straddling"] == 1 and s["straddling_detail"] == [_rd(5)[:4] + [0]] and s["shutdown_reads"] == 0
    assert s["inflight_at_stop"] == 1 and s["flags"] == [] and s["order"]["ok"] is True
    we2 = BASE + 5 * P5 + 1_000_000                      # read 5 starts after the work end, before the stop request
    s2 = en.dose_summary(_dose(we2, [_rd(k) for k in range(1, 6)], stop=we2 + 1_000_000, inflight=1), 500, BASE, we2)
    assert s2["shutdown_reads"] == 1 and s2["shutdown_detail"][0][0] == 5 and s2["straddling"] == 0
    assert (s2["eligible"], s2["delivered"]) == (4, 4) and s2["flags"] == [] and s2["order"]["ok"] is True


@strict
def test_A_active_poller_order_work_end_stop_join_is_enforced():
    we = BASE + 2_500_000_000 + 600_000
    reads = [_rd(k) for k in range(1, 5)]
    j = en.dose_summary(_dose(we, reads, stop=we + 20_000, joined=we + 10_000), 500, BASE, we)
    assert "DOSE_JOINED_BEFORE_STOP" in j["flags"] and j["order"]["ok"] is False
    st = en.dose_summary(_dose(we, reads, stop=we - 1, joined=we + 1_000), 500, BASE, we)
    assert "DOSE_STOPPED_BEFORE_WORK_END" in st["flags"] and st["order"]["ok"] is False
    miss = en.dose_summary(_dose(we, reads, joined=-1), 500, BASE, we)
    assert miss["flags"] == ["DOSE_TIMES_MISSING"] and miss["order"]["ok"] is False and miss["delivered_frac"] is None
    eq = en.dose_summary(_dose(we, reads, stop=we, joined=we), 500, BASE, we)          # equalities are allowed
    assert eq["order"]["ok"] is True and eq["flags"] == []


def _spec_builder():
    spec = importlib.util.spec_from_file_location("t_spec", REPO / "tests/v2/test_stage2_spec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._region_and_s2


@strict
def test_A_inactive_poller_E0_needs_no_stop_or_join_times():
    region, s2 = _spec_builder()(dose_ms=0, pp=True)
    assert all(g["dose"]["stop_req_ns"] == -1 and g["dose"]["joined_ns"] == -1 for g in s2["gpus"])
    m = en.s2_metrics(region, s2)
    assert m["flags"] == [] and all(p["dose"]["poller"] == "inactive" for p in m["per_gpu"])
    s5, why = S.smoke_s5(m["per_gpu"], {"dose_period_ms": 0}, S.Timing())
    assert why == [] and s5["poller"] == "inactive"
    s2["gpus"][0]["dose"]["stop_req_ns"] = 123                 # poller times without a dose request: inconsistent
    m2 = en.s2_metrics(region, s2)
    assert "DOSE_POLLER_TIMES_WITHOUT_DOSE_REQUEST" in m2["flags"]
    assert S.smoke_s5(m2["per_gpu"], {"dose_period_ms": 0}, S.Timing())[1] == ["S5_POLLER_TIMES_WITHOUT_DOSE_RANK0"]


@strict
def test_A_smoke_s5_threshold_unchanged_and_order_checked():
    def per(frac, ok=True, flags=()):
        return [{"rank": 0, "flags": list(flags), "dose": {"poller": "active", "order": {"ok": ok}, "delivered_frac": frac,
                                                           "flags": list(flags)}}]
    t, u = S.Timing(), {"dose_period_ms": 100}
    assert t.dose_delivery_min == 0.95
    assert S.smoke_s5(per(0.95), u, t)[1] == []
    assert S.smoke_s5(per(0.9499), u, t)[1] == ["S5_DELIVERY_BELOW_95PCT_RANK0"]
    assert S.smoke_s5(per(1.0, ok=False), u, t)[1] == ["S5_ORDER_WORK_END_STOP_JOIN_RANK0"]
    assert S.smoke_s5(per(1.0, flags=["DOSE_READ_BEFORE_BASE"]), u, t)[1] == ["S5_READ_BEFORE_C0_RANK0"]
    assert S.smoke_s5([{"rank": 0, "flags": [], "dose": None}], u, t)[1] == ["S5_NO_DOSE_SUMMARY_RANK0"]
    assert S.smoke_s5([{"rank": 0, "flags": [], "dose": None}], {"dose_period_ms": 0}, t)[1] == ["S5_NO_DOSE_SUMMARY_RANK0"]
    assert S.smoke_s5([{"rank": 0, "flags": [], "dose": {"poller": "inactive", "n_reads": 2}}], {"dose_period_ms": 0},
                      t)[1] == ["S5_POLLER_ACTIVE_WITHOUT_DOSE_RANK0"]


# ------------------------------------------------------------------ B: primary period guard

BRACKETS_MS = [(-0.1, 0.1), (99.9, 100.1), (199.9, 200.1), (300.0, 309.0)]


@strict
def test_B_ls_period_outside_the_feasible_interval_makes_aligned_hat_unavailable_without_clipping():
    e = ph.estimate_period_phase([(int(a * 1e6), int(b * 1e6)) for a, b in BRACKETS_MS])
    assert e["status"] == "OK" and e["k"] == [0, 1, 2, 3]
    lo, hi = (x / 1e6 for x in e["hard_period_ns"])
    assert abs(lo - 99.96667) < 1e-4 and abs(hi - 100.1) < 1e-6
    assert abs(e["period_ns"] / 1e6 - 101.35) < 1e-6                     # reported as computed, not clipped
    assert e["primary_period"]["status"] == "INCOMPATIBLE_WITH_FEASIBLE_MODEL"
    p, why = ph.aligned_hat_period(e)
    assert p is None and why.startswith("PRIMARY_PERIOD_INCOMPATIBLE") and "101.3500" in why
    ok = ph.estimate_period_phase([(int(a * 1e6), int(b * 1e6)) for a, b in
                                   [(-0.1, 0.1), (99.9, 100.1), (199.9, 200.1), (299.9, 300.1)]])
    assert ok["primary_period"]["status"] == "COMPATIBLE" and ph.aligned_hat_period(ok) == (ok["period_ns"], None)
    bad = ph.estimate_period_phase([(0, 2_000_000), (100_000_000, 102_000_000), (202_500_000, 204_500_000),
                                    (300_000_000, 302_000_000)])
    assert bad["status"] == "INFEASIBLE" and ph.aligned_hat_period(bad) == (None, "PERIOD_MODEL_INFEASIBLE")


@strict
def test_B_unit_energies_keep_raw_and_nominal_when_aligned_hat_is_unavailable():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as tmp:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(("FAKE_", "NCCLE_"))}
        env["PYTHONPATH"] = f"{REPO / 'tests/v2/fake_nvml'}:{REPO / 'src'}"
        p = subprocess.run([sys.executable, "-W", "error", str(REPO / "tests/v2/helpers/stage2_session.py"),
                            "calibration", tmp], env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-3000:]
        recs = K.load_units(Path(tmp) / "data/calibration")
        rec = next(r for r in recs if r["unit_id"] == "S0a-01")
        f = REPO / rec["dir"] / "region.jsonl"
        reg = json.loads(f.read_text().splitlines()[0])
        t0 = reg["gpus"][0]["edges"][0][0]
        reg["gpus"][0]["edges"] = [[t0 + int(a * 1e6), t0 + int(b * 1e6)] for a, b in BRACKETS_MS]
        f.write_text(json.dumps(reg) + "\n")                          # test copy only
        e = K.unit_energies(rec, REPO)
    g0, g1 = e["per_gpu"]
    assert g0["aligned_hat"] is None and g0["aligned_hat_unavailable_reason"].startswith("PRIMARY_PERIOD_INCOMPATIBLE")
    assert g0["aligned_nom"]["energy_J"] is not None and g0["raw_J"] is not None
    assert g1["aligned_hat"] is not None                                # the other GPU is unaffected
    assert e["E_aligned_hat_J"] is None and e["E_aligned_nom_J"] is not None and e["E_counter_raw_J"] is not None
    assert any(r.startswith("ALIGNED_HAT_UNAVAILABLE_RANK0") for r in e["reasons"])


@strict
def test_B_end_alignment_denominator_substitution_is_recorded():
    P = 100_000_000
    recs, ens = [], {}
    for i in range(21):
        t_c = 50_000_000_000 + i * 10**10
        raw = {r: {"t_loop_end_ns": t_c - 28_000_000, "t_commit_ns": t_c, "t_end_a_ns": t_c + 8_000_000,
                   "t_end_b_ns": t_c + 9_350_000, "t_post_a_ns": t_c + P + 8_000_000,
                   "t_post_b_ns": t_c + P + 9_350_000} for r in (0, 1)}
        br = [[t_c + (j + 2) * P - 1_100_000, t_c + (j + 2) * P + 1_100_000] for j in range(4)]
        recs.append({"unit_id": f"u{i}", "accepted": True, "gpus": [4, 5], "unit": {"post_probe": True, "length_s": 10.0},
                     "s2": {"per_gpu": [{"rank": r, "t_work_end_ns": t_c - 30_000_000, "post_probe": {"brackets": br}}
                                        for r in (0, 1)]}, "region": {"per_gpu": [{"rank": r} for r in (0, 1)]},
                     "_raw_region_gpus": raw})
        ens[f"u{i}"] = {"per_gpu": [{"rank": r, "aligned_nom": {"p_ctx_W": 80.0, "energy_J": 1500.0},
                                     "aligned_hat": None if (i == 0 and r == 0) else {"p_ctx_W": 80.0, "energy_J": 1500.0},
                                     "aligned_hat_unavailable_reason": "PRIMARY_PERIOD_INCOMPATIBLE: test"
                                     if (i == 0 and r == 0) else None} for r in (0, 1)]}
    b = K.end_alignment_timing(recs, ens)
    assert b["status"] == "CONDITIONAL"
    assert b["denominator_nominal_period_substitutions"] == [("u0", 0, "PRIMARY_PERIOD_INCOMPATIBLE: test")]


# ------------------------------------------------------------------ C: baseline raw evidence

@strict
def test_C_baseline_raw_evidence_saved_with_no_extra_counter_call():
    d = _session("calibration")
    assert d["counter_calls_total"] == 8 and d["counter_calls_in_units"] == 0      # unchanged: 2 baselines x 2 x 2
    for which in ("start", "end"):
        ev = d["baseline_evidence"][which]
        assert [g["nvml_index"] for g in ev["gpus"]] == [4, 5] and all(g["uuid"] and g["pci"] for g in ev["gpus"])
        assert "CLOCK_MONOTONIC" in ev["clock_domain"]["info"] and ev["units"]["value"].startswith("mJ")
        for key in ("e0", "e1"):
            rows = ev[key]
            assert [r[0] for r in rows] == [4, 5]
            assert all(isinstance(r[1], int) and r[2] == 0 and r[3] <= r[4] for r in rows)
        assert all(b[1] >= a[1] and b[3] > a[4] for a, b in zip(ev["e0"], ev["e1"]))


@strict
def test_C_first_read_evidence_is_kept_when_the_baseline_fails_later():
    d = _session("baseline_fail_after_first_read")
    assert d["stop_reason"].startswith("SESSION_START_BASELINE_FAILED: INVALID_DURING_MEASUREMENT")
    assert d["counter_calls_total"] == 2 and d["unit_dirs"] == []                  # first read only, no second read
    ev = d["baseline_evidence"]["start"]
    assert ev["e1"] is None and [r[0] for r in ev["e0"]] == [4, 5] and all(r[2] == 0 for r in ev["e0"])


# ------------------------------------------------------------------ the frozen 3-launch smoke schedule on CPU

@strict
def test_smoke_schedule_end_to_end_on_cpu_reports_S5_per_launch():
    d = _session("smoke")
    assert d["status"] == "COMPLETE" and d["n_accepted"] == 3 and d["smoke_verdict"] == "PASS"
    assert [u["id"] for u in d["units"]] == ["SMOKE-01", "SMOKE-02", "SMOKE-03"]
    e0, d100, d500 = d["smoke_checks"]
    assert e0["S5"]["poller"] == "inactive" and e0["S5"]["problems"] == []
    for c in (d100, d500):
        assert c["S5"]["poller"] == "active" and c["S5"]["problems"] == [] and c["S4_extra"] == []
        assert all(g["delivered_frac"] >= 0.95 and g["order"]["ok"] for g in c["S5"]["per_gpu"])
    assert d["loader"]["LD_PRELOAD"] == "" and d["counter_calls_in_units"] == 0


# ------------------------------------------------------------------ F: deadline and LD_PRELOAD

@strict
def test_F_smoke_deadline_is_the_earlier_of_reservation_end_and_start_plus_30_min():
    t0 = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
    def cfg(mode, minutes):
        return {"mode": mode, "allocation_confirmation": {"window_end_utc": (t0 + timedelta(minutes=minutes)).isoformat()}}
    a = S.execution_budget(cfg("STAGE2_SMOKE", 120), t0)
    assert a["budget_s"] == 1800.0 and a["binding"] == "smoke start + 30 min"
    b = S.execution_budget(cfg("STAGE2_SMOKE", 10), t0)
    assert b["budget_s"] == 600.0 and b["binding"] == "reservation end"
    c = S.execution_budget(cfg("STAGE2_CALIBRATION", 120), t0)
    assert c["budget_s"] == 7200.0 and c["smoke_ceiling_s"] is None


@strict
def test_F_LD_PRELOAD_is_refused_in_config_check_and_per_launch():
    old = os.environ.get("LD_PRELOAD")
    try:
        os.environ["LD_PRELOAD"] = "/tmp/libinterpose.so"
        assert "LD_PRELOAD must be unset or empty in the controller environment" in S.check_config({}, check_freeze=False)
        ctx = SimpleNamespace(uuids=lambda g: ["GPU-a", "GPU-b"], gpus=[4, 5], mode="calibration")
        try:
            S.stage2_env(ctx, {"arm": "B", "pause_ns": 1, "dose_period_ms": 0, "post_probe": False},
                         REPO / "logs" / "never_created")
            raise AssertionError("LD_PRELOAD accepted")
        except S.Stage2Stop as e:
            assert "LD_PRELOAD" in str(e)
        os.environ["LD_PRELOAD"] = ""
        assert not any("LD_PRELOAD" in x for x in S.check_config({}, check_freeze=False))
    finally:
        if old is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = old


# ------------------------------------------------------------------ G: idle-gate clock events (user-approved)

from ncclenergy.v2 import stage2_readiness as rd  # noqa: E402


def _idle_rows(n=900, dt=0.05, ce4=lambda i: 0x4 if i % 3 else 0x1, ce5=lambda i: 0x1):
    """Real-hardware-like P8 idle trace (gate 3b, driver 580): GPU 4 alternates 0x1 / 0x4, GPU 5 mostly 0x1."""
    return [{"t_s": i * dt, "gpu": g, "pstate": 8, "power_w": 25.0, "foreign": False, "temp_c": 30.0,
             "clock_events": (ce4 if g == 4 else ce5)(i)} for i in range(n) for g in (4, 5)]


@strict
def test_G_row_rule_tolerates_only_idle_bits_and_only_at_P8():
    base = {"temp_c": 30.0, "pstate": 8, "foreign": False}
    for ce in (0, 0x1, 0x4, 0x5):
        assert rd._row_problems(dict(base, clock_events=ce)) == [], ce
    for ce in (0x8, 0x10, 0x20, 0x40, 0x80, 0x41, 0x44):
        assert rd._row_problems(dict(base, clock_events=ce)) == ["CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING"], ce
    for ce in (None, True, 1.0, "1"):
        assert "CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING" in rd._row_problems(dict(base, clock_events=ce)), ce
    p2 = rd._row_problems(dict(base, pstate=2, clock_events=0x4))           # not idle: exact rule kept
    assert p2 == ["NOT_P8", "CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING"]
    assert rd.P8_TOLERATED_CLOCK_EVENTS == 0x5


@strict
def test_G_idle_trace_like_gate3b_passes_baseline_and_readiness_but_other_bits_fail():
    G = [4, 5]
    r = rd.p8_baseline(_idle_rows(), G)
    assert r["status"] == "OK", r
    assert rd.baseline_problems(_idle_rows(), G, 1.0, 40.0, 0.2) == []
    hot = _idle_rows(ce4=lambda i: 0x40 if i == 400 else 0x4)                 # one HW-thermal row at P8
    assert "CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING" in str(rd.baseline_problems(hot, G, 1.0, 40.0, 0.2))
    tr = [dict(x, t_s=x["t_s"] * 20) for x in _idle_rows(n=60)]               # 1 Hz-like readiness trace
    assert rd.gate_at(tr, G, tr[-1]["t_s"], 45.0, stable_window_s=30.0, max_gap_s=2.0, max_age_s=2.0)["status"] == "READY"
    sh = rd.clock_event_bit_shares(_idle_rows(n=300), G)
    assert sh["5"]["share_by_bit"] == {"0x1": 1.0} and sh["4"]["share_zero"] == 0.0
    assert abs(sh["4"]["share_by_bit"]["0x4"] - 2 / 3) < 0.01 and abs(sh["4"]["share_by_bit"]["0x1"] - 1 / 3) < 0.01


@strict
def test_G_sessions_idle_bits_pass_the_baselines_in_region_rule_unchanged_other_bits_refused():
    gi = _session("idle_ce_gpuidle")                     # 0x1 everywhere: masked in-region as before
    assert gi["status"] == "COMPLETE" and gi["n_accepted"] == 3 and gi["baseline_start"]["status"] == "OK"
    assert gi["baseline_end"]["status"] == "OK" and gi["counter_calls_total"] == 8
    assert gi["baseline_bit_shares"]["start"]["4"]["share_by_bit"] == {"0x1": 1.0}
    sp = _session("idle_ce_swpc")                        # 0x4 everywhere: idle gates pass, in-region QC still refuses
    assert sp["baseline_start"]["status"] == "OK" and sp["units"][0]["accepted"] is False
    assert "SW_POWER_CAP" in str(sp["unit_qc_reasons"][0]) and sp["status"] == "STOPPED"
    th = _session("idle_ce_thermal")                     # 0x40 at P8: no valid P8 entry, no counter read
    assert th["stop_reason"].startswith("SESSION_START_BASELINE_FAILED: NO_VALID_P8_ENTRY_WITHIN_CAP")
    assert "CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING" in th["stop_reason"] and th["counter_calls_total"] == 0
