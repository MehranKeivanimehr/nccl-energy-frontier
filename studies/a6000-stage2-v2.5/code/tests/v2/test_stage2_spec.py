"""Stage 2 specification tests (CPU only; no NVML, no GPU). Every test runs with warnings as errors (@strict);
warnings exercised on purpose are captured locally with expect_warning and asserted."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import expect_warning, strict  # noqa: E402

from ncclenergy.v2 import stage2_budget as budget  # noqa: E402
from ncclenergy.v2 import stage2_decisions as dec  # noqa: E402
from ncclenergy.v2 import stage2_energy as energy  # noqa: E402
from ncclenergy.v2 import stage2_phase as phase  # noqa: E402
from ncclenergy.v2 import stage2_readiness as rd  # noqa: E402
from ncclenergy.v2 import stage2_schedule as sched  # noqa: E402

CLUSTER_REPO = Path(os.environ.get("NCCLE_CLUSTER_REPO", Path.home() / "research/Overs/nccl-energy-frontier"))
PREP = Path(os.environ.get("NCCLE_STAGE2_PREP", Path(__file__).resolve().parents[3] / "prep"))


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# ---------------------------------------------------------------- units, clock domains, denominators
@strict
def test_counter_units_and_validity():
    assert energy.counter_delta_J(1_000_000, 1_250_500) == 250.5
    assert raises(energy.InvalidCounter, energy.counter_delta_J, 10, 5)               # reversal/wrap: never clamped
    assert raises(energy.InvalidCounter, energy.counter_delta_J, 0, 5, rc_end=3)       # NVML status error
    assert raises(energy.InvalidCounter, energy.counter_delta_J, 0.0, 5.0)            # raw counter must be integer mJ
    g = [{"c0_mJ": 0, "c1_mJ": 1000, "rc0": 0, "rc1": 0}, {"c0_mJ": 50, "c1_mJ": 2050}]
    assert energy.e_counter_raw_J(g) == 3.0


@strict
def test_clock_domains_are_not_mixed():
    a, b = energy.RealtimeUs(1_790_000_000_000_000), energy.RealtimeUs(1_790_000_000_500_000)
    assert b - a == 0.5
    assert raises(energy.ClockDomainError, lambda: a - 1234)
    assert raises(energy.ClockDomainError, lambda: 1234 - a)


@strict
def test_exact_operation_denominator_refuses_float_bool_and_nonfinite():
    assert energy.joules_per_op(764.769, 764769) == 0.001
    assert raises(ValueError, energy.joules_per_op, 1.0, 764769.0)
    assert raises(ValueError, energy.joules_per_op, 1.0, True)
    assert raises(ValueError, energy.joules_per_op, 1.0, 0)
    assert raises(ValueError, energy.joules_per_op, float("nan"), 10)
    assert raises(ValueError, energy.log_ratio, float("inf"), 1.0)
    assert raises(ValueError, energy.log_ratio, 0.0, 1.0)


@strict
def test_trapezoid_and_hold_known_answers():
    t = np.arange(0.0, 10.01, 0.05)
    assert abs(energy.sampled_energy_trapezoid_J(t, np.full_like(t, 100.0), 1.0, 9.0, 0.2) - 800.0) < 1e-9
    assert abs(energy.sampled_energy_trapezoid_J(t, 10.0 * t, 2.0, 6.0, 0.2) - 160.0) < 1e-9
    step = np.where(t < 5.0, 50.0, 150.0)
    assert abs(energy.sampled_energy_hold_J(t, step, 2.0, 8.0, 0.2) - 600.0) < 1e-9


@strict
def test_coverage_and_missing_samples_are_rejected_not_interpolated():
    t = np.arange(0.0, 10.01, 0.05)
    p = np.full_like(t, 100.0)
    assert raises(energy.CoverageError, energy.sampled_energy_trapezoid_J, t, p, -0.1, 5.0, 0.2)
    gap = np.concatenate([t[t < 3.0], t[t > 3.6]])
    assert raises(energy.CoverageError, energy.sampled_energy_trapezoid_J, gap, np.full_like(gap, 100.0), 1, 9, 0.2)
    nan = p.copy()
    nan[50] = np.nan
    assert raises(energy.CoverageError, energy.sampled_energy_hold_J, t, nan, 1, 9, 0.2)
    back = t.copy()
    back[10] = back[9]
    assert raises(energy.CoverageError, energy.sampled_energy_trapezoid_J, back, p, 1, 9, 0.2)
    tinf = t.copy()
    tinf[-1] = np.inf
    assert raises(energy.CoverageError, energy.sampled_energy_trapezoid_J, tinf, p, 1, 9, 0.2)
    assert raises(energy.CoverageError, energy.sampled_energy_trapezoid_J, t, p, 1, float("nan"), 0.2)


def _gpu():
    t_le = 10_000_000_000
    t_commit = t_le + 30_000_000
    return {"c0_mJ": 0, "c1_mJ": 1_352_400, "cpost_mJ": 1_360_400, "rc0": 0, "rc1": 0, "rcpost": 0,
            "t_loop_end_ns": t_le, "t_commit_ns": t_commit, "t_end_a_ns": t_commit + 8_000_000,
            "t_end_b_ns": t_commit + 8_000_000}


@strict
def test_aligned_correction_known_answer_and_period_variants():
    g = _gpu()
    r = energy.e_aligned_gpu_J(g, 100e6)
    assert abs(r["p_ctx_W"] - 8.0 / 0.092) < 1e-9 and abs(r["tail_s"] - 0.030) < 1e-12
    assert abs(r["energy_J"] - (1352.4 - 8.0 / 0.092 * 0.030)) < 1e-9
    r2 = energy.e_aligned_gpu_J(g, 99.9e6)
    assert r2["energy_J"] != r["energy_J"] and r2["period_ns"] == 99.9e6
    assert "TAIL_NEGATIVE" in energy.e_aligned_gpu_J(dict(g, t_commit_ns=g["t_loop_end_ns"] - 1_000_000), 100e6)["flags"]
    assert raises(energy.InvalidCounter, energy.e_aligned_gpu_J, dict(g, cpost_mJ=g["c1_mJ"] - 1), 100e6)
    assert raises(energy.InvalidCounter, energy.e_aligned_gpu_J, g, float("nan"))
    assert raises(ValueError, energy.e_aligned_J, [g, g], [100e6])


# ---------------------------------------------------------------- Stage 2 record: E_prep_to_C0, skew, dose, inventory
def _region_and_s2(dose_ms=0, pp=False, bad_inventory=False, prep_s=5.0):
    base = 50_000_000_000
    gpus_r, gpus_s = [], []
    for rank, off in ((0, 0), (1, 400_000)):            # GPU 1 reads 0.4 ms later at C0
        c0a = base + off
        t_work_end = c0a + 1_300_000 + 10_000_000_000 + 600_000      # GPU work ends 0.6 ms after the 100th request
        t_le = t_work_end + 2_000_000                                  # v2.1 loop end, after the poller join
        t_commit = t_le + 30_000_000
        stop = t_work_end + 20_000
        dose = {"base_ns": c0a + 1_300_000, "stop_req_ns": stop, "joined_ns": t_le - 10_000, "n": 0,
                "skipped": 0, "overflow": 0, "start_failed": 0, "inflight_at_stop": 0, "reads": []}
        if dose_ms:                     # native poller: a read at every request before the stop request
            P = dose_ms * 1_000_000
            k = 1
            while dose["base_ns"] + k * P < stop:
                req = dose["base_ns"] + k * P
                dose["reads"].append([k, req, req + 50_000, req + 1_400_000, 1000 + k, 0])
                k += 1
            dose["n"] = len(dose["reads"])
        else:                           # native v2.5: no poller started -> its times stay -1 (smoke-1 fixture fix)
            dose.update(base_ns=-1, stop_req_ns=-1, joined_ns=-1)
        post = {"start_ns": t_commit + 100_000_000 + 9_500_000, "end_ns": t_commit + 100_000_000 + 9_500_000 + 380_000_000,
                "n": 0, "edges": 0, "overflow": 0, "reads": []}
        if pp:
            t, v = post["start_ns"], 5000
            grid = t_commit + 200_000_000
            while t < post["end_ns"]:
                if t > grid:
                    v += 7
                    grid += 100_000_000
                post["reads"].append([t, t + 1_350_000, v, 0])
                t += 1_400_000
            post["n"] = len(post["reads"])
        probe_reads = 254
        n_reads = 1 + probe_reads + 1 + dose["n"] + 1 + 1 + post["n"] + (1 if bad_inventory else 0)
        t_pause_b = c0a - 50_000
        t_pause_a = t_pause_b - 23_000_000
        t_probe_end = t_pause_a - 50_000
        t_probe_start = t_probe_end - 350_000_000
        t_prep_end = t_probe_start - 100_000
        t_prep_start = t_prep_end - int(prep_s * 1e9)
        gpus_r.append({"rank": rank, "pci": f"0000000{rank}:81:00.0", "e_start_mJ": 1_000_000, "rc_start": 0,
                       "t_start_a_ns": c0a, "t_start_b_ns": c0a + 1_300_000, "t_loop_end_ns": t_le,
                       "t_end_a_ns": t_commit + 8_000_000, "t_end_b_ns": t_commit + 9_300_000,
                       "t_post_b_ns": t_commit + 109_300_000})
        gpus_s.append({"rank": rank, "pci": f"0000000{rank}:81:00.0", "n_counter_reads": n_reads, "e_prep0_mJ": 999_000,
                       "rc_prep0": 0, "t_prep0_a_ns": t_prep_start - 1_400_000, "t_prep0_b_ns": t_prep_start - 10_000,
                       "t_prep_start_ns": t_prep_start, "t_prep_end_ns": t_prep_end, "t_probe_start_ns": t_probe_start,
                       "t_probe_end_ns": t_probe_end, "probe_reads": probe_reads, "t_pause_a_ns": t_pause_a,
                       "t_pause_b_ns": t_pause_b, "t_work_end_ns": t_work_end, "dose": dose, "post_probe": post})
    region = {"pid": 7, "R": 191193, "warmup_iters": 95597, "coll": "all_gather", "size_bytes": 2097152, "gpus": gpus_r}
    s2 = {"schema": "nccle-s2-v1", "pid": 7, "R": 191193, "warmup_iters": 95597, "coll": "all_gather",
          "size_bytes": 2097152, "pause_req_ns": 22_900_000, "dose_period_ms": dose_ms, "post_probe": int(pp),
          "mode": "calibration", "gpus": gpus_s}
    return region, s2


@strict
def test_s2_metrics_prep_to_c0_skew_inventory_and_dose():
    region, s2 = _region_and_s2(dose_ms=100, pp=True)
    m = energy.s2_metrics(region, s2)
    assert m["flags"] == [], m["flags"]
    assert m["E_prep_to_C0_J"] == 2.0                          # 2 GPUs x (1_000_000 - 999_000) mJ
    assert abs(m["read_skew_ms"]["C0"] - 0.4) < 1e-9          # the actual per-GPU C0 read skew is recorded
    p0 = m["per_gpu"][0]
    assert abs(p0["prep_duration_s"] - 5.0) < 1e-9 and abs(p0["pause_dev_ms"] - 0.1) < 1e-9
    d = p0["dose"]                      # request 100 (0.6 ms before the work end) is not eligible; its read straddles
    assert d["eligible"] == 99 and d["delivered"] == 99 and d["delivered_frac"] == 1.0 and d["straddling"] == 1
    assert abs(p0["join_after_work_end_ms"] - 2.0) < 1e-9
    assert p0["n_counter_reads"] == p0["n_counter_reads_expected"]
    assert len(p0["post_probe"]["brackets"]) >= 3
    bad = energy.s2_metrics(*_region_and_s2(bad_inventory=True))
    assert "COUNTER_CALL_INVENTORY_MISMATCH" in bad["flags"]
    r2, s22 = _region_and_s2()
    s22["gpus"][0]["rc_prep0"] = 3                           # failed C_prep0: E_prep unavailable, not imputed
    m2 = energy.s2_metrics(r2, s22)
    assert m2["E_prep_to_C0_J"] is None and any(f.startswith("E_PREP_TO_C0_UNAVAILABLE") for f in m2["flags"])
    r3, s23 = _region_and_s2()
    s23["gpus"][0]["dose"]["reads"] = [[1, 1, 2, 3, 4, 0]]
    s23["gpus"][0]["dose"]["n"] = 1
    assert "DOSE_READS_WITHOUT_DOSE_REQUEST" in energy.s2_metrics(r3, s23)["flags"]
    r4, s24 = _region_and_s2()
    s24["pid"] = 8
    assert "S2_PID_MISMATCH" in energy.s2_metrics(r4, s24)["flags"]
    r5, s25 = _region_and_s2()
    del s25["gpus"][0]["t_work_end_ns"]                     # a v2.4 record has no work end: refused, not imputed
    assert "S2_NO_WORK_END" in energy.s2_metrics(r5, s25)["flags"]


@strict
def test_s2_parser_flags_unknown_and_partial_lines():
    good = json.dumps({"schema": "nccle-s2-v1", "pid": 1})
    recs, flags = energy.parse_s2_text(good + "\n" + json.dumps({"schema": "x"}) + "\n{partial")
    assert len(recs) == 1 and flags == ["S2_UNKNOWN_SCHEMA", "S2_PARTIAL_LOG"]
    assert energy.brackets_from_reads([(0, 1, 5, 0), (2, 3, 5, 0), (4, 5, 6, 0), (6, 7, 6, 3), (8, 9, 7, 0)]) == [(2, 5)]


# ---------------------------------------------------------------- period, phase, covariance, ambiguity
def _br(P=99.98e6, tau=5_000_000_123.0, n=4, h=1.1e6):
    return [(int(tau + k * P - h), int(tau + k * P + h)) for k in range(n)]


@strict
def test_period_feasible_model_hard_interval_and_labelled_approximation():
    est = phase.estimate_period_phase(_br())
    assert est["status"] == "OK" and est["k"] == [0, 1, 2, 3] and abs(est["period_ns"] - 99.98e6) < 2
    lo, hi = est["hard_period_ns"]
    assert lo < 99.98e6 < hi and abs(lo - 99.2467e6) < 1e3 and abs(hi - 100.7133e6) < 1e3   # exact LP range
    assert abs(est["approx"]["u_period_ns"] - 1.1e6 / math.sqrt(3) * math.sqrt(0.2)) < 50      # standard uncertainty
    assert "model error unknown" in est["approx"]["label"]
    assert phase.estimate_period_phase(_br()[:1])["status"] == "INSUFFICIENT_BRACKETS"
    assert phase.update_interval(phase.estimate_period_phase(_br()[:1]), 3) is None


@strict
def test_infeasible_constant_period_brackets_are_never_certified():
    """Counterexample: centres [0, 100, 202.5, 300] ms +- 1 ms. No constant period fits all four (k=1->2 needs
    P >= 100.5 ms, k=2->3 needs P <= 99.5 ms): the model is rejected and nothing is certified."""
    br = [(int((c - 1) * 1e6), int((c + 1) * 1e6)) for c in (0, 100, 202.5, 300)]
    est = phase.estimate_period_phase(br)
    assert est["status"] == "INFEASIBLE" and est["period_ns"] is None and "hard_period_ns" not in est
    assert phase.update_interval(est, 10) is None and phase.start_phase(est, 0, 1)["status"] == "UNKNOWN"
    assert phase.end_alignment(est, 0, 1, 2, 3)["status"] == "NOT_ESTABLISHED"
    two = [(0, 2_000_000), (200_000_000, 202_000_000)]                 # 200 ms apart: index 2 is forced by the prior
    e2 = phase.estimate_period_phase(two)
    assert e2["status"] == "OK" and e2["k"] == [0, 2]
    amb = phase.estimate_period_phase([(0, 2_000_000), (168_000_000, 180_000_000)], prior_ns=(55e6, 120e6))
    assert amb["status"] == "AMBIGUOUS_INDEX" and amb["n_feasible_assignments"] >= 2


@strict
def test_hard_interval_contains_every_feasible_truth_and_covariance_is_carried():
    """Unequal half-widths: the hard interval must contain the update time for random true grids inside the brackets;
    the approximate standard uncertainty uses the covariance-correct weights (matches Monte Carlo)."""
    hs = [0.2e6, 3.0e6, 3.0e6, 0.2e6, 0.2e6]
    P, tau = 100e6, 1e9
    br = [(int(tau + k * P - h), int(tau + k * P + h)) for k, h in enumerate(hs)]
    est = phase.estimate_period_phase(br)
    K = 30
    lo, hi = phase.update_interval(est, K)
    rng = np.random.default_rng(1)
    for _ in range(2000):                     # true (tau', P') consistent with every bracket -> inside the interval
        P2 = rng.uniform(*est["hard_period_ns"])
        t0 = rng.uniform(tau - 0.2e6, tau + 0.2e6)
        if all(b[0] <= t0 + k * P2 <= b[1] for k, b in enumerate(br)):
            assert lo - 1 <= t0 + K * P2 <= hi + 1
    ap = phase.approx_prediction(est, K)
    kc = np.array(est["k"]) - est["kbar"]
    c = kc / est["sxx"]
    e = rng.uniform(-1, 1, (100000, len(hs))) * np.array(hs)
    m = tau + np.arange(len(hs)) * P + e
    mc = float(np.std(m.mean(1) + (m @ c) * (K - est["kbar"])))
    assert abs(ap["u_ns"] / mc - 1) < 0.02


@strict
def test_probe_only_never_identifies_the_end_update_and_post_probe_does_under_the_model():
    est = phase.estimate_period_phase(_br())
    rows = {r["region_s"]: r for r in phase.drift_table(est)}
    assert not any(r["identified"] for r in rows.values())             # not even at 2.5 s (hard width ~48 ms)
    assert rows[2.5]["hard_width_ms"] > 40 and rows[40.0]["approx_u_ms"] > 100
    t_le = int(5_000_000_123 + 105 * 99.98e6)
    t_c = t_le + 30_000_000
    assert phase.end_alignment_probe_only(est, t_le, t_c, t_c + 8e6, t_c + 9.35e6)["status"] == "AMBIGUOUS_CYCLE"
    P = 100e6
    t_c = 70_000_000_000
    t_we = t_c - 30_000_000

    def post(offset):
        return [(int(t_c + offset + (j + 2) * P - 1.1e6), int(t_c + offset + (j + 2) * P + 1.1e6)) for j in range(4)]
    ok = phase.end_alignment_from_post_probe(post(0), t_we, t_c, t_c + 8e6, t_c + 9.35e6, t_c + P + 8e6, t_c + P + 9.35e6, 80.0)
    assert ok["status"] == "ALIGNED_UNDER_MODEL" and ok["hard_eps_ns"][0] < 0 < ok["hard_eps_ns"][1]
    lo_J, hi_J = ok["conditional_tail_timing_contribution_J"]
    assert lo_J < 0 < hi_J and "p_ctx from (Cpost - C1)" in ok["conditional_on"]
    late = phase.end_alignment_from_post_probe(post(20_000_000), t_we, t_c, t_c + 8e6, t_c + 9.35e6)
    assert late["status"] == "ALIGNMENT_NOT_ESTABLISHED" and late["checks"]["update_before_c1_read"] is False


@strict
def test_wrapped_phase_with_interval_below_zero_is_ambiguous():
    """Counterexample: a 1 ms start phase whose hard interval extends below zero (the preceding update might be after
    the read): AMBIGUOUS_CYCLE, never OK, even though the interval is far below P/2."""
    br2 = [(int(k * 100e6 - 2.0e6), int(k * 100e6 + 2.0e6)) for k in range(4)]
    e2 = phase.estimate_period_phase(br2)
    t0 = 3 * 100e6 + 1e6
    sp = phase.start_phase(e2, int(t0 - 100_000), int(t0 + 100_000))
    assert sp["status"] == "AMBIGUOUS_CYCLE" and sp["hard_phase_ns"] is None and "approx" in sp
    t1 = 3 * 100e6 + 23e6
    ok = phase.start_phase(e2, int(t1 - 675_000), int(t1 + 675_000))
    assert ok["status"] == "OK_UNDER_MODEL" and ok["hard_phase_ns"][0] > 0 and ok["hard_phase_ns"][0] < 23e6 < ok["hard_phase_ns"][1]


# ---------------------------------------------------------------- readiness: fail-closed numerics
def _trace(temps, pstate=8, foreign=False, gpus=(4, 5), dt=1.0, ce=0):
    return [{"t_s": i * dt, "gpu": g, "temp_c": T, "pstate": pstate, "foreign": foreign, "clock_events": ce}
            for i, T in enumerate(temps) for g in gpus]


@strict
def test_gate_uses_allocated_gpus_current_window_and_never_expiry():
    G = [4, 5]
    cooling = [45 - min(i, 120) * 0.08 for i in range(241)]
    r = rd.gate_over(_trace(cooling), G, 36.0, 240)
    assert r["status"] == "READY" and 30 <= r["t_ready_s"] <= 240
    assert rd.gate_over(_trace([44.0] * 300), G, 36.0, 240)["status"] == "NOT_ACHIEVED"      # expiry is not ready
    assert rd.gate_at(_trace([30.0] * 61), G, 60.0, 36.0)["status"] == "READY"
    assert rd.gate_at(_trace([30.0] * 61, gpus=(4,)), G, 60.0, 36.0)["reasons"] == ["NO_ROWS_GPU5"]   # missing GPU
    stale = _trace([30.0] * 101, gpus=(4,)) + _trace([30.0] * 41, gpus=(5,))
    assert "STALE_GPU5" in rd.gate_at(stale, G, 100.0, 36.0)["reasons"]
    hot_after = _trace([30.0] * 61 + [44.0] * 40)                        # an old ready interval followed by hot rows
    assert rd.gate_at(hot_after, G, 100.0, 36.0)["status"] == "NOT_READY"
    assert rd.gate_at(hot_after, G, 60.0, 36.0)["status"] == "READY"     # (it was ready at 60 s, not now)
    nan_mid = _trace([30.0] * 45 + [float("nan")] + [30.0] * 15)
    assert any(x.startswith("TEMP_INVALID") for x in rd.gate_at(nan_mid, G, 60.0, 36.0)["reasons"])
    none_t = [dict(x, temp_c=None) if x["t_s"] == 50 else x for x in _trace([30.0] * 61)]
    assert rd.gate_at(none_t, G, 60.0, 36.0)["status"] == "NOT_READY"
    gap = [x for x in _trace([30.0] * 61) if not 40 <= x["t_s"] <= 45]
    assert any(x.startswith("WINDOW_NOT_COVERED") for x in rd.gate_at(gap, G, 60.0, 36.0)["reasons"])
    assert rd.gate_at(_trace([30.0] * 61, ce=0x40), G, 60.0, 36.0)["status"] == "NOT_READY"   # clock events
    noce = [{k: v for k, v in x.items() if k != "clock_events"} for x in _trace([30.0] * 61)]
    assert rd.gate_at(noce, G, 60.0, 36.0)["status"] == "NOT_READY"                           # field missing
    unknown = _trace([30.0] * 61, foreign=None)
    assert rd.gate_at(unknown, G, 60.0, 36.0)["status"] == "NOT_READY"                        # unknown != no process
    assert rd.gate_at(_trace([30.0] * 61, pstate=2), G, 60.0, 36.0)["status"] == "NOT_READY"
    extra = _trace([30.0] * 61, gpus=(4, 5, 6))
    assert rd.gate_at(extra, G, 60.0, 36.0)["status"] == "NOT_READY"                         # unallocated GPU rows
    err = _trace([30.0] * 61) + [{"t_s": 55.0, "gpu": 4, "error": True, "foreign": None}]
    assert rd.gate_at(err, G, 60.0, 36.0)["status"] == "NOT_READY"                           # sampler error
    assert rd.gate_at(_trace([30.0] * 61), G, 60.0, float("nan"))["status"] == "NOT_READY"


def _p8_rows(n=900, dt=0.05, entry=3.0, power=25.0):
    return [{"t_s": i * dt, "gpu": g, "pstate": 8 if i * dt >= entry else 2, "power_w": power, "foreign": False,
             "temp_c": 30.0, "clock_events": 0} for i in range(n) for g in (4, 5)]


@strict
def test_p8_baseline_fail_closed_including_nonfinite_power_gaps_and_ownership():
    G = [4, 5]
    r = rd.p8_baseline(_p8_rows(), G)
    assert r["status"] == "OK" and abs(r["entry_s"] - 3.0) < 1e-9 and r["mean_power_W"][4] == 25.0
    assert rd.p8_baseline([dict(x, pstate=2) for x in _p8_rows()], G)["reason"] == "P8_NOT_REACHED_WITHIN_CAP"
    leaves = [dict(x, pstate=2) if abs(x["t_s"] - 20.0) < 1e-9 else x for x in _p8_rows()]
    assert "NOT_P8" in rd.p8_baseline(leaves, G)["reason"]
    nanp = [dict(x, power_w=float("nan")) if abs(x["t_s"] - 30.0) < 1e-9 else x for x in _p8_rows()]
    assert "NONFINITE_POWER" in rd.p8_baseline(nanp, G)["reason"]
    gap = [x for x in _p8_rows() if not 25.0 <= x["t_s"] <= 26.0]
    assert "BASELINE_SAMPLING_GAP" in rd.p8_baseline(gap, G)["reason"]
    unk = [dict(x, foreign=None) if 30 <= x["t_s"] <= 31 else x for x in _p8_rows()]
    assert "FOREIGN_OR_UNKNOWN_PROCESS" in rd.p8_baseline(unk, G)["reason"]
    assert rd.p8_baseline(_p8_rows(entry=70.0, n=2600), G, wait_cap_s=60)["reason"] == "P8_NOT_REACHED_WITHIN_CAP"
    assert rd.p8_baseline([], G)["status"] == "FAILED" and rd.p8_baseline(_p8_rows(), [])["status"] == "FAILED"
    only4 = [x for x in _p8_rows() if x["gpu"] == 4]
    assert rd.p8_baseline(only4, G)["status"] == "FAILED"                      # missing allocated GPU


def _cal(n=21, nan_at=None):
    out = []
    for i in range(n):
        g4 = {"t_launch_c": 36 + i % 3, "t_region_c": 40 + i % 4, "clk_mhz": 1920, "pstate": 2}
        if nan_at == i:
            g4 = dict(g4, t_launch_c=float("nan"))
        out.append({"gpus": {4: g4, 5: {"t_launch_c": 35 + i % 3, "t_region_c": 39 + i % 4, "clk_mhz": 1905, "pstate": 2}}})
    return out


def _wait(slope=0.1, flat_after=60, n_s=121, start=42.0, pstate_from=0, foreign_from=None):
    """A recorded 120 s fixed wait (1 Hz rows for GPUs 4 and 5)."""
    rows = []
    for t in range(n_s):
        T = start - min(t, flat_after) * slope
        for g in (4, 5):
            rows.append({"t_s": float(t), "gpu": g, "temp_c": T, "pstate": 8 if t >= pstate_from else 2,
                         "foreign": True if foreign_from is not None and t >= foreign_from else False, "clock_events": 0})
    return rows


@strict
def test_warning_mechanism_of_the_v2_spec_defect_is_reproduced_and_removed():
    """Packet-v2 spec readiness.derive_heldout_rules called np.quantile on wait times containing +inf: numpy computes
    inf - inf -> NaN with 'invalid value encountered in subtract'. The mechanism, captured on purpose here; the
    corrected rule uses nearest-rank order statistics."""
    waits = np.array([5.0] * 18 + [math.inf] * 3)
    with expect_warning(RuntimeWarning, "invalid value encountered in subtract"):
        q = float(np.quantile(waits, 0.9))
    assert math.isnan(q)
    assert rd.nearest_rank(waits, 0.9) == math.inf
    assert raises(rd.NonFinite, rd.nearest_rank, [1.0, float("nan")], 0.5)


@strict
def test_derivation_replays_the_full_gate_not_the_first_temperature_crossing():
    G = [4, 5]
    fast = [_wait() for _ in range(21)]                    # 42 -> 36 C at 60 s, then flat: full gate at 37 C at 80 s
    r = rd.derive_heldout_rules(_cal(), fast, G)
    assert r["status"] == "OK" and r["t_launch_c"] == 37 and r["q90_time_to_full_gate_s"] == 80.0
    assert rd.time_to_ready(_wait(), G, 37.0) == 80.0     # the first crossing (50 s) is not readiness: stability first
    assert r["bands"][4]["t_region_c"] == [39.0, 44.0]
    crossing_only = [_wait(pstate_from=200) for _ in range(21)]          # temperature crosses, P8 never reached
    nb = rd.derive_heldout_rules(_cal(), crossing_only, G)
    assert nb["status"] == "NOT_ACHIEVABLE" and nb["t_launch_c"] is None
    unstable = [_wait(slope=0.05, flat_after=120) for _ in range(21)]    # still falling 1.5 C per 30 s: never stable
    assert rd.derive_heldout_rules(_cal(), unstable, G)["status"] == "NOT_ACHIEVABLE"
    foreign = [_wait(foreign_from=10) for _ in range(21)]
    assert rd.derive_heldout_rules(_cal(), foreign, G)["status"] == "NOT_ACHIEVABLE"
    three = [_wait(start=60.0, flat_after=0) for _ in range(3)] + [_wait() for _ in range(18)]   # 3 of 21 censored
    t3 = rd.derive_heldout_rules(_cal(), three, G)
    assert t3["tried"][0]["q90_time_to_full_gate_s"] == math.inf
    assert rd.derive_heldout_rules(_cal(nan_at=5), fast, G)["status"] == "NOT_DERIVABLE"
    assert rd.derive_heldout_rules(_cal(n=14), fast, G)["reason"].startswith("INSUFFICIENT_CALIBRATION_LAUNCHES")
    assert rd.derive_heldout_rules(_cal(), fast[:10], G)["reason"].startswith("INSUFFICIENT_WAIT_TRACES")
    assert rd.derive_heldout_rules(_cal(), fast, [4, 6])["reason"] == "GPU_SET_DIFFERS_FROM_ALLOCATION"


@strict
def test_pre_region_check_and_per_gpu_contrast_balance_fail_closed():
    bands = {4: {"t_region_c": [38, 45], "clk_mhz": [1880, 1950], "pstates": [2]}}
    good = {4: {"t_region_c": 41.0, "clk_mhz": 1920, "pstate": 2, "clock_events": 0, "inst_age_s": 0.3}}
    assert rd.pre_region_check(good, bands, [4])["ok"]
    bad = {4: {"t_region_c": 47.0, "clk_mhz": 1800, "pstate": 0, "clock_events": 0x40, "inst_age_s": 2.0}}
    assert len(rd.pre_region_check(bad, bands, [4])["reasons"]) == 5
    assert rd.pre_region_check({4: dict(good[4], t_region_c=float("nan"))}, bands, [4])["reasons"] == \
        ["NONFINITE_PRE_REGION_STATE_GPU4"]
    assert not rd.pre_region_check({4: dict(good[4], clock_events=None)}, bands, [4])["ok"]
    assert not rd.pre_region_check(good, bands, [4, 5])["ok"]                       # missing allocated GPU
    mk = lambda a, t4, t5: {"session": 1, "length_s": 10.0, "arm": a, "t_region_gpu": {4: t4, 5: t5}}  # noqa: E731
    units = [mk("A", 40.0, 40.0), mk("B", 40.8, 39.2), mk("C", 40.1, 40.2)]          # B balanced on the mean only
    bal = rd.thermal_balance(units, [4, 5])
    assert bal[(1, 10.0, "B-A")]["ok"] is False and bal[(1, 10.0, "C-A")]["ok"] is True
    assert rd.thermal_balance([mk("A", 40.0, 40.0), mk("B", 40.1, 40.1)], [4, 5])[(1, 10.0, "C-A")]["reason"] == "ARM_MISSING"
    assert rd.thermal_balance([mk("A", 40.0, float("nan")), mk("B", 40.1, 40.1)], [4, 5])[(1, 10.0, "B-A")]["ok"] is False


# ---------------------------------------------------------------- budget and deadline
@strict
def test_budget_names_estimates_honestly_and_gate_reserves_cleanup():
    t = budget.budget_table()
    for part in ("calibration", "heldout_session"):
        assert abs(t[part]["expected_s"] - sum(t[part]["components"].values())) < 1e-9
        assert "worst_valid_s" not in t[part] and "cap_based_planning_estimate_s" in t[part]
    assert abs(t["calibration"]["expected_s"] - (3524.0 + 23 * 0.4)) < 0.1          # M4d adds 0.4 s per launch
    assert abs(t["calibration"]["cap_based_planning_estimate_s"] - (3634.0 + 23 * 0.4)) < 0.1
    assert abs(t["heldout_session"]["expected_s"] - 3774.4) < 0.1
    assert abs(t["heldout_session"]["cap_based_planning_estimate_s"] - 6764.4) < 0.1
    assert abs(t["heldout_session"]["floor_s"] - 1614.4) < 0.1
    assert raises(ValueError, budget.budget_table, None, budget.Design(post_probe_s=0.4))   # M4d never in held-out
    d = budget.Design()
    need = budget.unit_bound_s(40.0, d, True) + budget.end_baseline_bound_s(d) + budget.session_tail_reserved_s(d)
    assert budget.can_start_unit(need, 40.0, d, True) and not budget.can_start_unit(need - 1, 40.0, d, True)
    now = [1000.0]
    dl = budget.Deadline(need + 10, clock=lambda: now[0])
    assert dl.can_start_unit(40.0, d, True)
    now[0] += 11
    assert not dl.can_start_unit(40.0, d, True)
    now[0] = 1000.0 + need + 10 - 80
    assert dl.unit_wall_limit_s(40.0, d) == 80 - budget.session_tail_reserved_s(d)   # never past cleanup + verify
    assert raises(ValueError, budget.Deadline, 0.0)


# ---------------------------------------------------------------- decisions: difference vs equivalence, Holm
@strict
def test_nonfinite_missing_insufficient_and_zero_variance_never_yield_a_verdict():
    for bad in (float("nan"), float("inf"), None):
        assert raises(dec.NonFiniteContrast, dec.session_summary, {1: [0.001], 2: [bad], 3: [0.0]})
    two = dec.session_summary({1: [0.001], 2: [0.002]})
    assert two["status"] == "INSUFFICIENT_SESSIONS" and "ci95" not in two
    flat = dec.session_summary({1: [0.001], 2: [0.001], 3: [0.001, 0.001]})
    assert flat["status"] == "DEGENERATE_ZERO_VARIANCE" and "ci90" not in flat
    rounding = dec.session_summary({1: [0.1], 2: [0.1], 3: [0.3 - 0.2]})       # differ only by float rounding
    assert rounding["status"] == "DEGENERATE_ZERO_VARIANCE"
    fam = dec.family({10.0: flat, 20.0: two}, dec.D3)
    assert fam[10.0]["verdict"] == "DEGENERATE_ZERO_VARIANCE" and fam[20.0]["verdict"] == "INSUFFICIENT_SESSIONS"
    assert raises(dec.NonFiniteContrast, dec.holm, {"a": float("nan")})
    assert raises(dec.NonFiniteContrast, dec.holm, {"a": 1.5})


def _s(vals):
    return dec.session_summary({i + 1: [v] for i, v in enumerate(vals)})


@strict
def test_hd3_decisions_use_holm_and_unadjusted_intervals_are_descriptive():
    x10 = _s([0.002, 0.0026, 0.0014])          # unadjusted p = 0.029, 95 % CI excludes 0
    assert x10["p_two_sided"] < 0.05 and x10["ci95"][0] > 0
    alone = dec.family({10.0: x10}, dec.D3)[10.0]
    assert alone["verdict"] == "DETECTED_WITHIN_MARGIN"
    fam = dec.family({10.0: x10, 20.0: _s([0.0, 0.003, -0.003]), 40.0: _s([0.001, -0.002, 0.001])}, dec.D3)
    e = fam[10.0]
    assert abs(e["difference"]["p_holm"] - 3 * x10["p_two_sided"]) < 1e-12 and not e["decision"]["difference_detected"]
    assert e["difference"]["ci95_label"] == "descriptive (unadjusted)"
    assert e["verdict"] == "COMPATIBLE_WITH_EQUIVALENCE"            # equivalence evidence, no adjusted difference
    big = dec.family({10.0: _s([0.02, 0.021, 0.019])}, dec.D3)[10.0]
    assert big["verdict"] == "RESOLVED_MEANINGFUL_DIFFERENCE" and not big["decision"]["equivalent"]
    wide = dec.family({10.0: _s([0.01, -0.01, 0.0])}, dec.D3)[10.0]
    assert wide["verdict"] == "INSUFFICIENT_PRECISION" and wide["equivalence"]["within_margin"] is False
    h = dec.holm({"10": 0.01, "20": 0.04, "40": 0.03})
    assert abs(h["10"] - 0.03) < 1e-12 and abs(h["40"] - 0.06) < 1e-12 and abs(h["20"] - 0.06) < 1e-12
    shrink = dec.family({"a": _s([0.002, 0.0026, 0.0014]), "b": dec.session_summary({1: [0.1], 2: [0.1], 3: [0.1]}),
                         "c": _s([0.0, 0.003, -0.003])}, dec.D3)
    assert abs(shrink["a"]["difference"]["p_holm"] - 3 * x10["p_two_sided"]) < 1e-12   # family size never shrinks
    assert dec.family({10.0: x10}, dec.D3, overrides={10.0: "THERMAL_COMPARABILITY_NOT_ESTABLISHED"})[10.0]["verdict"] \
        == "THERMAL_COMPARABILITY_NOT_ESTABLISHED"


def _q1_round(d100_counter=0.99, d100_sampled=1.0, d500_counter=1.0, d500_sampled=1.0, j=0.0):
    base = 1e-3
    return {"E0": {"counter": base, "sampled": base * 1.01},
            "D100": {"counter": base * d100_counter * (1 + j), "sampled": base * 1.01 * d100_sampled},
            "D500": {"counter": base * d500_counter * (1 + j / 2), "sampled": base * 1.01 * d500_sampled}}


@strict
def test_q1_separate_doses_difference_and_bound_and_nonsignificance_is_not_absence():
    rounds = [_q1_round(j=0.0002), _q1_round(j=-0.0003), _q1_round(j=0.0001)]
    rounds = [dict(r, D500={"counter": r["D500"]["counter"] * (1 + 0.0004), "sampled": r["D500"]["sampled"]})
              for r in rounds]                                   # opposite sign to D100: allowed, no sign rule
    q = dec.q1(rounds)
    assert "probe" in q["scope_note"]
    d100, d500 = q["doses"]["D100"], q["doses"]["D500"]
    assert d100["counter"]["class"] == "DETECTED" and not d100["bounded_for_adequacy"]
    assert d100["interpretation"].startswith("COUNTER_SPECIFIC")
    assert d500["counter"]["class"] in ("BOUNDED_WITHIN_MARGIN", "DETECTED_AND_BOUNDED") and d500["bounded_for_adequacy"]
    assert d100["counter"]["summary"]["mean"] < 0 < d500["counter"]["summary"]["mean"]
    noisy = [_q1_round(d100_counter=1.0, j=0.01), _q1_round(d100_counter=1.0, j=-0.012), _q1_round(d100_counter=1.0, j=0.0)]
    qn = dec.q1(noisy)["doses"]["D100"]
    assert qn["counter"]["class"] == "INCONCLUSIVE" and not qn["bounded_for_adequacy"]   # not significant != absent
    missing = [dict(rounds[0], D100=None), rounds[1], rounds[2]]
    assert dec.q1(missing)["doses"]["D100"]["counter"]["class"] == "INSUFFICIENT_ROUNDS"
    flat = [_q1_round(), _q1_round(), _q1_round()]
    assert dec.q1(flat)["doses"]["D100"]["counter"]["class"] == "DEGENERATE_ZERO_VARIANCE"
    bad = [dict(rounds[0], E0={"counter": float("nan"), "sampled": 1e-3}), rounds[1], rounds[2]]
    assert raises(dec.NonFiniteContrast, dec.q1, bad)


def _eqfam(keys, vals=(0.0004, -0.0002, 0.0001), delta=dec.D3):
    return dec.family({k: _s(list(vals)) for k in keys}, delta)


def _adequacy_inputs(**over):
    tight = [0.0004, -0.0002, 0.0001]
    q = dec.q1([_q1_round(d100_counter=1.0, d500_counter=1.0, j=x) for x in (0.0002, -0.0001, 0.0)])
    bounded = {"status": "BOUNDED", "bound_rel": 0.0005, "basis": "test (hypothetical hard bounds)"}
    sysb = {c: {L: bounded for L in (2.5, 10.0, 20.0, 40.0)} for c in dec.SYSTEMATIC_COMPONENTS}
    r = {"HD1": _eqfam([(a, L) for a in "ABC" for L in dec.HD1_LENGTHS], tight, dec.D1_D2),
         "HD2": _eqfam("C", tight, dec.D1_D2), "HD3": _eqfam((10.0, 20.0, 40.0), tight),
         "HD4_precision": dec.precision(_s(tight)),
         "HD3_precision": {L: dec.precision(_s(tight)) for L in (10.0, 20.0, 40.0)}, "q1": q, "systematic": sysb,
         "r3_ok": {L: True for L in (2.5, 10.0, 20.0, 40.0)}, "control_valid": True}
    r.update(over)
    return r


@strict
def test_scope_specific_adequacy_and_unknown_systematics_are_not_zero():
    ok = _adequacy_inputs()
    assert dec.adequacy("BROAD", ok)["status"] == "ESTABLISHED"          # only if every component had a hard bound
    hd1 = dict(ok["HD1"])
    hd1[("A", 10.0)] = dec.family({("A", 10.0): _s([0.01, -0.01, 0.0])}, dec.D1_D2)[("A", 10.0)]   # a middle length
    b = dec.adequacy("BROAD", _adequacy_inputs(HD1=hd1))
    assert b["status"] == "NOT_ESTABLISHED" and b["unmet"] == ["H-D1(A, 10 s vs 40 s) equivalence"]
    r = _adequacy_inputs(HD1={}, HD2=dec.family({"C": _s([0.01, -0.01, 0.0])}, dec.D1_D2))
    assert dec.adequacy("SAME_EXEC_L40", r)["status"] == "ESTABLISHED"      # narrow scope does not need H-D1/H-D2
    cp = dec.adequacy("CROSS_PROTOCOL_40", r)
    assert cp["status"] == "NOT_ESTABLISHED" and cp["unmet"] == ["H-D2 equivalence"]
    real = {"end_alignment_timing": {L: {"status": "CONDITIONAL"} for L in (2.5, 10.0, 20.0, 40.0)}}  # realistic
    rr = dec.adequacy("SAME_EXEC_L40", _adequacy_inputs(systematic=real))
    assert rr["status"] == "NOT_ESTABLISHED" and rr["conditional_statistical"] == "MET"
    assert rr["unmet_systematic"] and not rr["unmet_statistical"]
    st = dec.systematic_total(real, 40.0)
    assert st["status"] == "UNKNOWN" and st["conditional_only"] == ["end_alignment_timing"] and len(st["unknown"]) == 5
    nobasis = {c: {40.0: {"status": "BOUNDED", "bound_rel": 0.0005}} for c in dec.SYSTEMATIC_COMPONENTS}
    assert dec.systematic_total(nobasis, 40.0)["status"] == "UNKNOWN"   # a number without a basis is not a bound
    flatq = dec.q1([_q1_round(), _q1_round(), _q1_round()])
    assert "Q1 bounded (D100 and D500)" in dec.adequacy("SAME_EXEC_L10", _adequacy_inputs(q1=flatq))["unmet"]
    degen = _adequacy_inputs(HD3=dec.family({L: dec.session_summary({1: [0.001], 2: [0.001], 3: [0.001]})
                                            for L in (10.0, 20.0, 40.0)}, dec.D3))
    assert dec.adequacy("SAME_EXEC_L10", degen)["status"] == "NOT_ESTABLISHED"
    assert raises(ValueError, dec.adequacy, "SAME_EXEC_L2.5", ok)
    assert raises(ValueError, dec.adequacy, "EVERYTHING", ok)


@strict
def test_historical_replay_reproduces_committed_decisions():
    camp = CLUSTER_REPO / "data/v2.2/raw/v2.2-measval-01"
    if not camp.exists():
        print("SKIP: complete historical data not available")
        return
    mod = dec.load_frozen_measval(CLUSTER_REPO)
    res = dec.replay(mod, camp, json.loads((CLUSTER_REPO / "configs/v2.2/measval01_frozen.json").read_text()))
    committed = json.loads((CLUSTER_REPO / "data/v2.2/processed/v2.2-measval-01/decisions.json").read_text())
    assert (res["n_processes"], res["n_accepted"]) == (72, 70)
    for k in ("D1", "D2", "D3", "D4"):
        assert res[k]["decision"] == committed[k]["decision"]


# ---------------------------------------------------------------- schedules and inventory
@strict
def test_calibration_schedule_dose_and_m4d_only_in_calibration():
    s = sched.calibration_schedule()
    assert len(s) == 23 and sched.poller_active_launches(s) == 7
    c2 = [u for u in s if u["part"].startswith("C2")]
    pos = {c: sorted(i % 3 for i, u in enumerate(c2) if u["dose"] == c) for c in sched.DOSES}
    assert all(p == [0, 1, 2] for p in pos.values())
    assert all(isinstance(u["pause_ns"], int) and 0 <= u["pause_ns"] < 100_000_000 for u in s)
    assert all(u["post_probe"] is True for u in s) and s == sched.calibration_schedule()
    assert not any(u["post_probe"] for u in sched.calibration_schedule(m4d=False))
    n = {x["call"]: x["n"] for x in sched.per_gpu_calls(10.0, "D100")}
    assert n["dose poller D100"] == 100
    assert {x["call"]: x["n"] for x in sched.per_gpu_calls(10.0, "D500")}["dose poller D500"] == 20
    assert not any(x["in_region"] is True for x in sched.per_gpu_calls(40.0, "E0", m4d=True))


@strict
def test_heldout_schedule_constraints_and_no_dose_fields():
    h = sched.heldout_schedule(n_candidates=400)
    plan = h["plan"]
    assert len(plan) == 72 and not any(k in u for u in plan for k in ("dose", "post_probe", "dose_period_ms"))
    for s in (1, 2, 3):
        for b in (1, 2):
            blk = [u for u in plan if u["session"] == s and u["block"] == b]
            assert sorted((u["arm"], u["length_s"]) for u in blk) == sorted((a, L) for a in "ABC" for L in sched.LENGTHS)
    cnt = {}
    for u in plan:
        cnt[(u["arm"], u["length_s"], u["triple"])] = cnt.get((u["arm"], u["length_s"], u["triple"]), 0) + 1
    assert max(cnt.values()) <= 2 and h == sched.heldout_schedule(n_candidates=400)


@strict
def test_schedules_reproduce_the_packet_v2_drafts():
    cal_p, ho_p = PREP / "drafts/calib01_schedule_DRAFT.json", PREP / "drafts/heldout_schedule_DRAFT.json"
    if not cal_p.exists():
        print("SKIP: packet-v2 drafts not available")
        return
    draft = json.loads(cal_p.read_text())["plan"]
    s = sched.calibration_schedule()
    assert [(u["launch_id"], u["length_s"], u["dose"]) for u in s] == [(u["launch_id"], u["length_s"], u["dose"]) for u in draft]
    assert all(abs(a["pause_ns"] / 1e6 - b["pause_ms"]) < 1e-6 for a, b in zip(s, draft))
    hd = json.loads(ho_p.read_text())
    h = sched.heldout_schedule()
    keys = ("session", "block", "pos", "triple", "arm", "length_s", "launch_id")
    assert [tuple(u[k] for k in keys) for u in h["plan"]] == [tuple(u[k] for k in keys) for u in hd["plan"]]
