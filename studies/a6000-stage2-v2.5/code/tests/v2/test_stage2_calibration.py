"""Stage 2 calibration analysis (K1): known-answer rules and an end-to-end pass over a fake calibration session
(CPU only; the session itself runs in a subprocess with the fake NVML, `python -W error`)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

from ncclenergy.v2 import stage2_calibration as K  # noqa: E402
from ncclenergy.v2.common import REPO  # noqa: E402

P = 100_000_000


def _m4c_rec(i, dev_ms=0.1, temp=40.0, clk=1900.0):
    req = (i * 4_300_000) % 100_000_000
    act = req + int(dev_ms * 1e6)
    return {"gpus": [4, 5], "s2": {"per_gpu": [{"rank": r, "pause_dev_ms": dev_ms, "pause_act_ns": act} for r in (0, 1)]},
            "pre_region_state": {"4": {"t_region_c": temp, "clk_mhz": clk}, "5": {"t_region_c": temp - 1, "clk_mhz": clk}}}


@strict
def test_m4c_decision_rule():
    ok = [_m4c_rec(i, temp=40.0 + (i % 3) * 0.1) for i in range(23)]
    assert K.m4c_decision(ok)["decision"] == "RETAINED"
    late = [_m4c_rec(i, dev_ms=3.0 if i < 2 else 0.1) for i in range(23)]
    r = K.m4c_decision(late)
    assert r["decision"] == "AMENDMENT_PROPOSED" and r["launches_dev_gt_2ms"] == 2
    one = [_m4c_rec(i, dev_ms=3.0 if i < 1 else 0.1) for i in range(23)]
    assert K.m4c_decision(one)["decision"] == "RETAINED"                  # at most 1 launch may deviate
    assoc = []
    for i in range(23):
        x = _m4c_rec(i)
        t = x["s2"]["per_gpu"][0]["pause_act_ns"] / 1e9
        x["pre_region_state"] = {"4": {"t_region_c": 40 + 10 * t, "clk_mhz": 1900}, "5": {"t_region_c": 39 + 10 * t, "clk_mhz": 1900}}
        assoc.append(x)
    a = K.m4c_decision(assoc)
    assert a["decision"] == "AMENDMENT_PROPOSED" and abs(a["association"]["temp_c"]["effect_over_100ms"] - 1.0) < 1e-6


def _align_rec(i, offset_ns=0, L=10.0):
    t_c = 50_000_000_000 + i * 10**10
    raw = {r: {"t_loop_end_ns": t_c - 28_000_000, "t_commit_ns": t_c, "t_end_a_ns": t_c + 8_000_000,
               "t_end_b_ns": t_c + 9_350_000, "t_post_a_ns": t_c + P + 8_000_000, "t_post_b_ns": t_c + P + 9_350_000}
           for r in (0, 1)}
    br = [[t_c + offset_ns + (j + 2) * P - 1_100_000, t_c + offset_ns + (j + 2) * P + 1_100_000] for j in range(4)]
    uid = f"u{i}"
    rec = {"unit_id": uid, "accepted": True, "gpus": [4, 5], "unit": {"post_probe": True, "length_s": L},
           "s2": {"per_gpu": [{"rank": r, "t_work_end_ns": t_c - 30_000_000, "post_probe": {"brackets": br}} for r in (0, 1)]},
           "region": {"per_gpu": [{"rank": r} for r in (0, 1)]}, "_raw_region_gpus": raw}
    en = {"per_gpu": [{"rank": r, "aligned_nom": {"p_ctx_W": 80.0, "energy_J": 1500.0},
                       "aligned_hat": {"p_ctx_W": 80.0, "energy_J": 1500.0}} for r in (0, 1)]}
    return rec, en


@strict
def test_end_alignment_is_a_conditional_timing_contribution_never_a_bound():
    recs, ens = zip(*[_align_rec(i) for i in range(21)])
    e = {r["unit_id"]: x for r, x in zip(recs, ens)}
    b = K.end_alignment_timing(list(recs), e)
    assert b["status"] == "CONDITIONAL" and "not a bound" in b["label"]
    assert b["by_length"][10.0]["status"] == "CONDITIONAL" and 0 < b["by_length"][10.0]["observed_max_rel"] < 1e-3
    assert b["by_length"][40.0]["status"] == "UNKNOWN"                       # no launch at that length
    from ncclenergy.v2 import stage2_decisions as dec
    budget = {"end_alignment_timing": b["by_length"]}
    assert dec.systematic_total(budget, 10.0)["status"] == "UNKNOWN"         # conditional is not a bound
    bad = list(recs)
    r2, _ = _align_rec(99, offset_ns=20_000_000)                              # update after the C1 read
    bad.append(r2)
    e[r2["unit_id"]] = ens[0]
    u = K.end_alignment_timing(bad, e)
    assert u["status"] == "ALIGNMENT_NOT_ESTABLISHED" and u["problems"][0][2] == "ALIGNMENT_NOT_ESTABLISHED"
    few = K.end_alignment_timing(list(recs[:10]), e)
    assert few["status"] == "ALIGNMENT_NOT_ESTABLISHED"


def _slope_recs(T_of_i, noise=None, n_rounds=3):
    import math
    recs, energies = [], {}
    rng = __import__("numpy").random.default_rng(3)
    for i, L in enumerate([2.5, 10.0, 20.0, 40.0] * n_rounds):
        T = T_of_i(i)
        uid = f"u{i}"
        recs.append({"unit_id": uid, "accepted": True, "gpus": [4, 5],
                     "unit": {"part": f"C1r{i // 4 + 1}", "length_s": L, "pos": i},
                     "pre_region_state": {"4": {"t_region_c": T}, "5": {"t_region_c": T}}})
        eps = 0.0 if noise is None else float(rng.normal(0, noise))
        energies[uid] = {"E_aligned_hat_J": 100.0 * L * math.exp(0.004 * T + eps), "R": 1000}
    return recs, energies


@strict
def test_thermal_slope_is_an_association_and_refuses_degenerate_or_confounded_fits():
    T_alt = lambda i: 40.0 + (0.6 if (i * 7) % 3 == 0 else -0.4) + (i % 4) * 0.05   # noqa: E731  not ordered in time
    recs, en = _slope_recs(T_alt, noise=0.0005)
    r = K.thermal_slope(recs, en)
    assert r["status"] == "ASSOCIATION_ONLY" and "not a bound" in r["label"] and r["ci95"][0] < 0.004 < r["ci95"][1]
    exact, en0 = _slope_recs(T_alt, noise=None)                              # zero residual variance
    assert K.thermal_slope(exact, en0)["status"] == "NOT_ESTIMABLE"
    drift, enD = _slope_recs(lambda i: 40.0 + 0.3 * i, noise=0.0005)         # exactly round + length: rank deficient
    assert K.thermal_slope(drift, enD)["status"] == "NOT_ESTIMABLE"
    jit = [0.03, -0.02, 0.01, 0.0, -0.03, 0.02, 0.0, 0.01, -0.01, 0.02, -0.02, 0.0]
    near, enN = _slope_recs(lambda i: 40.0 + 0.3 * i + jit[i], noise=0.0005)  # temperature ~ launch order
    c = K.thermal_slope(near, enN)
    assert c["status"] == "CONFOUNDED" and abs(c["corr_T_launch_order"]) > 0.99
    few, enF = _slope_recs(T_alt, noise=0.0005, n_rounds=2)
    assert K.thermal_slope(few, enF)["status"] == "NOT_ESTIMABLE"


@strict
def test_calibration_analysis_runs_on_a_fake_session_and_reports_insufficiency():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as tmp:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(("FAKE_", "NCCLE_"))}
        env["PYTHONPATH"] = f"{REPO / 'tests/v2/fake_nvml'}:{REPO / 'src'}"
        p = subprocess.run([sys.executable, "-W", "error", str(REPO / "tests/v2/helpers/stage2_session.py"),
                            "calibration", tmp], env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-3000:]
        res = K.analyse(Path(tmp) / "data/calibration", REPO)
    assert res["n_accepted"] == 3 and not res["excluded"]
    e = res["energies"]["S0a-01"]
    assert e["E_aligned_hat_J"] is not None and e["E_samp_inst_J"] is not None and e["E_counter_raw_J"] > e["E_aligned_hat_J"]
    assert res["q1_primary"]["doses"]["D100"]["counter"]["class"] == "INSUFFICIENT_ROUNDS"
    assert res["q3_readiness"]["status"] == "NOT_DERIVABLE"
    assert res["end_alignment_timing"]["status"] == "ALIGNMENT_NOT_ESTABLISHED"     # 3 launches < 18 per GPU
    assert res["thermal_slope"]["status"] == "NOT_ESTIMABLE"
    assert e["per_gpu"][0]["query_bracket_ms"]["median"] is not None
    assert "not established as acquisition time" in e["per_gpu"][0]["field_timestamp_us"]["label"]
    assert res["m4c"]["decision"] in ("RETAINED", "AMENDMENT_PROPOSED")
    json.dumps(res)


@strict
def test_q2_channel_description_known_answer():
    import numpy as np
    t = np.arange(0, 10, 0.05)
    v = np.floor(t / 0.5) * 3.0 + 100.0               # value changes every 500 ms
    ts = (np.floor(t / 0.5) * 500_000).astype(float)   # field timestamp advances only with the value
    r = K.q2_channels(t, v, ts)
    assert abs(r["change_interval_ms_median"] - 500) < 1e-6 and abs(r["repeated_fraction"] - 0.9) < 0.01
    assert r["run_length_median"] == 10 and r["field_ts_advances_on_repeated_value_frac"] == 0.0
    assert K.q2_channels(t[:2], v[:2], ts[:2])["status"] == "TOO_FEW_ROWS"


@strict
def test_q4_secondary_estimator_only_if_delay_interval_excludes_zero():
    def u(d):          # P_region 150 W, P_before 80 W: gap = d * 70 W
        return {"gpus": [{"e_aligned_J": 1500.0, "e_samp_J": 1500.0 - d * 70.0, "duration_s": 10.0, "p_before_W": 80.0}]}
    add = K.q4_delay([u(0.19), u(0.2), u(0.21), u(0.2)])
    assert add["decision"] == "ADD_SECONDARY_DELAY_SHIFTED_ESTIMATOR" and abs(add["d_eff_s"]["mean"] - 0.2) < 1e-9
    assert K.q4_delay([u(0.2), u(-0.2), u(0.01)])["decision"] == "NO_SECONDARY_ESTIMATOR"
    assert K.q4_delay([u(0.2), u(0.2)])["decision"] == "NO_SECONDARY_ESTIMATOR"            # insufficient launches
    flat = K.q4_delay([u(0.2), u(0.2), u(0.2)])
    assert flat["decision"] == "NO_SECONDARY_ESTIMATOR" and flat["d_eff_s"]["status"] == "DEGENERATE_ZERO_VARIANCE"
