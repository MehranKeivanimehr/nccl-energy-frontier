"""Pre-specified gate 3b decision logic (measval.py) on synthetic block data with known answers (CPU only)."""

import math

import numpy as np
import pandas as pd

from ncclenergy.v2.measval import LENGTHS, control_status, d1, d2, d3, d4

T_OP = {"A": 52.3e-6, "B": 52.3e-6, "C": 50.8e-6}


def synth(n_blocks=6, c_effect=1.012, b_effect=1.0, intercept_J=1.0, d_protocol=0.0, sd=0.0007, seed=1,
          bad_exec_blocks=()):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(1, n_blocks + 1):
        blk = rng.normal(0, sd)
        for arm in ("A", "B", "C"):
            for L in LENGTHS:
                R = math.ceil(L / T_OP[arm])
                jop = 13.4e-3 * {"A": 1.0, "B": b_effect, "C": c_effect}[arm] * math.exp(blk + rng.normal(0, sd))
                E = R * jop + intercept_J
                T = R * T_OP[arm]
                d = -0.02 - 2.0 / (T * 270.0) + (d_protocol if arm == "C" else 0.0) + rng.normal(0, 0.0005)
                ex = [("RING", "LL128" if arm == "C" else "SIMPLE", 4, 20 if arm == "C" else 17)]
                if arm == "B" and b in bad_exec_blocks:
                    ex = [("RING", "SIMPLE", 2, 17)]
                rows.append({"block": b, "arm": arm, "length_s": L, "R": R, "T_region_s": T, "E_counter": E,
                             "E_inst": E * math.exp(d), "E_lag": E * math.exp(d / 2), "exec": ex})
    return pd.DataFrame(rows)


def run(df):
    ctrl = control_status(df)
    r1, r2, r3 = d1(df), d2(df), d3(df, ctrl)
    return ctrl, r1, r2, r3, d4(df, r1, r2, r3)


def test_clean_case_passes_and_is_sufficient():
    ctrl, r1, r2, r3, r4 = run(synth())
    assert ctrl["status"] == "ESTABLISHED"
    assert r1["decision"] == "PASS" and r2["decision"] == "PASS" and r3["decision"] == "PASS", (r1["decision"], r2["decision"], r3["decision"])
    assert r4["decision"] == "SUFFICIENT"
    ca = r4["C_over_A_descriptive"][40.0]["counter"]
    assert abs(math.exp(ca["mean"]) - 1.012) < 0.002


def test_boundary_offset_is_a_length_effect():
    _, r1, _, _, r4 = run(synth(intercept_J=15.0))          # 15 J per region ~ 2 % at 2.5 s
    assert r1["decision"] == "LENGTH_EFFECT" and r4["decision"] == "NOT_SUFFICIENT"
    assert r1["per_config"]["A"]["intercept_J"]["mean"] > 10


def test_protocol_dependent_estimator_gap():
    _, _, r2, _, r4 = run(synth(d_protocol=-0.02))
    assert r2["decision"] == "PROTOCOL_DEPENDENT" and r4["decision"] == "NOT_SUFFICIENT"


def test_control_offset_and_not_established():
    _, _, _, r3, _ = run(synth(b_effect=1.01))
    assert r3["decision"] == "SYSTEMATIC_OFFSET"
    ctrl, _, _, r3, _ = run(synth(bad_exec_blocks=(2, 5)))
    assert ctrl["status"] == "CONTROL_NOT_ESTABLISHED" and r3["decision"].startswith("INCONCLUSIVE")
    ctrl, _, _, r3, _ = run(synth(bad_exec_blocks=(3,)))
    assert ctrl["status"] == "ESTABLISHED" and ctrl["failing_blocks"] == [3]
    assert r3["per_length"][40.0]["counter"]["n"] == 5                # the failing block is excluded from B/A


def test_noise_too_large_is_inconclusive_and_too_few_blocks():
    _, r1, _, r3, r4 = run(synth(sd=0.01, seed=3))
    assert r4["decision"] == "NOT_SUFFICIENT"
    assert r1["decision"] in ("INCONCLUSIVE", "LENGTH_EFFECT", "PASS") and r3["decision"] != "PASS"
    _, r1, r2, r3, r4 = run(synth(n_blocks=2))
    assert r1["decision"] == "INCONCLUSIVE" and r2["decision"] == "INCONCLUSIVE"
    assert r3["decision"] == "INCONCLUSIVE_INSUFFICIENT_BLOCKS" and r4["decision"] == "NOT_SUFFICIENT"


def test_measval_config_units_match_frozen_schedule():
    """Every unit run_block plans from configs/v2.2/measval01.yaml has a frozen entry, and vice versa."""
    import json
    from pathlib import Path

    import yaml

    from ncclenergy.v2.runner import unit_key
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "configs/v2.2/measval01.yaml").read_text())
    fr = json.loads((root / cfg["frozen_file"]).read_text())
    uids = [unit_key({"placement": p["id"], "coll": c, "config": lab, "size": s, "length_s": L})
            for p in cfg["placements"] for c in cfg["collectives"] for lab in cfg["configs"]
            for s in cfg["sizes"]["list"] for L in cfg["region_lengths_s"]]
    assert len(uids) == 12 == len(set(uids)) and set(uids) == set(fr["entries"])
    assert cfg["allocation"] == [4, 5] and cfg["max_attempts"] == 1 and cfg["max_blocks"] == 6


def test_descriptive_runs_on_a_frame_with_a_flags_column():
    """Regression (gate 3b): `df.flags` is pandas' DataFrame.flags attribute, not the 'flags' column."""
    import tempfile
    from pathlib import Path

    from ncclenergy.v2.measval import descriptive
    df = synth().assign(tail_J=0.1, A1_J=0.001, A2_J=0.001, B1_J=0.001, period_ms_median=100.0,
                        flags=["", "SAMPLE_GAP"] + [""] * 70)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "blocks").mkdir()
        out = descriptive(df, df, Path(tmp))
    assert out["flags"] == ["SAMPLE_GAP"] and len(out["by_arm_length"]) == 12 and out["idle_baselines"] == []


def test_posthoc_temperature_slope_recovery_and_adjustment():
    """measval_posthoc (exploratory): a known 0.4 %/C thermal effect riding on execution position is recovered, and
    adjusting for it removes the spurious B/A spread."""
    from ncclenergy.v2.measval_posthoc import analyse
    rng = np.random.default_rng(7)
    d = synth(b_effect=1.0, sd=0.0002)
    d["exec_pos"] = 0
    for b in d.block.unique():
        idx = d.index[d.block == b]
        d.loc[idx, "exec_pos"] = rng.permutation(len(idx)) + 1
    d["temp_c"] = 35.0 + d.exec_pos + rng.normal(0, 0.3, len(d))
    d["E_counter"] = d.E_counter * np.exp(0.004 * (d.temp_c - 40.0))
    d = d.assign(power_W=d.E_counter / d.T_region_s, us_per_op=1e6 * d.T_region_s / d.R, clk_gr_mhz=1900.0,
                 y=np.log(d.E_counter / d.R))
    r = analyse(d[d.length_s >= 10].reset_index(drop=True))
    assert abs(r["fit_all_lengths"]["temp_slope_ln_per_c"] - 0.004) < 0.0003
    ba = r["contrasts"]["D3_counter_B_over_A"][40.0]
    assert ba["temp_adjusted"]["half_width"] < ba["raw"]["half_width"]
