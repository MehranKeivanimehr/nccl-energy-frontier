"""CPU-only tests for the v2 block-level statistics."""

import math

import numpy as np
import pandas as pd

from ncclenergy.v2.analysis import choose_policies, collapse_effective, holm, paired_test, required_blocks, tost


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, 0.20]
    adj = holm(p)
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.20])
    assert all(a >= b for a, b in zip(adj, p))


def test_paired_test_detects_shift_and_reports_blocks():
    d = np.log(np.array([0.93, 0.94, 0.95, 0.94, 0.92, 0.95]))
    r = paired_test(d, "less")
    assert r["n"] == 6 and r["p"] < 0.001 and r["ci_hi"] < 0 and len(r["per_block"]) == 6
    assert math.isclose(r["mean"], d.mean())


def test_tost_equivalence_and_not_from_nonsignificance():
    tight = np.log(np.array([1.001, 0.999, 1.002, 0.998, 1.0]))
    assert tost(tight)["p"] < 0.05                       # within +/- 3 %
    noisy = np.log(np.array([1.10, 0.90, 1.08, 0.93, 1.0]))
    assert tost(noisy)["p"] > 0.05                       # mean ~0 but not shown equivalent
    assert paired_test(noisy, "two-sided")["p"] > 0.05   # and not significant either -> "inconclusive"


def test_required_blocks_rule():
    sd = np.full(10, 0.01)
    assert required_blocks(sd)["n"] <= 4
    r = required_blocks(np.full(10, 0.10))
    assert r["n"] == 12 and r.get("cap_binds")


def test_policy_choice_respects_latency_slo():
    tab = pd.DataFrame([
        dict(placement="P", coll="c", size_bytes=1, config="auto", logE=np.log(1.00), logT=np.log(1.00), actual="A"),
        dict(placement="P", coll="c", size_bytes=1, config="x", logE=np.log(0.80), logT=np.log(1.20), actual="X"),
        dict(placement="P", coll="c", size_bytes=1, config="y", logE=np.log(0.90), logT=np.log(1.04), actual="Y"),
        dict(placement="P", coll="c", size_bytes=1, config="z", logE=np.log(0.97), logT=np.log(0.95), actual="Z"),
    ])
    p = choose_policies(tab).iloc[0]
    assert p.en_opt == "x" and p.lat_opt == "z" and p.energy_slo == "y"   # x violates the 1.05 x auto latency SLO



def test_identical_executions_cannot_beat_auto():
    rows = [dict(placement="P", coll="c", size_bytes=1, config=cfg, actual=act, n_blocks=2, logE=np.log(e),
                 logT=np.log(t), sdlogE=0.01, sdlogT=0.01)
            for cfg, act, e, t in (("auto", "RING/LL/1", 1.00, 1.00), ("Ring/LL", "RING/LL/1", 0.97, 0.99),
                                   ("Ring/Simple", "RING/SIMPLE/1", 1.30, 1.40))]
    eff = collapse_effective(pd.DataFrame(rows))
    assert len(eff) == 2 and set(eff.config) == {"auto", "Ring/Simple"}
    assert eff[eff.config == "auto"].members.iloc[0] == "Ring/LL;auto"
    p = choose_policies(eff).iloc[0]
    assert p.en_opt == p.lat_opt == p.energy_slo == "auto"
