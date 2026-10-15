"""Corrective gate cg1: regression tests for the independent-review findings (IDs as in prep/cg1/FINDINGS_TABLE.md).
CPU only; every test runs with warnings as errors. Fixed seeds; Monte-Carlo tolerances are 3 standard errors."""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

from ncclenergy.v2 import stage2_decisions as dec  # noqa: E402
from ncclenergy.v2 import stage2_energy as en  # noqa: E402
from ncclenergy.v2 import stage2_runner as S  # noqa: E402
from ncclenergy.v2.common import REPO  # noqa: E402

CLUSTER_REPO = Path(os.environ.get("NCCLE_CLUSTER_REPO", Path.home() / "research/Overs/nccl-energy-frontier"))
PROTECTED = Path(os.environ.get("NCCLE_STAGE2_PREP", REPO.parent / "prep")) / "protected/cluster_checkout_d4c7f1c_files.sha256"


def _rates(mu, sd, n, delta, seed):
    rng = np.random.default_rng(seed)
    beyond = diff = tost = 0
    for _ in range(n):
        smy = dec.mean_summary(list(rng.normal(mu, sd, 3)), 3, "session means")
        ev = dec.evidence(smy, delta)
        beyond += ev["difference"]["p_beyond_margin"] < 0.05
        diff += ev["difference"]["p"] < 0.05
        tost += ev["equivalence"]["within_margin"]
    return beyond / n, diff / n, tost / n


@strict
def test_G1_minimum_effect_test_has_valid_size_at_both_margins():
    """H0: |M| <= delta. At M = +delta and M = -delta, for small and large noise, the rejection rate is <= alpha
    (the observed-sign one-tailed version exceeded alpha: 7.8 % at sd = 10 delta, pre-edit)."""
    n, d = 4000, dec.D3
    lim = 0.05 + 3 * math.sqrt(0.05 * 0.95 / n)
    for mu in (+d, -d):
        for sd in (10 * d, d / 2):
            b, _, _ = _rates(mu, sd, n, d, seed=hash((round(mu, 9), round(sd, 9))) % 2**32)
            assert b <= lim, (mu, sd, b)
    old = 0                                       # the pre-edit formula, for the record (not used by the code)
    rng = np.random.default_rng(20261012)
    for _ in range(n):
        smy = dec.mean_summary(list(rng.normal(d, 10 * d, 3)), 3, "s")
        old += float(__import__("scipy").stats.t.sf((abs(smy["mean"]) - d) / smy["se"], 2)) < 0.05
    assert old / n > lim                           # it really was invalid


@strict
def test_G1_null_sizes_difference_tost_and_holm_family():
    n, d = 4000, dec.D3
    lim = 0.05 + 3 * math.sqrt(0.05 * 0.95 / n)
    _, diff0, _ = _rates(0.0, d, n, d, seed=11)
    assert abs(diff0 - 0.05) <= 3 * math.sqrt(0.05 * 0.95 / n)            # two-sided test exact at M = 0
    for mu in (+d, -d):                                                    # TOST at both margin boundaries
        _, _, tost = _rates(mu, d / 2, n, d, seed=12 if mu > 0 else 13)
        assert tost <= lim, (mu, tost)
    rng = np.random.default_rng(14)                                        # Holm FWER over 3 null contrasts
    fw = 0
    for _ in range(n):
        fam = dec.family({L: dec.mean_summary(list(rng.normal(0, d, 3)), 3, "session means")
                          for L in dec.HD3_LENGTHS}, d)
        fw += any(e["decision"]["difference_detected"] for e in fam.values())
    assert fw / n <= lim


@strict
def test_G2_duration_family_covers_every_measured_length():
    assert dec.HD1_LENGTHS == (2.5, 10.0, 20.0)
    assert "every measured length against 40 s" in dec.__doc__


def _dose(work_end, reads, P=500_000_000, base=10_000_000_000, stop=None):
    return {"base_ns": base, "stop_req_ns": stop or work_end + 20_000, "joined_ns": work_end + 3_000_000,
            "n": len(reads), "skipped": 0, "overflow": 0, "start_failed": 0, "reads": reads}


@strict
def test_E1_dose_rule_boundary_cases_are_deterministic():
    P, base = 500_000_000, 10_000_000_000
    rd = lambda k, late=1_400_000, dur=1_300_000, rc=0: [k, base + k * P, base + k * P + late,  # noqa: E731
                                                         base + k * P + late + dur, 100 + k, rc]
    we = base + 2_500_000_000 + 600_000                                    # 2.5 s / D500 / 1.4 ms overshoot
    s = en.dose_summary(_dose(we, [rd(k) for k in range(1, 6)]), 500, base, we)
    assert (s["eligible"], s["delivered"], s["delivered_frac"], s["shutdown_reads"]) == (4, 4, 1.0, 1)
    we2 = base + 2 * P + 10_000_000                                        # request 2 exactly at the allowance edge
    s2 = en.dose_summary(_dose(we2, [rd(1), rd(2, dur=9_000_000)]), 500, base, we2)
    assert s2["eligible"] == 2 and s2["delivered"] == 1 and s2["straddling"] == 1   # eligible, but it straddles
    s3 = en.dose_summary(_dose(we, [rd(1), rd(2, late=25_000_000), rd(3), rd(4, rc=3)]), 500, base, we)
    assert s3["delivered"] == 2 and "DOSE_STATUS_ERROR" in s3["flags"]  # late start and status error not delivered
    s4 = en.dose_summary(_dose(base + 100_000_000, []), 500, base, base + 100_000_000)
    assert s4["eligible"] == 0 and s4["delivered_frac"] is None and "DOSE_NO_ELIGIBLE_REQUESTS" in s4["flags"]
    s5 = en.dose_summary(_dose(we, [rd(1), rd(1)]), 500, base, we)
    assert "DOSE_DUPLICATE_REQUEST" in s5["flags"]
    s6 = en.dose_summary(_dose(we, [rd(1)], stop=we - 1), 500, base, we)
    assert "DOSE_STOPPED_BEFORE_WORK_END" in s6["flags"]
    assert en.dose_summary(_dose(we, []), 500, base, None)["flags"] == ["DOSE_TIMES_MISSING"]


def _v3_rows(g=4, n=80, dt=0.05, t0=100.0, **bad):
    rows = []
    for i in range(n):
        t = t0 + i * dt
        ns = int(t * 1e9)
        r = [t, g, 150.0, 40, 1900, 8000, 50, 10, 0, 0, 300000, 1.0, 1.0, 151.0, ns, ns + 100_000, ns + 200_000,
             ns + 300_000, 1_790_000_000_000_000 + i, 120, 0, 1_790_000_000_000_000_000, ns]
        for k, (idx, val) in bad.items():
            if i == idx[0]:
                r[en.V3_COLUMNS.index(k)] = val
        rows.append(tuple(r))
    return rows


@strict
def test_D1_required_channels_are_enforced_in_counter_mode():
    win = {4: (100.5, 103.0)}
    assert en.required_channel_problems(_v3_rows(), [4], win, 0.05) == []
    assert "NONFINITE_POWER_INST_W_GPU4" in en.required_channel_problems(_v3_rows(power_inst_w=((30,), float("nan"))), [4], win, 0.05)
    assert "POWER_INSTANT_STATUS_GPU4" in en.required_channel_problems(_v3_rows(inst_status=((30,), 3)), [4], win, 0.05)
    assert "QUERY_BRACKET_REVERSED_GPU4" in en.required_channel_problems(_v3_rows(t_inst_a_ns=((30,), 10**15)), [4], win, 0.05)
    gap = [r for r in _v3_rows() if not 101.5 <= r[0] <= 102.0]
    assert "INST_SAMPLE_GAP_GPU4" in en.required_channel_problems(gap, [4], win, 0.05)
    assert en.required_channel_problems(_v3_rows(), [4, 5], {**win, 5: (100.5, 103.0)}, 0.05) == ["NO_V3_ROWS_GPU5"]
    short = [r for r in _v3_rows() if r[0] < 102.0]
    assert "INST_WINDOW_NOT_COVERED_GPU4" in en.required_channel_problems(short, [4], win, 0.05)
    assert "INVALID_PSTATE_GPU4" in en.required_channel_problems(_v3_rows(pstate=((30,), 8.5)), [4], win, 0.05)
    assert "TOTAL_POWER" not in " ".join(en.REQUIRED_CHANNELS) and "total_power_samples" in en.OPTIONAL_CHANNELS


@strict
def test_D2_smoke_criteria_pause_postprobe_and_dose():
    t = S.Timing()
    ok_pp = {"brackets": [(0, 1)] * 4, "span_s": 0.38, "statuses_ok": True}
    e0 = {"poller": "inactive", "n_reads": 0}         # smoke-1: s2_metrics now always carries the dose summary (S5)
    rec = {"s2": {"per_gpu": [{"rank": 0, "pause_dev_ms": 0.1, "post_probe": ok_pp, "dose": e0},
                              {"rank": 1, "pause_dev_ms": -0.2, "post_probe": ok_pp, "dose": e0}]}}
    assert S.smoke_problems(rec, t, {"post_probe": True}) == []
    bad = json.loads(json.dumps(rec))
    bad["s2"]["per_gpu"][0]["pause_dev_ms"] = 3.0
    bad["s2"]["per_gpu"][1]["post_probe"] = {"brackets": [(0, 1)] * 2, "span_s": 0.5, "statuses_ok": False}
    assert S.smoke_problems(bad, t, {"post_probe": True}) == ["PAUSE_DEVIATION_GT_2MS", "POST_PROBE_FEW_BRACKETS_RANK1",
                                                              "POST_PROBE_TOO_LONG_RANK1", "POST_PROBE_STATUS_RANK1"]
    assert S.smoke_problems({}, t, {}) == ["NO_STAGE2_RECORD"]


@strict
def test_D3_complete_is_execution_not_acceptance():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as tmp:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(("FAKE_", "NCCLE_"))}
        env["PYTHONPATH"] = f"{REPO / 'tests/v2/fake_nvml'}:{REPO / 'src'}"
        p = subprocess.run([sys.executable, "-W", "error", str(REPO / "tests/v2/helpers/stage2_session.py"),
                            "complete_not_accepted", tmp], env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-3000:]
        d = json.loads(p.stdout.strip().splitlines()[-1])
    assert d["status"] == "COMPLETE" and d["n_accepted"] == 2
    assert d["not_accepted"] == {"C2r1-03": ["STAGE2_RECORD_FLAGS:COUNTER_CALL_INVENTORY_MISMATCH"]}


@strict
def test_historical_replay_from_a_verified_complete_dataset():
    """Declared input root: the cluster checkout d4c7f1c (complete dataset). Every file of the campaign is checked
    against the protected manifest before the frozen code replays it; the expected decisions are explicit."""
    camp = CLUSTER_REPO / "data/v2.2/raw/v2.2-measval-01"
    if not camp.exists() or not PROTECTED.exists():
        print("SKIP: complete historical data or protected manifest not available")
        return
    man = {}
    for ln in PROTECTED.read_text().splitlines():
        h, f = ln.split("  ", 1)
        man[f[2:] if f.startswith("./") else f] = h
    files = [p for p in camp.rglob("*") if p.is_file()]
    assert len(files) == 702 and sum(1 for p in files if p.name == "region.jsonl") == 72
    for p in files:
        rel = str(p.relative_to(CLUSTER_REPO))
        assert man.get(rel) == hashlib.sha256(p.read_bytes()).hexdigest(), rel
    for rel in ("configs/v2.2/measval01_frozen.json", "configs/v2.2/frozen_code/measval_f4735b1.py"):
        assert man[rel] == hashlib.sha256((CLUSTER_REPO / rel).read_bytes()).hexdigest(), rel
    mod = dec.load_frozen_measval(CLUSTER_REPO)
    res = dec.replay(mod, camp, json.loads((CLUSTER_REPO / "configs/v2.2/measval01_frozen.json").read_text()))
    assert (res["n_processes"], res["n_accepted"]) == (72, 70)
    assert [res[k]["decision"] for k in ("D1", "D2", "D3", "D4")] == \
        ["INCONCLUSIVE", "INCONCLUSIVE", "SYSTEMATIC_OFFSET", "NOT_SUFFICIENT"]
