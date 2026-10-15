"""Held-out analysis (stage2_heldout), CPU only. Known answers come from synthetic fixtures built so that every block
contrast is fixed by construction; t probabilities with df = 2 are checked against the closed form
sf(t) = 1/2 - t / (2 sqrt(t^2 + 2)) and the textbook quantiles t(0.975, 2), t(0.95, 2), not against scipy. The fake-NVML
runner session is used only for the loader interface and for the period/counter/coverage/clock-domain cases with the
unchanged unit_energies. The real calibration is used only for interface/regression checks and is never relabeled as
held-out (the loader must refuse it)."""
import atexit
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

from ncclenergy.v2 import stage2_heldout as H  # noqa: E402
from ncclenergy.v2 import stage2_runner as S  # noqa: E402
from ncclenergy.v2.common import REPO, gpu_alias, redact  # noqa: E402

STAGE2 = REPO.parent
CAL_JSON = STAGE2 / "prep/calib1/run/calibration_analysis.json"
CAL_DIR = REPO / "data/stage2/v2.5-calib-01/calibration"
FREEZE_C = STAGE2 / "protocol/FREEZE_C_CALIB1.json"
GPUS = [4, 5]
DRAFTS = {s: yaml.safe_load((REPO / f"configs/stage2/heldout_s{s}_DRAFT.yaml").read_text()) for s in (1, 2, 3)}
COUNTS = DRAFTS[1]["counts"]
CAL_RULES = {"t_launch_c": 26, "bands": {"4": {"t_region_c": [28.0, 31.2], "clk_mhz": [1950.0, 1980.0], "pstates": [2]},
                                         "5": {"t_region_c": [29.0, 32.0], "clk_mhz": [1920.0, 1950.0], "pstates": [2]}}}
RAW = ["GPU-11111111-2222-3333-4444-555555555555", "GPU-66666666-7777-8888-9999-000000000000"]
SCOPE_CFG = {"gpu_name": "NVIDIA RTX A6000", "driver": "580.178.04", "link": "NV4", "uuids": RAW,      # configs: raw UUIDs
             "numa": {"4": 1, "5": 1}, "power_limit_mw": [300000, 300000], "max_clk_gr_mhz": [2100, 2100]}
SCOPE = dict(SCOPE_CFG, uuids=[gpu_alias(u) for u in RAW])          # session summaries: the runner's stable aliases
STATE_OK = {"4": {"t_region_c": 30.0, "clk_mhz": 1965.0, "pstate": 2, "clock_events": 0, "inst_age_s": 0.05},
            "5": {"t_region_c": 30.5, "clk_mhz": 1935.0, "pstate": 2, "clock_events": 0, "inst_age_s": 0.05}}
Q975, Q95 = 4.302652729911275, 2.919985580355516          # t quantiles, df = 2
D3, D12, D4 = math.log(1.005), math.log(1.003), math.log(1.004)
(REPO / "logs").mkdir(exist_ok=True)          # git-ignored


def sf2(t):
    return 0.5 - t / (2 * math.sqrt(t * t + 2))


def raises(exc, fn, *a, match=None, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return match is None or match in str(e)
    return False


# ------------------------------------------------------------------ synthetic fixture (contrasts fixed by construction)

LS = (2.5, 10.0, 20.0, 40.0)
BASE = {"X1": {(a, L): [[0.0, 0.0], [0.0005, 0.0005], [0.001, 0.001]] for a in "ABC" for L in (2.5, 10.0, 20.0)},
        "X3": {10.0: [[0.0005, 0.0015], [0.0015, 0.0025], [0.0025, 0.0035]],     # session means 0.001, 0.002, 0.003
               20.0: [[0.010, 0.012], [0.012, 0.012], [0.013, 0.013]],           # 0.011, 0.012, 0.013
               40.0: [[-0.0015, -0.0005], [0.0, 0.0], [0.0005, 0.0015]]},        # -0.001, 0, 0.001
        "X4": [[0.009, 0.011], [0.011, 0.011], [0.012, 0.012]],                  # 0.010, 0.011, 0.012
        "G": [[0.001, 0.001], [0.0015, 0.0015], [0.002, 0.002]],                 # H-D2: 0.001, 0.0015, 0.002
        "A": [[0.0, 0.01], [-0.02, 0.005], [0.03, -0.01]]}                       # block nuisance: cancels everywhere


def j_table(p, s, b):
    """J (J/op) of the 12 units of block (s, b), 0-based; every family contrast equals the matching X entry."""
    x1, x3 = (lambda a, L: p["X1"][(a, L)][s][b]), (lambda L: p["X3"][L][s][b])
    J = {("A", 40.0): 1e-3 * math.exp(p["A"][s][b])}
    for L in (2.5, 10.0, 20.0):
        J[("A", L)] = J[("A", 40.0)] * math.exp(x1("A", L))
    J[("B", 40.0)] = J[("A", 40.0)] * math.exp(x3(40.0))
    for L in (10.0, 20.0):
        J[("B", L)] = J[("A", L)] * math.exp(x3(L))
    J[("B", 2.5)] = J[("B", 40.0)] * math.exp(x1("B", 2.5))
    J[("C", 40.0)] = J[("A", 40.0)] * math.exp(p["X4"][s][b])
    for L in (2.5, 10.0, 20.0):
        J[("C", L)] = J[("C", 40.0)] * math.exp(x1("C", L))
    return J


def energies(p, s, b, arm, L, raw=lambda arm: 1.0):
    """Energies (J) by estimator name for one unit: E = J * R; the sampled one has the H-D2 structure exactly."""
    R = COUNTS[arm]["R"][f"{L:g}"]
    Ep = j_table(p, s - 1, b - 1)[(arm, L)] * R
    Es = Ep * math.exp(0.01 + 0.02 / L + (p["G"][s - 1][b - 1] if arm == "C" else 0.0))
    return R, {"primary": Ep, "raw_counter": Ep * raw(arm), "nominal_period": Ep * 0.999, "sampled_instant": Es}


def synth_units(p=BASE, sessions=(1, 2, 3), raw=lambda arm: 1.0):
    out = []
    for s in sessions:
        for b in (1, 2):
            for arm in "ABC":
                for L in LS:
                    R, E = energies(p, s, b, arm, L, raw)
                    out.append({"launch_id": f"s{s}-b{b}-{arm}{L:g}", "session": s, "block": b, "arm": arm,
                                "length_s": L, "status": "USED", "reasons": [], "R": R, "E": E,
                                "J": {k: v / R for k, v in E.items()}, "unavailable": {},
                                "exec_key": "RING/LL128/4" if arm == "C" else "RING/SIMPLE/4", "T_region_s": L,
                                "t_region_gpu": {4: 30.0, 5: 30.5}})
    return out


def find(units, s, b, arm, L):
    return next(u for u in units if (u["session"], u["block"], u["arm"], u["length_s"]) == (s, b, arm, L))


def tsum(means):
    m = sum(means) / 3
    se = math.sqrt(sum((x - m) ** 2 for x in means) / 2) / math.sqrt(3)
    return m, se


# ------------------------------------------------------------------ known answers

@strict
def test_known_answers_HD3_session_t_holm_tost_and_verdicts():
    r = H.analyse_units(synth_units(), GPUS, "primary")
    exp = {}
    for L, means in ((10.0, [0.001, 0.002, 0.003]), (20.0, [0.011, 0.012, 0.013]), (40.0, [-0.001, 0.0, 0.001])):
        sm = r["summaries"]["HD3"][L]
        assert all(abs(sm["session_means"][s] - v) < 1e-12 for s, v in zip((1, 2, 3), means))
        m, se = tsum(means)
        t = m / se
        exp[L] = {"m": m, "se": se, "p": 2 * sf2(abs(t)),
                  "beyond": min(1.0, 2 * min(sf2((m - D3) / se), 1 - sf2((m + D3) / se))),
                  "tost": max(sf2((m + D3) / se), 1 - sf2((m - D3) / se))}
        assert abs(sm["mean"] - m) < 1e-12 and abs(sm["se"] - se) < 1e-12 and sm["df"] == 2
        lo, hi = sm["ci95"]
        assert abs(lo - (m - Q975 * se)) < 1e-12 and abs(hi - (m + Q975 * se)) < 1e-12
        assert abs(sm["ci90"][1] - (m + Q95 * se)) < 1e-12
    assert abs(exp[10.0]["p"] - (1 - 2 * math.sqrt(3) / math.sqrt(14))) < 1e-12 and abs(exp[10.0]["p"] - 0.0741799) < 1e-6
    # Holm over the 3 lengths, both tests: p(20) < p(10) < p(40) = 1
    a20 = 3 * exp[20.0]["p"]
    a10 = max(a20, 2 * exp[10.0]["p"])
    hd3 = r["HD3"]
    for L, adj in ((20.0, a20), (10.0, a10), (40.0, 1.0)):
        e = hd3[L]
        assert abs(e["difference"]["p"] - exp[L]["p"]) < 1e-10 and abs(e["difference"]["p_holm"] - adj) < 1e-10
        assert abs(e["equivalence"]["p_tost"] - exp[L]["tost"]) < 1e-10          # unadjusted
        assert e["family_size"] == 3 and e["difference"]["ci95_label"] == "descriptive (unadjusted)"
    assert abs(hd3[20.0]["difference"]["p_beyond_margin_holm"] - 3 * exp[20.0]["beyond"]) < 1e-10
    assert [hd3[L]["verdict"] for L in (10.0, 20.0, 40.0)] == ["COMPATIBLE_WITH_EQUIVALENCE",
                                                               "RESOLVED_MEANINGFUL_DIFFERENCE",
                                                               "COMPATIBLE_WITH_EQUIVALENCE"]
    assert abs(r["HD3_precision"][10.0]["half_width"] - Q975 * exp[10.0]["se"]) < 1e-12 and r["HD3_precision"][10.0]["ok"]


@strict
def test_known_answers_HD1_HD2_HD4():
    r = H.analyse_units(synth_units(), GPUS, "primary")
    X1, X3 = BASE["X1"], BASE["X3"]
    for (a, L), sm in r["summaries"]["HD1"].items():
        for s in (1, 2, 3):
            via_a = a == "B" and L != 2.5                       # B at 10/20 s is built from A at L and H-D3 terms
            blocks = [X1[("A" if via_a else a, L)][s - 1][b] + ((X3[L][s - 1][b] - X3[40.0][s - 1][b]) if via_a else 0.0)
                      for b in (0, 1)]
            assert abs(sm["session_means"][s] - sum(blocks) / 2) < 1e-12
    assert len(r["HD1"]) == 9 and all(e["family_size"] == 9 for e in r["HD1"].values())
    assert all(r["HD1"][(a, L)]["verdict"] == "COMPATIBLE_WITH_EQUIVALENCE" for a in "AC" for L in (2.5, 10.0, 20.0))
    assert r["HD1"][("B", 20.0)]["verdict"] == "RESOLVED_MEANINGFUL_DIFFERENCE"
    # H-D2: the OLS coefficient of 1{C} recovers the constructed protocol term exactly in every block
    sm2 = r["summaries"]["HD2"]["C"]
    assert all(abs(sm2["session_means"][s] - v) < 1e-9 for s, v in zip((1, 2, 3), (0.001, 0.0015, 0.002)))
    assert r["HD2"]["C"]["family_size"] == 1 and r["HD2"]["C"]["delta"] == D12
    # H-D4: precision criterion only (95 % half-width <= ln 1.004); its evidence at ln 1.004 is unadjusted, family of 1
    m, se = tsum([0.010, 0.011, 0.012])
    assert abs(r["summaries"]["HD4"]["C/A@40"]["mean"] - m) < 1e-12
    assert r["HD4_precision"]["ok"] is True and abs(r["HD4_precision"]["half_width"] - Q975 * se) < 1e-12
    assert r["HD4"]["C/A@40"]["difference"]["ci95_label"] == "unadjusted"


# ------------------------------------------------------------------ sessions, missing primary, partial data

@strict
def test_three_sessions_are_required_and_never_replaced_by_blocks():
    r = H.analyse_units(synth_units(sessions=(1, 2)), GPUS, "primary")
    for f in ("HD1", "HD2", "HD3", "HD4"):
        assert all(e["verdict"] == "INSUFFICIENT_SESSIONS" for e in r[f].values())
    assert r["summaries"]["HD3"][10.0]["n"] == 2 and r["summaries"]["HD3"][10.0]["conditional_descriptive"]["n"] == 4
    assert r["HD4_precision"]["ok"] is False and r["r3_ok"][10.0] is False         # strict for adequacy
    adq = H.adequacy_all(r, {"doses": {"D100": {"bounded_for_adequacy": True}, "D500": {"bounded_for_adequacy": True}}})
    assert all(a["status"] == "NOT_ESTABLISHED" and a["conditional_statistical"] == "NOT_MET" for a in adq.values())
    # one member with no valid block in one session: only that member is INSUFFICIENT_SESSIONS; the family size stays 3
    u = [x for x in synth_units() if not (x["session"] == 2 and x["arm"] == "B" and x["length_s"] == 10.0)]
    r = H.analyse_units(u, GPUS, "primary")
    assert r["HD3"][10.0]["verdict"] == "INSUFFICIENT_SESSIONS" and r["HD3"][20.0]["family_size"] == 3
    m, se = tsum([0.011, 0.012, 0.013])
    assert abs(r["HD3"][20.0]["difference"]["p_holm"] - 3 * 2 * sf2(m / se)) < 1e-10      # missing member enters as p = 1


@strict
def test_missing_primary_stays_missing_and_partial_sessions_use_available_blocks():
    u = synth_units()
    x = find(u, 1, 1, "B", 10.0)
    x["J"]["primary"], x["E"]["primary"] = None, None
    x["unavailable"] = {"primary": ["RANK0: PRIMARY_PERIOD_INCOMPATIBLE"]}
    r, rr = H.analyse_units(u, GPUS, "primary"), H.analyse_units(u, GPUS, "raw_counter")
    assert abs(r["summaries"]["HD3"][10.0]["session_means"][1] - 0.0015) < 1e-12          # block 2 only
    assert ("HD3", 10.0, 1, 1, "MISSING") in r["excluded"]
    assert abs(rr["summaries"]["HD3"][10.0]["session_means"][1] - 0.001) < 1e-12         # raw keeps both blocks
    # partial session: session 2 block 2 not run -> session 2 mean is its block-1 contrast; nothing imputed
    u = [dict(x, status="NOT_RUN", J=None, E=None) if (x["session"], x["block"]) == (2, 2) else x for x in synth_units()]
    r = H.analyse_units(u, GPUS, "primary")
    assert abs(r["summaries"]["HD3"][10.0]["session_means"][2] - 0.0015) < 1e-12
    assert r["HD3"][10.0]["status"] == "OK" and len([e for e in r["excluded"] if e[2:4] == (2, 2)]) == 9 + 3 + 1 + 1
    # an unaccepted unit is missing data, never replaced
    u = synth_units()
    find(u, 3, 2, "A", 40.0)["status"] = "NOT_ACCEPTED"
    r = H.analyse_units(u, GPUS, "primary")
    assert abs(r["summaries"]["HD4"]["C/A@40"]["session_means"][3] - 0.012) < 1e-12        # block 1 only
    assert all(e[2:4] != (3, 2) or e[0] in ("HD1", "HD3", "HD4") for e in r["excluded"])


@strict
def test_estimator_separation_no_promotion_no_substitution():
    u = synth_units(raw=lambda arm: 1.02 if arm == "B" else 1.0)
    rp, rr = H.analyse_units(u, GPUS, "primary"), H.analyse_units(u, GPUS, "raw_counter")
    for L in (10.0, 20.0, 40.0):
        for s in (1, 2, 3):
            assert abs(rr["summaries"]["HD3"][L]["session_means"][s]
                       - rp["summaries"]["HD3"][L]["session_means"][s] - math.log(1.02)) < 1e-12
    assert rp["role"] == "PRIMARY" and rr["role"].startswith("SENSITIVITY")
    assert raises(ValueError, H.adequacy_all, rr, {})
    rs = H.analyse_units(u, GPUS, "sampled_instant")
    assert rs["HD2"].startswith("NOT_APPLICABLE")
    assert raises(ValueError, H.analyse_units, u, GPUS, "E_samp_hold_inst")
    assert "E_samp_hold_inst" in H.NOT_COMPUTED and "NO_SECONDARY_ESTIMATOR" in H.NOT_COMPUTED[
        "delay_shifted_secondary_estimator"]


# ------------------------------------------------------------------ thermal (R3), control validity, clock bands (R2)

@strict
def test_R3_is_per_gpu_and_overrides_only_the_between_arm_contrasts():
    u = synth_units()
    for b in (1, 2):                                        # GPU-mean balanced (-0.6 / +0.6) but per-GPU violated
        find(u, 2, b, "B", 20.0)["t_region_gpu"] = {4: 29.4, 5: 31.1}
        find(u, 1, b, "C", 40.0)["t_region_gpu"] = {4: 30.51, 5: 30.5}
        find(u, 3, b, "B", 10.0)["t_region_gpu"] = {4: 30.5, 5: 31.0}            # exactly 0.5 C: allowed
    r = H.analyse_units(u, GPUS, "primary")
    assert r["r3"][(2, 20.0, "B-A")]["ok"] is False and r["r3"][(3, 10.0, "B-A")]["ok"] is True
    assert r["HD3"][20.0]["verdict"] == "THERMAL_COMPARABILITY_NOT_ESTABLISHED"
    assert r["HD3"][10.0]["verdict"] == "COMPATIBLE_WITH_EQUIVALENCE"
    assert r["HD4"]["C/A@40"]["verdict"] == "THERMAL_COMPARABILITY_NOT_ESTABLISHED"
    assert r["HD1"][("B", 20.0)]["verdict"] == "RESOLVED_MEANINGFUL_DIFFERENCE"          # within-arm: no R3 override
    assert r["r3_ok"][20.0] is False and r["r3_ok"][40.0] is False and r["r3_ok"][10.0] is True
    adq = H.adequacy_all(r, {})
    assert any("R3 per-GPU balance at 20 s" in x for x in adq["SAME_EXEC_L20"]["unmet"])
    u = synth_units()
    find(u, 2, 1, "A", 10.0)["t_region_gpu"] = {4: float("nan"), 5: 30.5}             # evaluated, not establishable
    r = H.analyse_units(u, GPUS, "primary")
    assert r["r3"][(2, 10.0, "B-A")]["reason"] == "NONFINITE_OR_MISSING_T_REGION"
    assert r["HD3"][10.0]["verdict"] == "THERMAL_COMPARABILITY_NOT_ESTABLISHED"


@strict
def test_control_validity_includes_channels_and_excludes_the_block():
    u = synth_units()
    find(u, 3, 1, "A", 20.0)["exec_key"] = "RING/SIMPLE/2"           # same algorithm/protocol, other channel count
    r = H.analyse_units(u, GPUS, "primary")
    assert r["control"][(3, 1)] == {"valid": False, "differing_lengths": [20.0]} and r["control_valid"] is False
    assert all(r["HD3"][L]["verdict"] == "INVALID_CONTROL_OR_MEASUREMENT" for L in (10.0, 20.0, 40.0))
    assert ("HD3", 10.0, 3, 1, "CONTROL_INVALID_BLOCK") in r["excluded"]
    assert abs(r["summaries"]["HD3"][10.0]["session_means"][3] - 0.0035) < 1e-12           # valid block only
    assert r["HD4"]["C/A@40"]["verdict"] == "RESOLVED_MEANINGFUL_DIFFERENCE"
    assert "control validity" in H.adequacy_all(r, {})["SAME_EXEC_L10"]["unmet"]


# ------------------------------------------------------------------ session directories (loader) with fixed energies

def _cfgs(rules=CAL_RULES, scope=SCOPE_CFG):
    return {s: dict(DRAFTS[s], readiness_rules=json.loads(json.dumps(rules)), scope_expected=json.loads(json.dumps(scope)))
            for s in (1, 2, 3)}


def _write_session(root: Path, s: int, p=BASE, mode="STAGE2_HELDOUT", scope=SCOPE, mutate=None, drop=(), extra=(),
                   base=None, config_sha=None, summary=None):
    """A session directory in the runner's record format (the fields run_stage2_unit writes); unit dirs relative to
    `base` (default: root). Energies come from the synthetic fixture via FixedEnergies."""
    sdir = root / f"session_{s}"
    (sdir / "units").mkdir(parents=True)
    recs, en = [], {}
    for u in DRAFTS[s]["units"]:
        lid, arm, L = u["launch_id"], u["arm"], float(u["length_s"])
        if lid in drop:
            continue
        R, E = energies(p, s, u["block"], arm, L)
        d = sdir / "units" / lid
        d.mkdir()
        (d / "unit.json").write_text(json.dumps({"lid": lid}))
        key = {"A": "RING/SIMPLE/4", "B": "RING/SIMPLE/4", "C": "RING/LL128/4"}[arm]
        rec = {"launch_id": lid, "unit_id": lid, "attempt": 0, "implementation_id": S.IMPLEMENTATION_ID,
               "method_id": S.METHOD_ID, "campaign": DRAFTS[s]["campaign"], "mode": mode, "unit": {k: u[k] for k in u},
               "R": R, "w": COUNTS[arm]["w"], "gpus": list(GPUS), "dir": str(d.relative_to(base or root)),
               "readiness": {"status": "READY", "t_ready_s": 3.0}, "pre_region_state": json.loads(json.dumps(STATE_OK)),
               "evidence": {"status": "VERIFIED", "actual": [{"algo": key.split("/")[0], "proto": key.split("/")[1],
                                                               "nChannels": 4}],
                            "actual_key": key, "n_matches_expected": True},
               "region": {"R": R, "T_region_s": L}, "accepted": True, "not_accepted_reasons": [], "qc_status": "OK"}
        if mutate:
            mutate(rec)
        recs.append(rec)
        en[lid] = E
    for rec in extra:
        recs.append(rec)
    (sdir / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (sdir / "session_summary.json").write_text(json.dumps(dict({
        "mode": mode, "execution_status": "COMPLETE", "implementation_id": S.IMPLEMENTATION_ID, "method_id": S.METHOD_ID,
        "config_sha256": config_sha or f"cfg{s}", "scope_observed": scope}, **(summary or {}))))
    return sdir, en


class FixedEnergies:
    """unit_energies replaced by fixed per-launch energies (the loader logic is under test, not the estimators)."""

    def __init__(self, en):
        self.en = en

    def __call__(self, rec, repo):
        E = self.en[rec["unit_id"]]
        return {"status": "OK", "reasons": [], "per_gpu": [], "R": rec["region"]["R"],
                "E_aligned_hat_J": E["primary"], "E_counter_raw_J": E["raw_counter"],
                "E_aligned_nom_J": E["nominal_period"], "E_samp_inst_J": E["sampled_instant"]}

    def __enter__(self):
        self.orig, H.unit_energies = H.unit_energies, self
        return self

    def __exit__(self, *a):
        H.unit_energies = self.orig


def _cal(tmp: Path, bounded=False, rules=CAL_RULES):
    p = tmp / "cal.json"
    p.write_text(json.dumps({"q3_readiness": {"status": "OK", **rules},
                             "q1_primary": {"doses": {D: {"bounded_for_adequacy": bounded} for D in ("D100", "D500")}}}))
    return p


def _three(tmp: Path, **kw):
    dirs, en = {}, {}
    for s in (1, 2, 3):
        dirs[s], e = _write_session(tmp, s, **kw)
        en.update(e)
    return dirs, en


@strict
def test_end_to_end_on_session_directories_reproduces_the_known_answers_with_exact_accounting():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        dirs, en = _three(tmp)
        with FixedEnergies(en):
            out = H.analyse_sessions_unverified(dirs, tmp, _cfgs(), _cal(tmp), GPUS)
    assert out["provenance_status"].startswith("UNVERIFIED (low-level test interface")
    assert out["n_scheduled"] == 72 and out["unit_status_counts"] == {"USED": 72, "NOT_ACCEPTED": 0, "EXCLUDED": 0,
                                                                       "NOT_RUN": 0}
    for u in out["units"]:
        R = COUNTS[u["arm"]]["R"][f"{u['length_s']:g}"]
        assert u["R"] == R and isinstance(u["R"], int)
        assert all(u["J"][k] == u["E"][k] / R for k in H.ESTIMATORS)                    # exact denominator
    p = out["estimators"]["primary"]
    assert [p["HD3"][L]["verdict"] for L in ("10", "20", "40")] == ["COMPATIBLE_WITH_EQUIVALENCE",
                                                                    "RESOLVED_MEANINGFUL_DIFFERENCE",
                                                                    "COMPATIBLE_WITH_EQUIVALENCE"]
    assert abs(p["summaries"]["HD3"]["10"]["session_means"]["2"] - 0.002) < 1e-12
    assert out["scope"].startswith("conditional comparison of NVML-reported")
    assert all(a["status"] == "NOT_ESTABLISHED" for a in out["adequacy"].values())
    assert out["physical_adequacy"]["status"] == "NOT_ESTABLISHED"
    assert '"PASS"' not in json.dumps(out)                          # neither M4d nor adequacy is ever a PASS


@strict
def test_adequacy_is_not_established_even_when_every_statistical_criterion_is_met():
    flat = [[-0.0005, -0.0005], [0.0, 0.0], [0.0005, 0.0005]]
    p = dict(BASE, X3={10.0: flat, 20.0: flat, 40.0: flat}, G=[[0.0, 0.0], [0.0005, 0.0005], [0.001, 0.001]])
    r = H.analyse_units(synth_units(p), GPUS, "primary")
    q1 = {"doses": {D: {"bounded_for_adequacy": True} for D in ("D100", "D500")}}
    adq = H.adequacy_all(r, q1)
    assert all(a["conditional_statistical"] == "MET" for a in adq.values())             # intersection-union: all met
    assert all(a["status"] == "NOT_ESTABLISHED" and a["unmet_systematic"] for a in adq.values())
    x1 = dict(p["X1"])
    x1[("C", 2.5)] = [[0.004, 0.004], [0.0045, 0.0045], [0.005, 0.005]]
    p2 = dict(p, X1=x1)
    adq2 = H.adequacy_all(H.analyse_units(synth_units(p2), GPUS, "primary"), q1)
    assert adq2["BROAD"]["conditional_statistical"] == "NOT_MET"                         # one member fails -> no joint claim
    assert adq2["SAME_EXEC_L40"]["conditional_statistical"] == "MET"


@strict
def test_R1_and_R2_are_reverified_offline_with_per_gpu_bands():
    def mut(rec):
        lid = rec["unit_id"]
        st = rec["pre_region_state"]
        if lid == "s1-b1-p01":
            st["4"]["clk_mhz"] = 1935.0                     # inside GPU5's band, outside GPU4's: clock bands are per GPU
        if lid == "s1-b1-p02":
            st["4"]["t_region_c"] = 31.5                    # inside GPU5's band, outside GPU4's
        if lid == "s1-b1-p03":
            st["5"]["clock_events"] = 1                     # GpuIdle tolerated only at P8 idle gates; R2 needs exactly 0
        if lid == "s1-b1-p04":
            rec["readiness"] = {"kind": "fixed", "wait_s": 120.0}   # a fixed wait expiring is not readiness
        if lid == "s1-b1-p05":
            st["5"]["inst_age_s"] = 1.5
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        sdir, en = _write_session(tmp, 1, mutate=mut)
        with FixedEnergies(en):
            ld = H.load_session(sdir, tmp, 1, DRAFTS[1]["units"], CAL_RULES, GPUS, COUNTS, SCOPE_CFG)
    by = {u["launch_id"]: u for u in ld["units"]}
    assert "CLOCK_OUT_OF_BAND_GPU4" in by["s1-b1-p01"]["reasons"] and by["s1-b1-p01"]["status"] == "EXCLUDED"
    assert "T_REGION_OUT_OF_BAND_GPU4" in by["s1-b1-p02"]["reasons"]
    assert "CLOCK_EVENT_OR_MISSING_GPU5" in by["s1-b1-p03"]["reasons"]
    assert by["s1-b1-p04"]["reasons"] == ["ACCEPTANCE_INCONSISTENT", "R1_NOT_READY"]
    assert "STALE_POWER_GPU5" in by["s1-b1-p05"]["reasons"]
    assert all(by[f"s1-b1-p0{i}"]["reasons"][0] == "ACCEPTANCE_INCONSISTENT" for i in range(1, 6))
    assert sum(u["status"] == "USED" for u in ld["units"]) == 19


@strict
def test_readiness_rules_must_be_the_calibration_derived_ones():
    cal = {"q3_readiness": {"status": "OK", **CAL_RULES}}
    H.check_rules(CAL_RULES, cal)
    swapped = {"t_launch_c": 26, "bands": {"4": CAL_RULES["bands"]["5"], "5": CAL_RULES["bands"]["4"]}}
    assert raises(H.HeldoutInputError, H.check_rules, swapped, cal)
    assert raises(H.HeldoutInputError, H.check_rules, dict(CAL_RULES, t_launch_c=27), cal)
    assert raises(H.HeldoutInputError, H.check_rules, {}, cal)
    assert raises(H.HeldoutInputError, H.readiness_rules_from_calibration, {"q3_readiness": {"status": "NOT_DERIVABLE"}})


# ------------------------------------------------------------------ provenance and refusals

@strict
def test_schedule_is_the_frozen_72_launch_schedule():
    sched = H.schedule_from_configs(DRAFTS)
    assert sum(len(v) for v in sched.values()) == 72
    assert all(DRAFTS[s]["counts"] == COUNTS for s in (1, 2, 3))
    bad = copy.deepcopy(DRAFTS)
    bad[2]["units"][0]["arm"] = "B" if bad[2]["units"][0]["arm"] != "B" else "A"
    assert raises(H.HeldoutInputError, H.schedule_from_configs, bad)
    bad = copy.deepcopy(DRAFTS)
    bad[3]["units"] = bad[3]["units"][:-1]
    assert raises(H.HeldoutInputError, H.schedule_from_configs, bad)
    bad = copy.deepcopy(DRAFTS)
    bad[1]["units"][0]["launch_id"] = "s2-b1-p01"
    assert raises(H.HeldoutInputError, H.schedule_from_configs, bad)


@strict
def test_provenance_refusals_and_bindings():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        dirs, en = _three(tmp)
        cal, cfgs = _cal(tmp), _cfgs()
        with FixedEnergies(en):
            # calibration/smoke data are never relabeled as held-out
            cdir, _ = _write_session(tmp / "c", 1, mode="STAGE2_CALIBRATION")
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, {**dirs, 1: cdir}, tmp, cfgs, cal, GPUS,
                          match="never relabeled")
            sdir, _ = _write_session(tmp / "sm", 1, mode="STAGE2_SMOKE")
            assert raises(H.HeldoutInputError, H.load_session, sdir, tmp, 1, DRAFTS[1]["units"], CAL_RULES, GPUS, COUNTS)
            # scope: per-session expected vs observed, and the same pair in every session
            assert H.scope_check(SCOPE_CFG, SCOPE) == [] and SCOPE_CFG["uuids"] != SCOPE["uuids"]   # raw vs alias
            wdir, _ = _write_session(tmp / "w", 2, scope=dict(SCOPE, driver="550.54.15"))
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, {**dirs, 2: wdir}, tmp, cfgs, cal, GPUS,
                          match="SCOPE_MISMATCH driver")
            other_raw = ["GPU-aaaaaaaa-2222-3333-4444-555555555555", "GPU-bbbbbbbb-7777-8888-9999-000000000000"]
            odir, _ = _write_session(tmp / "o", 3, scope=dict(SCOPE, uuids=[gpu_alias(u) for u in other_raw]))
            c3 = {**cfgs, 3: dict(cfgs[3], scope_expected=dict(SCOPE_CFG, uuids=other_raw))}
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, {**dirs, 3: odir}, tmp, c3, cal, GPUS,
                          match="no pair substitution")
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, dirs, tmp, cfgs, cal, [5, 4])
            no_scope = {**cfgs, 1: dict(cfgs[1], scope_expected=None)}
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, dirs, tmp, no_scope, cal, GPUS,
                          match="no scope_expected")
            # readiness rules and counts are frozen
            c_bad = {**cfgs, 2: dict(cfgs[2], readiness_rules=dict(CAL_RULES, t_launch_c=25))}
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, dirs, tmp, c_bad, cal, GPUS)
            cnt = json.loads(json.dumps(COUNTS))
            cnt["A"]["R"]["40"] += 1
            assert raises(H.HeldoutInputError, H.analyse_sessions_unverified, dirs, tmp, {**cfgs, 3: dict(cfgs[3], counts=cnt)},
                          cal, GPUS, match="counts differ")
            # one session through the low-level interface: INSUFFICIENT_SESSIONS (Freeze-H binding: production tests)
            out = H.analyse_sessions_unverified({1: dirs[1]}, tmp, {1: cfgs[1]}, cal, GPUS)
            assert all(e["verdict"] == "INSUFFICIENT_SESSIONS" for e in out["estimators"]["primary"]["HD3"].values())
            # input hashes cover every file used
            h0 = out["provenance"]["sessions"]["1"]["input_sha256"]
            assert "units/s1-b1-p01/unit.json" in h0 and "attempts.jsonl" in h0 and len(h0) == 2 + 24
            (dirs[1] / "units/s1-b1-p01/unit.json").write_text("changed")
            h1 = H.analyse_sessions_unverified({1: dirs[1]}, tmp, {1: cfgs[1]}, cal,
                                               GPUS)["provenance"]["sessions"]["1"]["input_sha256"]
            assert h1["units/s1-b1-p01/unit.json"] != h0["units/s1-b1-p01/unit.json"]


@strict
def test_exact_operation_accounting_and_attempt_refusals():
    def run(**kw):
        with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
            tmp = Path(t)
            sdir, en = _write_session(tmp, 1, **kw)
            with FixedEnergies(en):
                return H.load_session(sdir, tmp, 1, DRAFTS[1]["units"], CAL_RULES, GPUS, COUNTS, SCOPE_CFG)

    def m(lid, f):
        return lambda rec: f(rec) if rec["unit_id"] == lid else None
    assert raises(H.HeldoutInputError, run, mutate=m("s1-b1-p02", lambda r: r.update(R=float(r["R"]))))
    assert raises(H.HeldoutInputError, run, mutate=m("s1-b1-p02", lambda r: r.update(R=r["R"] + 1)), match="frozen exact")
    assert raises(H.HeldoutInputError, run, mutate=m("s1-b1-p02", lambda r: r["region"].update(R=r["R"] - 1)))
    assert raises(H.HeldoutInputError, run, mutate=m("s1-b1-p02", lambda r: r["unit"].update(arm="B")),
                  match="differs from the frozen schedule")
    assert raises(H.HeldoutInputError, run, mutate=m("s1-b1-p02", lambda r: r.update(gpus=[4, 6])))
    dup = {"unit_id": "s1-b1-p01", "launch_id": "s1-b1-p01", "accepted": False, "qc_status": "INTERRUPTED_UNIT"}
    assert raises(H.HeldoutInputError, run, extra=[dup], match="no replacements")
    alien = {"unit_id": "C1r1-06", "launch_id": "C1r1-06", "accepted": True}
    assert raises(H.HeldoutInputError, run, extra=[alien], match="not in the frozen schedule")
    # a session stopped by the readiness gate: the unit is logged, later units are NOT_RUN; nothing is replaced
    stop = {"unit_id": "s1-b2-p01", "attempt": 0, "qc_status": "NOT_RUN_READINESS_NOT_ACHIEVED",
            "readiness": {"status": "NOT_READY"}}
    later = [u["launch_id"] for u in DRAFTS[1]["units"] if u["block"] == 2]
    ld = run(drop=later, extra=[stop])
    st = {u["launch_id"]: u["status"] for u in ld["units"]}
    assert st["s1-b2-p01"] == "NOT_ACCEPTED" and sum(v == "NOT_RUN" for v in st.values()) == 11
    assert sum(v == "USED" for v in st.values()) == 12
    # offline evidence checks (same rules as acceptance) on an accepted record
    ld = run(mutate=m("s1-b1-p01", lambda r: r["evidence"].update(status="UNVERIFIED")))
    assert "EVIDENCE_NOT_VERIFIED" in next(u for u in ld["units"] if u["launch_id"] == "s1-b1-p01")["reasons"]
    cl = next(u for u in DRAFTS[1]["units"] if u["arm"] == "C")["launch_id"]
    ld = run(mutate=m(cl, lambda r: r["evidence"].update(actual=[{"algo": "RING", "proto": "SIMPLE", "nChannels": 4}])))
    assert "EXECUTION_NOT_AS_FORCED" in next(u for u in ld["units"] if u["launch_id"] == cl)["reasons"]


@strict
def test_real_calibration_only_for_interface_and_regression_checks():
    if not CAL_JSON.exists() or not CAL_DIR.exists():
        return                                       # the frozen calibration is not present in this checkout
    cal = json.loads(CAL_JSON.read_text())
    rules = H.readiness_rules_from_calibration(cal)
    assert float(rules["t_launch_c"]) == 26.0 and H._norm_bands(rules["bands"]) == H._norm_bands(CAL_RULES["bands"])
    q1 = (cal.get("result") or cal)["q1_primary"]["doses"]
    assert not q1["D100"]["bounded_for_adequacy"] and not q1["D500"]["bounded_for_adequacy"]
    assert raises(H.HeldoutInputError, H.load_session, CAL_DIR, REPO, 1, DRAFTS[1]["units"], CAL_RULES, GPUS, COUNTS,
                  match="never relabeled")
    # scope rule on real records: the frozen calibration config (raw UUIDs) vs its session summary (aliases)
    cfg = yaml.safe_load((REPO / "configs/stage2/calib01.yaml").read_text())
    obs = json.loads((CAL_DIR / "session_summary.json").read_text())["scope_observed"]
    assert H.scope_check(cfg["scope_expected"], obs) == []
    assert H.scope_check(dict(cfg["scope_expected"], uuids=cfg["scope_expected"]["uuids"][::-1]), obs) != []


@strict
def test_module_is_analysis_only_and_the_shared_code_role_is_unchanged():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(REPO / "src")
    p = subprocess.run([sys.executable, "-W", "error", "-c",
                        "import sys, ncclenergy.v2.stage2_heldout; print('pynvml' in sys.modules)"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip() == "False", p.stderr[-2000:]
    src = (REPO / "src/ncclenergy/v2/stage2_runner.py").read_text()
    assert "import stage2_heldout" not in src and "from .stage2_heldout" not in src
    assert S.MANDATORY_ANALYSIS["Freeze-H"] == ("src/ncclenergy/v2/stage2_heldout.py", "src/ncclenergy/v2/stage2_decisions.py")
    if FREEZE_C.exists():
        fr = json.loads(FREEZE_C.read_text())
        for rel in fr["roles"]["code"]:
            assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == fr["files"][rel]["sha256"], rel
        assert "src/ncclenergy/v2/stage2_heldout.py" not in fr["roles"]["code"]


@strict
def test_labels_and_scope_statement():
    assert "unadjusted" in H.LABELS["tost"] and "intersection-union" in H.LABELS["tost"]
    assert H.LABELS["ci95"] == "descriptive (unadjusted)" and "Holm" in H.LABELS["difference"]
    assert "not global optima" in H.SCOPE_STATEMENT and "this pair and host" in H.SCOPE_STATEMENT
    assert all(v.startswith("NOT_COMPUTED") for v in H.NOT_COMPUTED.values())
    assert set(H.NOT_COMPUTED) >= {"K2_historical_replay", "secondary_covariate_analysis", "E_samp_hold_inst",
                                   "delay_shifted_secondary_estimator"}
    assert "sampled_energy_hold_J exists" in H.NOT_COMPUTED["E_samp_hold_inst"]            # wiring missing, not helper
    from ncclenergy.v2 import stage2_energy
    assert callable(stage2_energy.sampled_energy_hold_J)
    assert "sensor-only" in H.INTERPRETATION_LIMITS["HD1"] and "selected" in H.INTERPRETATION_LIMITS["HD4"]
    assert "recorded READY status only" in H.INTERPRETATION_LIMITS["R1"]


# ------------------------------------------------------------------ PRODUCTION analysis: strict provenance boundary

PROD_READY = all(p.exists() for p in (CAL_JSON, FREEZE_C, STAGE2 / "protocol/STAGE2_HELDOUT1_PROTOCOL.md",
                                       REPO / "configs/stage2/calib01.yaml"))
CAL_CFG = yaml.safe_load((REPO / "configs/stage2/calib01.yaml").read_text()) if PROD_READY else {}
FLAT = [[-0.0005, -0.0005], [0.0, 0.0], [0.0005, 0.0005]]
ALL_EQUIV = dict(BASE, X3={10.0: FLAT, 20.0: FLAT, 40.0: FLAT}, G=[[0.0, 0.0], [0.0005, 0.0005], [0.001, 0.001]])


def _rel(p):
    """Path relative to REPO, as the runner and the freeze records write it."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(Path("..") / p.relative_to(REPO.resolve().parent))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _final_cfg(s, **over):
    """DRAFT + only the finalization keys (calibration pair, calibration-derived rules); synthetic, not an authorization."""
    cfg = copy.deepcopy(DRAFTS[s])
    cfg.update(allocation=list(CAL_CFG["allocation"]), placements=[{"id": "P2_NVL", "gpus": list(CAL_CFG["allocation"])}],
               allocation_confirmation={"confirmed_by": "synthetic test fixture (not an authorization)"},
               scope_expected=copy.deepcopy(CAL_CFG["scope_expected"]), freeze_record="(synthetic test)",
               readiness_rules=copy.deepcopy(CAL_RULES))
    cfg.update(copy.deepcopy(over))
    return cfg


def _make_h(tmp, s, cfg_path, cal_path=CAL_JSON):
    """An approved Freeze-H for a synthetic test (temp dir only), binding the real frozen files like the gate builder."""
    fc = json.loads(FREEZE_C.read_text())
    analysis = [f"src/ncclenergy/v2/{m}.py" for m in H.ANALYSIS_MODULES]
    roles = {"config": _rel(cfg_path), "protocol": H.PROTOCOL_REL, **{k: fc["roles"][k] for k in S.SHARED_ROLES},
             "analysis": analysis, "calibration_analysis": _rel(cal_path),
             "schedule": [f"configs/stage2/heldout_s{s}_DRAFT.yaml"]}
    rels = [roles["config"], roles["protocol"], roles["calibration_analysis"]] + roles["schedule"] + analysis + \
        [r for k in S.SHARED_ROLES for r in H._listify(roles[k])]
    rec = {"schema": S.FREEZE_SCHEMA, "kind": "Freeze-H", "session": s,
           "approved_by_user": {"synthetic_test_fixture": "not a real approval"},
           "roles": roles, "files": {r: {"sha256": _sha(REPO / r)} for r in dict.fromkeys(rels)},
           "external": fc["external"], "parent": {"path": "../protocol/FREEZE_C_CALIB1.json", "sha256": _sha(FREEZE_C)}}
    p = tmp / f"freeze_h_s{s}.json"
    p.write_text(json.dumps(rec))
    return p


def _rewrite(path, f):
    rec = json.loads(Path(path).read_text())
    f(rec)
    Path(path).write_text(json.dumps(rec))


def _production(tmp, sessions=(1, 2, 3), with_data=None, p=BASE, mutate=None, summary=None, cfg_over=None,
                cal_path=CAL_JSON):
    """Real-format synthetic sessions + H-bound config files + approved test Freeze-H records, all in a temp dir."""
    with_data = sessions if with_data is None else with_data
    tmp.mkdir(parents=True, exist_ok=True)
    scope_obs = json.loads(redact(json.dumps(CAL_CFG["scope_expected"])))     # summaries store UUID aliases
    dirs, frs, en = {}, {}, {}
    for s in sessions:
        cp = tmp / f"heldout_s{s}.yaml"
        cp.write_text(yaml.safe_dump(_final_cfg(s, **((cfg_over or {}).get(s, {}))), sort_keys=False))
        frs[s] = _make_h(tmp, s, cp, cal_path)
        if s in with_data:
            dirs[s], e = _write_session(tmp / "data", s, p=p, scope=scope_obs, base=REPO, config_sha=_sha(cp),
                                        mutate=mutate, summary=summary)
            en.update(e)
    return dirs, frs, en


@strict
def test_production_happy_path_real_format_q1_ceiling_and_freeze_coverage():
    if not PROD_READY:
        return
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        dirs, frs, en = _production(tmp)
        cfg1 = yaml.safe_load((tmp / "heldout_s1.yaml").read_text())
        with FixedEnergies(en):
            out = H.analyse_heldout(dirs, frs, expected_cfgs={1: cfg1})
            part = H.analyse_heldout({1: dirs[1], 2: dirs[2]}, frs)            # H for session 3, no data: NOT_STARTED
            assert raises(H.HeldoutInputError, H.analyse_heldout, dirs, {1: frs[1], 2: frs[2]},
                          match="without its Freeze-H")                       # data without its H: refused
        dirs2, frs2, en2 = _production(tmp / "eq", p=ALL_EQUIV)
        with FixedEnergies(en2):
            eq = H.analyse_heldout(dirs2, frs2)
    assert out["provenance_status"].startswith("VERIFIED_OFFLINE") and out["provenance"]["trust_anchors"] == "frozen defaults"
    assert out["unit_status_counts"] == {"USED": 72, "NOT_ACCEPTED": 0, "EXCLUDED": 0, "NOT_RUN": 0}
    pr = out["estimators"]["primary"]
    assert [pr["HD3"][L]["verdict"] for L in ("10", "20", "40")] == ["COMPATIBLE_WITH_EQUIVALENCE",
                                                                     "RESOLVED_MEANINGFUL_DIFFERENCE",
                                                                     "COMPATIBLE_WITH_EQUIVALENCE"]  # known answers kept
    assert abs(pr["summaries"]["HD3"]["10"]["session_means"]["2"] - 0.002) < 1e-12
    for s in ("1", "2", "3"):
        nvo = out["provenance"]["freeze_h"][s]["not_verified_offline"]
        assert all("NOT_VERIFIED_OFFLINE" in x or "DIFFERS_ON_THIS_HOST" in x for x in nvo)
    assert set(out["not_computed"]) >= {"K2_historical_replay", "secondary_covariate_analysis", "E_samp_hold_inst",
                                        "delay_shifted_secondary_estimator"}
    assert part["provenance"]["sessions"]["3"]["execution_status"] == "NOT_STARTED"
    assert sum(u["reasons"] == ["SESSION_NOT_STARTED"] for u in part["units"]) == 24
    assert all(e["verdict"] == "INSUFFICIENT_SESSIONS" for e in part["estimators"]["primary"]["HD3"].values())
    # the actual frozen Q1 caps every aggregate scope, even when every held-out criterion succeeds
    assert eq["aggregate_adequacy_ceiling"]["conditional_statistical"].startswith("NOT_MET for every aggregate scope")
    for a in eq["adequacy"].values():
        assert a["conditional_statistical"] == "NOT_MET" and a["unmet_statistical"] == ["Q1 bounded (D100 and D500)"]
        assert a["status"] == "NOT_ESTABLISHED"
    assert all(e["verdict"] == "COMPATIBLE_WITH_EQUIVALENCE" for e in eq["estimators"]["primary"]["HD3"].values())


def _refuses(match, **kw):
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        edit = kw.pop("edit", None)
        dirs, frs, en = _production(tmp, **kw)
        if edit:
            edit(tmp, dirs, frs)
        with FixedEnergies(en):
            return raises(H.HeldoutInputError, H.analyse_heldout, dirs, frs, match=match)


def _on(lid, f):
    return lambda r: f(r) if r["unit_id"] == lid else None


@strict
def test_production_refuses_wrong_or_missing_identity_and_changed_w():
    if not PROD_READY:
        return
    assert _refuses("CONFIG_IDENTITY", cfg_over={2: {"implementation_id": "stage2-other"}})
    assert _refuses("permitted frozen identity", summary={"method_id": "v2.4-stage2-region"})
    assert _refuses("RECORD_METHOD_ID", mutate=_on("s1-b1-p03", lambda r: r.update(method_id="v2.4-stage2-region")))
    assert _refuses("RECORD_CAMPAIGN", mutate=_on("s2-b2-p05", lambda r: r.update(campaign="v2.5-calib-01")))
    assert _refuses("ACCEPTED_RECORD_LACKS_MODE", mutate=_on("s2-b1-p01", lambda r: r.pop("mode")))
    assert _refuses("ACCEPTED_RECORD_LACKS_IMPLEMENTATION_ID", mutate=_on("s3-b1-p07", lambda r: r.pop("implementation_id")))
    assert _refuses("analysed pair", mutate=_on("s1-b2-p02", lambda r: r.update(gpus=[5, 4])))
    assert _refuses("w 95605 is not the frozen exact count 95604",
                    mutate=_on(next(u["launch_id"] for u in DRAFTS[1]["units"] if u["arm"] == "A"),
                               lambda r: r.update(w=r["w"] + 1)))
    assert _refuses("accepted record without its executed unit", mutate=_on("s1-b1-p04", lambda r: r.pop("unit")))


@strict
def test_production_config_is_the_bound_bytes_and_equals_the_frozen_draft():
    if not PROD_READY:
        return
    cnt = copy.deepcopy(COUNTS)
    cnt["B"]["w"] += 1
    assert _refuses("CONFIG_DIFFERS_FROM_FROZEN_DRAFT counts", cfg_over={1: {"counts": cnt}})
    units = copy.deepcopy(DRAFTS[3]["units"])
    units[4]["pause_ns"] += 1
    assert _refuses("CONFIG_DIFFERS_FROM_FROZEN_DRAFT units", cfg_over={3: {"units": units}})
    assert _refuses("READINESS_RULES", cfg_over={1: {"readiness_rules": dict(CAL_RULES, t_launch_c=27)}})
    assert _refuses("ALLOCATION", cfg_over={2: {"allocation": [6, 7], "placements": [{"id": "X", "gpus": [6, 7]}]}})
    assert _refuses("SCOPE_EXPECTED", cfg_over={2: {"scope_expected": dict(CAL_CFG.get("scope_expected") or {},
                                                                          driver="550.54.15")}})
    assert _refuses("not the Freeze-H-bound config", summary={"config_sha256": "0" * 64})

    def changed_after_freeze(tmp, dirs, frs):
        with open(tmp / "heldout_s2.yaml", "a") as f:
            f.write("# edited after the freeze\n")
    assert _refuses("BOUND_FILE_CHANGED", edit=changed_after_freeze)
    assert _refuses("BOUND_FILE_MISSING", edit=lambda tmp, dirs, frs: (tmp / "heldout_s3.yaml").unlink())
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        dirs, frs, en = _production(tmp)
        altered = yaml.safe_load((tmp / "heldout_s1.yaml").read_text())
        altered["counts"]["A"]["R"]["40"] += 1                              # dictionary differs from the bound bytes
        with FixedEnergies(en):
            assert raises(H.HeldoutInputError, H.analyse_heldout, dirs, frs, expected_cfgs={1: altered},
                          match="differs from the Freeze-H-bound config bytes")


@strict
def test_production_freeze_h_coverage_validity_parent_and_analysis_binding():
    if not PROD_READY:
        return
    assert _refuses("without its Freeze-H", edit=lambda tmp, dirs, frs: frs.pop(2))
    assert _refuses("FREEZE_H_NOT_APPROVED",
                    edit=lambda tmp, dirs, frs: _rewrite(frs[2], lambda r: r.update(approved_by_user={})))
    assert _refuses("FREEZE_H_KIND", edit=lambda tmp, dirs, frs: _rewrite(frs[1], lambda r: r.update(kind="Freeze-C")))
    assert _refuses("FREEZE_H_SESSION", edit=lambda tmp, dirs, frs: _rewrite(frs[2], lambda r: r.update(session=3)))

    def token_only(tmp, dirs, frs):                     # a JSON carrying only a matching config hash token
        frs[1].write_text(json.dumps({"kind": "Freeze-H", "files": {_rel(tmp / "heldout_s1.yaml"):
                                                                    {"sha256": _sha(tmp / "heldout_s1.yaml")}}}))
    assert _refuses("FREEZE_H_NOT_A_RECORD", edit=token_only)
    assert _refuses("FREEZE_H_UNREADABLE", edit=lambda tmp, dirs, frs: frs.update({3: tmp / "missing.json"}))
    fs = STAGE2 / "protocol/FREEZE_S_SMOKE1.json"
    assert _refuses("PARENT_IS_NOT_THE_FROZEN_FREEZE_C", edit=lambda tmp, dirs, frs: _rewrite(
        frs[3], lambda r: r.update(parent={"path": "../protocol/FREEZE_S_SMOKE1.json", "sha256": _sha(fs)})))
    assert _refuses("ANALYSIS_ROLE_LACKS src/ncclenergy/v2/stage2_heldout.py", edit=lambda tmp, dirs, frs: _rewrite(
        frs[1], lambda r: r["roles"].update(analysis=r["roles"]["analysis"][1:])))

    def other_analysis(rec):
        rec["files"]["src/ncclenergy/v2/stage2_heldout.py"]["sha256"] = "0" * 64
    assert _refuses("ANALYSIS_IS_NOT_THE_RUNNING_CODE", edit=lambda tmp, dirs, frs: _rewrite(frs[2], other_analysis))

    def other_binary(rec):
        rec["files"][rec["roles"]["binary"]]["sha256"] = "0" * 64
    assert _refuses("SHARED_ROLE_DIFFERS_FROM_PARENT binary", edit=lambda tmp, dirs, frs: _rewrite(frs[3], other_binary))


@strict
def test_production_refuses_changed_calibration_q1_with_unchanged_q3():
    if not PROD_READY:
        return
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        tmp = Path(t)
        cal = json.loads(CAL_JSON.read_text())
        for D in ("D100", "D500"):
            cal["result"]["q1_primary"]["doses"][D]["bounded_for_adequacy"] = True     # Q3 untouched
        changed = tmp / "calibration_analysis.json"
        changed.write_text(json.dumps(cal))
        assert H.readiness_rules_from_calibration(cal) == H.readiness_rules_from_calibration(
            json.loads(CAL_JSON.read_text()))
        dirs, frs, en = _production(tmp, cal_path=changed)
        with FixedEnergies(en):
            assert raises(H.HeldoutInputError, H.analyse_heldout, dirs, frs,
                          match="CALIBRATION_ANALYSIS_NOT_THE_FROZEN_BYTES")


@strict
def test_unfrozen_regression_drafts_refused_as_drafts_and_executed_configs_only_as_expired():
    """Replaces nothing: the frozen test_cli_check_refuses_the_drafts stays unchanged (and fails since smoke01.yaml and
    calib01.yaml were finalized). This shows the correct current behaviour of the runner's `check` command, run like the
    frozen test in a fresh process that loads only the runner's code (never the analysis module)."""
    seen = set()
    for c in sorted((REPO / "configs/stage2").glob("*.yaml")):
        if not (c.name.endswith("_DRAFT.yaml") or c.name in ("smoke01.yaml", "calib01.yaml")):
            continue
        p = subprocess.run([sys.executable, "-W", "error", "-m", "ncclenergy.v2.stage2_runner", "check", "--campaign", str(c)],
                           cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 1, (c, p.stderr[-2000:])
        r = json.loads(p.stdout)["refusals"]
        if c.name.endswith("_DRAFT.yaml"):
            assert any("allocation_confirmation missing" in x for x in r) and \
                any("freeze_record missing" in x for x in r) and any("scope_expected" in x for x in r), (c, r)
        else:
            assert r == ["now is outside the confirmed access window"], (c, r)
        seen.add(c.name)
    assert {"smoke01.yaml", "calib01.yaml", "heldout_s1_DRAFT.yaml"} <= seen


# ------------------------------------------------------------------ the unchanged unit_energies on runner-format data

_FAKE = {}


def _fake_session() -> Path:
    """One fake-NVML held-out session from the real runner code (2 launches, A and C at 2.5 s), made once."""
    if "root" not in _FAKE:
        root = Path(tempfile.mkdtemp(dir=REPO / "logs", prefix="heldout_iface_"))
        atexit.register(shutil.rmtree, root, True)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith(("FAKE_", "NCCLE_"))}
        env["PYTHONPATH"] = f"{REPO / 'tests/v2/fake_nvml'}:{REPO / 'src'}"
        p = subprocess.run([sys.executable, "-W", "error", str(REPO / "tests/v2/helpers/stage2_session.py"), "heldout",
                            str(root / "run")], env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-4000:]
        _FAKE["root"] = root
    return _FAKE["root"] / "run/data/session_1"


FAKE_RULES = {"t_launch_c": 45, "bands": {g: {"t_region_c": [38, 45], "clk_mhz": [1880, 1950], "pstates": [8]}
                                          for g in ("4", "5")}}
FAKE_COUNTS = {a: {"w": 95597, "R": {"2.5": 47799, "10": 191193, "20": 382385, "40": 764769}} for a in "ABC"}


def _fake_copy(edit=None) -> tuple[Path, list[dict]]:
    """A copy of the fake session (test copy only) with optional edits of one unit's files; attempts point to the copy."""
    src = _fake_session()
    dst = Path(tempfile.mkdtemp(dir=src.parents[2], prefix="copy_")) / "session_1"
    shutil.copytree(src, dst)
    old, new = str(src.relative_to(REPO)), str(dst.relative_to(REPO))
    lines = (dst / "attempts.jsonl").read_text().replace(old, new)
    (dst / "attempts.jsonl").write_text(lines)
    recs = [json.loads(x) for x in lines.splitlines()]
    if edit:
        edit(dst / "units" / recs[0]["unit_id"])
    return dst, [dict(r["unit"], block=1) for r in recs]


def _load(sdir, sched):
    scope = json.loads((sdir / "session_summary.json").read_text())["scope_observed"]
    return {u["launch_id"]: u for u in H.load_session(sdir, REPO, 1, sched, FAKE_RULES, GPUS, FAKE_COUNTS, scope)["units"]}


def _edit_region(f):
    def ed(d):
        reg = json.loads((d / "region.jsonl").read_text().splitlines()[0])
        f(reg)
        (d / "region.jsonl").write_text(json.dumps(reg) + "\n")
    return ed


def _edit_telemetry(f):
    def ed(d):
        with gzip.open(d / "telemetry.csv.gz", "rt") as fh:
            rows = list(csv.reader(fh))
        rows = [rows[0]] + f(rows[0], rows[1:])
        with gzip.open(d / "telemetry.csv.gz", "wt", newline="") as fh:
            csv.writer(fh).writerows(rows)
    return ed


@strict
def test_interface_on_runner_format_data_all_estimators_and_exact_R():
    sdir, sched = _fake_copy()
    by = _load(sdir, sched)
    assert [u["status"] for u in by.values()] == ["USED", "USED"]
    for u in by.values():
        assert u["R"] == 47799 and all(u["J"][k] == u["E"][k] / 47799 for k in H.ESTIMATORS) and not u["unavailable"]
    assert {u["exec_key"] for u in by.values()} == {"RING/SIMPLE/2", "RING/LL128/2"}


@strict
def test_invalid_period_counter_and_coverage_make_only_that_estimator_unavailable():
    brackets = [(-0.1, 0.1), (99.9, 100.1), (199.9, 200.1), (300.0, 309.0)]       # period outside the hard interval

    def period(reg):
        t0 = reg["gpus"][0]["edges"][0][0]
        reg["gpus"][0]["edges"] = [[t0 + int(a * 1e6), t0 + int(b * 1e6)] for a, b in brackets]
    u = next(iter(_load(*_fake_copy(_edit_region(period))).values()))
    assert u["J"]["primary"] is None and u["unavailable"]["primary"][0].startswith("RANK0: PRIMARY_PERIOD_INCOMPATIBLE")
    assert all(u["J"][k] is not None for k in ("raw_counter", "nominal_period", "sampled_instant"))

    def counter(reg):
        reg["gpus"][1]["e_end_mJ"] = reg["gpus"][1]["e_start_mJ"] - 5
    u = next(iter(_load(*_fake_copy(_edit_region(counter))).values()))
    assert u["J"]["raw_counter"] is None and u["J"]["primary"] is None and u["J"]["nominal_period"] is None
    assert "RANK1: counter decreased" in u["unavailable"]["primary"][0]
    assert u["J"]["sampled_instant"] is not None

    def gap(head, rows):
        i, g = head.index("t_inst_a_ns"), head.index("gpu")
        ts = sorted(int(r[i]) for r in rows if r[g] == "5")
        mid = ts[len(ts) // 2]
        return [r for r in rows if not (r[g] == "5" and mid - 200_000_000 < int(r[i]) < mid + 200_000_000)]
    u = next(iter(_load(*_fake_copy(_edit_telemetry(gap))).values()))
    assert u["J"]["sampled_instant"] is None and u["unavailable"]["sampled_instant"][0].startswith("RANK1: ")
    assert all(u["J"][k] is not None for k in ("primary", "raw_counter", "nominal_period"))


def _ramp(head, rows):
    """GPU 4 POWER_INSTANT made a ramp in its own row order (test copy), so a time shift changes the integral."""
    w, g = head.index("power_inst_w"), head.index("gpu")
    out, k = [], 0
    for r in rows:
        r = list(r)
        if r[g] == "4":
            r[w], k = f"{100.0 + 2.0 * k:.3f}", k + 1
        out.append(r)
    return out


def _shift(col_a, col_b, delta, gpu=None):
    def f(head, rows):
        idx = [head.index(c) for c in (col_a, col_b) if c]
        g = head.index("gpu")
        out = []
        for r in _ramp(head, rows):
            if gpu is None or r[g] == gpu:
                for i in idx:
                    r[i] = str(int(r[i]) + delta)
            out.append(r)
        return out
    return f


@strict
def test_clock_domains_field_timestamps_never_used_as_sample_time():
    base = next(iter(_load(*_fake_copy(_edit_telemetry(_ramp))).values()))
    assert base["J"]["sampled_instant"] is not None
    # CLOCK_REALTIME-epoch field timestamps (us) shifted by 1 h: the sampled estimate is identical
    u = next(iter(_load(*_fake_copy(_edit_telemetry(_shift("inst_field_ts_us", None, 3_600_000_000)))).values()))
    assert u["E"] == base["E"]
    # CLOCK_MONOTONIC query brackets of GPU 4 shifted by 0.2 s: the sampled estimate changes (or loses coverage)
    u = next(iter(_load(*_fake_copy(_edit_telemetry(_shift("t_inst_a_ns", "t_inst_b_ns", 200_000_000, "4")))).values()))
    assert u["J"]["sampled_instant"] is None or abs(u["E"]["sampled_instant"] / base["E"]["sampled_instant"] - 1) > 1e-6
    assert u["E"]["primary"] == base["E"]["primary"] and u["E"]["raw_counter"] == base["E"]["raw_counter"]
