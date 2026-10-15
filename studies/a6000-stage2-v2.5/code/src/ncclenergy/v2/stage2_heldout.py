"""Stage 2 held-out analysis (offline, CPU only). Never imported by the runner; it imports only pure constants and pure
rules from stage2_runner (pynvml is imported there only inside acquisition functions, never called here).

PRODUCTION entry point: analyse_heldout(session_dirs, freeze_records). It reads every config and the calibration
analysis only from the files bound by approved Freeze-H records and refuses the analysis on any identity, binding,
config, count or provenance inconsistency (portable artifact checks only: no NVML, no GPU, no live environment; the
identities that cannot be verified offline are listed in the output). LOW-LEVEL / synthetic test interfaces, whose
output is labelled UNVERIFIED: analyse_sessions_unverified, load_session, analyse_units.

What it computes, from the three frozen held-out sessions (3 sessions x 2 blocks x 12 launches = 72):
  * unit quantities J = E / R (exact integer R, stage2_energy.joules_per_op); E is the participating-GPU sum from the
    unchanged stage2_calibration.unit_energies: primary E_aligned(P_hat); sensitivities E_counter_raw,
    E_aligned(P_nom), E_samp,trap,inst. Only runner-accepted units whose acceptance re-verifies offline: the runner's
    RECORDED R1 status is READY (no readiness-trace replay), R2 recomputed from the recorded pre-region state with the
    frozen bands, evidence, exact R and w, work end. An unavailable estimator stays unavailable with its own reason;
    nothing is substituted, imputed, trimmed or replaced;
  * H-D1 (9 contrasts ln(J_L/J_40), L in 2.5/10/20 x A/B/C, Holm, ln 1.003), H-D2 (per-block coefficient of 1{C}
    in OLS ln(E_samp,trap,inst / E) ~ 1 + 1/T + 1{C}, ln 1.003), H-D3 (ln(J_B/J_A) at 10/20/40 s, control-valid
    blocks, Holm, ln 1.005), H-D4 (ln(J_C/J_A) at 40 s, precision 95 % half-width <= ln 1.004);
  * block contrast -> session mean of available blocks -> t on the 3 session means (df 2; S = 3 required, otherwise
    INSUFFICIENT_SESSIONS) with the unchanged stage2_decisions helpers. Difference and minimum-effect p are Holm-adjusted
    within each family; 95 % intervals are descriptive (unadjusted); each TOST p is unadjusted; a joint equivalence
    claim exists only through the intersection-union adequacy rule;
  * R3 (stage2_readiness.thermal_balance, per session/length/pair/GPU) and control validity as verdict overrides;
  * adequacy (stage2_decisions.adequacy) for every predefined scope. With the actual frozen calibration Q1 (primary not
    bounded) every aggregate scope is conditional_statistical NOT_MET and physical adequacy NOT_ESTABLISHED whatever the
    held-out results (reported as aggregate_adequacy_ceiling); individual H-D1..H-D4 results can still be informative.
Scope of every result: a conditional comparison of NVML-reported participating-GPU board energy of the tested arms on
the calibration pair and host (pilot). Not global optima, not externally calibrated physical energy, no training claim.
NOT_COMPUTED (accepted prospective scope reductions, also in the machine-readable output): K2 historical replay,
secondary covariate analysis, sampled-hold sensitivity, delay-shifted estimator, systematic bounds. Rules where the
protocol was silent: prep/heldout1/CHANGE_LEDGER.md (N1-N18, N11a); production provenance:
prep/heldout1_r1/CHANGE_LEDGER_r1.md (P1-P7, D1-D7)."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .common import REPO, redact
from .stage2_calibration import load_units, unit_energies
from .stage2_decisions import (D1_D2, D3, D4_HW, HD1_LENGTHS, HD3_LENGTHS, SYSTEMATIC_COMPONENTS, adequacy, family,
                               precision, session_summary)
from .stage2_energy import joules_per_op
from .stage2_readiness import pre_region_check, thermal_balance
from .stage2_runner import (ARM_EXPECTED, FREEZE_SCHEMA, IMPLEMENTATION_ID, MANDATORY_ANALYSIS, METHOD_ID, SHARED_ROLES,
                            scope_problems)

SESSIONS = (1, 2, 3)
BLOCKS = (1, 2)
ARMS = ("A", "B", "C")
LENGTHS = (2.5, 10.0, 20.0, 40.0)
UNITS_PER_SESSION = 24
ESTIMATORS = {"primary": "E_aligned_hat_J", "raw_counter": "E_counter_raw_J", "nominal_period": "E_aligned_nom_J",
              "sampled_instant": "E_samp_inst_J"}
SAMPLED = "sampled_instant"
SCOPES = ("BROAD", "SAME_EXEC_L10", "SAME_EXEC_L20", "SAME_EXEC_L40", "CROSS_PROTOCOL_40")
H2_MIN_UNITS = 4
SCOPE_STATEMENT = ("conditional comparison of NVML-reported participating-GPU board energy of the tested arms on this "
                   "pair and host (pilot, 3 sessions); not global optima, not externally calibrated physical accuracy, "
                   "no training or real-model claim")
LABELS = {"difference": "two-sided and minimum-effect p Holm-adjusted within the family (p_holm, "
                        "p_beyond_margin_holm); decisions use adjusted p only",
          "ci95": "descriptive (unadjusted)",
          "tost": "per-contrast TOST p, unadjusted; a joint equivalence claim exists only via the intersection-union "
                  "adequacy rule",
          "conditional_descriptive": "t over all block contrasts, conditional on these sessions (descriptive)",
          "nonsignificance": "nonsignificance is never equivalence"}
NOT_COMPUTED = {
    "K2_historical_replay": "NOT_COMPUTED (accepted prospective scope reduction): the frozen measval_f4735b1.py reads the "
                            "historical campaign layout; no approved adapter to the Stage 2 session layout exists",
    "secondary_covariate_analysis": "NOT_COMPUTED (accepted prospective scope reduction): the protocol does not freeze "
                                    "the block-resampling count or seed",
    "E_samp_hold_inst": "NOT_COMPUTED (accepted prospective scope reduction): the frozen helper "
                        "stage2_energy.sampled_energy_hold_J exists, but its wiring into stage2_calibration.unit_energies "
                        "is missing and is not added in this stage",
    "delay_shifted_secondary_estimator": "NOT_COMPUTED: refused by calibration Q4 (NO_SECONDARY_ESTIMATOR)",
    "systematic_bounds": "NOT_COMPUTED: no hard-bound rule was frozen; every systematic component is UNKNOWN"}
INTERPRETATION_LIMITS = {
    "scope": SCOPE_STATEMENT,
    "HD1": "length dependence of J/op does not by itself identify a sensor-only mechanism: thermal and clock changes "
           "over the region can also change board energy",
    "HD4": "one selected C/A comparison (AllGather 2 MiB, 40 s, this pair); not evidence that global latency and energy "
           "optima differ",
    "R1": "offline R1 verifies the runner's recorded READY status only (no readiness-trace replay); R2 is recomputed "
          "from the recorded pre-region state with the frozen bands",
    "adequacy": "with the actual frozen calibration Q1 (primary not bounded) no aggregate scope can meet its statistical "
                "criteria and physical adequacy cannot be established; individual H-D1..H-D4 results can still be "
                "informative"}
# Production trust anchors (frozen with this module): permitted identity, parent Freeze-C, calibration analysis bytes and
# the cg1 schedule files. A caller may pass other anchors only for synthetic tests; the output then says so.
ANCHORS = {"implementation_id": IMPLEMENTATION_ID, "method_id": METHOD_ID,
           "parent_freeze_c_sha256": "45b127f3f29a65f64552f00525e13b26670226dac6ccd765cd78ae6aa390278c",
           "calibration_analysis_sha256": "152dd0a5a8cac1e8f587941eef1ac490c8095de073b2e0422580045fc59d89f4",
           "draft_sha256": {1: "2200fad1ff3ed25d9e3c0ed53e8eb3144b4f1bcdc579c9d221365e60078b00d3",
                            2: "28fbbb5a1d6d3cbb359ee9825a7ede227333b396181c461f8f4b4d1bfbeaddfb",
                            3: "e4656b69fbaf2b704bd5c7ae4b767c823b2569dfb359032bbdaae6bcbe10ae3d"}}
PROTOCOL_REL = "../protocol/STAGE2_HELDOUT1_PROTOCOL.md"
ANALYSIS_MODULES = ("stage2_heldout", "stage2_decisions", "stage2_calibration", "stage2_energy", "stage2_phase",
                    "stage2_readiness")
FINALIZATION_KEYS = ("allocation", "placements", "allocation_confirmation", "scope_expected", "freeze_record",
                     "readiness_rules")
OFFLINE_OPTIONAL_ROLES = ("binary", "plugin")      # compiled build outputs: verified when present, else listed


class HeldoutInputError(ValueError):
    """Inputs that are not the frozen held-out sessions (or fail provenance) are refused, never adapted."""


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _lkey(L) -> str:
    return f"{float(L):g}"


# ------------------------------------------------------------------ frozen inputs

def readiness_rules_from_calibration(cal: dict) -> dict:
    """The held-out readiness settings derived by the frozen calibration Q3 (status OK only)."""
    q3 = (cal.get("result") or cal).get("q3_readiness") or {}
    if q3.get("status") != "OK":
        raise HeldoutInputError(f"calibration Q3 status {q3.get('status')!r}: no readiness rules (amendment needed)")
    return {"t_launch_c": q3["t_launch_c"], "bands": q3["bands"]}


def _norm_bands(bands: dict) -> dict:
    return {int(g): {"t_region_c": [float(x) for x in b["t_region_c"]], "clk_mhz": [float(x) for x in b["clk_mhz"]],
                     "pstates": [int(x) for x in b["pstates"]]} for g, b in bands.items()}


def check_rules(rules: dict, cal: dict) -> None:
    exp = readiness_rules_from_calibration(cal)
    try:
        same = float(rules["t_launch_c"]) == float(exp["t_launch_c"]) and \
            _norm_bands(rules["bands"]) == _norm_bands(exp["bands"])
    except (KeyError, TypeError, ValueError):
        same = False
    if not same:
        raise HeldoutInputError("readiness rules differ from the calibration-derived rules frozen for held-out")


def schedule_from_configs(cfgs: dict) -> dict:
    """{session: [unit, ...]} from the held-out session configs (the frozen schedule). Each session: 24 launches with
    unique ids s<n>-..., blocks 1-2, one launch per arm x length in each block."""
    out = {}
    for s, cfg in cfgs.items():
        units = cfg.get("units") or []
        ids = [u.get("launch_id") for u in units]
        if len(units) != UNITS_PER_SESSION or len(set(ids)) != len(ids) or \
                not all(str(i).startswith(f"s{s}-") for i in ids) or sorted({u.get("block") for u in units}) != list(BLOCKS):
            raise HeldoutInputError(f"session {s}: schedule is not 24 unique s{s}- launches in blocks 1-2")
        for b in BLOCKS:
            cells = sorted((u["arm"], float(u["length_s"])) for u in units if u["block"] == b)
            if cells != sorted((a, L) for a in ARMS for L in LENGTHS):
                raise HeldoutInputError(f"session {s} block {b}: not one launch per arm x length")
        out[int(s)] = units
    return out


# ------------------------------------------------------------------ loading (provenance-checked)

def scope_check(expected: dict, observed: dict) -> list[str]:
    """The runner's scope rule. Configs carry raw GPU UUIDs; session summaries store them as the runner's stable aliases
    (common.redact), so the expected scope is aliased the same way before comparing."""
    return scope_problems(json.loads(redact(json.dumps(expected))), observed)


def _bad_energy(E) -> bool:
    return E is None or isinstance(E, bool) or not isinstance(E, (int, float)) or not math.isfinite(E) or E <= 0


def _estimator_reasons(en: dict, name: str) -> list[str]:
    """The specific reason an estimator is unavailable for this unit (per GPU), from unit_energies' per-GPU record."""
    why = []
    for p in en.get("per_gpu") or []:
        r = p.get("rank")
        if name == "primary" and p.get("aligned_hat") is None:
            why.append(f"RANK{r}: " + (p.get("aligned_hat_unavailable_reason") or p.get("error") or "ALIGNED_HAT_NONE"))
        elif name == "nominal_period" and p.get("aligned_nom") is None:
            why.append(f"RANK{r}: " + (p.get("error") or "ALIGNED_NOM_NONE"))
        elif name == "raw_counter" and p.get("raw_J") is None:
            why.append(f"RANK{r}: " + (p.get("error") or "RAW_COUNTER_INVALID (decrease or reset flag)"))
        elif name == "sampled_instant" and p.get("samp_inst_J") is None:
            why.append(f"RANK{r}: " + (p.get("samp_error") or "SAMPLED_NONE"))
    return why or [f"{ESTIMATORS[name]} unavailable"]


def _verify_evidence(rec: dict, arm: str) -> list[str]:
    """The frozen selected-execution checks, re-applied offline (same rules as the runner's acceptance)."""
    ev, why = rec.get("evidence") or {}, []
    act = {(x.get("algo"), x.get("proto")) for x in (ev.get("actual") or [])}
    if ev.get("status") != "VERIFIED":
        why.append("EVIDENCE_NOT_VERIFIED")
    if ARM_EXPECTED[arm] is not None and act != {ARM_EXPECTED[arm]}:
        why.append("EXECUTION_NOT_AS_FORCED")
    if len(act) != 1 or not ev.get("actual_key"):
        why.append("NOT_A_SINGLE_EXECUTION")
    if ev.get("n_matches_expected") is not True:
        why.append("OP_COUNT_NOT_R_PLUS_W_PLUS_1")
    return why


def _identity_problems(r: dict, identity: dict, gpus: list[int]) -> list[str]:
    """P2: an executed unit record (one with a `unit`) must carry the permitted identity, mode, campaign and GPU order.
    Required on accepted records; on other executed records a present value must still be consistent."""
    want = {"implementation_id": identity["implementation_id"], "method_id": identity["method_id"],
            "mode": "STAGE2_HELDOUT", "campaign": identity["campaign"], "gpus": list(gpus)}
    why = []
    for k, w in want.items():
        v = r.get(k)
        if v is None:
            if r.get("accepted"):
                why.append(f"ACCEPTED_RECORD_LACKS_{k.upper()}")
        elif (list(v) if k == "gpus" else v) != w:
            why.append(f"RECORD_{k.upper()}_{v!r}_IS_NOT_{w!r}")
    return why


def load_session(session_dir: Path, repo: Path, session: int, schedule_units: list[dict], rules: dict,
                 gpus: list[int], counts: dict, scope_expected: dict | None = None,
                 identity: dict | None = None) -> dict:
    """LOW-LEVEL (also used by the production path). Unit entries of one held-out session. Refuses anything that is not
    that frozen held-out session (calibration and smoke directories, other modes, unscheduled or duplicated launches, a
    changed unit definition, another GPU set, a scope mismatch, a non-integer or wrong R or w). With `identity`
    ({implementation_id, method_id, campaign}; the production path always passes it) the session summary and every
    executed record must carry that identity (P2). Accepted units whose acceptance does not re-verify offline are
    EXCLUDED with an ACCEPTANCE_INCONSISTENT reason (reported, never repaired). Every input file used is hashed."""
    session_dir, repo = Path(session_dir), Path(repo)
    summ_p = session_dir / "session_summary.json"
    if not summ_p.exists() or not (session_dir / "attempts.jsonl").exists():
        raise HeldoutInputError(f"{session_dir}: no session_summary.json / attempts.jsonl")
    summ = json.loads(summ_p.read_text())
    if summ.get("mode") != "STAGE2_HELDOUT":
        raise HeldoutInputError(f"{session_dir}: mode {summ.get('mode')!r} is not STAGE2_HELDOUT (calibration and "
                                "smoke data are never relabeled as held-out)")
    if identity is not None and (summ.get("implementation_id"), summ.get("method_id")) != \
            (identity["implementation_id"], identity["method_id"]):
        raise HeldoutInputError(f"{session_dir}: session identity {summ.get('implementation_id')!r}/"
                                f"{summ.get('method_id')!r} is not the permitted frozen identity")
    if scope_expected is not None:
        sp = scope_check(scope_expected, summ.get("scope_observed") or {})
        if sp:
            raise HeldoutInputError(f"session {session}: {sp}")
    recs = load_units(session_dir)
    sched = {u["launch_id"]: u for u in schedule_units}
    seen, out = set(), []
    hashes = {"session_summary.json": sha256_file(summ_p), "attempts.jsonl": sha256_file(session_dir / "attempts.jsonl")}
    bands = _norm_bands(rules["bands"])
    for r in recs:
        lid = r.get("unit_id") or r.get("launch_id")
        if r.get("mode") not in (None, "STAGE2_HELDOUT"):
            raise HeldoutInputError(f"{lid}: record mode {r.get('mode')!r}")
        if lid not in sched:
            raise HeldoutInputError(f"{lid}: not in the frozen schedule of session {session}")
        if lid in seen:
            raise HeldoutInputError(f"{lid}: more than one attempt (max_attempts is 1; no replacements)")
        seen.add(lid)
        su, ru = sched[lid], r.get("unit")
        if ru is not None and any(ru.get(k) != su.get(k) for k in ("arm", "length_s", "pause_ns")):
            raise HeldoutInputError(f"{lid}: executed unit differs from the frozen schedule")
        if r.get("gpus") is not None and list(r["gpus"]) != list(gpus):
            raise HeldoutInputError(f"{lid}: GPU set {r['gpus']} is not the analysed pair {gpus}")
        if identity is not None:
            if r.get("accepted") and ru is None:
                raise HeldoutInputError(f"{lid}: accepted record without its executed unit")
            if ru is not None and _identity_problems(r, identity, gpus):
                raise HeldoutInputError(f"{lid}: {_identity_problems(r, identity, gpus)}")
        Rn = int(counts[su["arm"]]["R"][_lkey(su["length_s"])])
        wn = int(counts[su["arm"]]["w"])
        for k, want in (("R", Rn), ("w", wn)):           # P6: checked against the config, never changed
            v = r.get(k)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v != want):
                raise HeldoutInputError(f"{lid}: {k} {v!r} is not the frozen exact count {want}")
        e = {"launch_id": lid, "session": session, "block": int(su["block"]), "arm": su["arm"],
             "length_s": float(su["length_s"]), "status": "USED", "reasons": []}
        if not r.get("accepted"):
            e.update(status="NOT_ACCEPTED", reasons=list(r.get("not_accepted_reasons") or [r.get("qc_status")]))
            out.append(e)
            continue
        R = r.get("R")
        if R is None or r.get("w") is None or (r.get("region") or {}).get("R") != Rn:
            raise HeldoutInputError(f"{lid}: accepted record R {R!r} / w {r.get('w')!r} / region R "
                                    f"{(r.get('region') or {}).get('R')!r} is not the frozen exact count {Rn} / {wn}")
        why = []
        if (r.get("readiness") or {}).get("status") != "READY":
            why.append("R1_NOT_READY")
        state = {int(g): v for g, v in (r.get("pre_region_state") or {}).items()}
        r2 = pre_region_check(state, bands, gpus)
        if not r2["ok"]:
            why += ["R2"] + r2["reasons"]
        why += _verify_evidence(r, su["arm"])
        if why:
            e.update(status="EXCLUDED", reasons=["ACCEPTANCE_INCONSISTENT"] + why)
            out.append(e)
            continue
        en = unit_energies(r, repo)
        if en.get("status") != "OK" or en.get("R") != Rn or any(x.startswith("NO_WORK_END") for x in en.get("reasons", [])):
            e.update(status="EXCLUDED", reasons=["ACCEPTANCE_INCONSISTENT", f"ENERGY_INPUTS {en.get('status')}"]
                     + [x for x in en.get("reasons", []) if x.startswith("NO_WORK_END")])
            out.append(e)
            continue
        d = repo / r["dir"]
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            hashes[f"units/{lid}/{f.relative_to(d)}"] = sha256_file(f)
        J, E, unavailable = {}, {}, {}
        for name, key in ESTIMATORS.items():
            v = en.get(key)
            if _bad_energy(v):
                J[name] = E[name] = None
                unavailable[name] = _estimator_reasons(en, name) if v is None else [f"NONPOSITIVE_OR_NONFINITE {v!r}"]
            else:
                E[name], J[name] = float(v), joules_per_op(float(v), R)
        reg = r.get("region") or {}
        e.update(R=R, E=E, J=J, unavailable=unavailable, exec_key=(r.get("evidence") or {}).get("actual_key"),
                 T_region_s=reg.get("T_region_s"), t_region_gpu={g: (state.get(g) or {}).get("t_region_c") for g in gpus})
        out.append(e)
    for lid, su in sched.items():
        if lid not in seen:
            out.append({"launch_id": lid, "session": session, "block": int(su["block"]), "arm": su["arm"],
                        "length_s": float(su["length_s"]), "status": "NOT_RUN", "reasons": ["NOT_RUN"]})
    return {"session": session, "units": out, "execution_status": summ.get("execution_status"),
            "stop_reason": summ.get("stop_reason"), "implementation_id": summ.get("implementation_id"),
            "method_id": summ.get("method_id"), "config_sha256": summ.get("config_sha256"),
            "scope_observed": summ.get("scope_observed"), "input_sha256": hashes}


# ------------------------------------------------------------------ contrasts

def _index(units: list[dict]) -> dict:
    idx = {}
    for u in units:
        if u.get("status") == "USED":
            k = (u["session"], u["block"], u["arm"], float(u["length_s"]))
            if k in idx:
                raise HeldoutInputError(f"duplicate unit {k}")
            idx[k] = u
    return idx


def _J(idx, s, b, arm, L, est):
    u = idx.get((s, b, arm, float(L)))
    return None if u is None else (u.get("J") or {}).get(est)


def _ratio(num, den):
    return None if num is None or den is None else math.log(num / den)


def control_validity(units: list[dict]) -> dict:
    """Per (session, block) present in units: invalid if, at any length, A and B are both used and their verified
    execution keys (algorithm/protocol/channels) differ."""
    idx = _index(units)
    out = {}
    for s, b in sorted({(u["session"], u["block"]) for u in units}):
        bad = [L for L in LENGTHS if (s, b, "A", L) in idx and (s, b, "B", L) in idx
               and idx[(s, b, "A", L)].get("exec_key") != idx[(s, b, "B", L)].get("exec_key")]
        out[(s, b)] = {"valid": not bad, "differing_lengths": bad}
    return out


def h2_coefficient(rows: list[dict], est: str):
    """Coefficient of 1{C} in OLS ln(E_samp,trap,inst / E_est) ~ 1 + 1/T + 1{C} over one block's used units with both
    energies available (T = measured region duration)."""
    rows = [u for u in rows if (u.get("E") or {}).get(SAMPLED) and (u.get("E") or {}).get(est)
            and isinstance(u.get("T_region_s"), (int, float)) and math.isfinite(u["T_region_s"]) and u["T_region_s"] > 0]
    if len(rows) < H2_MIN_UNITS or not any(u["arm"] == "C" for u in rows) or all(u["arm"] == "C" for u in rows):
        return None, "TOO_FEW_UNITS_OR_NO_C_AND_NON_C"
    y = np.array([math.log(u["E"][SAMPLED] / u["E"][est]) for u in rows])
    X = np.column_stack([np.ones(len(rows)), [1.0 / u["T_region_s"] for u in rows],
                         [1.0 if u["arm"] == "C" else 0.0 for u in rows]])
    if np.linalg.matrix_rank(X) < 3:
        return None, "RANK_DEFICIENT"
    return float(np.linalg.lstsq(X, y, rcond=None)[0][2]), None


def block_contrasts(units: list[dict], est: str) -> dict:
    """{family: {member: {session: [x per block]}}} for one estimator, plus a log of every missing block contrast."""
    idx, ctrl = _index(units), control_validity(units)
    fam = {"HD1": {(a, L): {s: [] for s in SESSIONS} for a in ARMS for L in HD1_LENGTHS},
           "HD3": {L: {s: [] for s in SESSIONS} for L in HD3_LENGTHS},
           "HD4": {"C/A@40": {s: [] for s in SESSIONS}}}
    if est != SAMPLED:
        fam["HD2"] = {"C": {s: [] for s in SESSIONS}}
    log = []

    def put(f, m, s, b, x, why="MISSING"):
        (fam[f][m][s].append(x) if x is not None else log.append((f, m, s, b, why)))
    for s in SESSIONS:
        for b in BLOCKS:
            for a in ARMS:
                for L in HD1_LENGTHS:
                    put("HD1", (a, L), s, b, _ratio(_J(idx, s, b, a, L, est), _J(idx, s, b, a, 40.0, est)))
            for L in HD3_LENGTHS:
                if not ctrl.get((s, b), {"valid": True})["valid"]:
                    log.append(("HD3", L, s, b, "CONTROL_INVALID_BLOCK"))
                else:
                    put("HD3", L, s, b, _ratio(_J(idx, s, b, "B", L, est), _J(idx, s, b, "A", L, est)))
            put("HD4", "C/A@40", s, b, _ratio(_J(idx, s, b, "C", 40.0, est), _J(idx, s, b, "A", 40.0, est)))
            if "HD2" in fam:
                sb = [u for u in units if u.get("status") == "USED" and u["session"] == s and u["block"] == b]
                x, why = h2_coefficient(sb, est)
                put("HD2", "C", s, b, x, why)
    return {"families": fam, "excluded": log}


def analyse_units(units: list[dict], gpus: list[int], est: str = "primary") -> dict:
    """All held-out decisions for one estimator from unit entries (load_session output or synthetic fixtures)."""
    if est not in ESTIMATORS:
        raise ValueError(f"unknown estimator {est!r}")
    bc = block_contrasts(units, est)
    summ = {f: {m: session_summary(sv) for m, sv in members.items()} for f, members in bc["families"].items()}
    used = [u for u in units if u.get("status") == "USED"]
    r3 = thermal_balance([{"session": u["session"], "length_s": float(u["length_s"]), "arm": u["arm"],
                           "t_region_gpu": u.get("t_region_gpu") or {}} for u in used], gpus)

    def r3ok(L, pair):              # strict (adequacy): every session evaluated and balanced
        return all(r3.get((s, float(L), pair), {"ok": False})["ok"] for s in SESSIONS)

    def r3_failed(L, pair):         # override: evaluated and failed; a missing arm/session is missing data instead
        return any(not v["ok"] and v.get("reason") != "ARM_MISSING" for (s, L2, p), v in r3.items()
                   if L2 == float(L) and p == pair)
    ctrl = control_validity(units)
    control_ok = all(v["valid"] for v in ctrl.values())
    r3_ok = {L: r3ok(L, "B-A") and r3ok(L, "C-A") for L in LENGTHS}
    ov3 = {L: "INVALID_CONTROL_OR_MEASUREMENT" if not control_ok else
           ("THERMAL_COMPARABILITY_NOT_ESTABLISHED" if r3_failed(L, "B-A") else None) for L in HD3_LENGTHS}
    ov4 = "THERMAL_COMPARABILITY_NOT_ESTABLISHED" if r3_failed(40.0, "C-A") else None
    return {"estimator": est, "estimator_key": ESTIMATORS[est],
            "role": "PRIMARY" if est == "primary" else "SENSITIVITY (never promoted to primary; no decision or adequacy)",
            "labels": LABELS, "summaries": summ,
            "HD1": family(summ["HD1"], D1_D2),
            "HD2": family(summ["HD2"], D1_D2) if "HD2" in summ else "NOT_APPLICABLE (sampled sensitivity)",
            "HD3": family(summ["HD3"], D3, overrides={L: o for L, o in ov3.items() if o}),
            "HD4": family(summ["HD4"], D4_HW, overrides={"C/A@40": ov4} if ov4 else None),
            "HD4_precision": precision(summ["HD4"]["C/A@40"]),
            "HD3_precision": {L: precision(summ["HD3"][L]) for L in HD3_LENGTHS},
            "r3": r3, "r3_ok": r3_ok, "control": ctrl, "control_valid": control_ok, "excluded": bc["excluded"]}


def systematic_unknown() -> dict:
    return {c: {L: {"status": "UNKNOWN", "reason": "no hard-bound rule frozen"} for L in LENGTHS}
            for c in SYSTEMATIC_COMPONENTS}


def adequacy_all(primary: dict, q1: dict) -> dict:
    if primary.get("estimator") != "primary":
        raise ValueError("adequacy uses the primary estimator only")
    r = {"HD1": primary["HD1"], "HD2": primary["HD2"] if isinstance(primary["HD2"], dict) else {},
         "HD3": primary["HD3"], "HD4_precision": primary["HD4_precision"], "HD3_precision": primary["HD3_precision"],
         "q1": q1, "systematic": systematic_unknown(), "r3_ok": primary["r3_ok"], "control_valid": primary["control_valid"]}
    return {sc: adequacy(sc, r) for sc in SCOPES}


def _k(k) -> str:
    if isinstance(k, tuple):
        return "|".join(f"{x:g}" if isinstance(x, float) else str(x) for x in k)
    return f"{k:g}" if isinstance(k, float) else str(k)


def _jsonable(x):
    if isinstance(x, dict):
        return {_k(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ------------------------------------------------------------------ output assembly (shared)

def aggregate_adequacy_ceiling(calr: dict) -> dict:
    """D1: what the frozen calibration Q1 allows before any held-out data are seen."""
    doses = (calr.get("q1_primary") or {}).get("doses") or {}
    q1 = {D: {"bounded_for_adequacy": bool((doses.get(D) or {}).get("bounded_for_adequacy")),
              "counter_class": ((doses.get(D) or {}).get("counter") or {}).get("class")} for D in ("D100", "D500")}
    met = all(v["bounded_for_adequacy"] for v in q1.values())
    return {"q1_primary": q1,
            "conditional_statistical": "may be MET only if every scope criterion holds" if met else
            "NOT_MET for every aggregate scope (Q1 bounded is required by every scope; the frozen calibration Q1 primary "
            "is not bounded), whatever the held-out results",
            "physical": "NOT_ESTABLISHED (Q1 not bounded; every systematic component UNKNOWN)" if not met else
            "NOT_ESTABLISHED (every systematic component UNKNOWN)",
            "individual_results": "H-D1..H-D4 verdicts can still be informative as conditional NVML-reported comparisons"}


def _assemble(units: list[dict], gpus: list[int], calr: dict, prov: dict, sessions: list[int],
              provenance_status: str) -> dict:
    results = {est: analyse_units(units, gpus, est) for est in ESTIMATORS}
    adeq = adequacy_all(results["primary"], calr.get("q1_primary") or {})
    status = {st: sum(1 for u in units if u["status"] == st) for st in ("USED", "NOT_ACCEPTED", "EXCLUDED", "NOT_RUN")}
    out = {"provenance_status": provenance_status, "scope": SCOPE_STATEMENT, "sessions_present": sessions,
           "n_scheduled": UNITS_PER_SESSION * len(sessions), "unit_status_counts": status,
           "units": [{k: u.get(k) for k in ("launch_id", "session", "block", "arm", "length_s", "status", "reasons",
                                             "R", "E", "J", "unavailable", "exec_key")} for u in units],
           "estimators": results, "adequacy": adeq, "aggregate_adequacy_ceiling": aggregate_adequacy_ceiling(calr),
           "physical_adequacy": {"status": "ESTABLISHED" if any(a["status"] == "ESTABLISHED" for a in adeq.values())
                                 else "NOT_ESTABLISHED",
                                 "systematic": "every component UNKNOWN (no hard-bound rule frozen)"},
           "not_computed": NOT_COMPUTED, "interpretation_limits": INTERPRETATION_LIMITS, "provenance": prov}
    return json.loads(json.dumps(_jsonable(out)))


# ------------------------------------------------------------------ LOW-LEVEL entry point (synthetic tests only)

def analyse_sessions_unverified(session_dirs: dict, repo: Path, session_cfgs: dict, cal_analysis_path: Path,
                                gpus: list[int]) -> dict:
    """LOW-LEVEL / SYNTHETIC TEST INTERFACE. No Freeze-H, identity, config-bytes or calibration-anchor verification:
    the output is labelled provenance_status UNVERIFIED and is never a held-out result. Production: analyse_heldout."""
    if not session_dirs or not set(session_dirs) <= set(SESSIONS) or set(session_dirs) != set(session_cfgs):
        raise HeldoutInputError("sessions must be a non-empty subset of 1-3 with one config each")
    calr = json.loads(Path(cal_analysis_path).read_text())
    calr = calr.get("result") or calr
    sched = schedule_from_configs(session_cfgs)
    prov = {"calibration_analysis_sha256": sha256_file(cal_analysis_path), "sessions": {}}
    units, scopes, counts0 = [], [], None
    for s, d in sorted(session_dirs.items()):
        cfg = session_cfgs[s]
        if cfg.get("mode") != "STAGE2_HELDOUT" or cfg.get("session") != s:
            raise HeldoutInputError(f"session {s}: config is not the held-out config of that session")
        if not cfg.get("scope_expected"):
            raise HeldoutInputError(f"session {s}: config has no scope_expected (the pair identity is required)")
        if counts0 is None:
            counts0 = cfg.get("counts")
        if cfg.get("counts") != counts0 or not counts0:
            raise HeldoutInputError(f"session {s}: frozen counts differ between sessions")
        check_rules(cfg.get("readiness_rules") or {}, calr)
        ld = load_session(Path(d), Path(repo), s, sched[s], cfg["readiness_rules"], gpus, cfg["counts"],
                          cfg.get("scope_expected"))
        prov["sessions"][s] = {k: ld[k] for k in ("execution_status", "stop_reason", "implementation_id", "method_id",
                                                  "config_sha256", "input_sha256")}
        scopes.append(ld["scope_observed"])
        units += ld["units"]
    if any(sc != scopes[0] for sc in scopes):
        raise HeldoutInputError("sessions were not run on the same pair/scope (no pair substitution)")
    return _assemble(units, gpus, calr, prov, sorted(session_dirs),
                     "UNVERIFIED (low-level test interface; no Freeze-H, identity or config-bytes verification)")


# ------------------------------------------------------------------ PRODUCTION: portable Freeze-H verification

def _listify(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else ([v] if v else [])


def _module_file(name: str) -> Path:
    return Path(importlib.import_module(f"{__package__}.{name}").__file__)


def _role_hashes(roles: dict, files: dict, role: str) -> list:
    return sorted((r, (files.get(r) or {}).get("sha256")) for r in _listify(roles.get(role)))


def freeze_h_problems(record_path: Path, session: int, base: Path = REPO, anchors: dict = ANCHORS) -> dict:
    """Portable verification of one Freeze-H record (P3): no NVML, no GPU, no live environment. Returns {"problems",
    "not_verified_offline", "record", "parent_record", "sha256"}. Every bound file must be present with its hash except
    the compiled binary/plugin (absent -> listed NOT_VERIFIED_OFFLINE) and the external libnccl (verified only if its
    recorded path exists on this host; otherwise listed)."""
    base, out = Path(base), {"problems": [], "not_verified_offline": [], "record": None, "parent_record": None,
                             "sha256": None}
    p, nvo = out["problems"], out["not_verified_offline"]
    try:
        out["sha256"] = sha256_file(record_path)
        rec = json.loads(Path(record_path).read_text())
    except (OSError, ValueError) as e:
        p.append(f"FREEZE_H_UNREADABLE: {e}")
        return out
    if not isinstance(rec, dict) or not isinstance(rec.get("roles"), dict) or not isinstance(rec.get("files"), dict):
        p.append("FREEZE_H_NOT_A_RECORD (roles/files missing)")
        return out
    out["record"] = rec
    roles, files = rec["roles"], rec["files"]
    if rec.get("schema") != FREEZE_SCHEMA:
        p.append(f"FREEZE_H_SCHEMA {rec.get('schema')!r}")
    if rec.get("kind") != "Freeze-H":
        p.append(f"FREEZE_H_KIND {rec.get('kind')!r}")
    if not isinstance(rec.get("approved_by_user"), dict) or not rec["approved_by_user"]:
        p.append("FREEZE_H_NOT_APPROVED (no user approval entry)")
    if rec.get("session") != session:
        p.append(f"FREEZE_H_SESSION {rec.get('session')!r} != {session}")
    for role in ("config", "protocol", "analysis", "calibration_analysis", "schedule") + tuple(SHARED_ROLES):
        if not _listify(roles.get(role)):
            p.append(f"ROLE_MISSING {role}")
        for r in _listify(roles.get(role)):
            if r not in files:
                p.append(f"ROLE_FILE_NOT_IN_MANIFEST {role}: {r}")
    optional = {r for role in OFFLINE_OPTIONAL_ROLES for r in _listify(roles.get(role))}
    for rel, h in files.items():
        want = h.get("sha256") if isinstance(h, dict) else None
        f = base / rel
        if not want:
            p.append(f"NO_HASH {rel}")
        elif not f.is_file():
            if rel in optional:
                nvo.append(f"{rel}: NOT_VERIFIED_OFFLINE (compiled build output absent)")
            else:
                p.append(f"BOUND_FILE_MISSING {rel}")
        elif sha256_file(f) != want:
            p.append(f"BOUND_FILE_CHANGED {rel}")
    ana = _listify(roles.get("analysis"))
    for rel in MANDATORY_ANALYSIS["Freeze-H"]:
        if rel not in ana:
            p.append(f"MANDATORY_ANALYSIS_NOT_BOUND {rel}")
    for m in ANALYSIS_MODULES:
        rel = f"src/ncclenergy/v2/{m}.py"
        if rel not in ana:
            p.append(f"ANALYSIS_ROLE_LACKS {rel}")
        elif (files.get(rel) or {}).get("sha256") != sha256_file(_module_file(m)):
            p.append(f"ANALYSIS_IS_NOT_THE_RUNNING_CODE {rel}")
    if roles.get("protocol") != PROTOCOL_REL:
        p.append(f"PROTOCOL_ROLE {roles.get('protocol')!r} != {PROTOCOL_REL}")
    sch = _listify(roles.get("schedule"))
    if len(sch) != 1 or (files.get(sch[0]) or {}).get("sha256") != anchors["draft_sha256"].get(session):
        p.append("SCHEDULE_NOT_THE_FROZEN_DRAFT")
    if (files.get(roles.get("calibration_analysis")) or {}).get("sha256") != anchors["calibration_analysis_sha256"]:
        p.append("CALIBRATION_ANALYSIS_NOT_THE_FROZEN_BYTES")
    ext = (rec.get("external") or {}).get("libnccl") or {}
    if not ext.get("path") or not ext.get("sha256"):
        p.append("EXTERNAL_LIBNCCL_IDENTITY_MISSING")
    elif not Path(ext["path"]).is_file():
        nvo.append(f"external libnccl {ext['sha256'][:16]}: NOT_VERIFIED_OFFLINE (recorded path absent on this host)")
    elif sha256_file(Path(ext["path"])) != ext["sha256"]:
        nvo.append(f"external libnccl {ext['sha256'][:16]}: DIFFERS_ON_THIS_HOST (informational; the run-time identity "
                   "check is the runner's live verify_freeze)")
    par = rec.get("parent") or {}
    if par.get("sha256") != anchors["parent_freeze_c_sha256"]:
        p.append("PARENT_IS_NOT_THE_FROZEN_FREEZE_C")
    pf = base / str(par.get("path") or "")
    if not par.get("path") or not pf.is_file() or sha256_file(pf) != par.get("sha256"):
        p.append("PARENT_FILE_MISSING_OR_CHANGED")
        return out
    prec = json.loads(pf.read_text())
    out["parent_record"] = prec
    pfiles, proles = prec.get("files") or {}, prec.get("roles") or {}
    if prec.get("kind") != "Freeze-C" or not prec.get("approved_by_user"):
        p.append("PARENT_IS_NOT_AN_APPROVED_FREEZE_C")
    for role in SHARED_ROLES:
        if _role_hashes(roles, files, role) != _role_hashes(proles, pfiles, role):
            p.append(f"SHARED_ROLE_DIFFERS_FROM_PARENT {role}")
    gp = prec.get("parent") or {}
    if not gp.get("path") or not (base / gp["path"]).is_file() or sha256_file(base / gp["path"]) != gp.get("sha256"):
        p.append("FREEZE_S_FILE_MISSING_OR_CHANGED")
    ccfg = proles.get("config")
    if not ccfg or not (base / ccfg).is_file() or sha256_file(base / ccfg) != (pfiles.get(ccfg) or {}).get("sha256"):
        p.append("FREEZE_C_CONFIG_MISSING_OR_CHANGED")
    return out


def _config_problems(cfg: dict, draft: dict, s: int, calr: dict, cal_cfg: dict, anchors: dict) -> list[str]:
    """P4: the run config is the frozen DRAFT plus only the finalization keys, with calibration-derived rules and the
    calibration pair."""
    p = []
    if not isinstance(cfg, dict) or not isinstance(draft, dict):
        return ["CONFIG_OR_DRAFT_NOT_A_MAPPING"]
    if cfg.get("mode") != "STAGE2_HELDOUT" or cfg.get("session") != s:
        p.append("CONFIG_MODE_OR_SESSION")
    if (cfg.get("implementation_id"), cfg.get("method_id")) != (anchors["implementation_id"], anchors["method_id"]):
        p.append(f"CONFIG_IDENTITY {cfg.get('implementation_id')!r}/{cfg.get('method_id')!r}")
    for k in sorted(set(draft) | set(cfg)):
        if k in FINALIZATION_KEYS:
            continue
        if k not in draft or k not in cfg or cfg[k] != draft[k]:
            p.append(f"CONFIG_DIFFERS_FROM_FROZEN_DRAFT {k}")
    try:
        check_rules(cfg.get("readiness_rules") or {}, calr)
    except HeldoutInputError as e:
        p.append(f"READINESS_RULES: {e}")
    if cfg.get("allocation") != cal_cfg.get("allocation"):
        p.append(f"ALLOCATION {cfg.get('allocation')!r} is not the calibration pair {cal_cfg.get('allocation')!r}")
    if cfg.get("scope_expected") != cal_cfg.get("scope_expected"):
        p.append("SCOPE_EXPECTED differs from the calibration pair's")
    if [pl.get("gpus") for pl in cfg.get("placements") or []] != [cfg.get("allocation")]:
        p.append("PLACEMENT is not exactly the allocated pair")
    return p


# ------------------------------------------------------------------ PRODUCTION entry point

def analyse_heldout(session_dirs: dict, freeze_records: dict, base: Path = REPO, expected_cfgs: dict | None = None,
                    anchors: dict = ANCHORS) -> dict:
    """PRODUCTION held-out analysis (portable, offline). freeze_records: {session: approved Freeze-H record path}, one
    per session that was frozen; session_dirs: {session: session directory} for the sessions with data (a frozen
    session without a directory is reported NOT_STARTED; a directory without its Freeze-H is refused). Configs and the
    calibration analysis are read only from the H-bound bytes; expected_cfgs, if given, must equal them. Any failed
    check raises HeldoutInputError; nothing is adapted."""
    base = Path(base)
    if not freeze_records or not set(freeze_records) <= set(SESSIONS):
        raise HeldoutInputError("freeze_records must name sessions 1-3, one approved Freeze-H each")
    if set(session_dirs) - set(freeze_records):
        raise HeldoutInputError(f"session data without its Freeze-H: {sorted(set(session_dirs) - set(freeze_records))}")
    checks = {s: freeze_h_problems(Path(freeze_records[s]), s, base, anchors) for s in sorted(freeze_records)}
    bad = [f"session {s}: {x}" for s, c in checks.items() for x in c["problems"]]
    if bad:
        raise HeldoutInputError("Freeze-H verification failed: " + "; ".join(bad))
    recs = {s: c["record"] for s, c in checks.items()}
    s0 = min(recs)
    for s, rec in recs.items():
        for role in ("protocol", "analysis", "calibration_analysis") + tuple(SHARED_ROLES):
            if _role_hashes(rec["roles"], rec["files"], role) != _role_hashes(recs[s0]["roles"], recs[s0]["files"], role):
                raise HeldoutInputError(f"session {s}: Freeze-H role {role} differs from session {s0}'s")
        if rec["parent"] != recs[s0]["parent"]:
            raise HeldoutInputError(f"session {s}: Freeze-H parent differs from session {s0}'s")
    if len({rec["roles"]["config"] for rec in recs.values()}) != len(recs):
        raise HeldoutInputError("two sessions bind the same config file")
    ca = base / recs[s0]["roles"]["calibration_analysis"]
    cal = json.loads(ca.read_text())
    pfiles = checks[s0]["parent_record"]["files"]
    for mod, h in (cal.get("analysis_module_sha256") or {}).items():
        rel = "src/" + mod.replace(".", "/") + ".py"
        if (pfiles.get(rel) or {}).get("sha256") != h:
            raise HeldoutInputError(f"calibration analysis was not produced by the Freeze-C code ({rel})")
    if not cal.get("analysis_module_sha256"):
        raise HeldoutInputError("calibration analysis lacks its analysis_module_sha256 provenance")
    calr = cal.get("result") or {}
    cal_cfg = yaml.safe_load((base / checks[s0]["parent_record"]["roles"]["config"]).read_bytes())
    cfgs = {}
    for s, rec in recs.items():
        cp = base / rec["roles"]["config"]
        if not cp.is_file():
            raise HeldoutInputError(f"session {s}: config file {rec['roles']['config']} does not exist")
        cfg = yaml.safe_load(cp.read_bytes())
        draft = yaml.safe_load((base / rec["roles"]["schedule"][0]).read_bytes())
        cp_ = _config_problems(cfg, draft, s, calr, cal_cfg, anchors)
        if cp_:
            raise HeldoutInputError(f"session {s}: {cp_}")
        if expected_cfgs is not None and s in expected_cfgs and expected_cfgs[s] != cfg:
            raise HeldoutInputError(f"session {s}: supplied config differs from the Freeze-H-bound config bytes")
        cfgs[s] = cfg
    sched = schedule_from_configs(cfgs)
    gpus = list(cfgs[s0]["allocation"])
    prov = {"freeze_h": {s: {"sha256": c["sha256"], "not_verified_offline": c["not_verified_offline"]}
                         for s, c in checks.items()},
            "parent_freeze_c_sha256": recs[s0]["parent"]["sha256"],
            "calibration_analysis_sha256": sha256_file(ca),
            "analysis_sha256": {m: sha256_file(_module_file(m)) for m in ANALYSIS_MODULES},
            "trust_anchors": "frozen defaults" if anchors is ANCHORS else "NON-DEFAULT (synthetic test only)",
            "checks": "portable offline artifact checks (no NVML, no GPU, no live verify_freeze)", "sessions": {}}
    units, scopes = [], []
    for s in sorted(recs):
        cfg, h_cfg = cfgs[s], recs[s]["files"][recs[s]["roles"]["config"]]["sha256"]
        if s not in session_dirs:
            units += [{"launch_id": u["launch_id"], "session": s, "block": int(u["block"]), "arm": u["arm"],
                       "length_s": float(u["length_s"]), "status": "NOT_RUN", "reasons": ["SESSION_NOT_STARTED"]}
                      for u in sched[s]]
            prov["sessions"][s] = {"execution_status": "NOT_STARTED", "config_sha256": h_cfg}
            continue
        ld = load_session(Path(session_dirs[s]), base, s, sched[s], cfg["readiness_rules"], gpus, cfg["counts"],
                          cfg["scope_expected"], identity={"implementation_id": anchors["implementation_id"],
                                                           "method_id": anchors["method_id"],
                                                           "campaign": cfg["campaign"]})
        if ld["config_sha256"] != h_cfg:
            raise HeldoutInputError(f"session {s}: the run's config_sha256 is not the Freeze-H-bound config")
        prov["sessions"][s] = {k: ld[k] for k in ("execution_status", "stop_reason", "implementation_id", "method_id",
                                                  "config_sha256", "input_sha256")}
        scopes.append(ld["scope_observed"])
        units += ld["units"]
    if any(sc != scopes[0] for sc in scopes):
        raise HeldoutInputError("sessions were not run on the same pair/scope (no pair substitution)")
    return _assemble(units, gpus, calr, prov, sorted(recs), "VERIFIED_OFFLINE (production; see provenance)")
