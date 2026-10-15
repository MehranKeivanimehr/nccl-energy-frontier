"""Stage 2 statistical decisions (CPU only). Historical D1-D4 replay is the frozen measval_f4735b1.py, unchanged.

Prospective decisions, fixed before any Stage 2 data:
  * unit energy E = E_aligned(P_hat) (primary); J = E / R (exact integer R);
  * block contrasts -> session means m_s -> overall M = mean(m_s); S = 3 sessions required;
  * primary inference on the 3 session means, t with df = 2;
  * DIFFERENCE evidence and EQUIVALENCE evidence are reported separately:
      difference: two-sided t-test of M = 0 (p) and the minimum-effect test of the composite null |M| <= delta
                  against |M| > delta. That test is valid in both directions: p_beyond = min(1, 2 min(p_up, p_low))
                  with p_up for H0: M <= delta and p_low for H0: M >= -delta (equivalently: the two-sided 95 % interval
                  lies entirely beyond +delta or -delta). A one-sided version is used only for a direction declared
                  in the frozen protocol (none is declared). Within a family (H-D1 over 3 arms x 3 lengths, H-D3 over
                  3 lengths, Q1 over 2 doses) both are Holm-adjusted and decisions use the adjusted p only; the
                  unadjusted 95 % intervals of a family are labelled descriptive;
      equivalence: TOST at alpha = 0.05 (equivalently the 90 % interval inside [-delta, delta]); a joint claim over a
                  family (all arms / all lengths) is an intersection-union test and needs every member equivalent;
  * non-finite or missing contrasts are errors, never filtered; zero between-session variance is DEGENERATE (no
    interval, never acceptable); fewer than 3 sessions is INSUFFICIENT_SESSIONS.
Nonsignificance never demonstrates absence: only the equivalence evidence can support a bound.
Sessions: three sessions on one pair and one host. The randomized order removes a fixed position-length/arm
association; it does not make the sessions independent replicates of the population of sessions, hosts or pairs, and
the between-session variance from S = 3 is itself very uncertain.
H-D1 (duration dependence) is a frozen family over every measured length against 40 s: ln(J_L / J_40) for
L in HD1_LENGTHS and arms A, B, C (9 contrasts). "No duration dependence" refers to exactly these contrasts.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

D1_D2 = math.log(1.003)
D3 = math.log(1.005)
D4_HW = math.log(1.004)
Q1_MARGIN = math.log(1.003)
HD1_LENGTHS = (2.5, 10.0, 20.0)         # each against 40 s
HD3_LENGTHS = (10.0, 20.0, 40.0)
ALPHA = 0.05
N_SESSIONS = 3
N_ROUNDS = 3
ZERO_SPREAD_REL = 1e-9


class NonFiniteContrast(ValueError):
    pass


# ------------------------------------------------------------------ historical replay (frozen code, unchanged)

def load_frozen_measval(repo: Path):
    sys.path.insert(0, str(repo / "src"))
    spec = importlib.util.spec_from_file_location("ncclenergy.v2.measval_f4735b1",
                                                  repo / "configs/v2.2/frozen_code/measval_f4735b1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def replay(mod, campaign: Path, frozen_schedule: dict) -> dict:
    df = mod.accept(mod.load(campaign, frozen_schedule))
    acc = df[df.accepted]
    ctrl = mod.control_status(acc)
    r1, r2, r3 = mod.d1(acc), mod.d2(acc), mod.d3(acc, ctrl)
    r4 = mod.d4(acc, r1, r2, r3)
    return {"label": "historical D1-D4 rules, frozen code", "n_processes": int(len(df)), "n_accepted": int(len(acc)),
            "control": ctrl, "D1": r1, "D2": r2, "D3": r3, "D4": r4}


# ------------------------------------------------------------------ summaries

def _check(values, what: str) -> list[float]:
    out = []
    for v in values:
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)):
            raise NonFiniteContrast(f"{what}: missing or non-numeric value {v!r}")
        if not math.isfinite(float(v)):
            raise NonFiniteContrast(f"{what}: non-finite value {v!r}")
        out.append(float(v))
    return out


def mean_summary(values, n_required: int, label: str) -> dict:
    """t summary of n_required independent values (session means or rounds), df = n - 1."""
    x = _check(values, label)
    out = {"label": label, "n": len(x), "values": x}
    if len(x) != n_required:
        out["status"] = "INSUFFICIENT_SESSIONS" if label.startswith("session") else "INSUFFICIENT_ROUNDS"
        return out
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    out.update(mean=m, sd=sd)
    # zero spread, including spread that is only floating-point rounding of identical values (relative 1e-9 is far
    # below the counter's 1 mJ resolution on ~10^2-10^3 J), gives no interval: t would be infinite
    if not sd > ZERO_SPREAD_REL * max(1e-300, max(abs(v) for v in x)):
        out["status"] = "DEGENERATE_ZERO_VARIANCE"
        return out
    se = sd / math.sqrt(len(x))
    df = len(x) - 1
    q95, q90 = float(stats.t.ppf(0.975, df)), float(stats.t.ppf(0.95, df))
    out.update(status="OK", se=se, df=df, ci95=(m - q95 * se, m + q95 * se), ci90=(m - q90 * se, m + q90 * se),
               p_two_sided=float(2 * stats.t.sf(abs(m) / se, df)))
    return out


def session_summary(block_contrasts: dict) -> dict:
    """block_contrasts: {session: [x for each block with a valid contrast]} (finite numbers only). Session mean = mean
    of that session's blocks; primary summary over the session means; conditional block interval descriptive."""
    for s, v in block_contrasts.items():
        _check(v, f"session {s} block contrasts")
    ms = {s: float(np.mean(v)) for s, v in block_contrasts.items() if len(v)}
    out = mean_summary(list(ms.values()), N_SESSIONS, "session means")
    out["session_means"] = ms
    allx = [float(x) for v in block_contrasts.values() for x in v]
    cond = mean_summary(allx, len(allx), "block contrasts (conditional on these sessions)") if len(allx) >= 2 else None
    out["conditional_descriptive"] = cond
    out["session_range"] = [min(ms.values()), max(ms.values())] if ms else None
    return out


def evidence(summary: dict, delta: float, direction: str | None = None) -> dict:
    """Difference and equivalence evidence for one contrast, unadjusted.

    Minimum-effect test of the composite null |M| <= delta. Without a predeclared direction: p_beyond =
    min(1, 2 min(p_up, p_low)), p_up = P(T >= t | M = +delta) (H0: M <= delta), p_low = P(T <= t | M = -delta)
    (H0: M >= -delta); its size is <= alpha over the whole null. direction '+' or '-' (only if frozen in the protocol)
    uses the corresponding one-sided p. TOST: p_tost = max of the two one-sided equivalence p-values."""
    if summary.get("status") != "OK":
        return {"status": summary.get("status"), "difference": None, "equivalence": None}
    if direction not in (None, "+", "-"):
        raise ValueError("direction must be None, '+' or '-'")
    m, se, df = summary["mean"], summary["se"], summary["df"]
    p_eq_lower = float(stats.t.sf((m + delta) / se, df))       # TOST part, H0: M <= -delta
    p_eq_upper = float(stats.t.cdf((m - delta) / se, df))      # TOST part, H0: M >= +delta
    p_up = float(stats.t.sf((m - delta) / se, df))             # minimum effect, H0: M <= +delta
    p_low = float(stats.t.cdf((m + delta) / se, df))           # minimum effect, H0: M >= -delta
    p_beyond = {None: min(1.0, 2.0 * min(p_up, p_low)), "+": p_up, "-": p_low}[direction]
    return {"status": "OK", "delta": delta, "direction": direction or "two-direction",
            "difference": {"p": summary["p_two_sided"], "p_beyond_margin": p_beyond, "ci95": summary["ci95"]},
            "equivalence": {"p_tost": max(p_eq_lower, p_eq_upper), "ci90": summary["ci90"],
                            "within_margin": max(p_eq_lower, p_eq_upper) < ALPHA}}


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values; refuses non-finite or out-of-range input."""
    for k, p in pvalues.items():
        if p is None or not math.isfinite(p) or not 0 <= p <= 1:
            raise NonFiniteContrast(f"p-value for {k} is invalid: {p!r}")
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def verdict(ev: dict, override: str | None = None) -> str:
    """Verdict from ADJUSTED decisions (ev from family()). Order: override, status, beyond margin, equivalence,
    difference, otherwise insufficient precision."""
    if override:
        return override
    if ev.get("status") != "OK":
        return ev.get("status") or "INSUFFICIENT_SESSIONS"
    d = ev["decision"]
    if d["beyond_margin"]:
        return "RESOLVED_MEANINGFUL_DIFFERENCE"
    if d["equivalent"]:
        return "DETECTED_WITHIN_MARGIN" if d["difference_detected"] else "COMPATIBLE_WITH_EQUIVALENCE"
    if d["difference_detected"]:
        return "DIFFERENCE_MAGNITUDE_UNRESOLVED"
    return "INSUFFICIENT_PRECISION"


def family(summaries: dict, delta: float, overrides: dict | None = None) -> dict:
    """Decisions for a family of contrasts (keys: arms, lengths or doses). Holm over ALL members of the family for
    both the two-sided difference test and the minimum-effect test; a member without a valid summary enters with
    p = 1 (the family size never shrinks). With one member Holm is the identity."""
    overrides = overrides or {}
    evs = {k: evidence(s, delta) for k, s in summaries.items()}
    ok = {k for k, e in evs.items() if e["status"] == "OK"}
    adj_d = holm({k: (e["difference"]["p"] if k in ok else 1.0) for k, e in evs.items()}) if evs else {}
    adj_b = holm({k: (e["difference"]["p_beyond_margin"] if k in ok else 1.0) for k, e in evs.items()}) if evs else {}
    out = {}
    for k, e in evs.items():
        if e["status"] == "OK":
            e["difference"]["p_holm"] = adj_d[k]
            e["difference"]["p_beyond_margin_holm"] = adj_b[k]
            e["difference"]["ci95_label"] = "descriptive (unadjusted)" if len(summaries) > 1 else "unadjusted"
            e["decision"] = {"difference_detected": adj_d[k] < ALPHA, "beyond_margin": adj_b[k] < ALPHA,
                             "equivalent": e["equivalence"]["within_margin"]}
        e["verdict"] = verdict(e, overrides.get(k))
        e["family_size"] = len(summaries)
        out[k] = e
    return out


def precision(summary: dict, max_half_width: float = D4_HW) -> dict:
    if summary.get("status") != "OK":
        return {"ok": False, "status": summary.get("status")}
    lo, hi = summary["ci95"]
    hw = (hi - lo) / 2
    return {"ok": hw <= max_half_width, "half_width": hw, "status": "OK"}


# ------------------------------------------------------------------ Q1 read dose (calibration C2)

def q1(rounds: list[dict], margin: float = Q1_MARGIN) -> dict:
    """rounds: one dict per C2 round r with accepted energies per condition and estimator:
         {"E0": {"counter": J, "sampled": J}, "D100": {...}, "D500": {...}} (J per op; a missing condition = None).
    Per dose D: d = ln(J_D / J_E0) on the counter estimator (primary E_aligned(P_hat); call again for E_counter_raw),
    s = the same ratio on the sampled estimator, did = d - s (counter-specific part). D100/E0 and D500/E0 are
    reported separately; no sign agreement is required. Each gets difference evidence (Holm over the two doses) and
    equivalence evidence (TOST at `margin`). A dose is BOUNDED only by equivalence evidence; nonsignificance is
    never a bound. E0 still contains the v2.1 pre-region probe and the C0/C1/Cpost reads: Q1 tests only ADDITIONAL
    in-region reads."""
    res = {"scope_note": "E0 includes the pre-region probe (~250 reads) and the boundary reads; Q1 tests only the "
                         "additional in-region reads of D100 (100 per 10 s) and D500 (20 per 10 s)", "doses": {}}
    def lr(r, D, est):
        """ln(J_D / J_E0) for one round; None if the unit or the estimator is unavailable (excluded, never imputed)."""
        if r.get(D) is None or r.get("E0") is None or r[D].get(est) is None or r["E0"].get(est) is None:
            return None
        a, b = _check([r[D][est], r["E0"][est]], f"Q1 {D} {est}")
        if a <= 0 or b <= 0:
            raise NonFiniteContrast(f"Q1 {D} {est}: non-positive energy")
        return math.log(a / b)
    for est in ("counter", "sampled", "did"):
        sums = {}
        for D in ("D100", "D500"):
            vals = []
            for r in rounds:
                c, sm = lr(r, D, "counter"), lr(r, D, "sampled")
                v = {"counter": c, "sampled": sm, "did": None if c is None or sm is None else c - sm}[est]
                if v is not None:
                    vals.append(v)
            sums[D] = mean_summary(vals, N_ROUNDS, f"rounds {est}")
        fam = family(sums, margin)
        for D, e in fam.items():
            res["doses"].setdefault(D, {})[est] = {"summary": sums[D], **e, "class": q1_class(e)}
    for D, r in res["doses"].items():
        r["interpretation"] = q1_interpretation(r)
        r["bounded_for_adequacy"] = r["counter"]["class"] in ("BOUNDED_WITHIN_MARGIN", "DETECTED_AND_BOUNDED")
    return res


def q1_class(e: dict) -> str:
    if e.get("status") != "OK":
        return e.get("status")
    d = e["decision"]
    if d["difference_detected"] and d["equivalent"]:
        return "DETECTED_AND_BOUNDED"
    if d["equivalent"]:
        return "BOUNDED_WITHIN_MARGIN"
    if d["difference_detected"]:
        return "DETECTED"
    return "INCONCLUSIVE"


def q1_interpretation(r: dict) -> str:
    """Descriptive, from the classes only; no mechanism is named."""
    c, s, did = r["counter"]["class"], r["sampled"]["class"], r["did"]["class"]
    if c in ("DETECTED", "DETECTED_AND_BOUNDED") and did in ("DETECTED", "DETECTED_AND_BOUNDED"):
        return "COUNTER_SPECIFIC_DOSE_DEPENDENCE (counter differs from the sampled channel; mechanism not established)"
    if c in ("DETECTED", "DETECTED_AND_BOUNDED") and did in ("BOUNDED_WITHIN_MARGIN",):
        return "SHARED_WITH_SAMPLED_CHANNEL (dose changes both channels alike)"
    if c in ("BOUNDED_WITHIN_MARGIN", "DETECTED_AND_BOUNDED"):
        return "COUNTER_TOTAL_BOUNDED_WITHIN_MARGIN"
    return "UNRESOLVED"


# ------------------------------------------------------------------ scope-specific adequacy

# Every component that can bias a contrast. Each must carry a HARD bound with a stated basis (status BOUNDED) for a
# systematic total to exist. A model-conditional timing contribution (status CONDITIONAL), an association, a
# calibration maximum or a missing component is not a bound: the total is then UNKNOWN.
SYSTEMATIC_COMPONENTS = ("end_alignment_timing", "start_phase_imbalance", "thermal_imbalance", "context_power_model",
                         "sensor_and_counter_accuracy", "load_extrapolation")


def systematic_total(budget: dict, length_s: float, required=SYSTEMATIC_COMPONENTS) -> dict:
    """budget: {component: {length_s: {"status": "BOUNDED", "bound_rel": x, "basis": ...} | {"status":
    "CONDITIONAL"|"UNKNOWN", ...}}}. Bounds add linearly (no cancellation assumed). Unknown is not zero."""
    tot, unknown, conditional = 0.0, [], []
    for c in required:
        e = (budget.get(c) or {}).get(length_s)
        st = (e or {}).get("status")
        if st == "BOUNDED" and isinstance(e.get("bound_rel"), (int, float)) and math.isfinite(e["bound_rel"]) \
                and e["bound_rel"] >= 0 and e.get("basis"):
            tot += e["bound_rel"]
        elif st == "CONDITIONAL":
            conditional.append(c)
        else:
            unknown.append(c)
    missing = unknown + conditional
    return {"status": "UNKNOWN" if missing else "BOUNDED", "unknown": unknown, "conditional_only": conditional,
            "bound_rel": None if missing else tot}


def _eq(fam_entry) -> bool:
    return bool(fam_entry) and fam_entry.get("status") == "OK" and fam_entry["decision"]["equivalent"] and \
        fam_entry.get("verdict") in ("COMPATIBLE_WITH_EQUIVALENCE", "DETECTED_WITHIN_MARGIN")


def adequacy(scope: str, r: dict) -> dict:
    """Scopes are fixed now, always for estimator E_aligned(P_hat), the frozen pair identity and the Freeze-C software.

      BROAD     lengths 2.5-40 s, arms A/B/C. Statistical part: every H-D1(arm, L) equivalent (L in HD1_LENGTHS vs 40 s:
                "no duration dependence" means exactly these 9 contrasts), H-D2 equivalent, H-D3 equivalent at 10, 20,
                40 s, H-D4 95 % half-width <= ln 1.004, Q1 bounded for D100 and D500, R3 and control validity at every
                length. Systematic part: a BOUNDED total <= ln 1.005 at every length.
      SAME_EXEC_L<L>  (L in 10, 20, 40) identical-execution resolution at one length (A vs B): H-D3(L) equivalent and
                its half-width <= ln 1.004, Q1 bounded, R3 and control valid at L; systematic total at L <= ln 1.005.
      CROSS_PROTOCOL_40   SAME_EXEC_L40 plus H-D4 precision and H-D2 equivalent.

    Returns the statistical criteria separately ("conditional_statistical") so they can be reported even when the
    systematic total is UNKNOWN; status ESTABLISHED requires both. r keys: HD1 (family keyed (arm, L)), HD2 ({"C"}),
    HD3 (keyed L), HD4_precision, HD3_precision {L}, q1, systematic, r3_ok {L} (all contrasts at L, per GPU),
    control_valid. Missing inputs count as unmet."""
    unmet_stat, unmet_sys = [], []

    def need(cond, label, sys_=False):
        if not cond:
            (unmet_sys if sys_ else unmet_stat).append(label)
    q1_ok = all((r.get("q1") or {}).get("doses", {}).get(D, {}).get("bounded_for_adequacy") for D in ("D100", "D500"))
    need(r.get("control_valid") is True, "control validity")
    if scope == "BROAD":
        for a in ("A", "B", "C"):
            for L in HD1_LENGTHS:
                need(_eq((r.get("HD1") or {}).get((a, L))), f"H-D1({a}, {L:g} s vs 40 s) equivalence")
        need(_eq((r.get("HD2") or {}).get("C")), "H-D2 equivalence")
        for L in HD3_LENGTHS:
            need(_eq((r.get("HD3") or {}).get(L)), f"H-D3({L:g} s) equivalence")
        need((r.get("HD4_precision") or {}).get("ok") is True, "H-D4 precision")
        need(q1_ok, "Q1 bounded (D100 and D500)")
        lengths = (2.5, 10.0, 20.0, 40.0)
    elif scope.startswith("SAME_EXEC_L") or scope == "CROSS_PROTOCOL_40":
        L = 40.0 if scope == "CROSS_PROTOCOL_40" else float(scope[len("SAME_EXEC_L"):])
        if L not in HD3_LENGTHS:
            raise ValueError(f"undefined scope {scope}")
        need(_eq((r.get("HD3") or {}).get(L)), f"H-D3({L:g} s) equivalence")
        need(((r.get("HD3_precision") or {}).get(L) or {}).get("ok") is True, f"H-D3({L:g} s) precision")
        need(q1_ok, "Q1 bounded (D100 and D500)")
        if scope == "CROSS_PROTOCOL_40":
            need((r.get("HD4_precision") or {}).get("ok") is True, "H-D4 precision")
            need(_eq((r.get("HD2") or {}).get("C")), "H-D2 equivalence")
        lengths = (L,)
    else:
        raise ValueError(f"undefined scope {scope}")
    for L in lengths:
        need((r.get("r3_ok") or {}).get(L) is True, f"R3 per-GPU balance at {L:g} s")
        st = systematic_total(r.get("systematic") or {}, L)
        need(st["status"] == "BOUNDED" and st["bound_rel"] <= D3,
             f"systematic total at {L:g} s ({st['status']}; unknown {st['unknown']}; conditional only "
             f"{st['conditional_only']})", sys_=True)
    return {"scope": scope, "status": "NOT_ESTABLISHED" if unmet_stat or unmet_sys else "ESTABLISHED",
            "conditional_statistical": "MET" if not unmet_stat else "NOT_MET", "unmet": unmet_stat + unmet_sys,
            "unmet_statistical": unmet_stat, "unmet_systematic": unmet_sys}
