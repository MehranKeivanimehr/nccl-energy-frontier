"""Counter update period and phase from probe brackets (CPU only; never reads NVML).

A bracket (lo, hi) [CLOCK_MONOTONIC ns] contains one counter update: the value differed between the read that started
at lo and the read that ended at hi.

Model M (an assumption, stated wherever a result is used): updates occur at t(k) = tau + k P with ONE constant
period P inside the prior range PRIOR_PERIOD_NS, and each bracket contains the update of its integer index k_i.

Hard (model-conditional) statements, exact for M:
  * feasibility: M is feasible iff some (tau, P) satisfies lo_i <= tau + k_i P <= hi_i for all brackets (a 2-variable
    linear program). Infeasible brackets REJECT the model: no period, phase or bound is reported;
  * index identification: every index assignment compatible with the prior and the bracket gaps is tested; more than
    one feasible assignment = AMBIGUOUS_INDEX;
  * prediction: for any index K, [min, max] of tau + K P over the feasible set (two LPs). This is the hard interval of
    that update time under M; it is not a probability statement and it does not cover model error;
  * adjacency: "the first update after t" is identified only if max t(K-1) < t <= min t(K); "the last update before
    an interval [a, b]" only if max t(K) <= a and min t(K+1) >= b. A bound smaller than P/2 is NOT sufficient by itself.

Statistical approximation (labelled "approx", model error unknown): least-squares P_hat and the covariance-correct
standard uncertainty of a prediction (weights w_i = 1/n + c_i (K - kbar); u^2 = sum w_i^2 h_i^2 / 3). It is reported
next to, never instead of, the hard interval. Standard uncertainty, confidence interval, observed maximum and hard
bound are different quantities and are named differently in every output.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

PRIOR_PERIOD_NS = (90e6, 110e6)       # assumption; historical intervals 93.5-101.8 ms (idle and load)
SQRT3 = math.sqrt(3.0)
MAX_ASSIGNMENTS = 64
PERIOD_COMPAT_TOL_MS = 1e-6           # 1 ns: numerical tolerance of the LP period bounds


def _lp(k: np.ndarray, lo: np.ndarray, hi: np.ndarray, prior_ms, c) -> float | None:
    """Optimum of c . (tau, P) subject to lo_i <= tau + k_i P <= hi_i, P in prior (ms units). None if infeasible."""
    A = np.vstack([np.column_stack([np.ones_like(k), k]), -np.column_stack([np.ones_like(k), k])])
    b = np.concatenate([hi, -lo])
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), prior_ms], method="highs")
    return float(r.fun) if r.status == 0 else None


def _feasible(k, lo, hi, prior_ms) -> bool:
    return _lp(np.asarray(k, float), lo, hi, prior_ms, [0.0, 0.0]) is not None


def _assignments(lo, hi, prior_ms) -> list[list[int]]:
    """All integer index assignments (k_0 = 0, increasing) compatible with the prior and feasible under M."""
    n, out = len(lo), []

    def dfs(ks):
        if len(out) > MAX_ASSIGNMENTS:
            return
        j = len(ks)
        if j == n:
            out.append(list(ks))
            return
        d_lo = max(1, math.ceil((lo[j] - hi[j - 1]) / prior_ms[1] - 1e-9))
        d_hi = math.floor((hi[j] - lo[j - 1]) / prior_ms[0] + 1e-9)
        for d in range(d_lo, d_hi + 1):
            cand = ks + [ks[-1] + d]
            if _feasible(cand, lo[:j + 1], hi[:j + 1], prior_ms):
                dfs(cand)
    dfs([0])
    return out


def estimate_period_phase(brackets_ns: list[tuple[int, int]], p_nom_ns: float = 100e6,
                          prior_ns: tuple[float, float] = PRIOR_PERIOD_NS) -> dict:
    b = sorted((int(lo), int(hi)) for lo, hi in brackets_ns)
    base = {"n": len(b), "period_ns": None, "model": "constant period, prior "
            f"[{prior_ns[0] / 1e6:g}, {prior_ns[1] / 1e6:g}] ms, one update per bracket"}
    if len(b) < 2:
        return {"status": "INSUFFICIENT_BRACKETS", **base}
    if any(hi < lo for lo, hi in b) or any(b[i + 1][0] < b[i][0] for i in range(len(b) - 1)):
        return {"status": "MALFORMED_BRACKET", **base}
    origin = b[0][0]
    lo = np.array([(x - origin) / 1e6 for x, _ in b])
    hi = np.array([(y - origin) / 1e6 for _, y in b])
    prior_ms = (prior_ns[0] / 1e6, prior_ns[1] / 1e6)
    asg = _assignments(lo, hi, prior_ms)
    out = {**base, "origin_ns": origin, "lo_ms": lo.tolist(), "hi_ms": hi.tolist(), "prior_ms": list(prior_ms),
           "n_feasible_assignments": len(asg)}
    if not asg:
        out.update(status="INFEASIBLE", note="no (tau, P) in the prior fits every bracket: the constant-period model "
                                             "is rejected; nothing is certified")
        return out
    if len(asg) > 1:
        out.update(status="AMBIGUOUS_INDEX", assignments=asg[:8])
        return out
    k = np.array(asg[0], float)
    p_lo, p_hi = _lp(k, lo, hi, prior_ms, [0.0, 1.0]), -_lp(k, lo, hi, prior_ms, [0.0, -1.0])
    # statistical approximation (labelled; model error unknown)
    m, h = (lo + hi) / 2, (hi - lo) / 2
    kbar = float(k.mean())
    kc = k - kbar
    sxx = float(np.sum(kc ** 2))
    c = kc / sxx
    P_hat = float(np.sum(c * m))
    u_P = math.sqrt(float(np.sum(c ** 2 * (h / SQRT3) ** 2)))
    # guard: the primary (least-squares) estimate must lie in the hard feasible period interval of model M; it is
    # never clipped into it. Outside it, the estimate contradicts the model it is conditional on, so the primary
    # aligned estimator E_aligned(P_hat) is unavailable (aligned_hat_period). Tolerance: 1 ns (LP numerics).
    compatible = p_lo - PERIOD_COMPAT_TOL_MS <= P_hat <= p_hi + PERIOD_COMPAT_TOL_MS
    primary = {"period_ns": P_hat * 1e6, "hard_period_ns": [p_lo * 1e6, p_hi * 1e6],
               "status": "COMPATIBLE" if compatible else "INCOMPATIBLE_WITH_FEASIBLE_MODEL"}
    if not compatible:
        primary["reason"] = (f"least-squares P_hat {P_hat:.4f} ms lies outside the hard feasible period interval "
                             f"[{p_lo:.4f}, {p_hi:.4f}] ms of the constant-period model; not clipped")
    out.update(status="OK", k=k.astype(int).tolist(), kbar=kbar, sxx=sxx, mid_ms=m.tolist(), half_ms=h.tolist(),
               hard_period_ns=[p_lo * 1e6, p_hi * 1e6],
               approx={"label": "statistical approximation (least squares, uniform bracket errors); model error unknown",
                       "period_ns": P_hat * 1e6, "u_period_ns": u_P * 1e6, "a_ms": float(m.mean())},
               period_ns=P_hat * 1e6, primary_period=primary)
    return out


def aligned_hat_period(est: dict) -> tuple[float | None, str | None]:
    """Period for the primary aligned estimator E_aligned(P_hat): (P_hat in ns, None) only if the model is identified
    (status OK) and P_hat is compatible with its hard feasible interval; otherwise (None, explicit reason). The
    raw-counter and nominal-period estimators do not depend on this and are always computed."""
    if est.get("status") != "OK":
        return None, f"PERIOD_MODEL_{est.get('status')}"
    pp = est.get("primary_period") or {}
    if pp.get("status") != "COMPATIBLE":
        return None, "PRIMARY_PERIOD_INCOMPATIBLE: " + str(pp.get("reason", "no compatibility record"))
    return float(pp["period_ns"]), None


def update_interval(est: dict, K: int) -> tuple[float, float] | None:
    """Hard interval [min, max] (ns, absolute) of update K under M, or None if the model is not identified."""
    if est.get("status") != "OK":
        return None
    k = np.array(est["k"], float)
    lo, hi = np.array(est["lo_ms"]), np.array(est["hi_ms"])
    pr = tuple(est["prior_ms"])
    tmin = _lp(k, lo, hi, pr, [1.0, float(K)])
    tmax = _lp(k, lo, hi, pr, [-1.0, -float(K)])
    if tmin is None or tmax is None:
        return None
    return est["origin_ns"] + tmin * 1e6, est["origin_ns"] - tmax * 1e6


def approx_prediction(est: dict, K: int) -> dict:
    """Statistical approximation of update K: LS point and covariance-correct standard uncertainty (model error
    unknown)."""
    if est.get("status") != "OK":
        return {"t_ns": None, "u_ns": None}
    m, h = np.array(est["mid_ms"]), np.array(est["half_ms"])
    kc = np.array(est["k"], float) - est["kbar"]
    w = 1.0 / len(m) + (kc / est["sxx"]) * (K - est["kbar"])
    return {"t_ns": est["origin_ns"] + float(np.sum(w * m)) * 1e6,
            "u_ns": math.sqrt(float(np.sum(w ** 2 * (h / SQRT3) ** 2))) * 1e6, "cycles": float(K - est["kbar"]),
            "label": est["approx"]["label"]}


def _k_guess(est: dict, t_ns: float) -> int:
    return int(math.floor((t_ns - est["origin_ns"]) / est["period_ns"]))


def first_update_after(est: dict, t_ns: float) -> int | None:
    """Index K with max t(K-1) < t <= min t(K) under M; None if not identified (cycle ambiguity)."""
    if est.get("status") != "OK":
        return None
    g = _k_guess(est, t_ns)
    for K in range(g - 2, g + 4):
        a, b = update_interval(est, K - 1), update_interval(est, K)
        if a is None or b is None:
            return None
        if a[1] < t_ns <= b[0]:
            return K
    return None


def last_update_before(est: dict, t_a_ns: float, t_b_ns: float) -> int | None:
    """Index K with max t(K) <= t_a and min t(K+1) >= t_b under M (the update preceding a read interval, with no
    update inside it); None if not identified."""
    if est.get("status") != "OK":
        return None
    g = _k_guess(est, t_a_ns)
    for K in range(g - 2, g + 3):
        a, b = update_interval(est, K), update_interval(est, K + 1)
        if a is None or b is None:
            return None
        if a[1] <= t_a_ns and b[0] >= t_b_ns:
            return K
    return None


def start_phase(est: dict, t_c0_a_ns: int, t_c0_b_ns: int) -> dict:
    """Phase of the C0 read after the preceding update, as a hard interval under M. AMBIGUOUS_CYCLE when the
    preceding update is not identified (e.g. a 1 ms phase whose interval extends below zero)."""
    if est.get("status") != "OK":
        return {"status": "UNKNOWN", "model_status": est.get("status"), "phase_ns": None, "hard_phase_ns": None}
    K = last_update_before(est, t_c0_a_ns, t_c0_b_ns)
    ap = approx_prediction(est, _k_guess(est, (t_c0_a_ns + t_c0_b_ns) / 2))
    if K is None:
        return {"status": "AMBIGUOUS_CYCLE", "phase_ns": None, "hard_phase_ns": None,
                "approx": {"phase_ns": (t_c0_a_ns + t_c0_b_ns) / 2 - ap["t_ns"], "u_ns": ap["u_ns"], "label": ap["label"]}}
    lo, hi = update_interval(est, K)
    mid = (t_c0_a_ns + t_c0_b_ns) / 2
    apK = approx_prediction(est, K)
    return {"status": "OK_UNDER_MODEL", "index": K, "hard_phase_ns": [mid - hi, mid - lo],
            "phase_ns": mid - apK["t_ns"],
            "approx": {"phase_ns": mid - apK["t_ns"], "u_ns": math.sqrt(apK["u_ns"] ** 2 + ((t_c0_b_ns - t_c0_a_ns) / 2 / SQRT3) ** 2),
                       "label": apK["label"]}}


def drift_table(est: dict, region_lengths_s=(2.5, 10.0, 20.0, 40.0), lead_s: float = 0.5) -> list[dict]:
    """Width of the hard interval of the update at the end of a region (and the approximate standard uncertainty),
    for the protocol table. 'identified' = the first update after the region end is identified under M."""
    out = []
    if est.get("status") != "OK":
        return out
    last_hi = est["origin_ns"] + est["hi_ms"][-1] * 1e6
    for L in region_lengths_s:
        t_end = last_hi + (L + lead_s) * 1e9
        K = _k_guess(est, t_end) + 1
        iv = update_interval(est, K)
        ap = approx_prediction(est, K)
        out.append({"region_s": L, "cycles": ap["cycles"], "hard_width_ms": (iv[1] - iv[0]) / 1e6,
                    "approx_u_ms": ap["u_ns"] / 1e6, "identified": first_update_after(est, t_end) is not None})
    return out


def end_alignment(est: dict, t_work_end_ns: float, t_commit_used_ns: float, t_c1_a_ns: float, t_c1_b_ns: float,
                  t_post_a_ns: float | None = None, t_post_b_ns: float | None = None, p_ctx_W: float | None = None,
                  source: str = "") -> dict:
    """Whether C1 (and Cpost) bracket the first update after the GPU-work end, under M, from the given bracket model.
    The tail timing contribution p_ctx * (t_commit_used - t_update) is reported as an interval CONDITIONAL on M and
    on p_ctx; it is not a bound on the energy error (context-power model, sensor and load extrapolation are not
    included)."""
    if est.get("status") != "OK":
        return {"status": "NOT_ESTABLISHED", "reason": f"model {est.get('status')}", "source": source}
    K = first_update_after(est, t_work_end_ns)
    if K is None:
        return {"status": "AMBIGUOUS_CYCLE", "reason": "the first update after the work end is not identified under M",
                "source": source}
    u0, u1, u2 = update_interval(est, K), update_interval(est, K + 1), update_interval(est, K + 2)
    checks = {"update_before_c1_read": u0[1] <= t_c1_a_ns, "next_update_after_c1_read": u1[0] >= t_c1_b_ns}
    if t_post_a_ns is not None and t_post_b_ns is not None:
        checks.update(next_update_before_post_read=u1[1] <= t_post_a_ns, second_update_after_post_read=u2[0] >= t_post_b_ns)
    out = {"source": source, "index": K, "hard_update_ns": list(u0), "checks": checks,
           "hard_eps_ns": [t_commit_used_ns - u0[1], t_commit_used_ns - u0[0]],
           "status": "ALIGNED_UNDER_MODEL" if all(checks.values()) else "ALIGNMENT_NOT_ESTABLISHED"}
    if p_ctx_W is not None and math.isfinite(p_ctx_W) and out["status"] == "ALIGNED_UNDER_MODEL":
        e0, e1 = out["hard_eps_ns"]
        out["conditional_tail_timing_contribution_J"] = sorted([p_ctx_W * e0 / 1e9, p_ctx_W * e1 / 1e9])
        out["conditional_on"] = ["constant-period model M and its prior", "p_ctx from (Cpost - C1)",
                                 "this launch only (no extrapolation)"]
    return out


def end_alignment_probe_only(pre_est, t_work_end_ns, t_commit_used_ns, t_c1_a_ns, t_c1_b_ns, t_post_a_ns=None,
                             t_post_b_ns=None, p_ctx_W=None) -> dict:
    """What the pre-region probe alone says about the end read (held-out has nothing else)."""
    return end_alignment(pre_est, t_work_end_ns, t_commit_used_ns, t_c1_a_ns, t_c1_b_ns, t_post_a_ns, t_post_b_ns,
                         p_ctx_W, "pre-region probe")


def end_alignment_from_post_probe(post_brackets, t_work_end_ns, t_commit_used_ns, t_c1_a_ns, t_c1_b_ns,
                                  t_post_a_ns=None, t_post_b_ns=None, p_ctx_W=None) -> dict:
    """Calibration only (M4d): the post-region probe sees updates ~2 periods after the predicted one; under M with
    the prior the short backward extrapolation usually identifies the update after the work end."""
    est = estimate_period_phase(post_brackets)
    out = end_alignment(est, t_work_end_ns, t_commit_used_ns, t_c1_a_ns, t_c1_b_ns, t_post_a_ns, t_post_b_ns, p_ctx_W,
                        "M4d post-probe")
    out["post_probe_model_status"] = est.get("status")
    return out
