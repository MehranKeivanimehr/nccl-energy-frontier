"""Numerical checks for the synthetic training workload (pure Python; no torch import).

Finite losses and gradients are a necessary sanity condition, not evidence that
two NCCL policies train equivalently. Policy comparison uses the recorded loss
trajectories and final parameter norms of runs with identical declared
semantics (mode, world size, seed, batch, steps, warm-up, model shape, dtype).
Different reduction algorithms change floating-point summation order, so exact
equality is not expected. The tolerances are declared assumptions, to be checked
against same-policy repeat runs before they are used to accept equivalence.
"""

from __future__ import annotations

import math

SEMANTIC_KEYS = ("mode", "world", "seed", "batch", "global_batch", "steps", "warmup", "hidden", "layers", "dtype",
                 "data")
DEFAULT_RTOL_LOSS = 1e-3
DEFAULT_RTOL_PARAM = 1e-4


def finite_summary(values) -> dict:
    vals = [float(v) for v in values]
    bad = [i for i, v in enumerate(vals) if not math.isfinite(v)]
    return {"n": len(vals), "all_finite": not bad, "nonfinite_steps": bad[:20],
            "first": vals[0] if vals else None, "last": vals[-1] if vals else None}


def global_param_norm(mode: str, per_rank_sq_sums: list[float]) -> float | None:
    """DDP replicates parameters (use rank 0); FSDP shards them (sum over ranks)."""
    if not per_rank_sq_sums or any(not math.isfinite(x) for x in per_rank_sq_sums):
        return None
    s = per_rank_sq_sums[0] if mode.startswith("ddp") else sum(per_rank_sq_sums)
    return math.sqrt(s)


def _rel(a: float, b: float) -> float:
    d = max(abs(a), abs(b))
    return 0.0 if d == 0 else abs(a - b) / d


def compare_numerics(a: dict, b: dict, rtol_loss: float = DEFAULT_RTOL_LOSS,
                     rtol_param: float = DEFAULT_RTOL_PARAM) -> dict:
    """Compare two workload results' "semantics" and "numerics" sections.

    Returns comparable (same declared semantics, both finite, same trajectory length), the maximum relative
    loss difference over the measured steps, the relative final-parameter-norm difference and
    within_tolerance. Not comparable -> within_tolerance is None (never True by default)."""
    reasons = []
    sa, sb = a.get("semantics") or {}, b.get("semantics") or {}
    for k in SEMANTIC_KEYS:
        if sa.get(k) is None or sb.get(k) is None:
            reasons.append(f"SEMANTICS_MISSING:{k}")
        elif sa.get(k) != sb.get(k):
            reasons.append(f"SEMANTICS_DIFFER:{k}")
    na, nb = a.get("numerics") or {}, b.get("numerics") or {}
    for tag, n in (("a", na), ("b", nb)):
        if not n.get("loss_all_finite") or not n.get("grads_all_finite"):
            reasons.append(f"NONFINITE:{tag}")
        if not n.get("ranks_consistent", False):
            reasons.append(f"RANKS_INCONSISTENT:{tag}")
    la, lb = na.get("loss_trajectory") or [], nb.get("loss_trajectory") or []
    if not la or len(la) != len(lb):
        reasons.append("TRAJECTORY_LENGTH")
    out = {"comparable": not reasons, "reasons": reasons, "rtol_loss": rtol_loss, "rtol_param": rtol_param,
           "max_rel_loss_diff": None, "param_norm_rel_diff": None, "within_tolerance": None}
    if la and len(la) == len(lb) and all(math.isfinite(x) for x in la + lb):
        out["max_rel_loss_diff"] = max(_rel(x, y) for x, y in zip(la, lb))
    pa, pb = na.get("param_norm"), nb.get("param_norm")
    if pa is not None and pb is not None:
        out["param_norm_rel_diff"] = _rel(pa, pb)
    if out["comparable"] and out["max_rel_loss_diff"] is not None and out["param_norm_rel_diff"] is not None:
        out["within_tolerance"] = out["max_rel_loss_diff"] <= rtol_loss and out["param_norm_rel_diff"] <= rtol_param
    return out
