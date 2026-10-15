"""Stage 2 calibration and held-out schedules and the counter-call inventory (CPU only; deterministic from seeds).

Every launch gets an M4c pause derived from (campaign, launch id) by SHA-256: d_req = floor(u * 1e8) ns with u in
[0, 1). The requested pause is NOT claimed to produce a uniform actual counter phase.
Dose (M4b) and the post-region probe (M4d) exist only in the calibration schedule; held-out units never carry them.
"""

from __future__ import annotations

import hashlib
import math
import random

import numpy as np

LENGTHS = (2.5, 10.0, 20.0, 40.0)
ARMS = ("A", "B", "C")
DOSES = ("E0", "D100", "D500")
DOSE_PERIOD_MS = {"E0": 0, "D100": 100, "D500": 500}
PAUSE_RANGE_NS = 100_000_000


def pause_ns(campaign: str, launch_id: str) -> int:
    h = int(hashlib.sha256(f"{campaign}|{launch_id}|m4c".encode()).hexdigest()[:16], 16)
    return int((h / 2 ** 64) * PAUSE_RANGE_NS)


def calibration_schedule(campaign: str = "v2.4-calib-01", m4d: bool = True) -> list[dict]:
    out = [{"part": "S0a", "length_s": 2.5, "dose": "E0"}, {"part": "S0b", "length_s": 2.5, "dose": "D100"}]
    latin = [("E0", "D100", "D500"), ("D100", "D500", "E0"), ("D500", "E0", "D100")]
    for r in range(3):
        rng = random.Random(f"{campaign}/C1/{r + 1}")
        order = list(LENGTHS)
        rng.shuffle(order)
        out += [{"part": f"C1r{r + 1}", "length_s": L, "dose": "E0"} for L in order]
        out += [{"part": f"C2r{r + 1}", "length_s": 10.0, "dose": c} for c in latin[r]]
    for i, u in enumerate(out):
        lid = f"{u['part']}-{i + 1:02d}"
        u.update(pos=i + 1, arm="B", launch_id=lid, pause_ns=pause_ns(campaign, lid),
                 dose_period_ms=DOSE_PERIOD_MS[u["dose"]], poller_active=u["dose"] != "E0", post_probe=bool(m4d))
    return out


def _latin(rng: random.Random) -> list[list[float]]:
    """Random 4x4 Latin square over LENGTHS (cyclic square with permuted rows, columns and symbols)."""
    rows, cols, sym = rng.sample(range(4), 4), rng.sample(range(4), 4), rng.sample(LENGTHS, 4)
    return [[sym[(rows[i] + cols[j]) % 4] for j in range(4)] for i in range(4)]


def _blocks(rng: random.Random, n_blocks: int) -> list[list[tuple[str, float, int]]]:
    """n_blocks (= 6) blocks of 12 units = 4 consecutive triples, one unit of each arm per triple. For each arm the
    block -> (triple -> length) orders are the 4 rows of one random Latin square plus 2 rows of a second one, in random
    block order, so each (arm, length) occupies each triple position at most twice across the 6 blocks."""
    orders = {}
    for a in ARMS:
        rows = _latin(rng) + rng.sample(_latin(rng), n_blocks - 4)
        rng.shuffle(rows)
        orders[a] = rows
    blocks = []
    for b in range(n_blocks):
        units = []
        for t in range(4):
            arms = list(ARMS)
            rng.shuffle(arms)
            units += [(a, orders[a][b][t], t) for a in arms]
        blocks.append(units)
    return blocks


def _abs_corr(pos, x) -> float:
    pos, x = np.asarray(pos, float), np.asarray(x, float)
    if x.std() == 0 or pos.std() == 0:      # a constant column has no correlation; never divide by zero
        return 0.0
    return float(abs(np.mean((pos - pos.mean()) * (x - x.mean())) / (pos.std() * x.std())))


def _stats(blocks, sessions) -> dict:
    rows = [(s, i + 1, a, L) for b, s in zip(blocks, sessions) for i, (a, L, _) in enumerate(b)]
    res = {}
    for scope, sel in [("overall", rows)] + [(f"session{s}", [r for r in rows if r[0] == s]) for s in sorted(set(sessions))]:
        pos = [r[1] for r in sel]
        res[scope] = max([_abs_corr(pos, [r[3] for r in sel])] +
                         [_abs_corr(pos, [1.0 if r[2] == a else 0.0 for r in sel]) for a in ARMS])
    return res


def heldout_schedule(seed: str = "v2.4-heldout", n_candidates: int = 10000, sessions: int = 3, blocks_per_session: int = 2):
    rng = random.Random(seed)
    sess = [s + 1 for s in range(sessions) for _ in range(blocks_per_session)]
    best, best_score = None, None
    for _ in range(n_candidates):
        blocks = _blocks(rng, len(sess))
        cnt = {}
        for b in blocks:
            for a, L, t in b:
                cnt[(a, L, t)] = cnt.get((a, L, t), 0) + 1
        if max(cnt.values()) > 2:                   # each (arm, length) in each triple position at most twice
            continue
        score = max(_stats(blocks, sess).values())
        if best_score is None or score < best_score:
            best, best_score = blocks, score
    if best is None:
        raise RuntimeError("no admissible held-out schedule")
    plan = []
    for bi, (b, s) in enumerate(zip(best, sess)):
        for i, (a, L, t) in enumerate(b):
            lid = f"s{s}-b{bi % blocks_per_session + 1}-p{i + 1:02d}"
            plan.append({"session": s, "block": bi % blocks_per_session + 1, "pos": i + 1, "triple": t + 1, "arm": a,
                         "length_s": L, "launch_id": lid, "pause_ns": pause_ns(f"v2.4-heldout-s{s}", lid)})
    return {"seed": seed, "n_candidates": n_candidates, "max_abs_corr": best_score, "balance": _stats(best, sess),
            "plan": plan}


# ------------------------------------------------------------------ counter-call inventory (expected; v2.4 logs it)

PROBE_READS_EST = {"median": 254, "p10": 96, "p90": 300}   # gate 3b estimate; v2.4 logs the exact count


def per_gpu_calls(region_s: float, dose: str = "E0", m4d: bool = False) -> list[dict]:
    """Chronological cumulative-counter calls in one launch for ONE GPU (each GPU thread does the same)."""
    calls = [{"call": "C_prep0", "n": 1, "where": "after data init + sync, before the warm-up", "in_region": False},
             {"call": "probe (v2.1)", "n": f"~{PROBE_READS_EST['median']} (logged exactly)",
              "where": "after warm-up + sync + barrier; GPU idle; before the M4c pause", "in_region": False},
             {"call": "C0", "n": 1, "where": "region start (after pause + barrier)", "in_region": "boundary"}]
    if dose != "E0":
        period = DOSE_PERIOD_MS[dose] / 1000.0
        calls.append({"call": f"dose poller {dose}", "n": math.floor(region_s / period),
                      "where": "between C0 and the loop end (calibration S0b / C2 only)", "in_region": True})
    calls += [{"call": "C1", "n": 1, "where": "after loop end, 8 ms after the predicted update", "in_region": False},
              {"call": "Cpost", "n": 1, "where": "one nominal period after C1's predicted update", "in_region": False}]
    if m4d:
        calls.append({"call": "post-probe (M4d)", "n": "~100-300 (logged exactly, <= 2048)",
                      "where": "after Cpost; outside C0..Cpost", "in_region": False})
    return calls


CONTROLLER_CALLS = [
    {"call": "session-start idle baseline", "n_per_gpu": 2, "where": "GPUs idle, no experiment process"},
    {"call": "session-end idle baseline", "n_per_gpu": 2, "where": "GPUs idle, no experiment process"},
    {"call": "preflight / readiness / sampler / watchdog / per-launch idle window", "n_per_gpu": 0,
     "where": "none of these read the cumulative counter"},
]


def poller_active_launches(schedule: list[dict]) -> int:
    return sum(1 for u in schedule if u.get("dose", "E0") != "E0")
