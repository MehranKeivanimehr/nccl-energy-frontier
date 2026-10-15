"""In-process energy meters for the Phase 10 workload.

AlignedMeter (method "v2.3-aligned-meter-logged-qc") uses the same commit-aligned
accounting as v2.1 (patch v2.1 / the v2.1 AlignedMeter), and now also keeps
every underlying value: raw counter readings C0 (start), C1 (end), Cpost (post
window), NVML return status of every read, read-call brackets (monotonic ns
before and after each call), probe edges, phase, predicted commits, the period
used and the period estimated from the probe edges, and QC flags. The energy is
recomputable offline from `per_gpu[g]["raw"]` with `aligned_energy()`.

Interpretation (supported by the retained local experiments of Phase 4 on
driver 580; not an NVML guarantee): the counter value advances in steps about
every 100 ms, and a read discards energy accumulated since the last step, so the
post window starts at the end read. The period is configurable and is checked
against the edges observed in the probe; a mismatch is flagged.

SampledMeter (method "v2.3-sampled-meter-qc") never calls the cumulative counter:
energy is the trapezoidal integral of sampled power (POWER_INSTANT if supported,
else PowerUsage shifted by a declared lag) over [start, stop].
"""

from __future__ import annotations

import time
import math

ALIGNED_METHOD = "v2.3-aligned-meter-logged-qc"
SAMPLED_METHOD = "v2.3-sampled-meter-qc"
P_NS = 100_000_000
GUARD_NS, MARGIN_NS = 2_000_000, 8_000_000
BRACKET_MAX_NS = 20_000_000            # same threshold as region.py COUNTER_PHASE_IMPRECISE
PERIOD_TOL_NS = 2_000_000
TAIL_P_MIN_W, TAIL_P_MAX_W = 5.0, 250.0  # same plausibility range as region.py


def _N():
    import pynvml
    return pynvml


def _now() -> int:
    return time.monotonic_ns()


def _sleep_until(t_ns: int) -> None:
    d = t_ns - _now()
    if d > 0:
        time.sleep(d / 1e9)


def _mid(br):
    return (br[0] + br[1]) // 2


def aligned_energy(raw: dict) -> dict:
    """Commit-aligned region energy of one GPU from raw readings. Same formula as the v2.1 meter:
    energy = (C1 - C0) - p_ctx * tail, p_ctx = (Cpost - C1) / (commit + P - t1), tail = commit - t_end."""
    flags = []
    vals = (raw.get("c0_mJ"), raw.get("c1_mJ"), raw.get("cpost_mJ"))
    if any(v is None for v in vals) or any(raw.get(k) != 0 for k in ("rc0", "rc1", "rcpost")):
        return {"energy_J": None, "flags": ["READ_ERROR"]}
    required = ("period_ns", "commit_ns", "t_end_ns", "br0", "br1", "brpost", "bracket_ns")
    try:
        for k in required:
            v = raw[k]
            vs = v if k.startswith("br") and k != "bracket_ns" else [v]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vs):
                raise ValueError(k)
        if any(len(raw[k]) != 2 or raw[k][1] < raw[k][0] for k in ("br0", "br1", "brpost")):
            raise ValueError("read brackets")
        if raw["period_ns"] <= 0 or raw["bracket_ns"] < 0:
            raise ValueError("period/bracket")
    except (KeyError, TypeError, ValueError):
        return {"energy_J": None, "flags": ["MALFORMED_TIMING"]}
    P = raw["period_ns"]
    c0, c1, cp = vals
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in vals):
        return {"energy_J": None, "flags": ["NONFINITE_OR_NEGATIVE_COUNTER"]}
    if raw.get("phase_ns") is None:
        return {"energy_J": None, "flags": ["PHASE_UNKNOWN"]}
    if raw["bracket_ns"] > BRACKET_MAX_NS:
        flags.append("PHASE_IMPRECISE")
    est = raw.get("period_est_ns")
    if est is not None and abs(est - P) > PERIOD_TOL_NS:
        flags.append("PERIOD_MISMATCH")
    commit, t_end = raw["commit_ns"], raw["t_end_ns"]
    t1, tp = _mid(raw["br1"]), _mid(raw["brpost"])
    duration_s = (t_end - _mid(raw["br0"])) / 1e9
    if raw["br0"][1] > t_end or duration_s <= 0 or raw["br1"][0] < t_end \
            or raw["brpost"][0] < raw["br1"][1]:
        flags.append("TIME_ORDER_INVALID")
    if raw["br1"][0] < commit:
        flags.append("END_READ_BEFORE_COMMIT")
    if raw["brpost"][0] < commit + P:
        flags.append("POST_READ_BEFORE_COMMIT")
    if c1 < c0 or cp < c1:
        flags.append("REVERSAL_OR_WRAP")
    if c1 == c0:
        flags.append("COUNTER_NOT_UPDATING")
    post_s = (commit + P - t1) / 1e9
    p_ctx = (cp - c1) / 1000.0 / post_s if post_s > 0 else None
    tail_s = (commit - t_end) / 1e9
    if p_ctx is None or not (TAIL_P_MIN_W <= p_ctx <= TAIL_P_MAX_W) or tail_s < 0:
        flags.append("TAIL_IMPLAUSIBLE")
    aligned = (c1 - c0) / 1000.0
    energy = aligned - p_ctx * tail_s if p_ctx is not None else None
    if energy is None or not math.isfinite(energy) or energy <= 0:
        flags.append("ENERGY_INVALID")
    # v2.2 release audit: every flag here corresponds to a hard COUNTER_* flag on the region path (region.py), so a
    # flagged reading yields no energy; the unvalidated value is kept for diagnosis only
    return {"aligned_J": aligned, "p_ctx_W": p_ctx, "tail_s": tail_s, "energy_J": None if flags else energy,
            "energy_J_unvalidated": energy,
            "duration_s": duration_s, "bracket_ms": raw["bracket_ns"] / 1e6,
            "post_window_s": post_s, "flags": flags}


def period_from_edges(edges: list[tuple[int, int]]) -> int | None:
    """Median spacing of consecutive probe-edge midpoints, in ns (None if fewer than 2 edges)."""
    mids = sorted((a + b) // 2 for a, b in edges)
    if len(mids) < 2:
        return None
    d = sorted(b - a for a, b in zip(mids, mids[1:]))
    return d[len(d) // 2]


class AlignedMeter:
    method = ALIGNED_METHOD

    def __init__(self, gpus: list[int], period_ns: int = P_NS):
        N = _N()
        N.nvmlInit()
        self.N = N
        self.gpus = gpus
        self.P = period_ns
        self.h = {g: N.nvmlDeviceGetHandleByIndex(g) for g in gpus}
        self.phase, self.bracket, self.edges = {}, {}, {}
        self.r0 = {}

    def _read(self, g):
        """(value_mJ or None, status 0 or error, (t_before_ns, t_after_ns))."""
        a = _now()
        try:
            v, rc = self.N.nvmlDeviceGetTotalEnergyConsumption(self.h[g]), 0
        except Exception as e:  # noqa: BLE001 - pynvml raises NVMLError_*; keep its value/name
            v, rc = None, getattr(e, "value", repr(e))
        return v, rc, (a, _now())

    def probe(self, max_periods: int = 4) -> dict:
        for g in self.gpus:
            edges, prev, prev_a = [], None, None
            t_stop = _now() + max_periods * self.P
            while _now() < t_stop and len(edges) < 4:
                v, rc, (a, b) = self._read(g)
                if rc != 0:
                    break
                if prev is not None and v != prev:
                    edges.append((prev_a, b))
                prev, prev_a = v, a
            self.edges[g] = edges
            if edges:
                lo, hi = min(edges, key=lambda e: e[1] - e[0])
                self.phase[g], self.bracket[g] = (lo + hi) // 2, hi - lo
            else:
                self.phase[g], self.bracket[g] = None, None
        return {"phase_ns": dict(self.phase), "bracket_ns": dict(self.bracket),
                "edges": {g: list(e) for g, e in self.edges.items()},
                "period_est_ns": {g: period_from_edges(e) for g, e in self.edges.items()}}

    def start(self) -> None:
        for g in self.gpus:
            self.r0[g] = self._read(g)

    def stop(self) -> dict:
        t_end = _now()
        commits = {}
        for g in self.gpus:
            if self.phase.get(g) is None:
                commits[g] = t_end + 2 * self.P   # phase unknown: flagged below, energy not computed
                continue
            k = -(-(t_end + GUARD_NS - self.phase[g]) // self.P)   # ceil
            commits[g] = self.phase[g] + k * self.P
        r1, rp = {}, {}
        for g in sorted(self.gpus, key=lambda x: commits[x]):
            _sleep_until(commits[g] + MARGIN_NS)
            r1[g] = self._read(g)
        for g in sorted(self.gpus, key=lambda x: commits[x]):
            _sleep_until(commits[g] + self.P + MARGIN_NS)
            rp[g] = self._read(g)
        out = {"method": self.method, "t_start_ns": {g: _mid(self.r0[g][2]) for g in self.gpus}, "t_end_ns": t_end,
               "period_ns": self.P, "guard_ns": GUARD_NS, "margin_ns": MARGIN_NS, "per_gpu": {}}
        for g in self.gpus:
            raw = {"c0_mJ": self.r0[g][0], "rc0": self.r0[g][1], "br0": list(self.r0[g][2]),
                   "c1_mJ": r1[g][0], "rc1": r1[g][1], "br1": list(r1[g][2]),
                   "cpost_mJ": rp[g][0], "rcpost": rp[g][1], "brpost": list(rp[g][2]),
                   "phase_ns": self.phase.get(g), "bracket_ns": self.bracket.get(g), "edges": self.edges.get(g, []),
                   "period_ns": self.P, "period_est_ns": period_from_edges(self.edges.get(g, [])),
                   "commit_ns": commits[g], "t_end_ns": t_end}
            res = aligned_energy(raw)
            out["per_gpu"][g] = {**res, "raw": raw}
        vals = [v["energy_J"] for v in out["per_gpu"].values()]
        out["flags"] = sorted({f for v in out["per_gpu"].values() for f in v["flags"]})
        out["energy_J"] = sum(vals) if all(v is not None for v in vals) else None
        return out


class SampledMeter:
    """Energy from integrated sampled power; never calls nvmlDeviceGetTotalEnergyConsumption."""
    method = SAMPLED_METHOD

    def __init__(self, gpus: list[int], period_s: float = 0.05, lag_s: float = 0.0, tail_s: float = 0.4):
        from .telemetry import Sampler
        self.gpus, self.lag_s, self.tail_s = gpus, lag_s, tail_s
        self.s = Sampler(gpus, period_s=period_s)
        self.t0 = self.t1 = None

    def probe(self) -> dict:
        self.s.start()
        time.sleep(0.5)
        return {"sampler_started": True}

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> dict:
        from .qc import telemetry_problems
        from .telemetry import window_summary
        self.t1 = time.perf_counter()
        time.sleep(self.tail_s + self.lag_s)
        self.s.stop()
        arr = self.s.array()
        per, flags = {}, []
        for g in self.gpus:
            w = window_summary(arr, g, self.t0, self.t1, self.s.period_s, lag_s=self.lag_s)
            ei = w.get("energy_int_inst_J")
            if ei is not None and math.isfinite(ei):
                e, src = ei, "POWER_INSTANT"
            elif self.lag_s:
                e, src = w.get("energy_int_lag_J"), "PowerUsage_lag"
            else:
                e, src = None, "NONE"
            per[g] = {"energy_J": e, "source": src, "window": w}
        flags = telemetry_problems({"errors": len(self.s.errors), "truncated": self.s.truncated},
                                   {g: v["window"] for g, v in per.items()}, len(self.gpus), "sampled")
        vals = [v["energy_J"] for v in per.values()]
        ok = not flags and all(v is not None and math.isfinite(v) and v > 0 for v in vals)
        return {"method": self.method, "t_start_s": self.t0, "t_end_s": self.t1, "lag_s": self.lag_s,
                "per_gpu": per, "flags": flags, "energy_J": sum(vals) if ok else None}
