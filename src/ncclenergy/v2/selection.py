"""Allowed configuration set from NCCL's init tuning table (NCCL_DEBUG_SUBSYS=TUNING).

NCCL 2.27.5 prints, per communicator at init, a table of latency/bandwidth per
(function, algorithm, protocol). A (function, algorithm, protocol) with
bandwidth 0.0 is not enabled. Run with NCCL_ALGO/NCCL_PROTO unset, the table
gives the set NCCL enables by default for that placement.
"""

from __future__ import annotations

import re

_PREFIX = re.compile(r"^.*?NCCL INFO ")
FUNCS = ("Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce")


def parse_tuning_table(log_text: str) -> dict:
    """Return {func: {algo: {proto: (lat_us, bw)}}} from the first complete table in the log."""
    lines = [_PREFIX.sub("", ln) for ln in log_text.splitlines()]
    out: dict = {}
    algos: list[str] = []
    protos: list[str] = []
    for ln in lines:
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) < 2:
            continue
        head = cells[0]
        if head == "Algorithm":
            algos = [c for c in cells[1:] if c]
            protos = []
            continue
        if head == "Protocol":
            protos = [c for c in cells[1:] if c]
            continue
        if head in FUNCS and algos and protos:
            vals = [c for c in cells[1:] if c]
            per_algo = len(protos) // max(1, len(algos))
            for k, v in enumerate(vals):
                m = re.match(r"([\d.]+)\s*/\s*([\d.]+)", v)
                if not m or per_algo == 0:
                    continue
                algo = algos[k // per_algo]
                proto = protos[k]
                out.setdefault(head, {}).setdefault(algo, {})[proto] = (float(m[1]), float(m[2]))
    return out


def enabled(table: dict, func: str, algo: str, proto: str) -> bool | None:
    """True/False from the table; None if the table does not contain the entry."""
    try:
        return table[func][algo][proto][1] > 0.0
    except KeyError:
        return None
