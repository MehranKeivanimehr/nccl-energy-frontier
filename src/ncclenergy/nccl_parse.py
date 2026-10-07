"""Parsers for nccl-tests stdout and NCCL_DEBUG=INFO logs.

Nothing here invents values: every field is either parsed from text produced
by nccl-tests / NCCL or left as None.
"""

from __future__ import annotations

import bisect
import re

# nccl-tests stdout

RE_VERSION = re.compile(r"^# nccl-tests version (\S+) \((\w+)\) nccl-headers=(\d+) nccl-library=(\d+)")
RE_PARAMS = re.compile(
    r"^# nThread (\d+) nGpus (\d+) minBytes (\d+) maxBytes (\d+) step: (\d+)\((\w+)\) "
    r"warmup iters: (\d+) iters: (\d+) agg iters: (\d+) validation: (\d+) graph: (\d+)")
RE_DEVICE = re.compile(
    r"^#\s+Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+(\d+)\s+on\s+(\S+)\s+device\s+(\d+)\s+\[([0-9a-fA-F:.]+)\]\s+(.*?)\s*$")
RE_ROW = re.compile(r"^\s*\d+\s+\d+\s+\w+\s+\S+\s+-?\d+(\s+\S+){8}\s*$")
RE_OOB = re.compile(r"^# Out of bounds values : (\d+) (\w+)")
RE_AVGBW = re.compile(r"^# Avg bus bandwidth\s*:\s*(\S+)")
RE_FAILURE = re.compile(r"Test failure (\S+)")

ROW_KEYS = ["size_bytes", "count", "type", "redop", "root"]
BODY_KEYS = ["time_us", "algbw_gbs", "busbw_gbs", "wrong"]


def _num(tok: str):
    if tok == "N/A":
        return None
    try:
        return int(tok)
    except ValueError:
        return float(tok)


def parse_row(line: str) -> dict:
    toks = line.split()
    rec = dict(zip(ROW_KEYS, [_num(toks[0]), _num(toks[1]), toks[2], toks[3], _num(toks[4])]))
    for prefix, chunk in (("oop", toks[5:9]), ("ip", toks[9:13])):
        for k, v in zip(BODY_KEYS, chunk):
            rec[f"{prefix}_{k}"] = _num(v)
    return rec


def parse_stdout(text: str) -> dict:
    out = {"version": None, "params": None, "devices": [], "rows": [], "oob": None,
           "avg_busbw": None, "failures": []}
    for line in text.splitlines():
        if m := RE_VERSION.match(line):
            out["version"] = {"nccl_tests": m[1], "commit": m[2], "nccl_headers": int(m[3]), "nccl_library": int(m[4])}
        elif m := RE_PARAMS.match(line):
            out["params"] = dict(zip(
                ["nthread", "ngpus", "minbytes", "maxbytes", "step", "step_kind", "warmup_iters", "iters",
                 "agg_iters", "validation", "graph"],
                [int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), m[6], int(m[7]), int(m[8]),
                 int(m[9]), int(m[10]), int(m[11])]))
        elif m := RE_DEVICE.match(line):
            out["devices"].append({"rank": int(m[1]), "group": int(m[2]), "pid": int(m[3]), "host": m[4],
                                   "cuda_dev": int(m[5]), "pci": m[6].lower(), "name": m[7]})
        elif RE_ROW.match(line):
            out["rows"].append(parse_row(line))
        elif m := RE_OOB.match(line):
            out["oob"] = {"count": int(m[1]), "status": m[2]}
        elif m := RE_AVGBW.match(line):
            out["avg_busbw"] = float(m[1])
        if m := RE_FAILURE.search(line):
            out["failures"].append(m[1])
    return out


def region_offsets(text: str) -> list[dict]:
    """Byte offsets at which each region-delimiting token of a result row ends.

    nccl-tests writes a row in three printf calls: preamble (size..root), the
    out-of-place body and the in-place body.  The out-of-place measurement
    region runs from the end of the preamble to the end of the out-of-place
    body; the in-place region from there to the end of the in-place body.
    Offsets are in bytes of the UTF-8 encoded stdout stream.
    """
    res = []
    pos = 0
    data = text.encode()
    for raw in data.splitlines(keepends=True):
        line = raw.decode()
        if RE_ROW.match(line.rstrip("\n")):
            toks = [m for m in re.finditer(r"\S+", line)]
            # character == byte offsets because the row is pure ASCII
            res.append({
                "size_bytes": int(toks[0].group()),
                "pre_end": pos + toks[4].end(),
                "oop_end": pos + toks[8].end(),
                "ip_end": pos + toks[12].end(),
            })
        pos += len(raw)
    return res


def assign_events(offsets: list[dict], chunk_ends: list[int]) -> list[dict]:
    """Map token-end offsets to the index of the stdout chunk that delivered them."""
    out = []
    for o in offsets:
        rec = {"size_bytes": o["size_bytes"]}
        for k in ("pre_end", "oop_end", "ip_end"):
            i = bisect.bisect_left(chunk_ends, o[k])
            rec[k.replace("_end", "_chunk")] = i if i < len(chunk_ends) else None
        out.append(rec)
    return out


# NCCL debug log

RE_NCCL_VERSION = re.compile(r"NCCL INFO NCCL version (\S+)")
RE_DRIVER = re.compile(r"NCCL INFO cudaDriverVersion (\d+)")
RE_ENV = re.compile(r"NCCL INFO (NCCL_\w+) set by environment to (\S+)")
RE_INIT = re.compile(r"ncclCommInitRankConfig comm \S+ rank (\d+) nranks (\d+) cudaDev (\d+) nvmlDev (\d+) busId (\w+)")
RE_TRANSPORT = re.compile(r"Channel (\d+)(?:/\d+)? : (\d+)\[(\d+)\] -> (\d+)\[(\d+)\] via (.+?)\s*$")
RE_TUNING = re.compile(
    r"NCCL INFO (\w+): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}")
RE_WARN = re.compile(r"NCCL WARN (.*)$")
RE_CHANNELS = re.compile(r"(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels")
RE_MATRIX_ROW = re.compile(
    r"^\s*(Broadcast|Reduce|AllGather|ReduceScatter|AllReduce) \|\s+(\d)\s+(\d)\s+(\d)\s+\|\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s*$")
RE_INIT_TIME = re.compile(r"Init timings - ncclCommInitRankConfig: rank (\d+) nranks (\d+) total ([\d.]+)")
MATRIX_PROTOS = ["LL", "LL128", "Simple"]
MATRIX_ALGOS = ["Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT"]


def parse_nccl_log(text: str) -> dict:
    out = {"nccl_version": None, "cuda_driver": None, "env": {}, "ranks": [], "transports": set(),
           "tuning": [], "warnings": [], "channels": None, "enabled_matrix": {}, "init_total_s": []}
    for line in text.splitlines():
        if m := RE_TUNING.search(line):
            out["tuning"].append({"func": m[1], "bytes": int(m[2]), "algo": m[3], "proto": m[4],
                                  "ch_lo": int(m[5]), "ch_hi": int(m[6])})
            continue
        if m := RE_NCCL_VERSION.search(line):
            out["nccl_version"] = m[1]
        elif m := RE_DRIVER.search(line):
            out["cuda_driver"] = int(m[1])
        elif m := RE_ENV.search(line):
            out["env"][m[1]] = m[2]
        elif m := RE_INIT.search(line):
            out["ranks"].append({"rank": int(m[1]), "nranks": int(m[2]), "cuda_dev": int(m[3]),
                                 "nvml_dev": int(m[4]), "bus_id": m[5]})
        elif m := RE_TRANSPORT.search(line):
            out["transports"].add((int(m[3]), int(m[5]), m[6]))
        elif m := RE_WARN.search(line):
            out["warnings"].append(m[1].strip())
        elif m := RE_CHANNELS.search(line):
            out["channels"] = {"coll": int(m[1]), "collnet": int(m[2]), "nvls": int(m[3]), "p2p": int(m[4])}
        elif m := RE_MATRIX_ROW.match(line):
            vals = [int(x) for x in m.groups()[1:]]
            out["enabled_matrix"][m[1]] = {"proto": dict(zip(MATRIX_PROTOS, vals[:3])),
                                           "algo": dict(zip(MATRIX_ALGOS, vals[3:]))}
        elif m := RE_INIT_TIME.search(line):
            out["init_total_s"].append(float(m[3]))
    # dedupe ranks (one INIT START + one COMPLETE line per rank)
    seen, ranks = set(), []
    for r in out["ranks"]:
        if r["rank"] not in seen:
            seen.add(r["rank"])
            ranks.append(r)
    out["ranks"] = sorted(ranks, key=lambda r: r["rank"])
    out["transports"] = sorted(out["transports"])
    return out


def unsupported_reason(log: dict, stdout: dict, stderr_text: str) -> str | None:
    """Return a reason string if NCCL rejected the requested algorithm/protocol."""
    for w in log["warnings"]:
        if "no algorithm/protocol available" in w:
            return w
    if "no algorithm/protocol available" in stderr_text:
        return "no algorithm/protocol available (stderr)"
    return None


def selection_by_size(tuning: list[dict], func: str) -> dict[int, list[tuple]]:
    """Group TUNING lines by NCCL byte count -> list of distinct (algo, proto, ch_lo, ch_hi)."""
    res: dict[int, set] = {}
    for t in tuning:
        if t["func"] != func:
            continue
        res.setdefault(t["bytes"], set()).add((t["algo"], t["proto"], t["ch_lo"], t["ch_hi"]))
    return {k: sorted(v) for k, v in res.items()}
