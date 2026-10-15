"""Methodology-v2 runner: probe, calibrate, freeze, block.

    python -m ncclenergy.v2.runner preflight --alloc 4,5,6,7
    python -m ncclenergy.v2.runner probe     --campaign configs/v2/setup.yaml
    python -m ncclenergy.v2.runner calibrate --campaign configs/v2/setup.yaml
    python -m ncclenergy.v2.runner freeze    --campaign configs/v2/setup.yaml --T-star 2.5
    python -m ncclenergy.v2.runner block     --campaign configs/v2/<campaign>.yaml

One experiment process at a time. Every attempt is appended to the block's
attempts.jsonl and never rewritten. A block stops (artifacts kept) on any
stop condition of docs/v2/protocol.md section 9.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .attempts import AttemptLog, final_attempt, should_retry
from .common import COLL_FUNC, HAS_OP, REPO, append_jsonl, git_provenance, norm_pci, redact, sha256_bytes, sha256_file, utc_now
from .evidence import load_profiler_dir, verify
from .preflight import allocation_problems, light_health, poll_xid, snapshot, xid_event_test
from .procctl import ControllerSignal, OwnedProcess, SafetyStop, controller_signals
from .qc import QC_METHOD, bad_clock_events, classify, telemetry_problems
from .watchdog import DEFAULT_BAD_BITS, DiskFloorCheck, SamplerHealthCheck, WallClockCheck, Watchdog
from .region import parse_region_text, region_metrics
from .selection import enabled, parse_tuning_table
from .telemetry import Sampler, window_summary

BIN_DIR = REPO / "third_party/nccl-tests-v2/build"
NCCL_LIB = REPO / "third_party/nccl_home/lib"
PLUGIN = REPO / "native/nccl_profiler_algo/build/libnccl-profiler-nccle.so"
PATCH = REPO / "patches/nccl-tests-afd59ab-v2.1-region.patch"   # commit-aligned end read (Phase 4)
ERROR_PAT = re.compile(r"(NCCL WARN.*(error|fail|Error|Fail))|Cuda failure|CUDA failure|NCCL failure|Test failure|"
                       r"out of memory|Xid", re.I)
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\S+)\s+(-?\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")


class StopBlock(Exception):
    pass


# ------------------------------------------------------------------ config

def load_campaign(path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw), sha256_bytes(raw)


def sizes(cfg) -> list[int]:
    s = cfg["sizes"]
    if "list" in s:
        return [int(x) for x in s["list"]]
    out, v = [], int(s["min"])
    while v <= int(s["max"]):
        out.append(v)
        v *= int(s.get("factor", 2))
    return out


def parse_cfg_label(label: str) -> tuple[str, str]:
    if label == "auto":
        return "default", "default"
    a, p = label.split("/")
    return a, p


def unit_id(placement, coll, cfg_label, size) -> str:
    return f"{placement}__{coll}__{cfg_label.replace('/', '-')}__{size}"


def unit_key(u: dict) -> str:
    """unit id of a planned unit; v2.2 campaigns with region_lengths_s add the target length (e.g. __T2.5s).
    Without region lengths the id is unchanged (historical campaigns)."""
    base = unit_id(u["placement"], u["coll"], u["config"], u["size"])
    return base if u.get("length_s") is None else f"{base}__T{float(u['length_s']):g}s"


# ------------------------------------------------------------------ process launch

def build_env(uuids: list[str], cfg_label: str, d: Path, debug_subsys="INIT,ENV", extra=None,
              plugin=True, energy_source="counter") -> dict:
    if energy_source not in ("counter", "sampled"):
        raise ValueError(f"unsupported energy_source: {energy_source}")
    if not uuids or len(set(uuids)) != len(uuids):
        raise ValueError("participating GPU UUIDs must be nonempty and unique")
    reserved = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_PRELOAD", "LD_LIBRARY_PATH",
                "NCCL_ALGO", "NCCL_PROTO", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE",
                "NCCL_PROFILER_PLUGIN", "NCCL_TUNER_PLUGIN"}
    forbidden = sorted(k for k in (extra or {}) if k in reserved or k.startswith("NCCLE_"))
    if forbidden:
        raise ValueError(f"env_extra cannot override allocation/measurement controls: {forbidden}")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("NCCL_", "NCCLE_")) and k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    env["LD_LIBRARY_PATH"] = str(NCCL_LIB) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    env["NCCLE_REGION_FILE"] = str(d / "region.jsonl")
    env["NCCLE_PLACEMENT"] = "oop"
    env["NCCLE_ENERGY_SOURCE"] = energy_source
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = debug_subsys
    env["NCCL_DEBUG_FILE"] = str(d / "nccl.log")
    if plugin:
        env["NCCL_PROFILER_PLUGIN"] = str(PLUGIN)
        env["NCCLE_PROFILER_DIR"] = str(d / "prof")
    algo, proto = parse_cfg_label(cfg_label)
    if algo != "default":
        env["NCCL_ALGO"] = algo
    if proto != "default":
        env["NCCL_PROTO"] = proto
    for k, v in (extra or {}).items():
        env[k] = str(v)
    return env


def build_cmd(coll: str, ngpu: int, minb: int, maxb: int, R: int, w: int, check: int, numa: int | None,
              bin_dir: Path | None = None) -> list[str]:
    """bin_dir: Stage 2 only (its own patched build); None keeps the historical BIN_DIR."""
    cmd = [str((bin_dir if bin_dir is not None else BIN_DIR) / f"{coll}_perf"), "-t", str(ngpu), "-g", "1", "-b", str(minb), "-e", str(maxb), "-f", "2",
           "-n", str(R), "-w", str(w), "-c", str(check), "-d", "float", "-m", "1", "-G", "0", "-z", "0"]
    if HAS_OP[coll]:
        cmd += ["-o", "sum"]
    pre = ["nice", "-n", "10"]
    if numa is not None:
        pre = ["numactl", f"--cpunodebind={numa}", f"--membind={numa}"] + pre
    return pre + cmd


def run_process(cmd, env, d: Path, timeout_s: float, watch: dict | None = None) -> dict:
    """Run one experiment command as an owned process tree (procctl.OwnedProcess, v2.2).

    watch: {"gpus", "temp_stop_c", "disk_floor_gb", "bad_bits"?, "sampler"?, "unit_wall_limit_s"?}. With watch, a
    watchdog cancels the run on contention, health events, temperature, telemetry failure, low disk or (if set) a
    per-unit wall-clock limit (reason in res["stop_reason"]).
    Owned processes are always terminated and reaped (res["cleanup"]); SIGINT/SIGTERM/SIGHUP to the controller
    propagate as procctl.ControllerSignal after cleanup.
    """
    d.mkdir(parents=True, exist_ok=True)
    (d / "prof").mkdir(exist_ok=True)
    own_sampler, wd = None, None
    with open(d / "stdout.txt", "wb") as so, open(d / "stderr.txt", "wb") as se:
        op = OwnedProcess(cmd, env=env, cwd=d, stdout=so, stderr=se, timeout_s=timeout_s)
        try:
            if watch:
                smp = watch.get("sampler")
                if smp is None:
                    smp = own_sampler = Sampler(watch["gpus"], period_s=0.05).start()

                def owned():
                    op.snapshot()
                    return op.ever_owned_pids() | {os.getpid()}

                checks = [SamplerHealthCheck(smp, watch["gpus"], owned, watch["temp_stop_c"],
                                             watch.get("bad_bits", DEFAULT_BAD_BITS))]
                if watch.get("disk_floor_gb") is not None:
                    checks.append(DiskFloorCheck(REPO, watch["disk_floor_gb"]))
                if watch.get("unit_wall_limit_s"):
                    checks.append(WallClockCheck(float(watch["unit_wall_limit_s"])))
                wd = op.watchdog = Watchdog(checks).start()
            res = op.run()
        finally:
            if wd is not None:
                wd.stop()
            if own_sampler is not None:
                own_sampler.stop()
            # persisted on every exit path (also when a controller signal propagates): exit status, stop reason,
            # watchdog trip and the cleanup report of the owned process tree
            r = getattr(op, "result", None) or {}
            try:
                (d / "owned_process.json").write_text(json.dumps(
                    {k: r.get(k) for k in ("pid", "rc", "timed_out", "stop_reason", "exception", "cleanup", "wall_s",
                                           "subreaper", "signals_during_cleanup")}
                    | {"watchdog_trip": wd.tripped if wd is not None else None,
                       "watchdog_trip_mono": wd.trip_time if wd is not None else None,
                       "written_utc": utc_now()}, indent=1, default=str))
            except Exception:  # noqa: BLE001 - never mask the original outcome
                pass
    res["watchdog_trip"] = wd.tripped if wd is not None else None
    return res


def safety_problem(res: dict) -> str | None:
    """Reason to stop the campaign after a run (watchdog trip or failed cleanup), else None."""
    if (res.get("stop_reason") or "").startswith("WATCHDOG"):
        return res["stop_reason"]
    if res.get("watchdog_trip"):          # tripped after the process exited but within the unit: fail closed
        return f"WATCHDOG_AFTER_EXIT:{res['watchdog_trip']}"
    cl = res.get("cleanup") or {}
    if not cl.get("ok", False):
        return f"CLEANUP_FAILED:survivors={cl.get('survivors')} errors={cl.get('errors')}"
    return None


def parse_stdout_rows(text: str) -> list[dict]:
    rows = []
    for ln in text.splitlines():
        m = ROW.match(ln)
        if m:
            wrong = None if m[9] == "N/A" else int(float(m[9]))
            rows.append({"size_bytes": int(m[1]), "count": int(m[2]), "time_us": float(m[6]), "wrong": wrong})
    return rows


def error_hits(d: Path) -> list[str]:
    hits = []
    for name in ("stderr.txt", "nccl.log"):
        p = d / name
        if p.exists():
            for ln in p.read_text(errors="replace").splitlines():
                if ERROR_PAT.search(ln):
                    hits.append(f"{name}: {ln.strip()[:300]}")
    return hits[:20]


# ------------------------------------------------------------------ context

class Ctx:
    """Allocation, placements and provenance shared by all commands."""

    def __init__(self, cfg: dict, cfg_hash: str):
        self.cfg, self.cfg_hash = cfg, cfg_hash
        self.alloc = list(cfg["allocation"])
        if not self.alloc or len(set(self.alloc)) != len(self.alloc):
            raise SystemExit("allocation must be nonempty and contain unique GPU indices")
        if len(self.alloc) > 6:
            raise SystemExit("allocation larger than 6 GPUs is not allowed")
        ids = [p["id"] for p in cfg["placements"]]
        if not ids or len(set(ids)) != len(ids):
            raise SystemExit("placement IDs must be nonempty and unique")
        for p in cfg["placements"]:
            gp = list(p["gpus"])
            if not gp or len(set(gp)) != len(gp) or not set(gp) <= set(self.alloc):
                raise SystemExit(f"placement {p['id']} requires unique GPUs within the allocation")
        self.snap = snapshot(self.alloc)
        self.by_index = {g["index"]: g for g in self.snap["gpus"]}
        self.placements = {p["id"]: list(p["gpus"]) for p in cfg["placements"]}
        self.placement_env = {p["id"]: dict(p.get("env") or {}) for p in cfg["placements"]}
        for pid, gp in self.placements.items():
            if not set(gp) <= set(self.alloc):
                raise SystemExit(f"placement {pid} uses GPUs outside the allocation")
        self.out = REPO / cfg.get("output_root", "data/v2/raw") / cfg["campaign"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.git = git_provenance()
        self.versions = {"driver": self.snap["driver"], "nvml": self.snap["nvml"], "cuda_driver": self.snap["cuda_driver"],
                         "nccl_lib_sha256": sha256_file(NCCL_LIB / "libnccl.so.2"),
                         "nccl_tests_commit": "afd59abab774a6a4b492eb344bd0c845a0b7bb4d",
                         "patch_sha256": sha256_file(PATCH), "plugin_sha256": sha256_file(PLUGIN),
                         "binary_sha256": {c: sha256_file(BIN_DIR / f"{c}_perf") for c in COLL_FUNC}}

    def uuids(self, gpus):
        return [self.by_index[g]["uuid"] for g in gpus]

    def numa(self, gpus):
        nodes = {self.snap["topology"].get("numa", {}).get(g) for g in gpus}
        return nodes.pop() if len(nodes) == 1 and None not in nodes else None

    def placement_meta(self, pid):
        gpus = self.placements[pid]
        from .topology import fingerprint
        topo = {"links": {tuple(map(int, k.split("-"))): v for k, v in self.snap["topology"]["links"].items()},
                "numa": self.snap["topology"]["numa"]}
        fp = fingerprint(gpus, topo, {g: self.by_index[g]["pci"] for g in gpus}, self.snap["nvlink_active"],
                         "2.27.5")
        return {"id": pid, "gpus": gpus, "env": self.placement_env.get(pid, {}), "pci": [self.by_index[g]["pci"] for g in gpus],
                "aliases": [self.by_index[g]["alias"] for g in gpus],
                "cuda_visible_devices_rank_to_alias": {r: self.by_index[g]["alias"] for r, g in enumerate(gpus)},
                "fingerprint": fp}

    def watch(self, gpus, sampler=None) -> dict:
        st = self.cfg.get("stop", {})
        return {"gpus": list(gpus), "sampler": sampler, "temp_stop_c": st.get("temp_c", 80),
                "disk_floor_gb": st.get("disk_floor_gb", 20),
                "unit_wall_limit_s": st.get("unit_wall_limit_s")}   # smoke-test option: self-initiated cancel

    def check_allocation(self) -> list[str]:
        h = light_health(self.alloc)
        return allocation_problems(h, self.alloc, self.cfg.get("stop", {}).get("mem_used_mib_max", 16))


# ------------------------------------------------------------------ probe / calibrate / freeze

def cmd_probe(ctx: Ctx, only: list[str] | None = None, timeout_s: float = 120):
    """Per (placement, collective): auto probe with the init tuning table; plus rejected-config probes.
    Stops at the first timeout (hang) without probing further."""
    out = ctx.out / "probe"
    res = {}
    for pid, gpus in ctx.placements.items():
        if only and pid not in only:
            continue
        for coll in ctx.cfg["collectives"]:
            if ctx.check_allocation():
                raise StopBlock(f"allocation busy: {ctx.check_allocation()}")
            probes = [("auto", "auto")]
            if coll != "all_reduce":
                probes.append(("Tree-rejection", "Tree/Simple"))
            for tag, label in probes:
                d = out / f"{pid}__{coll}__{tag}"
                if d.exists():
                    raise SystemExit(f"{d} exists; probes are append-only")
                env = build_env(ctx.uuids(gpus), label, d, debug_subsys="INIT,ENV,TUNING", energy_source=ctx.cfg.get("energy_source", "counter"),
                                extra={**ctx.placement_env.get(pid, {}), **(ctx.cfg.get("env_extra") or {})})
                cmd = build_cmd(coll, len(gpus), 1 << 20, 1 << 20, 1, 1, 1, ctx.numa(gpus))
                r = run_process(cmd, env, d, timeout_s, watch=ctx.watch(gpus))
                if safety_problem(r) and not r["timed_out"]:
                    raise StopBlock(f"SAFETY_STOP probe {pid} {coll} {tag}: {safety_problem(r)}")
                log = (d / "nccl.log").read_text(errors="replace") if (d / "nccl.log").exists() else ""
                table = parse_tuning_table(log)
                rec = {"placement": ctx.placement_meta(pid), "coll": coll, "probe": tag, "requested": label,
                       "rc": r["rc"], "timed_out": r["timed_out"], "table": table,
                       "rejected": "no algorithm/protocol available" in log,
                       "warn": [ln for ln in log.splitlines() if "NCCL WARN" in ln][:5]}
                append_jsonl(out / "probe.jsonl", rec)
                res[(pid, coll, tag)] = rec
                print(f"probe {pid} {coll} {tag}: rc={r['rc']} rejected={rec['rejected']} "
                      f"table_funcs={sorted(table)}", flush=True)
                if r["timed_out"]:
                    raise StopBlock(f"probe {pid} {coll} {tag} exceeded {timeout_s} s (hang); stopping all GPU work")
    return res


def allowed_configs(ctx: Ctx, probe_dir: Path) -> dict:
    """{(placement, coll): {cfg_label: (allowed:bool, reason)}} from the auto probes' tuning tables."""
    recs = [json.loads(ln) for ln in (probe_dir / "probe.jsonl").read_text().splitlines() if ln.strip()]
    out = {}
    for r in recs:
        if r["probe"] != "auto":
            continue
        key = (r["placement"]["id"], r["coll"])
        func = COLL_FUNC[r["coll"]]
        out[key] = {"auto": (True, "auto")}
        for label in ctx.cfg["configs"]:
            if label == "auto":
                continue
            a, p = label.split("/")
            ok = enabled(r["table"], func, a, p)
            out[key][label] = (bool(ok), "enabled_by_default" if ok else
                               ("NOT_ENABLED_BY_NCCL" if ok is False else "NOT_IN_TABLE"))
    return out


def cmd_calibrate(ctx: Ctx, setup_dir: Path):
    allowed = allowed_configs(ctx, setup_dir / "probe")
    out = setup_dir / "calibration"
    for (pid, coll), cfgs in sorted(allowed.items()):
        if pid not in ctx.placements:     # e.g. abandoned placements whose probes remain on record
            continue
        gpus = ctx.placements[pid]
        for label, (ok, why) in cfgs.items():
            if not ok:
                continue
            for lo, hi, n, w in ((min(sizes(ctx.cfg)), 1 << 20, 2000, 20), (2 << 20, max(sizes(ctx.cfg)), 10, 2)):
                if lo > hi:
                    continue
                if ctx.check_allocation():
                    raise StopBlock(f"allocation busy: {ctx.check_allocation()}")
                d = out / f"{pid}__{coll}__{label.replace('/', '-')}__{lo}_{hi}"
                if d.exists():
                    continue
                env = build_env(ctx.uuids(gpus), label, d, energy_source=ctx.cfg.get("energy_source", "counter"),
                                extra={**ctx.placement_env.get(pid, {}), **(ctx.cfg.get("env_extra") or {})})
                r = run_process(build_cmd(coll, len(gpus), lo, hi, n, w, 0, ctx.numa(gpus)), env, d, 900,
                                watch=ctx.watch(gpus))
                if safety_problem(r) and not r["timed_out"]:
                    raise StopBlock(f"SAFETY_STOP calibration {pid} {coll} {label}: {safety_problem(r)}")
                if r["timed_out"]:
                    raise StopBlock(f"calibration {pid} {coll} {label} exceeded 900 s (hang)")
                regs, flags = parse_region_text((d / "region.jsonl").read_text() if (d / "region.jsonl").exists() else "")
                for reg in regs:
                    m = region_metrics(reg)
                    append_jsonl(out / "calibration.jsonl", {
                        "placement": pid, "coll": coll, "config": label, "size_bytes": m["size_bytes"],
                        "count": m["count"], "R": m["R"], "t_gpu_s": m.get("t_gpu_s"), "T_region_s": m.get("T_region_s"),
                        "flags": m["flags"], "parse_flags": flags, "rc": r["rc"], "dir": str(d.relative_to(REPO))})
                print(f"calibrate {pid} {coll} {label} {lo}-{hi}: rc={r['rc']} regions={len(regs)} {flags}", flush=True)


def cmd_freeze(setup_dir: Path, T_star: float, warm_s: float = 0.2, R_min: int = 10, w_min: int = 5):
    recs = [json.loads(ln) for ln in (setup_dir / "calibration/calibration.jsonl").read_text().splitlines() if ln.strip()]
    entries = {}
    for r in recs:
        if not calibration_usable(r):
            continue
        uid = unit_id(r["placement"], r["coll"], r["config"], r["size_bytes"])
        t = r["t_gpu_s"]
        entries[uid] = {"R": max(R_min, math.ceil(T_star / t)), "w": max(w_min, math.ceil(warm_s / t)),
                        "t_gpu_cal_s": t, "count": r["count"]}
    body = json.dumps(entries, sort_keys=True)
    frozen = {"frozen_utc": utc_now(), "T_star_s": T_star, "warmup_s": warm_s, "R_min": R_min, "w_min": w_min,
              "entries_sha256": sha256_bytes(body.encode()), "entries": entries}
    p = setup_dir / "calibration_frozen.json"
    if not entries:
        raise SystemExit("no successful, QC-clean calibration entries; refusing to freeze")
    if p.exists():
        raise SystemExit(f"{p} exists; frozen schedules are never overwritten")
    p.write_text(json.dumps(frozen, indent=1, sort_keys=True))
    print(f"froze {len(entries)} entries, sha256 {frozen['entries_sha256'][:16]}")


def calibration_usable(record: dict) -> bool:
    """A failed invocation or malformed region cannot seed a future schedule.

    Counter flags are deliberately not ignored: sampled-mode calibration needs
    its own declared timing-only acceptance protocol rather than silent reuse.
    """
    t = record.get("t_gpu_s")
    return (record.get("rc") == 0 and not record.get("flags") and not record.get("parse_flags")
            and isinstance(t, (int, float)) and math.isfinite(t) and t > 0)


# ------------------------------------------------------------------ idle baseline

def idle_baseline(ctx: Ctx, gpus: list[int], d: Path, duration: float, settle: float, p8_timeout: float) -> dict:
    import pynvml as N
    hs = [N.nvmlDeviceGetHandleByIndex(g) for g in gpus]
    t_wait = time.perf_counter()
    while True:
        ps = [N.nvmlDeviceGetPerformanceState(h) for h in hs]
        if all(p == 8 for p in ps) or time.perf_counter() - t_wait > p8_timeout:
            break
        time.sleep(1)
    reached = all(p == 8 for p in ps)
    t_p8 = time.perf_counter()
    time.sleep(settle)
    s = Sampler(gpus, period_s=ctx.cfg.get("sampler_period_s", 0.05)).start()
    time.sleep(0.5)
    sampled_mode = ctx.cfg.get("energy_source", "counter") == "sampled"
    t0 = time.perf_counter()
    if sampled_mode:   # v2.2: the declared fallback never calls the cumulative counter (e.g. Pascal boards)
        e0 = e1 = [None] * len(hs)
        time.sleep(duration)
    else:
        e0 = [N.nvmlDeviceGetTotalEnergyConsumption(h) for h in hs]
        time.sleep(duration)
        e1 = [N.nvmlDeviceGetTotalEnergyConsumption(h) for h in hs]
    t1 = time.perf_counter()
    time.sleep(0.5)
    s.stop()
    d.mkdir(parents=True, exist_ok=True)
    s.write(d / "telemetry.csv.gz")
    arr = s.array()
    per = {}
    for g, a, b in zip(gpus, e0, e1):
        w = window_summary(arr, g, t0, t1, s.period_s)
        ei = w.get("energy_int_inst_J")
        per[norm_pci(ctx.by_index[g]["pci"])] = {
            "gpu": g, "alias": ctx.by_index[g]["alias"],
            "counter_W": None if a is None else (b - a) / 1000 / (t1 - t0), "delta_mJ": None if a is None else b - a,
            "counter_status": "NOT_READ_SAMPLED_MODE" if a is None else "READ",
            "sampled_mean_W": w.get("power_mean_w"),
            "sampled_inst_W": (ei / (t1 - t0)) if ei is not None and np.isfinite(ei) else None,
            "temp_mean_c": w.get("temp_mean_c"), "pstates": w.get("pstates"), "n_samples": w.get("n_samples")}
    rec = {"utc": utc_now(), "reached_p8": reached, "p8_wait_s": t_p8 - t_wait, "settle_s": settle,
           "duration_s": t1 - t0, "energy_source": "sampled" if sampled_mode else "counter",
           "sampler": {"errors": len(s.errors), "truncated": s.truncated}, "per_gpu": per}
    (d / "idle.json").write_text(json.dumps(rec, indent=1))
    return rec


# ------------------------------------------------------------------ unit / block

def run_unit(ctx: Ctx, bctx: dict, unit: dict, attempt: int, parent: int | None) -> dict:
    pid, coll, label, size = unit["placement"], unit["coll"], unit["config"], unit["size"]
    gpus = ctx.placements[pid]
    uid = unit_key(unit)
    fr = bctx["frozen"]["entries"].get(uid)
    d = bctx["dir"] / "units" / f"{uid}__a{attempt}"
    if d.exists():
        raise SystemExit(f"{d} exists; attempts are append-only")
    rec = {"unit_id": uid, "attempt": attempt, "parent_attempt": parent, "campaign": ctx.cfg["campaign"],
           "mode": ctx.cfg["mode"], "block": bctx["block"], "seed": bctx["seed"], "planned_pos": unit["pos"],
           "exec_pos": bctx["exec_counter"], "placement": bctx["placement_meta"][pid], "coll": coll,
           "requested": label, "size_bytes": size, "length_s": unit.get("length_s"), "dir": str(d.relative_to(REPO))}
    if fr is None:
        rec.update(qc_status="NOT_RUN_NO_CALIBRATION", retryable=False)
        return rec
    R, w = fr["R"], fr["w"]
    rec.update(R=R, w=w, t_gpu_cal_s=fr["t_gpu_cal_s"])
    probs = ctx.check_allocation()
    if probs:
        rec.update(qc_status="FOREIGN_PROCESS", qc_reasons=probs, retryable=False, stop_block=True)
        return rec
    env = build_env(ctx.uuids(gpus), label, d, energy_source=ctx.cfg.get("energy_source", "counter"),
                    extra={**ctx.placement_env.get(pid, {}), **(ctx.cfg.get("env_extra") or {})})
    cmd = build_cmd(coll, len(gpus), size, size, R, w, 1, ctx.numa(gpus))
    to = ctx.cfg["timeouts"]
    if "add_s" in to:   # v2.2: factor x planned (warm-up + timed) + add_s
        tmo = to["factor"] * (R + w) * fr["t_gpu_cal_s"] + to["add_s"]
    else:
        tmo = max(to["min_s"], to["factor"] * (10 + (R + w) * fr["t_gpu_cal_s"]))
    sampler = Sampler(gpus, period_s=ctx.cfg.get("sampler_period_s", 0.05)).start()
    time.sleep(0.2)
    err0 = len(sampler.errors)
    proc = run_process(cmd, env, d, tmo, watch=ctx.watch(gpus, sampler))
    time.sleep(0.4)   # sampler tail: process exit already trails the loop end by >= 0.4 s (end wait, post window, data check)
    sampler.stop()
    sampler.write(d / "telemetry.csv.gz")
    regs, rflags = parse_region_text((d / "region.jsonl").read_text() if (d / "region.jsonl").exists() else "")
    rows = parse_stdout_rows((d / "stdout.txt").read_text(errors="replace"))
    metrics = None
    if len(regs) == 1:
        metrics = region_metrics(regs[0], expected_pci=[ctx.by_index[g]["pci"] for g in gpus], expected_R=R)
    elif len(regs) > 1:
        rflags.append("MULTIPLE_REGIONS")
    files, pflags = load_profiler_dir(d / "prof")
    ev = verify(files, COLL_FUNC[coll], regs[0]["count"] if regs else -1, len(gpus), *parse_cfg_label(label),
                min_expected=R, exact_expected=R + w + 1) if regs else {"status": "UNVERIFIED", "reasons": ["NO_REGION"]}
    arr = sampler.array()
    tel = {}
    clock_or, temp_max, pfrac, pfrac_known = 0, None, 0.0, True
    if metrics:
        for p in metrics["per_gpu"]:
            g = gpus[p["rank"]]
            s = window_summary(arr, g, p["t_start_ns"] * 1e-9, p["t_end_ns"] * 1e-9, sampler.period_s,
                               lag_s=ctx.cfg.get("power_lag_s", 0.0))
            s["counter_J"] = p["delta_J"]
            s["int_over_counter"] = (s["energy_int_J"] / p["delta_J"]) if p["delta_J"] else None
            if s.get("energy_int_lag_J") is not None and p["delta_J"]:
                s["int_lag_over_counter"] = s["energy_int_lag_J"] / p["delta_J"]
            tel[str(g)] = s
            clock_or |= s.get("clock_events_or", 0)
            temp_max = max(temp_max or -1, s.get("temp_max_c", -1))
            lim = (s.get("enforced_limit_mw") or [0])[0] / 1000.0
            if lim and s.get("power_mean_w") is not None:
                pfrac = max(pfrac, s["power_mean_w"] / lim)
            else:
                pfrac_known = False
    foreign = sorted({p for (_, _, pids) in sampler.proc_events if pids is not None for p in pids if p != proc["pid"]})
    polled = {g for (_, g, pids) in sampler.proc_events if pids is not None}
    plist_problems = sorted({f"PROCESS_LIST_UNKNOWN_GPU{g}" for (_, g, pids) in sampler.proc_events if pids is None}
                            | {f"PROCESS_LIST_MISSING_GPU{g}" for g in gpus if g not in polled})
    hits = error_hits(d)
    wrong = rows[0]["wrong"] if len(rows) == 1 else None
    src = ctx.cfg.get("energy_source", "counter")
    sampled = None
    if metrics and tel:
        vals = [t.get("energy_int_inst_J") if np.isfinite(t.get("energy_int_inst_J") or np.nan) else t.get("energy_int_lag_J")
                for t in tel.values()]
        sampled = sum(vals) if all(v is not None and np.isfinite(v) for v in vals) else None
    tprob = telemetry_problems({"errors": len(sampler.errors) - err0, "truncated": sampler.truncated},
                               tel if metrics else None, len(gpus), src)
    if tprob is not None and metrics:
        tprob = tprob + plist_problems
    qc = classify({"exit_code": proc["rc"], "timed_out": proc["timed_out"], "region": metrics, "wrong": wrong,
                   "region_parse_flags": rflags,
                   "sampled_energy_J": sampled, "telemetry_problems": tprob, "evidence_status": ev.get("status"),
                   "foreign_pids": foreign, "clock_events_or": clock_or & ~0x1, "temp_max_c": temp_max,
                   "power_frac_max": pfrac if pfrac_known else None,
                   "temp_stop_c": ctx.cfg["stop"]["temp_c"]}, ctx.cfg["T_min_s"], energy_source=src)
    safety = safety_problem(proc)
    if safety and not proc["timed_out"]:
        qc = {"qc_status": "SAFETY_STOP", "reasons": [safety], "retryable": False, "stop_block": True}
    stop = qc.get("stop_block", False)
    if proc["timed_out"]:
        stop = True
    if hits:
        stop = True
        if qc["qc_status"] == "OK":         # v2.2 release audit: errors in the logs never leave a unit OK
            qc = {"qc_status": "EXECUTION_ERROR", "reasons": [], "retryable": False, "stop_block": True}
        qc.setdefault("reasons", []).append("ERROR_IN_LOGS")
    rec.update(proc=proc, region=metrics, region_flags=rflags, stdout_rows=rows, evidence=ev, energy_source=src,
               sampled_energy_J=sampled,
               profiler_flags=pflags, telemetry=tel, sampler={"overruns": sampler.overruns, "errors": len(sampler.errors),
                                                               "truncated": sampler.truncated, "n": len(sampler.rows)},
               foreign_pids=foreign, error_hits=hits, qc_status=qc["qc_status"], qc_reasons=qc.get("reasons", []),
               qc_method=QC_METHOD, implementation_id="release-corrections-2026-10-11", telemetry_problems=tprob,
               retryable=qc.get("retryable", False) and not stop, stop_block=stop,
               nccl_env={k: v for k, v in env.items() if k.startswith(("NCCL_", "NCCLE_", "CUDA_"))})
    (d / "unit.json").write_text(redact(json.dumps(rec, indent=1, default=str)))
    return rec


def run_block(ctx: Ctx, setup_dir: Path) -> dict:
    cfg = ctx.cfg
    frozen = json.loads((REPO / cfg["frozen_file"]).read_text() if cfg.get("frozen_file")
                        else (setup_dir / "calibration_frozen.json").read_text())
    allowed = allowed_configs(ctx, setup_dir / "probe")
    blocks = sorted((ctx.out / "blocks").glob("block_*")) if (ctx.out / "blocks").exists() else []
    b = len(blocks) + 1
    if b > cfg.get("max_blocks", 99):
        raise SystemExit("campaign already has max_blocks blocks")
    bdir = ctx.out / "blocks" / f"block_{b:02d}"
    bdir.mkdir(parents=True)
    seed = f"{cfg['campaign']}-{cfg['seed']}-{b}"
    units = []
    for pid in ctx.placements:
        for coll in cfg["collectives"]:
            for label in cfg["configs"]:
                ok, why = allowed.get((pid, coll), {}).get(label, (False, "NO_PROBE"))
                for s in sizes(cfg):
                    if cfg.get("cells") and f"{pid}|{coll}|{s}" not in cfg["cells"]:
                        continue
                    if cfg.get("cell_configs") and label not in cfg["cell_configs"].get(f"{pid}|{coll}|{s}", [label]):
                        continue
                    for L in (cfg.get("region_lengths_s") or [None]):
                        units.append({"placement": pid, "coll": coll, "config": label, "size": s, "allowed": ok,
                                      "why": why, "length_s": L})
    rng = random.Random(seed)
    rng.shuffle(units)
    for i, u in enumerate(units):
        u["pos"] = i
    (bdir / "order_planned.json").write_text(json.dumps([unit_key(u) for u in units], indent=0))
    log = AttemptLog(bdir / "attempts.jsonl")
    bctx = {"dir": bdir, "block": b, "seed": seed, "frozen": frozen, "exec_counter": 0,
            "placement_meta": {p: ctx.placement_meta(p) for p in ctx.placements}}
    meta = {"campaign": cfg["campaign"], "mode": cfg["mode"], "block": b, "seed": seed, "utc_start": utc_now(),
            "config_sha256": ctx.cfg_hash, "frozen_sha256": frozen["entries_sha256"], "git": ctx.git,
            "versions": ctx.versions, "n_units": len(units)}
    summary = {"status": "RUNNING", **meta}
    xid = xid_event_test(ctx.alloc)
    es = xid.pop("event_set", None)
    try:
        pre = snapshot(ctx.alloc)
        (bdir / "preflight_pre.json").write_text(redact(json.dumps(pre, indent=1, default=str)))
        probs = allocation_problems(pre, ctx.alloc, cfg["stop"]["mem_used_mib_max"])
        if probs:
            raise StopBlock(f"allocation busy at block start: {probs}")
        if pre["host"]["disk_free_gb"] < cfg["stop"]["disk_floor_gb"]:
            raise StopBlock("disk below floor")
        hot = [g for g in pre["gpus"] if g["index"] in ctx.alloc and isinstance(g["temp_c"], int) and g["temp_c"] >= 45]
        if hot:
            raise StopBlock(f"stability rule: GPUs >= 45 C at start {[(g['index'], g['temp_c']) for g in hot]}")
        limits0 = {g["index"]: (g["enforced_limit_mw"], g["max_clk_gr_mhz"], g["app_clk_gr_mhz"]) for g in pre["gpus"]
                   if g["index"] in ctx.alloc}
        ib = cfg["idle"]
        summary["idle_pre"] = idle_baseline(ctx, ctx.alloc, bdir / "idle_pre", ib["duration_s"], ib["settle_s"],
                                            ib["p8_timeout_s"])
        consecutive_fail, consecutive_counter = 0, 0
        retry_queue = []
        queue = list(units)
        while queue or retry_queue:
            u = queue.pop(0) if queue else retry_queue.pop(0)   # retries only after all first attempts
            uid = unit_key(u)
            prev = log.attempts_for(uid)
            attempt = len(prev)
            if not u["allowed"]:
                log.append({"unit_id": uid, "attempt": 0, "qc_status": f"NOT_RUN_{u['why']}", "retryable": False,
                            "placement": u["placement"], "coll": u["coll"], "requested": u["config"],
                            "size_bytes": u["size"], "block": b})
                continue
            bctx["exec_counter"] += 1
            rec = run_unit(ctx, bctx, u, attempt, prev[-1]["attempt"] if prev else None)
            log.append(rec)
            print(f"[{time.strftime('%H:%M:%S')}] b{b} {bctx['exec_counter']}/{len(units)} {uid} a{attempt}: "
                  f"{rec['qc_status']} {rec.get('evidence', {}).get('status', '')} "
                  f"J/coll={(rec.get('region') or {}).get('J_per_coll')} T={(rec.get('region') or {}).get('T_region_s')}",
                  flush=True)
            if es is not None:
                x = poll_xid(es)
                if x:
                    raise StopBlock(f"Xid events {x}")
            if rec.get("stop_block"):
                raise StopBlock(f"stop condition at {uid}: {rec['qc_status']} {rec.get('qc_reasons')}")
            if rec["qc_status"] == "MEASUREMENT_INVALID":
                consecutive_counter += 1
                if consecutive_counter >= 2:
                    raise StopBlock("counter/measurement invalid in two consecutive units")
            else:
                consecutive_counter = 0
            consecutive_fail = consecutive_fail + 1 if rec["qc_status"] not in ("OK",) else 0
            if consecutive_fail >= 3:
                raise StopBlock("three consecutive failed units")
            if cfg.get("max_attempts", 2) > 1 and should_retry(log.attempts_for(uid)):   # v2.2 campaigns: 1
                retry_queue.append(u)
            post = light_health(ctx.alloc)
            for g in post["gpus"]:
                if g["index"] in limits0 and (g["enforced_limit_mw"], g["max_clk_gr_mhz"], g["app_clk_gr_mhz"]) != limits0[g["index"]]:
                    raise StopBlock(f"power limit / clock policy changed on GPU{g['index']}")
            if post["host"]["disk_free_gb"] < cfg["stop"]["disk_floor_gb"]:
                raise StopBlock("disk below floor")
        summary["idle_post"] = idle_baseline(ctx, ctx.alloc, bdir / "idle_post", ib["duration_s"], ib["settle_s"],
                                             ib["p8_timeout_s"])
        summary["status"] = "COMPLETE"
    except StopBlock as e:
        summary["status"] = "STOPPED"
        summary["stop_reason"] = str(e)
        print(f"BLOCK STOPPED: {e}", flush=True)
    except (ControllerSignal, KeyboardInterrupt, SafetyStop) as e:
        summary["status"] = "INTERRUPTED" if not isinstance(e, SafetyStop) else "STOPPED_SAFETY"
        summary["stop_reason"] = f"{type(e).__name__}: {e}"
        print(f"BLOCK {summary['status']}: {e}", flush=True)
        raise
    finally:
        try:
            post = snapshot(ctx.alloc)
            (bdir / "preflight_post.json").write_text(redact(json.dumps(post, indent=1, default=str)))
            pre_s = json.loads((bdir / "preflight_pre.json").read_text()) if (bdir / "preflight_pre.json").exists() else None
            if pre_s:
                c0 = {g["index"]: g.get("throttle_counters_ns") or {} for g in pre_s["gpus"]}
                delta = {}
                for g in post["gpus"]:
                    c1 = g.get("throttle_counters_ns") or {}
                    if g["index"] in ctx.alloc:
                        delta[g["index"]] = {k: c1[k] - c0[g["index"]].get(k, 0) for k in c1
                                             if isinstance(c1.get(k), int) and isinstance(c0[g["index"]].get(k), int)}
                summary["throttle_counter_delta_ns"] = delta
                # SW thermal / SW power cap counters tick at idle on GPUs 4/6 with driver 580 (Phase 4), so only
                # the HW counters are health events; SW activity during regions is caught by per-sample reasons.
                summary["health_events"] = [f"GPU{g}: {k} +{v / 1e9:.3f} s" for g, d in delta.items()
                                            for k, v in d.items() if k in ("hw_thermal", "hw_power_brake") and v > 0]
        except Exception as e:  # noqa: BLE001
            summary["post_snapshot_error"] = str(e)
        atts = log.records()
        finals = {}
        for a in atts:
            finals.setdefault(a["unit_id"], []).append(a)
        fin = [final_attempt(v) for v in finals.values()]
        summary.update(utc_end=utc_now(), xid=xid, n_attempts=len(atts),
                       final_status_counts={s: sum(1 for f in fin if f["qc_status"] == s) for s in
                                            sorted({f["qc_status"] for f in fin})},
                       evidence_counts={s: sum(1 for f in fin if (f.get("evidence") or {}).get("status") == s)
                                        for s in ("VERIFIED", "UNVERIFIED")})
        (bdir / "block_summary.json").write_text(redact(json.dumps(summary, indent=1, default=str)))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ncclenergy.v2.runner")
    ap.add_argument("cmd", choices=["preflight", "probe", "calibrate", "freeze", "block"])
    ap.add_argument("--campaign", type=Path)
    ap.add_argument("--setup", type=Path, default=REPO / "data/v2/raw/setup")
    ap.add_argument("--alloc", default=None, help="explicitly allocated NVML GPU indices (required for preflight)")
    ap.add_argument("--T-star", type=float)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--placements", default=None, help="comma-separated placement ids (probe only)")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        if not a.alloc:
            raise SystemExit("--alloc is required: an idle GPU is not an allocation")
        alloc = [int(x) for x in a.alloc.split(",")]
        snap = snapshot(alloc)
        snap["xid_events"] = {k: v for k, v in xid_event_test(alloc).items() if k != "event_set"}
        snap["allocation_problems"] = allocation_problems(snap, alloc)
        txt = redact(json.dumps(snap, indent=1, default=str))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(txt)
        print(txt if not a.out else f"wrote {a.out}")
        return
    if a.cmd == "freeze":
        cmd_freeze(a.setup, a.T_star)
        return
    cfg, h = load_campaign(a.campaign)
    with controller_signals():   # SIGTERM/SIGHUP outside an owned run also unwind through the cleanup paths
        _run_gpu_command(a, cfg, h)


def _run_gpu_command(a, cfg, h):
    if cfg.get("implementation_id") != "release-corrections-2026-10-11":
        raise SystemExit("Historical campaign configs are read-only. A newly approved campaign must declare "
                         "implementation_id: release-corrections-2026-10-11; do not append to old data.")
    if cfg.get("energy_source", "counter") == "sampled":
        raise SystemExit("Native sampled collection is disabled pending CUDA/Pascal validation of the new "
                         "v2.3-sampled-gate patch. CPU fallback tests do not validate the GPU executable.")
    ctx = Ctx(cfg, h)
    if a.cmd == "probe":
        try:
            cmd_probe(ctx, a.placements.split(",") if a.placements else None, a.timeout)
        except StopBlock as e:
            print(f"STOPPED: {e}", flush=True)
            sys.exit(2)
    elif a.cmd == "calibrate":
        try:
            cmd_calibrate(ctx, a.setup)
        except StopBlock as e:
            print(f"STOPPED: {e}", flush=True)
            sys.exit(2)
    elif a.cmd == "block":
        s = run_block(ctx, a.setup)
        print(json.dumps({k: s[k] for k in ("status", "block", "final_status_counts", "evidence_counts")
                          if k in s}, default=str))
        if s["status"] != "COMPLETE":
            sys.exit(2)


if __name__ == "__main__":
    main()
