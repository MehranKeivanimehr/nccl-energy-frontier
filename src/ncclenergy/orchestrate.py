"""Experiment orchestration.

Phases (all results are appended to <run_dir>/launches.jsonl and never
overwritten; a re-invocation resumes and skips launches already recorded):

  probe       one nccl-tests sweep per condition with NCCL_DEBUG_SUBSYS=TUNING
              and data checking (-c 1).  Records the algorithm/protocol NCCL
              actually selected for every message size, whether NCCL rejected
              the requested configuration, and whether results were correct.
  calibrate   short sweeps (-c 0) to estimate per-op latency so that the
              timed launches can be sized to >= timing.target_region_s.
  trials      T repeated trial blocks; in each, every supported
              (collective, gpu_set, algo, proto, size) unit is launched once,
              in an order shuffled with a seed derived from the config seed.
              Idle power is measured at the start/end of each trial and
              periodically in between.
  probe_end   the probe pass repeated after the trials (selection stability).
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import os
import random
import resource
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pynvml as N
import yaml

from . import metadata as md
from .energy import foreign_processes, integrate_power_j, measure_idle, read_counters_mj
from .nccl_parse import (assign_events, parse_nccl_log, parse_stdout, region_offsets,
                         selection_by_size, unsupported_reason)
from .nvml_sampler import NvmlSampler, handle, nvml_init

REPO = Path(__file__).resolve().parents[2]
NCCL_FUNC = {"all_reduce": "AllReduce", "all_gather": "AllGather", "reduce_scatter": "ReduceScatter"}
HAS_OP = {"all_reduce": True, "all_gather": False, "reduce_scatter": True}

_CHILD: subprocess.Popen | None = None


# Configuration and experiment plan

def load_config(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def sizes_from_config(cfg: dict) -> list[int]:
    s = cfg["sizes"]
    if "list" in s:
        return sorted(int(x) for x in s["list"])
    out, v = [], int(s["min_bytes"])
    while v <= int(s["max_bytes"]):
        out.append(v)
        v *= int(s.get("factor", 2))
    return out


def cond_id(c: dict) -> str:
    return f"{c['coll']}__{c['gpu_set']}__{c['algo']}__{c['proto']}"


def conditions(cfg: dict) -> list[dict]:
    sets = [g for g in cfg["gpu_sets"] if g.get("enabled", True)]
    out = []
    for coll, gs, algo, proto in itertools.product(cfg["collectives"], sets, cfg["algorithms"], cfg["protocols"]):
        out.append({"coll": coll, "gpu_set": gs["name"], "gpus": list(gs["gpus"]),
                    "numa_bind": gs.get("numa_bind"), "algo": algo, "proto": proto})
    return out


# Run-directory helpers

class RunLog:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.launches = run_dir / "launches.jsonl"
        self.events = run_dir / "events.jsonl"

    def append(self, path: Path, rec: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def event(self, kind: str, **kw) -> None:
        rec = {"t_wall": time.time(), "kind": kind, **kw}
        self.append(self.events, rec)
        msg = " ".join(f"{k}={v}" for k, v in kw.items() if k not in ("detail",))
        print(f"[{time.strftime('%H:%M:%S')}] {kind} {msg}", flush=True)

    def done_ids(self) -> set[str]:
        if not self.launches.exists():
            return set()
        ids = set()
        for line in self.launches.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("final", True) and r.get("status") != "interrupted":
                ids.add(r["launch_id"])
        return ids

    def records(self, phase: str | None = None) -> list[dict]:
        if not self.launches.exists():
            return []
        out = []
        for line in self.launches.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if phase is None or r.get("phase") == phase:
                out.append(r)
        return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# Single nccl-tests launch

def fresh_dir(d: Path) -> Path:
    """Return d, or d with a numeric suffix if d already holds files from an earlier attempt."""
    k = 1
    out = d
    while out.exists() and any(out.iterdir()):
        out = d.with_name(f"{d.name}__dup{k}")
        k += 1
    return out


def _preexec():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def build_env(cfg: dict, cond: dict, uuids: list[str], debug_subsys: str, log_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("NCCL_") and k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    lib = str(REPO / cfg["nccl_tests"]["nccl_lib_dir"])
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = debug_subsys
    env["NCCL_DEBUG_FILE"] = str(log_path)
    for k, v in (cfg["nccl_tests"].get("extra_env") or {}).items():
        env[k] = str(v)
    if cond["algo"] != "default":
        env["NCCL_ALGO"] = cond["algo"]
    if cond["proto"] != "default":
        env["NCCL_PROTO"] = cond["proto"]
    return env


def nccl_args(cfg: dict, cond: dict, minb: int, maxb: int, n: int, w: int, check: int) -> list[str]:
    nt = cfg["nccl_tests"]
    binary = str(REPO / nt["build_dir"] / f"{cond['coll']}_perf")
    a = [binary, "-t", str(len(cond["gpus"])), "-g", "1", "-b", str(minb), "-e", str(maxb), "-f", "2",
         "-n", str(n), "-w", str(w), "-c", str(check), "-d", nt.get("datatype", "float")]
    if HAS_OP[cond["coll"]]:
        a += ["-o", nt.get("op", "sum")]
    return a


def run_launch(cfg: dict, cond: dict, uuids: list[str], out_dir: Path, args: list[str], debug_subsys: str,
               timeout_s: float, sample: bool) -> dict:
    """Run one nccl-tests process; record stdout chunks with energy-counter reads."""
    global _CHILD
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "nccl.log"
    env = build_env(cfg, cond, uuids, debug_subsys, log_path)
    cmd = ["stdbuf", "-o0"] + args
    if cond.get("numa_bind") is not None:
        nb = str(cond["numa_bind"])
        cmd = ["numactl", f"--cpunodebind={nb}", f"--membind={nb}"] + cmd
    handles = [handle(g) for g in cond["gpus"]]
    sampler = NvmlSampler(gpus=cond["gpus"], period_s=cfg["sampler"]["period_s"],
                          proc_period_s=cfg["sampler"].get("proc_period_s", 1.0)) if sample else None
    if sampler:
        sampler.start()
        time.sleep(0.15)  # sampler coverage before the process starts
    stderr_f = open(out_dir / "stderr.txt", "wb")
    t_start_wall, t_start = time.time(), time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_f, env=env, cwd=out_dir,
                            preexec_fn=_preexec, start_new_session=True)
    _CHILD = proc
    fd = proc.stdout.fileno()
    buf = bytearray()
    chunks = []  # (end_offset, t_read, t_after_counters, counters_mJ)
    deadline = t_start + timeout_s
    end_reason = "eof"
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            end_reason = "timeout"
            break
        r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        if not r:
            continue
        data = os.read(fd, 1 << 16)
        t_read = time.perf_counter()
        ctr = read_counters_mj(handles) if sample else None
        t_ctr = time.perf_counter()
        if not data:
            break
        buf += data
        chunks.append((len(buf), t_read, t_ctr, ctr))
    if end_reason == "timeout":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    rc = proc.wait()
    _CHILD = None
    t_end = time.perf_counter()
    stderr_f.close()
    if sampler:
        time.sleep(0.15)
        sampler.stop()
        sampler.write_csv_gz(out_dir / "power.csv.gz")
    stdout_text = buf.decode(errors="replace")
    (out_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
    (out_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    stderr_text = (out_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    res = {
        "cmd": cmd, "env_nccl": {k: v for k, v in env.items() if k.startswith(("NCCL_", "CUDA_"))},
        "pid": proc.pid, "rc": rc, "end_reason": end_reason, "t_wall_start": t_start_wall,
        "wall_s": t_end - t_start, "stdout": parse_stdout(stdout_text), "nccl": parse_nccl_log(log_text),
        "stderr_tail": stderr_text[-2000:],
    }
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, "rb") as src, gzip.open(str(log_path) + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        log_path.unlink()
    if sampler:
        res["sampler"] = sampler.stats()
        res["regions"] = compute_regions(stdout_text, chunks, sampler, cond["gpus"], proc.pid)
        res["foreign_pids_during"] = sorted({p["pid"] for ev in sampler.proc_events for p in ev["procs"]
                                             if p.get("pid") not in (None, proc.pid)})
    return res


def compute_regions(stdout_text: str, chunks: list, sampler: NvmlSampler, gpus: list[int], child_pid: int) -> list[dict]:
    """Energy/telemetry for each out-of-place and in-place region in the stdout."""
    offs = region_offsets(stdout_text)
    ends = [c[0] for c in chunks]
    ev = assign_events(offs, ends)
    rows = np.array([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in sampler.rows]) \
        if sampler.rows else np.zeros((0, 9))
    out = []
    for o, e in zip(offs, ev):
        for placement, a, b in (("oop", "pre_chunk", "oop_chunk"), ("ip", "oop_chunk", "ip_chunk")):
            ia, ib = e[a], e[b]
            reg = {"size_bytes": o["size_bytes"], "placement": placement, "chunk_start": ia, "chunk_end": ib}
            if ia is None or ib is None or ia == ib:
                reg["valid_boundaries"] = False
                out.append(reg)
                continue
            t0, t1 = chunks[ia][1], chunks[ib][1]
            c0, c1 = chunks[ia][3], chunks[ib][3]
            reg.update(valid_boundaries=True, t0=t0, t1=t1, duration_s=t1 - t0, per_gpu={})
            for k, g in enumerate(gpus):
                m = rows[:, 1] == g if len(rows) else np.zeros(0, bool)
                t, p = rows[m, 0], rows[m, 2]
                inwin = (t >= t0) & (t <= t1)
                e_ctr = None if (c0 is None or c1 is None or c0[k] is None or c1[k] is None) \
                    else (c1[k] - c0[k]) / 1000.0
                sel = rows[m][inwin]
                reg["per_gpu"][str(g)] = {
                    "energy_counter_j": e_ctr,
                    "energy_integrated_j": integrate_power_j(t, p, t0, t1),
                    "energy_integrated_lag05_j": integrate_power_j(t, p, t0 + 0.5, t1 + 0.5),
                    "n_samples": int(inwin.sum()),
                    "power_sampled_mean_w": float(sel[:, 2].mean()) if len(sel) else None,
                    "util_gpu_mean": float(sel[:, 3].mean()) if len(sel) else None,
                    "util_mem_mean": float(sel[:, 4].mean()) if len(sel) else None,
                    "temp_mean_c": float(sel[:, 5].mean()) if len(sel) else None,
                    "temp_max_c": float(sel[:, 5].max()) if len(sel) else None,
                    "clk_gr_mean_mhz": float(sel[:, 6].mean()) if len(sel) else None,
                    "clk_mem_mean_mhz": float(sel[:, 7].mean()) if len(sel) else None,
                    "pstates": sorted({int(x) for x in sel[:, 8]}) if len(sel) else [],
                }
            out.append(reg)
    return out


# Experiment phases

class Orchestrator:
    def __init__(self, config_path: Path, run_name: str | None = None):
        self.cfg, self.cfg_hash = load_config(config_path)
        self.config_path = Path(config_path)
        name = run_name or self.cfg["name"]
        self.run_dir = REPO / self.cfg.get("output_root", "data/raw") / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = RunLog(self.run_dir)
        nvml_init()
        self.uuid = {}
        for i in range(N.nvmlDeviceGetCount()):
            u = N.nvmlDeviceGetUUID(N.nvmlDeviceGetHandleByIndex(i))
            self.uuid[i] = u.decode() if isinstance(u, bytes) else u
        self.sizes = sizes_from_config(self.cfg)
        self.conds = conditions(self.cfg)
        self._done = self.log.done_ids()
        self._launch_counter = 0
        self._save_config()

    # -- bookkeeping
    def _save_config(self):
        dst = self.run_dir / "config.yaml"
        if dst.exists():
            old = hashlib.sha256(dst.read_bytes()).hexdigest()
            if old != self.cfg_hash:
                raise SystemExit(f"{dst} exists with a different config hash; use a new --run-name")
        else:
            shutil.copy(self.config_path, dst)

    def uuids(self, cond):
        return [self.uuid[g] for g in cond["gpus"]]

    def gpu_set_status(self) -> dict:
        """Skip GPU sets with foreign processes at plan time (recorded, not silent)."""
        st = {}
        for gs in self.cfg["gpu_sets"]:
            if not gs.get("enabled", True):
                st[gs["name"]] = {"status": "disabled_in_config"}
                continue
            missing = [g for g in gs["gpus"] if g not in self.uuid]
            if missing:
                st[gs["name"]] = {"status": "gpus_not_present", "missing": missing}
                continue
            fp = foreign_processes(gs["gpus"])
            st[gs["name"]] = {"status": "foreign_process", "foreign": fp} if fp else {"status": "ok"}
        return st

    def _wait_free(self, gpus) -> tuple[bool, float, dict]:
        t0 = time.time()
        fp = foreign_processes(gpus)
        while fp and time.time() - t0 < self.cfg["foreign_process"]["wait_s"]:
            self.log.event("foreign_wait", gpus=gpus, foreign=fp)
            time.sleep(self.cfg["foreign_process"]["poll_s"])
            fp = foreign_processes(gpus)
        return (not fp), time.time() - t0, fp

    # -- metadata
    def capture_metadata(self, tag: str):
        d = self.run_dir / f"metadata_{tag}"
        if (d / "machine.json").exists():
            return
        self.log.event("metadata", tag=tag)
        md.collect(d, REPO, self.cfg)

    # -- idle
    def idle(self, tag: str, gpus: list[int]):
        iid = f"idle-{tag}"
        if iid in self._done:
            return
        icfg = self.cfg["idle"]
        rec = measure_idle(gpus, icfg["duration_s"], icfg["settle_timeout_s"], self.cfg["sampler"]["period_s"],
                           icfg.get("min_settle_s", 5.0))
        rec.update(launch_id=iid, phase="idle", final=True)
        self.log.append(self.log.launches, rec)
        self._done.add(iid)
        per = rec.get("per_gpu", {})
        self.log.event("idle", tag=tag, status=rec["status"],
                       watts={g: round(v["power_counter_w"], 2) for g, v in per.items() if v.get("power_counter_w")})

    # -- probe
    def probe(self, tag: str):
        for cond in self.conds:
            if cond["gpu_set"] not in self.active_sets:
                continue
            lid = f"{tag}__{cond_id(cond)}"
            if lid in self._done:
                continue
            remaining = list(self.sizes)
            per_size = {}
            attempt = 0
            launches = []
            res = None
            while remaining:
                minb, maxb = remaining[0], remaining[-1]
                ok, waited, fp = self._wait_free(cond["gpus"])
                if not ok:
                    for s in remaining:
                        per_size[str(s)] = {"status": "skipped_foreign_process"}
                    break
                d = fresh_dir(self.run_dir / tag / cond_id(cond) / f"attempt{attempt}")
                args = nccl_args(self.cfg, cond, minb, maxb, n=1, w=0, check=1)
                res = run_launch(self.cfg, cond, self.uuids(cond), d, args, "INIT,ENV,GRAPH,TUNING",
                                 timeout_s=self.cfg["timing"].get("probe_timeout_s", 600), sample=False)
                launches.append({"dir": str(d.relative_to(self.run_dir)), "rc": res["rc"],
                                 "end_reason": res["end_reason"], "range": [minb, maxb]})
                sel = selection_by_size(res["nccl"]["tuning"], NCCL_FUNC[cond["coll"]])
                rows = {r["size_bytes"]: r for r in res["stdout"]["rows"]}
                wsize = 4 if self.cfg["nccl_tests"].get("datatype", "float") in ("float", "int32", "int") else None
                for s in remaining:
                    if s not in rows:
                        continue
                    r = rows[s]
                    nbytes = r["count"] * wsize if wsize else None
                    picks = sel.get(nbytes, [])
                    wrong = [x for x in (r.get("oop_wrong"), r.get("ip_wrong")) if x is not None]
                    per_size[str(s)] = {
                        "status": "ok" if wrong and max(wrong) == 0 else ("incorrect_results" if wrong else "no_check"),
                        "selected": [{"algo": a, "proto": p, "ch_lo": lo, "ch_hi": hi} for a, p, lo, hi in picks],
                        "nccl_bytes": nbytes, "oop_wrong": r.get("oop_wrong"), "ip_wrong": r.get("ip_wrong"),
                        "probe_oop_time_us": r.get("oop_time_us"),
                    }
                done_sizes = [s for s in remaining if s in rows]
                remaining = [s for s in remaining if s not in rows]
                if remaining:
                    # first size without a complete row failed; record reason, continue with the next size
                    s_fail = remaining[0]
                    reason = unsupported_reason(res["nccl"], res["stdout"], res["stderr_tail"])
                    per_size[str(s_fail)] = {
                        "status": "unsupported" if reason else ("timeout" if res["end_reason"] == "timeout" else "error"),
                        "reason": reason or (res["nccl"]["warnings"][-3:] if res["nccl"]["warnings"] else
                                             res["stdout"]["failures"] or res["stderr_tail"][-500:]),
                        "rc": res["rc"],
                    }
                    remaining = remaining[1:]
                attempt += 1
                if not done_sizes and attempt > len(self.sizes) + 2:
                    break
            statuses = {v["status"] for v in per_size.values()}
            rec = {"launch_id": lid, "phase": tag, "final": True, "cond": cond, "per_size": per_size,
                   "launches": launches, "status": "ok" if statuses == {"ok"} else "+".join(sorted(statuses))}
            if res is not None:
                rec.update(nccl_env=res["nccl"]["env"], transports=res["nccl"]["transports"],
                           channels=res["nccl"]["channels"], enabled_matrix=res["nccl"]["enabled_matrix"],
                           nccl_version=res["nccl"]["nccl_version"], ranks=res["nccl"]["ranks"],
                           devices=res["stdout"]["devices"], nccl_tests_version=res["stdout"]["version"])
            self.log.append(self.log.launches, rec)
            self._done.add(lid)
            self.log.event("probe", cond=cond_id(cond), status=rec["status"])

    def probe_table(self, tag: str = "probe") -> dict:
        return {cond_id(r["cond"]): r for r in self.log.records(tag)}

    # -- calibration
    def calibrate(self):
        ccfg = self.cfg["calibration"]
        probes = self.probe_table()
        for cond in self.conds:
            if cond["gpu_set"] not in self.active_sets:
                continue
            cid = cond_id(cond)
            lid = f"calibrate__{cid}"
            if lid in self._done:
                continue
            ok_sizes = [s for s in self.sizes
                        if probes.get(cid, {}).get("per_size", {}).get(str(s), {}).get("status") == "ok"]
            est = {}
            launches = []
            for lo_hi, n, w in (((None, ccfg["small_max_bytes"]), ccfg["small_iters"], ccfg["small_warmup"]),
                                ((ccfg["small_max_bytes"] * 2, None), ccfg["large_iters"], ccfg["large_warmup"])):
                group = [s for s in ok_sizes if (lo_hi[0] is None or s >= lo_hi[0]) and (lo_hi[1] is None or s <= lo_hi[1])]
                attempt = 0
                while group:
                    ok, _, _ = self._wait_free(cond["gpus"])
                    if not ok:
                        break
                    d = fresh_dir(self.run_dir / "calibrate" / cid / f"{group[0]}_{group[-1]}_a{attempt}")
                    args = nccl_args(self.cfg, cond, group[0], group[-1], n=n, w=w, check=0)
                    res = run_launch(self.cfg, cond, self.uuids(cond), d, args, "INIT,ENV",
                                     timeout_s=self.cfg["timing"].get("probe_timeout_s", 600), sample=False)
                    launches.append({"dir": str(d.relative_to(self.run_dir)), "rc": res["rc"]})
                    rows = {r["size_bytes"]: r for r in res["stdout"]["rows"]}
                    for s in group:
                        if s in rows:
                            est[str(s)] = {"oop_time_us": rows[s]["oop_time_us"], "ip_time_us": rows[s]["ip_time_us"],
                                           "iters": n, "warmup": w}
                    nxt = [s for s in group if s not in rows]
                    if nxt:
                        est[str(nxt[0])] = {"error": True, "rc": res["rc"]}
                        nxt = nxt[1:]
                    group = nxt
                    attempt += 1
            self._cal_cache = None
            rec = {"launch_id": lid, "phase": "calibrate", "final": True, "cond": cond, "estimates": est,
                   "launches": launches}
            self.log.append(self.log.launches, rec)
            self._done.add(lid)
            self.log.event("calibrate", cond=cid, n_sizes=len(est))

    def iteration_plan(self, cid: str, size: int) -> tuple[int, int] | None:
        if getattr(self, "_cal_cache", None) is None:
            self._cal_cache = {cond_id(r["cond"]): r for r in self.log.records("calibrate")}
        cal = self._cal_cache.get(cid)
        if not cal:
            return None
        e = cal["estimates"].get(str(size))
        if not e or e.get("error"):
            return None
        t = max(e["oop_time_us"], e["ip_time_us"]) * 1e-6
        tc = self.cfg["timing"]
        n = max(tc.get("min_iters", 5), math.ceil(tc["target_region_s"] / t))
        w = max(tc.get("min_warmup_iters", 1), math.ceil(tc["warmup_s"] / t))
        return n, w

    # -- timed trials
    def units(self) -> list[dict]:
        probes = self.probe_table()
        out = []
        for cond in self.conds:
            if cond["gpu_set"] not in self.active_sets:
                continue
            for s in self.sizes:
                p = probes.get(cond_id(cond), {}).get("per_size", {}).get(str(s), {"status": "not_probed"})
                out.append({"cond": cond, "size": s, "probe_status": p["status"]})
        return out

    def trial(self, k: int):
        tc = self.cfg["timing"]
        all_gpus = sorted({g for gs in self.cfg["gpu_sets"] if gs["name"] in self.active_sets for g in gs["gpus"]})
        units = self.units()
        rng = random.Random(f"{self.cfg['seed']}-trial-{k}")
        rng.shuffle(units)
        order_path = self.run_dir / f"order_trial{k}.json"
        if not order_path.exists():
            order_path.write_text(json.dumps([f"{cond_id(u['cond'])}__{u['size']}" for u in units], indent=0),
                                  encoding="utf-8")
        self.idle(f"t{k}-start", all_gpus)
        every = self.cfg["idle"].get("every_n_launches", 0)
        n_since_idle = 0
        deferred = []
        for i, u in enumerate(units):
            lid = f"t{k}__{cond_id(u['cond'])}__{u['size']}"
            if lid in self._done:
                continue
            if u["probe_status"] != "ok":
                rec = {"launch_id": lid, "phase": "trial", "trial": k, "final": True, "cond": u["cond"],
                       "size_bytes": u["size"], "status": f"not_run_probe_{u['probe_status']}"}
                self.log.append(self.log.launches, rec)
                self._done.add(lid)
                continue
            if every and n_since_idle >= every:
                self.idle(f"t{k}-mid{i}", all_gpus)
                n_since_idle = 0
            st = self._timed(k, u, lid, i, len(units))
            n_since_idle += 1
            if st == "skipped_foreign_process":
                deferred.append(u)
        for u in deferred:  # one more attempt at the end of the trial
            lid = f"t{k}__{cond_id(u['cond'])}__{u['size']}"
            if lid not in self._done:
                self._timed(k, u, lid, -1, len(units), last_chance=True)
        self.idle(f"t{k}-end", all_gpus)

    def _timed(self, k, u, lid, i, total, last_chance=False) -> str:
        cond, size = u["cond"], u["size"]
        cid = cond_id(cond)
        plan = self.iteration_plan(cid, size)
        if plan is None:
            rec = {"launch_id": lid, "phase": "trial", "trial": k, "final": True, "cond": cond,
                   "size_bytes": size, "status": "not_run_no_calibration"}
            self.log.append(self.log.launches, rec)
            self._done.add(lid)
            return rec["status"]
        n, w = plan
        tc = self.cfg["timing"]
        for attempt in range(tc.get("max_retries_short", 1) + 1):
            ok, waited, fp = self._wait_free(cond["gpus"])
            if not ok:
                if not last_chance:
                    self.log.event("defer_foreign", launch=lid, foreign=fp)
                    return "skipped_foreign_process"
                rec = {"launch_id": lid, "phase": "trial", "trial": k, "final": True, "cond": cond,
                       "size_bytes": size, "status": "skipped_foreign_process", "foreign": fp}
                self.log.append(self.log.launches, rec)
                self._done.add(lid)
                return rec["status"]
            est_s = (n + w) * 2 * (tc["target_region_s"] / max(n, 1))
            timeout = max(tc.get("launch_timeout_min_s", 120), tc.get("launch_timeout_factor", 4) * est_s)
            d = fresh_dir(self.run_dir / "launches" / f"{lid}__a{attempt}")
            args = nccl_args(self.cfg, cond, size, size, n=n, w=w, check=0)
            others = {g: foreign_processes([g]).get(g, []) for g in self.uuid if g not in cond["gpus"]}
            res = run_launch(self.cfg, cond, self.uuids(cond), d, args, "INIT,ENV", timeout_s=timeout, sample=True)
            status, flags, timed = classify(res, n, w, size, tc)
            final = status not in RETRY_STATUSES or attempt == tc.get("max_retries_short", 1)
            rec = {"launch_id": lid, "phase": "trial", "trial": k, "attempt": attempt, "final": final,
                   "cond": cond, "size_bytes": size, "iters": n, "warmup": w, "status": status, "flags": flags,
                   "timed_s": timed, "dir": str(d.relative_to(self.run_dir)), "rc": res["rc"],
                   "end_reason": res["end_reason"], "wall_s": res["wall_s"], "t_wall_start": res["t_wall_start"],
                   "rows": res["stdout"]["rows"], "regions": res.get("regions"), "sampler": res.get("sampler"),
                   "foreign_pids_during": res.get("foreign_pids_during"), "nccl_env": res["nccl"]["env"],
                   "transports": res["nccl"]["transports"], "nccl_warnings": res["nccl"]["warnings"],
                   "ranks": res["nccl"]["ranks"], "init_total_s": res["nccl"]["init_total_s"],
                   "other_gpu_processes": {str(g): v for g, v in others.items() if v},
                   "load_avg": os.getloadavg(), "foreign_wait_s": waited}
            (d / "result.json").write_text(json.dumps(rec, indent=1, default=_json_default), encoding="utf-8")
            self.log.append(self.log.launches, rec)
            if final:
                self._done.add(lid)
                e_op = None
                regs = [r for r in (res.get("regions") or []) if r.get("placement") == "oop" and r.get("valid_boundaries")]
                if regs:
                    ej = [v["energy_counter_j"] for v in regs[0]["per_gpu"].values()]
                    if all(x is not None for x in ej):
                        e_op = sum(ej) / (n + w)
                self._launch_counter += 1
                self.log.event("launch", trial=k, i=f"{i + 1}/{total}", cond=cid, size=size, status=status,
                               timed_s=round(min(timed.values()), 2) if timed else None,
                               mJ_per_op=None if e_op is None else round(e_op * 1e3, 4))
                return status
            self.log.event("retry", launch=lid, status=status, attempt=attempt)
            if status == "short_region":  # scale iterations so the timed part reaches the target
                shortest = min(timed.values())
                n = math.ceil(n * tc["target_region_s"] / shortest * 1.05)
        return status

    # -- driver
    def run(self, phases: list[str]):
        signal.signal(signal.SIGTERM, _terminate)
        signal.signal(signal.SIGINT, _terminate)
        status_path = self.run_dir / "gpu_set_status.json"
        if status_path.exists():  # resumed run: keep the matrix decided at the first start
            set_status = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            set_status = self.gpu_set_status()
            status_path.write_text(json.dumps(set_status, indent=1, default=str), encoding="utf-8")
        self.active_sets = {k for k, v in set_status.items() if v["status"] == "ok"}
        self.log.event("start", run_dir=str(self.run_dir), phases=phases, gpu_sets=set_status,
                       config_sha256=self.cfg_hash)
        self.capture_metadata("start")
        if "probe" in phases:
            self.probe("probe")
        if "calibrate" in phases:
            self.calibrate()
        if "trials" in phases:
            for k in range(1, self.cfg["trials"] + 1):
                self.trial(k)
        if "probe_end" in phases:
            self.probe("probe_end")
            self.capture_metadata("end")
        self.log.event("finished", run_dir=str(self.run_dir))


RETRY_STATUSES = {"short_region", "error_rc", "timeout", "region_boundary_error", "parse_error"}


def classify(res: dict, n: int, w: int, size: int, tc: dict) -> tuple[str, list[str], dict]:
    flags = []
    timed = {}
    if res["end_reason"] == "timeout":
        return "timeout", flags, timed
    if res["rc"] != 0:
        return "error_rc", flags, timed
    rows = [r for r in res["stdout"]["rows"] if r["size_bytes"] == size]
    if len(rows) != 1:
        return "parse_error", flags, timed
    r = rows[0]
    timed = {"oop": n * r["oop_time_us"] * 1e-6, "ip": n * r["ip_time_us"] * 1e-6}
    regs = [g for g in (res.get("regions") or []) if g["size_bytes"] == size]
    if len(regs) != 2 or not all(g.get("valid_boundaries") for g in regs):
        return "region_boundary_error", flags, timed
    for g in regs:
        if any(v["energy_counter_j"] is None for v in g["per_gpu"].values()):
            flags.append(f"no_counter_{g['placement']}")
        wall_per_op = g["duration_s"] / (n + w)
        t_op = r[f"{g['placement']}_time_us"] * 1e-6
        if abs(wall_per_op / t_op - 1) > tc.get("wall_vs_reported_tol", 0.10):
            flags.append(f"wall_vs_reported_{g['placement']}")
        if any(v.get("n_samples", 0) < 5 for v in g["per_gpu"].values()):
            flags.append(f"few_samples_{g['placement']}")
    if res.get("foreign_pids_during"):
        flags.append("foreign_process_during")
    if res["sampler"]["overruns"] > 0.05 * max(1, sum(v["n"] for v in res["sampler"]["per_gpu"].values())):
        flags.append("sampler_overruns")
    if min(timed.values()) < tc["min_region_s"]:
        return "short_region", flags, timed
    return ("ok_flagged" if flags else "ok"), flags, timed


def _terminate(signum, frame):
    if _CHILD is not None and _CHILD.poll() is None:
        try:
            os.killpg(_CHILD.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print("terminated; launches already recorded are kept, re-run the same command to resume", flush=True)
    sys.exit(130)
