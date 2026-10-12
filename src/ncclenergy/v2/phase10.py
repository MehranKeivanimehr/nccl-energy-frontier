"""Phase 10: tuner configuration check, tuner overhead, synthetic-workload A/B blocks.

    python -m ncclenergy.v2.phase10 verify-tuner --policy-dir data/v2/processed/confirmatory --tag verify_tuner_r2
    python -m ncclenergy.v2.phase10 overhead     --policy-dir data/v2/processed/confirmatory
    python -m ncclenergy.v2.phase10 workload     --policy-dir data/v2/processed/confirmatory --blocks 5

GPU commands need --alloc: the explicitly allocated NVML indices (an idle GPU is not an allocation); every
placement used must lie inside it. One owned experiment process tree at a time (procctl.OwnedProcess) with an
active watchdog; a safety stop ends the command, nothing is retried. `analyze` needs no GPU and no NVML.
Raw output: data/v2/raw/phase10/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from .common import COLL_FUNC, REPO, append_jsonl, redact, utc_now
from .evidence import load_profiler_dir
from .preflight import allocation_problems, light_health
from .region import parse_region_text, region_metrics
from .procctl import SafetyStop
from .runner import NCCL_LIB, PLUGIN, build_cmd, build_env, run_process, safety_problem
from .telemetry import Sampler, window_summary

OUT = REPO / "data/v2/raw/phase10"
TUNER = REPO / "native/nccl_tuner_example/build/libnccl-tuner-example.so"
PLACEMENTS = {"P4_MIX": [4, 5, 6, 7], "P2_NVL": [4, 5]}   # physical placements of the measured profiles
ALLOC: list[int] | None = None                             # set from --alloc; never assumed
TEMP_STOP_C, DISK_FLOOR_GB = 80, 20


def require_validated_gpu_protocol() -> None:
    """Historical harness is retained for analysis, not authorized new collection."""
    raise SystemExit("Phase 10 GPU collection is disabled in this pilot release: validate the real guarded "
                     "plugin and preregister numerical/training acceptance before enabling a new campaign. "
                     "Historical analysis remains available; no GPU was queried or launched.")


def set_allocation(alloc: list[int], needed: list[list[int]]) -> None:
    global ALLOC
    if len(alloc) > 6:
        raise SystemExit("allocation larger than 6 GPUs is not allowed")
    for gp in needed:
        if not set(gp) <= set(alloc):
            raise SystemExit(f"placement GPUs {gp} are not inside the declared allocation {alloc}")
    ALLOC = list(alloc)


def uuids(gpus):
    import pynvml as N
    N.nvmlInit()
    return [N.nvmlDeviceGetUUID(N.nvmlDeviceGetHandleByIndex(g)) for g in gpus]


def check_free():
    if not ALLOC:
        raise SystemExit("no allocation declared (--alloc)")
    p = allocation_problems(light_health(ALLOC), ALLOC)
    if p:
        raise SystemExit(f"STOP: allocation busy {p}")


def watch(gpus, sampler=None):
    return {"gpus": list(gpus), "sampler": sampler, "temp_stop_c": TEMP_STOP_C, "disk_floor_gb": DISK_FLOOR_GB}


def parse_rules(conf: Path) -> list[tuple]:
    rules = []
    for ln in conf.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        c, lo, hi, algo, proto, ch, nn, nr, *_ = ln.split(",")
        rules.append((c, int(lo), int(hi), algo.upper(), proto.upper(), int(nr)))
    return rules


def expected(rules, func_lc, nbytes, nranks):
    for c, lo, hi, a, p, nr in rules:
        if c == func_lc and lo <= nbytes <= hi and nr in (-1, nranks):
            return a, p
    return None


VERIFY_PLACEMENTS = {"P4_MIX": ([4, 5, 6, 7], {}), "P2_NVL": ([4, 5], {}), "P2_PCI_SHM": ([5, 6], {"NCCL_P2P_DISABLE": "1"})}
CFG_LABEL = {"tree": "Tree", "ring": "Ring", "simple": "Simple", "ll": "LL", "ll128": "LL128"}


def reference_actual() -> dict:
    """(placement, coll, size_bytes, config) -> executed ALGO/PROTO/nChannels in the measurement campaigns
    (exploratory cell table + confirmatory units). Used to check that the tuner reproduces measured executions."""
    import pandas as pd
    ref = {}
    for f in (REPO / "data/v2/processed/exploratory/cell_config.csv", REPO / "data/v2/processed/confirmatory/units.csv"):
        t = pd.read_csv(f)
        if "qc_status" in t:
            t = t[(t.qc_status == "OK") & (t.evidence == "VERIFIED")]
        for r in t.itertuples():
            ref.setdefault((r.placement, r.coll, int(r.size_bytes), r.config), set()).add(r.actual)
    return ref


def cmd_verify(policy_dir: Path, tag: str):
    """Run each collective over 1 KiB..1 GiB with the Energy-SLO tuner file. Per size compare the executed
    ALGO/PROTO/nChannels with (a) the rule, (b) the execution of the same requested configuration in the
    measurement campaigns (ruled sizes) or of NCCL automatic (sizes without a rule)."""
    require_validated_gpu_protocol()
    names = {"all_reduce": "allreduce", "all_gather": "allgather", "reduce_scatter": "reducescatter"}
    ref = reference_actual()
    root = OUT / tag
    res = []
    for pl, (gpus, pl_env) in VERIFY_PLACEMENTS.items():
        conf = policy_dir / f"nccl_tuner_{pl}.conf"
        rules = parse_rules(conf)
        R = len(gpus)
        for coll in COLL_FUNC:
            check_free()
            d = root / f"{pl}__{coll}"
            if d.exists():
                raise SystemExit(f"{d} exists")
            env = build_env(uuids(gpus), "auto", d, extra={**pl_env, "NCCL_TUNER_PLUGIN": str(TUNER),
                                                           "NCCL_TUNER_CONFIG_FILE": str(conf)})
            r = run_process(build_cmd(coll, R, 1024, 1 << 30, 2, 1, 1, 1), env, d, 900, watch=watch(gpus))
            if safety_problem(r):
                raise SafetyStop(safety_problem(r))
            files, _ = load_profiler_dir(d / "prof")
            log = (d / "nccl.log").read_text(errors="replace") if (d / "nccl.log").exists() else ""
            loaded = "Using tuner plugin" in log
            seen = {}
            for f in files:
                for c in f["colls"]:
                    nb = int(c["count"]) * {"ncclInt8": 1, "ncclFloat32": 4}.get(c["datatype"], 4)
                    size = nb if coll == "all_reduce" else nb * R      # profiler count is per rank for AG/RS
                    seen.setdefault(size, set()).add(f"{c['algo']}/{c['proto']}/{c['nChannels']}")
            for size, acts in sorted(seen.items()):
                exp = expected(rules, names[coll], size, R)      # tuner nBytes == nccl-tests size (total)
                cfg = "auto" if exp is None else f"{CFG_LABEL[exp[0].lower()]}/{CFG_LABEL[exp[1].lower()]}"
                refs = ref.get((pl, coll, size, cfg), set())
                rule_ok = None if exp is None else all(a.split("/")[:2] == list(exp) for a in acts)
                res.append({"placement": pl, "coll": coll, "size_bytes": size, "rule": None if exp is None else "/".join(exp),
                            "actual": "|".join(sorted(acts)), "reference_config": cfg, "reference_actual": "|".join(sorted(refs)),
                            "rule_match": rule_ok, "reference_match": (acts <= refs) if refs else None,
                            "rc": r["rc"], "tuner_loaded": loaded})
            print(f"verify {pl} {coll}: rc={r['rc']} tuner_loaded={loaded} sizes={len(seen)}", flush=True)
            if r["rc"] != 0:
                for x in res:
                    append_jsonl(root / "verify.jsonl", x)
                raise SystemExit(f"STOP: rc={r['rc']} in {d}")
    for x in res:
        append_jsonl(root / "verify.jsonl", x)
    ruled = [x for x in res if x["rule"] is not None]
    withref = [x for x in res if x["reference_match"] is not None]
    summ = {"sizes": len(res), "ruled_sizes": len(ruled), "rule_matched": sum(bool(x["rule_match"]) for x in ruled),
            "sizes_with_reference": len(withref), "reference_matched": sum(bool(x["reference_match"]) for x in withref),
            "rule_mismatches": [x for x in ruled if not x["rule_match"]],
            "reference_mismatches": [x for x in withref if not x["reference_match"]]}
    (root / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if not k.endswith("mismatches")}, indent=1))
    print("rule mismatches:", len(summ["rule_mismatches"]), "reference mismatches:", len(summ["reference_mismatches"]))
    for x in summ["rule_mismatches"][:10] + summ["reference_mismatches"][:20]:
        print(x)


def region_run(d, gpus, coll, size, R, w, extra, cfg_label="auto"):
    require_validated_gpu_protocol()
    check_free()
    env = build_env(uuids(gpus), cfg_label, d, extra=extra)
    s = Sampler(gpus, period_s=0.05).start()
    time.sleep(0.2)
    try:
        r = run_process(build_cmd(coll, len(gpus), size, size, R, w, 1, 1), env, d, 300, watch=watch(gpus, s))
    finally:
        time.sleep(0.4)
        s.stop()
    if safety_problem(r):
        raise SafetyStop(safety_problem(r))
    s.write(d / "telemetry.csv.gz")
    regs, _ = parse_region_text((d / "region.jsonl").read_text() if (d / "region.jsonl").exists() else "")
    m = region_metrics(regs[0]) if regs else {}
    files, _ = load_profiler_dir(d / "prof")
    acts = sorted({f"{c['algo']}/{c['proto']}/{c['nChannels']}" for f in files for c in f["colls"]
                   if int(c["count"]) * 4 == size})
    return {"rc": r["rc"], "t_gpu_s": m.get("t_gpu_s"), "J_per_coll": m.get("J_per_coll"), "flags": m.get("flags"),
            "T_region_s": m.get("T_region_s"), "actual": "|".join(acts)}


def cmd_overhead(policy_dir: Path, reps: int = 3):
    """Per-collective cost of the tuner plugin at latency-bound AllReduce sizes on P4_MIX (ABAB, interleaved).
    The comparison arm executes the same configuration without the plugin, so only the plugin path differs:
      4 KiB : no rule matches -> tuner arm runs NCCL automatic; comparison arm = automatic, no plugin.
      64 KiB: rule Tree/LL    -> comparison arm = NCCL_ALGO=Tree NCCL_PROTO=LL, no plugin.
    The executed ALGO/PROTO/nChannels of both arms is checked from the profiler records."""
    require_validated_gpu_protocol()
    conf = policy_dir / "nccl_tuner_P4_MIX.conf"
    rules = parse_rules(conf)
    root = OUT / "tuner_overhead"
    for rep in range(reps):
        for size, R in ((4096, 200000), (65536, 120000)):
            exp = expected(rules, "allreduce", size, 4)
            plain_cfg = "auto" if exp is None else f"{CFG_LABEL[exp[0].lower()]}/{CFG_LABEL[exp[1].lower()]}"
            arms = [("tuner", "auto", {"NCCL_TUNER_PLUGIN": str(TUNER), "NCCL_TUNER_CONFIG_FILE": str(conf)}),
                    ("plain", plain_cfg, {})]
            if rep % 2:
                arms.reverse()
            for arm, cfg, extra in arms:
                d = root / f"rep{rep}_{size}_{arm}"
                if d.exists():
                    raise SystemExit(f"{d} exists")
                o = region_run(d, PLACEMENTS["P4_MIX"], "all_reduce", size, R, 2000, extra, cfg)
                o.update(rep=rep, size=size, arm=arm, cfg_label=cfg, rule=None if exp is None else "/".join(exp))
                append_jsonl(root / "overhead.jsonl", o)
                print(f"overhead rep{rep} {size} {arm:5s} ({cfg}): t_gpu={o['t_gpu_s'] * 1e6:.3f} us "
                      f"J/coll={o['J_per_coll']:.3e} actual={o['actual']} rc={o['rc']}", flush=True)
                if o["rc"] != 0:
                    raise SystemExit(f"STOP rc={o['rc']}")


def torchrun(mode, policy, conf, gpus, out_json, d, steps, warmup, batch=128, meter="aligned", tuner=None):
    """One owned torchrun (all workers tracked by procctl, watchdog active). Returns rc, wall, bad clock events,
    max temperature and the run record. Safety problems raise procctl.SafetyStop (the campaign stops)."""
    require_validated_gpu_protocol()
    check_free()
    d.mkdir(parents=True, exist_ok=False)
    (d / "prof").mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "NCCLE_")) and k != "CUDA_VISIBLE_DEVICES"}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(uuids(gpus)),
               PYTHONPATH=str(REPO / "src"), NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,ENV",
               NCCL_DEBUG_FILE=str(d / "nccl.%h.%p.log"), NCCL_PROFILER_PLUGIN=str(PLUGIN), NCCLE_PROFILER_DIR=str(d / "prof"),
               OMP_NUM_THREADS="4")
    if conf is not None:
        env.update(NCCL_TUNER_PLUGIN=str(tuner or TUNER), NCCL_TUNER_CONFIG_FILE=str(conf))
    cmd = ["numactl", "--cpunodebind=1", "--membind=1", "nice", "-n", "10", sys.executable, "-m", "torch.distributed.run",
           "--standalone", f"--nproc_per_node={len(gpus)}", "-m", "ncclenergy.v2.workload", "--mode", mode,
           "--nvml-gpus", ",".join(map(str, gpus)), "--policy", policy, "--out", str(out_json),
           "--steps", str(steps), "--warmup", str(warmup), "--batch", str(batch), "--meter", meter]
    s = Sampler(gpus, period_s=0.05).start()
    t0 = time.time()
    try:
        r = run_process(cmd, env, d, 900, watch=watch(gpus, s))
    finally:
        s.stop()
        s.write(d / "telemetry.csv.gz")
    arr = s.array()
    ce = 0
    tmax = -1
    for g in gpus:
        w = window_summary(arr, g, arr[:, 0].min(), arr[:, 0].max(), 0.05) if len(arr) else {}
        ce |= w.get("clock_events_or", 0)
        tmax = max(tmax, w.get("temp_max_c", -1))
    rc = "TIMEOUT" if r["timed_out"] else r["rc"]
    if safety_problem(r) and not r["timed_out"]:
        raise SafetyStop(safety_problem(r), {"dir": str(d)})
    return rc, time.time() - t0, ce & ~0x1 & ~0x4, tmax, r


# Workload units (docs/v2/phase10_plan.md, amendment A1): (arm name, torch mode, batch, policy, measured steps)
WORKLOAD_UNITS = ([(f"ddp_{pol}", "ddp", 128, pol, 100) for pol in ("auto", "lat_opt", "energy_slo")]
                  + [(f"fsdp_{pol}", "fsdp", 128, pol, 100) for pol in ("auto", "lat_opt", "energy_slo")]
                  + [("ddp_nosync_auto", "ddp_nosync", 128, "auto", 1000)]
                  + [(f"ddpb2048_{pol}", "ddp", 2048, pol, 100) for pol in ("auto", "energy_slo")]
                  + [("ddpb2048_nosync_auto", "ddp_nosync", 2048, "auto", 100)])


def cmd_workload(policy_dir: Path, blocks: int, warmup: int = 10, smoke: bool = False):
    """Workload A/B blocks. --smoke: one run of each new configuration into workload_smoke/ (not analysed)."""
    require_validated_gpu_protocol()
    pl, gpus = "P4_MIX", PLACEMENTS["P4_MIX"]
    root = OUT / ("workload_smoke" if smoke else "workload")
    lat_conf = policy_dir / "nccl_tuner_P4_MIX_latency.conf"
    slo_conf = policy_dir / f"nccl_tuner_{pl}.conf"
    units = list(WORKLOAD_UNITS)
    if smoke:
        units, blocks = [u for u in units if u[0] in ("ddp_nosync_auto", "ddpb2048_energy_slo", "ddpb2048_nosync_auto")], 1
    for b in range(1, blocks + 1):
        order = list(units)
        if not smoke:
            random.Random(f"phase10-workload-{b}").shuffle(order)
        bdir = root / f"block_{b:02d}"
        if not smoke and bdir.exists():
            raise SystemExit(f"{bdir} exists")
        for k, (arm, mode, batch, pol, steps) in enumerate(order):
            conf = {"auto": None, "lat_opt": lat_conf, "energy_slo": slo_conf}[pol]
            d = bdir / f"{k:02d}_{arm}" if not smoke else root / "block_02" / f"{k:02d}_{arm}"
            out_json = d / "result.json"
            rc, wall, ce, tmax, _ = torchrun(mode, pol, conf, gpus, out_json, d, steps, warmup, batch)
            rec = {"block": b, "pos": k, "arm": arm, "mode": mode, "batch": batch, "policy": pol, "steps": steps,
                   "rc": rc, "wall_s": wall, "utc": utc_now(), "clock_events_bad": ce, "temp_max_c": tmax,
                   "dir": str(d.relative_to(REPO))}
            if out_json.exists():
                r = json.loads(out_json.read_text())
                rec.update({k2: r[k2] for k2 in ("step_time_s", "throughput_samples_s", "energy_per_step_J", "nccl")})
                rec["region_s"] = r["step_time_s"] * steps
            append_jsonl(root / "runs.jsonl", rec)
            print(f"[{time.strftime('%H:%M:%S')}] wl b{b} {arm:22s} rc={rc} step={rec.get('step_time_s')} "
                  f"E/step={rec.get('energy_per_step_J')} region={rec.get('region_s')}", flush=True)
            if rc != 0 or ce:
                raise SystemExit(f"STOP: rc={rc} clock_events={ce}")


def _rank_execution(f: dict, world: int) -> dict:
    out = {}
    for c in f["colls"]:
        nb = int(c["count"]) * {"ncclInt8": 1, "ncclUint8": 1, "ncclFloat16": 2, "ncclBfloat16": 2}.get(c["datatype"], 4)
        if c["func"] in ("AllGather", "ReduceScatter"):
            nb *= world
        k = f"{c['func']}:{nb}"
        cfg = f"{c['algo']}/{c['proto']}/{c['nChannels']}"
        out.setdefault(k, {}).setdefault(cfg, 0)
        out[k][cfg] += int(c.get("n", 1))
    return out


def run_execution(d: Path, world: int) -> dict:
    """Executed configurations of one torchrun, from rank 0's profiler records: (func, total bytes) -> {cfg: n}."""
    files, _ = load_profiler_dir(d / "prof")
    out = {}
    for f in files:
        if f["rank"] != 0:
            continue
        for k, v in _rank_execution(f, world).items():
            for cfg, n in v.items():
                out.setdefault(k, {}).setdefault(cfg, 0)
                out[k][cfg] += n
    return out


def all_rank_execution(d: Path, world: int) -> dict:
    """v2.2: per-rank executed configurations. consistent = every rank of every communicator recorded the same
    (func, bytes) -> configuration histogram and all `world` ranks are present for the main communicator."""
    files, _ = load_profiler_dir(d / "prof")
    by_comm: dict = {}
    for f in files:
        key = f.get("comm_hash") or f.get("commHash") or "?"
        by_comm.setdefault(key, {})
        ex = _rank_execution(f, world)
        prev = by_comm[key].get(f["rank"])
        if prev is not None:
            for k, v in ex.items():
                for cfg, n in v.items():
                    prev.setdefault(k, {}).setdefault(cfg, 0)
                    prev[k][cfg] += n
        else:
            by_comm[key][f["rank"]] = ex
    consistent, problems = True, []
    if not by_comm:
        consistent, problems = False, ["NO_PROFILER_FILES"]
    for key, ranks in by_comm.items():
        hist = list(ranks.values())
        if any(h != hist[0] for h in hist):
            consistent = False
            problems.append(f"RANKS_DIFFER:{key}")
    if by_comm and max(len(r) for r in by_comm.values()) != world:
        consistent = False
        problems.append(f"MAIN_COMM_RANKS={max(len(r) for r in by_comm.values())}!={world}")
    return {"consistent": consistent, "problems": problems, "n_comms": len(by_comm),
            "ranks_per_comm": {k: sorted(v) for k, v in by_comm.items()}}


def meter_qc(runs) -> tuple[dict, bool]:
    """{"b<block>:<arm>": flags} for runs whose v2.2 meter record is flagged or has no energy; and whether any run
    carries a v2.2 meter record (v2.1 records have no "flags")."""
    mq, v22 = {}, False
    for r in runs.itertuples():
        meter = json.loads((REPO / r.dir / "result.json").read_text()).get("meter") or {}
        if "flags" in meter:
            v22 = True
            bad = sorted(set(meter.get("flags") or []) | {f for v in (meter.get("per_gpu") or {}).values()
                                                          for f in (v.get("flags") or [])})
            e = meter.get("energy_J")
            import math
            if bad or e is None or not math.isfinite(e) or e <= 0:
                mq[f"b{r.block}:{r.arm}"] = bad or ["NO_ENERGY"]
    return mq, v22


def prospective_acceptance(runs, all_rank: dict) -> tuple[list[bool], dict]:
    """Fail-closed numerical sanity/execution gate; NOT training equivalence.

    Historical results without raw numerical records cannot pass. Every policy
    is compared with its within-block auto reference. The declared default
    tolerances need independent same-policy calibration before a training claim.
    """
    import math
    from .numerics import compare_numerics
    payloads = {(r.block, r.arm): json.loads((REPO / r.dir / "result.json").read_text())
                for r in runs.itertuples()}
    decisions, accepted = {}, []
    for r in runs.itertuples():
        key = (r.block, r.arm)
        reasons = []
        p = payloads[key]
        if not all_rank.get(key, {}).get("consistent", False):
            reasons.append("EXECUTION_INCONSISTENT_OR_MISSING")
        meter = p.get("meter") or {}
        e = meter.get("energy_J")
        if "flags" not in meter or meter.get("flags") or e is None or not math.isfinite(e) or e <= 0:
            reasons.append("METER_INVALID_OR_LEGACY")
        per_gpu = meter.get("per_gpu") or {}
        if len(per_gpu) != 4:
            reasons.append("PER_GPU_METER_INCOMPLETE")
        for v in per_gpu.values():
            eg = v.get("energy_J")
            if "flags" not in v or v.get("flags") or eg is None or not math.isfinite(eg) or eg <= 0:
                reasons.append("PER_GPU_METER_INVALID")
        for col in ("energy_per_step_J", "step_time_s"):
            value = getattr(r, col, None)
            if value is None or not math.isfinite(value) or value <= 0:
                reasons.append(f"METRIC_INVALID:{col}")
        family = next((f for f in ("ddpb2048", "ddp", "fsdp")
                       if r.arm in (f"{f}_auto", f"{f}_lat_opt", f"{f}_energy_slo")), None)
        baseline = payloads.get((r.block, f"{family}_auto")) if family else None
        comparison = compare_numerics(p, baseline or {})
        if not baseline or not comparison["comparable"] or comparison["within_tolerance"] is not True:
            reasons.append("NUMERICAL_SANITY_NOT_PASSED")
        ok = not reasons
        accepted.append(ok)
        decisions[f"b{r.block}:{r.arm}"] = {"accepted": ok, "reasons": sorted(set(reasons)),
                                             "numerical_comparison": comparison}
    return accepted, decisions


def cmd_analyze(out_dir: Path, figs_dir: Path | None = None, analysis_mode="historical-descriptive"):
    """Pre-specified analysis (docs/v2/phase10_plan.md) of the workload blocks and the tuner overhead runs.
    v2.2 adds all-rank execution consistency and, for runs that recorded them, numerical policy comparisons.
    Figures go to figs_dir (default: figures/v2/phase10 only for the default output dir, else out_dir/figures)."""
    import math
    import numpy as np
    import pandas as pd
    from .analysis import paired_test, tost
    from .common import read_jsonl
    if analysis_mode not in ("historical-descriptive", "strict-sanity"):
        raise ValueError(f"unknown analysis_mode: {analysis_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    runs = pd.DataFrame(read_jsonl(OUT / "workload" / "runs.jsonl")[0])
    runs = runs[runs.rc == 0]
    # v2.2 release audit: a run whose v2.2 meter record carries QC flags or no energy is excluded and listed. The
    # historical v2.1 meter records carry no flags at all: for them nothing is excluded and no key is added, so the
    # historical summary is unchanged (its meter QC is limited to what v2.1 recorded).
    mq, v22_meter = meter_qc(runs)
    if v22_meter:
        res["meter_qc"] = {"excluded_runs": mq}
        runs = runs[[f"b{b}:{a_}" not in mq for b, a_ in zip(runs.block, runs.arm)]]
    ex, allr = {}, {}
    for r in runs.itertuples():
        ex[(r.block, r.arm)] = run_execution(REPO / r.dir, 4)
        allr[(r.block, r.arm)] = all_rank_execution(REPO / r.dir, 4)
    if analysis_mode == "strict-sanity":
        mask, decisions = prospective_acceptance(runs, allr)
        res["acceptance"] = {"mode": analysis_mode, "training_equivalence_validated": False, "runs": decisions}
        runs = runs[mask]
        allowed_keys = set(zip(runs.block, runs.arm))
        ex = {k: v for k, v in ex.items() if k in allowed_keys}
        allr = {k: v for k, v in allr.items() if k in allowed_keys}
        if runs.empty:
            res.update(contrasts=[], arms=[], all_rank_execution={"runs": 0, "consistent": 0, "inconsistent": {}},
                       numerics_comparisons=[])
            (out_dir / "phase10_summary.json").write_text(json.dumps(res, indent=1, default=str))
            runs.to_csv(out_dir / "workload_runs.csv", index=False)
            print("No runs passed strict numerical-sanity acceptance; no savings estimates produced.")
            return
    res["all_rank_execution"] = {"runs": len(allr), "consistent": sum(v["consistent"] for v in allr.values()),
                                 "inconsistent": {f"b{b}:{arm}": v["problems"] for (b, arm), v in allr.items()
                                                  if not v["consistent"]}}
    from .numerics import compare_numerics
    comps = []
    for r in runs.itertuples():
        if not r.arm.endswith("_energy_slo"):
            continue
        fam = r.arm[: -len("_energy_slo")]
        base = runs[(runs.block == r.block) & (runs.arm == f"{fam}_auto")]
        if base.empty:
            continue
        ra = json.loads((REPO / r.dir / "result.json").read_text())
        rb = json.loads((REPO / base.iloc[0]["dir"] / "result.json").read_text())
        if "numerics" not in ra or "numerics" not in rb:
            comps.append({"block": r.block, "family": fam, "status": "NOT_RECORDED"})
            continue
        comps.append({"block": r.block, "family": fam, "status": "COMPARED", **compare_numerics(ra, rb)})
    res["numerics_comparisons"] = comps
    exrows = []
    for (b, arm), e in ex.items():
        for k, cfgs in e.items():
            func, nb = k.split(":")
            exrows.append({"block": b, "arm": arm, "func": func, "bytes": int(nb),
                           "executed": "|".join(sorted(cfgs)), "n": sum(cfgs.values())})
    exdf = pd.DataFrame(exrows)
    exdf.to_csv(out_dir / "workload_execution.csv", index=False)
    big = exdf[exdf.bytes >= (1 << 20)]
    res["execution_large_collectives"] = (big.groupby(["arm", "func", "bytes"]).executed
                                          .agg(lambda x: "|".join(sorted(set(x)))).reset_index().to_dict("records"))
    contrasts = []
    for fam in ("ddp", "fsdp", "ddpb2048"):
        for a_pol, b_pol in (("energy_slo", "auto"), ("lat_opt", "auto"), ("lat_opt", "energy_slo")):
            A = runs[runs.arm == f"{fam}_{a_pol}"].set_index("block")
            B = runs[runs.arm == f"{fam}_{b_pol}"].set_index("block")
            j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
            if len(j) < 2:
                continue
            row = {"family": fam, "contrast": f"{a_pol}/{b_pol}", "n_blocks": len(j)}
            for name, col in (("E", "energy_per_step_J"), ("T", "step_time_s")):
                d = np.log(j[f"{col}_a"] / j[f"{col}_b"]).to_numpy()
                t = paired_test(d, "two-sided")
                row.update({f"{name}_ratio": math.exp(t["mean"]), f"{name}_ci": [math.exp(t["ci_lo"]), math.exp(t["ci_hi"])],
                            f"{name}_p_t_two_sided": t.get("p"), f"{name}_wilcoxon_p": t.get("wilcoxon_p_two_sided"),
                            f"{name}_tost_3pct_p": tost(d).get("p"),
                            f"{name}_per_block": [round(math.exp(x), 5) for x in d]})
            contrasts.append(row)
    res["contrasts"] = contrasts
    desc = runs.groupby("arm").agg(n=("block", "size"), E_step_J=("energy_per_step_J", "mean"),
                                   E_step_sd=("energy_per_step_J", "std"), t_step_s=("step_time_s", "mean"),
                                   t_step_sd=("step_time_s", "std"), region_s=("region_s", "mean"),
                                   temp_max_c=("temp_max_c", "max")).reset_index()
    res["arms"] = desc.to_dict("records")
    # Historical no_sync arithmetic is retained for reproduction, not an
    # isolated communication-energy decomposition (power/overlap regimes differ).
    res["sync_energy_share"] = {}
    for fam, nos in (("ddp", "ddp_nosync_auto"), ("ddpb2048", "ddpb2048_nosync_auto")):
        n_ = runs[runs.arm == nos].set_index("block").energy_per_step_J
        a_ = runs[runs.arm == f"{fam}_auto"].set_index("block").energy_per_step_J
        jj = pd.concat([n_, a_], axis=1, keys=["nosync", "auto"]).dropna()
        if len(jj):
            share = ((jj["auto"] - jj.nosync) / jj["auto"]).to_numpy()
            res["sync_energy_share"][fam] = {"mean": float(share.mean()), "per_block": [round(x, 4) for x in share],
                                             "naive_predicted_E_ratio": float(1 - 0.346 * share.mean()),
                                             "note": "plan estimate 1 - (1 - 0.654) * share; not a measurement"}
    ov = pd.DataFrame(read_jsonl(OUT / "tuner_overhead" / "overhead.jsonl")[0]) if (OUT / "tuner_overhead" / "overhead.jsonl").exists() else pd.DataFrame()
    if len(ov):
        rows = []
        for size, g in ov.groupby("size"):
            A = g[g.arm == "tuner"].set_index("rep")
            B = g[g.arm == "plain"].set_index("rep")
            j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
            dT = np.log(j.t_gpu_s_a / j.t_gpu_s_b).to_numpy()
            dE = np.log(j.J_per_coll_a / j.J_per_coll_b).to_numpy()
            tT, tE = paired_test(dT, "two-sided"), paired_test(dE, "two-sided")
            rows.append({"size": int(size), "n_reps": len(j), "same_execution": bool((j.actual_a == j.actual_b).all()),
                         "executed": "|".join(sorted(set(j.actual_a) | set(j.actual_b))),
                         "t_tuner_us": float(j.t_gpu_s_a.mean() * 1e6), "t_plain_us": float(j.t_gpu_s_b.mean() * 1e6),
                         "T_ratio": math.exp(tT["mean"]), "T_ci": [math.exp(tT["ci_lo"]), math.exp(tT["ci_hi"])],
                         "E_ratio": math.exp(tE["mean"]), "E_ci": [math.exp(tE["ci_lo"]), math.exp(tE["ci_hi"])],
                         "T_per_rep": [round(math.exp(x), 5) for x in dT]})
        res["tuner_overhead"] = rows
    (out_dir / "phase10_summary.json").write_text(json.dumps(res, indent=1, default=str))
    runs.to_csv(out_dir / "workload_runs.csv", index=False)
    from . import plots_v2
    figs = figs_dir or (REPO / "figures/v2/phase10" if out_dir.resolve() == (REPO / "data/v2/processed/phase10").resolve()
                        else out_dir / "figures")
    figs.mkdir(parents=True, exist_ok=True)
    plots_v2.workload_blocks(runs, figs)
    print(json.dumps({k: v for k, v in res.items() if k != "execution_large_collectives"}, indent=1, default=str))
    print(json.dumps(res.get("execution_large_collectives"), indent=0, default=str)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify-tuner", "overhead", "workload", "analyze"])
    ap.add_argument("--policy-dir", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="verify_tuner", help="output subdirectory of data/v2/raw/phase10 (verify-tuner)")
    ap.add_argument("--alloc", default=None, help="explicitly allocated NVML GPU indices, e.g. 4,5,6,7 (GPU commands)")
    ap.add_argument("--out", type=Path, default=None, help="analyze: output directory (default data/v2/processed/phase10)")
    ap.add_argument("--analysis-mode", choices=["historical-descriptive", "strict-sanity"], default="historical-descriptive",
                    help="historical-descriptive preserves unverified pilot results; strict-sanity excludes missing/failed numerical evidence")
    a = ap.parse_args()
    a.policy_dir = a.policy_dir.resolve()   # processes run with cwd = their output dir; NCCL needs absolute paths
    if a.cmd == "analyze":
        if a.analysis_mode == "strict-sanity" and a.out is None:
            raise SystemExit("strict-sanity requires a new --out directory; do not replace historical outputs")
        cmd_analyze(a.out or (REPO / "data/v2/processed/phase10"), analysis_mode=a.analysis_mode)
        return
    require_validated_gpu_protocol()
    for f in ("nccl_tuner_P4_MIX.conf", "nccl_tuner_P2_NVL.conf", "nccl_tuner_P2_PCI_SHM.conf", "nccl_tuner_P4_MIX_latency.conf"):
        if not (a.policy_dir / f).is_file():
            raise SystemExit(f"missing {a.policy_dir / f}")
    if not a.alloc:
        raise SystemExit("--alloc is required for GPU commands: an idle GPU is not an allocation")
    needed = {"verify-tuner": [g for g, _ in VERIFY_PLACEMENTS.values()], "overhead": [PLACEMENTS["P4_MIX"]],
              "workload": [PLACEMENTS["P4_MIX"]]}[a.cmd]
    set_allocation([int(x) for x in a.alloc.split(",")], needed)
    import pynvml as N
    from .procctl import controller_signals
    N.nvmlInit()
    with controller_signals():
        {"verify-tuner": lambda: cmd_verify(a.policy_dir, a.tag), "overhead": lambda: cmd_overhead(a.policy_dir),
         "workload": lambda: cmd_workload(a.policy_dir, a.blocks, smoke=a.smoke)}[a.cmd]()


if __name__ == "__main__":
    main()
