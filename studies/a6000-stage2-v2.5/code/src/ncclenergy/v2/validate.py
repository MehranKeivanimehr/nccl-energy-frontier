"""Phase 4 validation experiments (small, GPUs 4-7 only, one process at a time).

    python -m ncclenergy.v2.validate --exp all

Each run writes data/v2/raw/validation/<exp>/<run>/ and one line to
data/v2/raw/validation/validation.jsonl. Nothing here is trial data.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import pynvml as N

from .common import COLL_FUNC, REPO, append_jsonl, redact, utc_now
from .evidence import load_profiler_dir, verify
from .preflight import allocation_problems, light_health, snapshot
from .region import parse_region_text, region_metrics
from .runner import build_cmd, build_env, parse_cfg_label, parse_stdout_rows, run_process
from .selection import parse_tuning_table
from .telemetry import Sampler, window_summary

OUT = REPO / "data/v2/raw/validation"
P2 = [4, 5]
P4 = [4, 5, 6, 7]
NUMA = 1


class Busy(Exception):
    pass


def require_validated_protocol():
    raise SystemExit("Historical Phase 4 GPU validation is archived and disabled: this path lacks explicit "
                     "allocation/watchdog protection. Design a new controlled validation campaign first.")


def uuids(gpus):
    require_validated_protocol()
    N.nvmlInit()
    return [N.nvmlDeviceGetUUID(N.nvmlDeviceGetHandleByIndex(g)) for g in gpus]


def pci(g):
    return N.nvmlDeviceGetPciInfo(N.nvmlDeviceGetHandleByIndex(g)).busId


def check_free():
    probs = allocation_problems(light_health(P4), P4)
    if probs:
        raise Busy(str(probs))


class Reader(threading.Thread):
    """Extra NVML reads on one GPU during a run (dose test). kind: 'counter' | 'instant' | None."""

    def __init__(self, gpu, kind, hz, delay_s, duration_s):
        require_validated_protocol()
        super().__init__(daemon=True)
        self.h = N.nvmlDeviceGetHandleByIndex(gpu)
        self.kind, self.hz, self.delay, self.dur = kind, hz, delay_s, duration_s
        self.samples = []

    def run(self):
        time.sleep(self.delay)
        t_end = time.perf_counter() + self.dur
        k = 0
        t0 = time.perf_counter()
        while time.perf_counter() < t_end:
            t = time.perf_counter()
            if self.kind == "counter":
                v = N.nvmlDeviceGetTotalEnergyConsumption(self.h)
            else:
                f = N.nvmlDeviceGetFieldValues(self.h, [N.NVML_FI_DEV_POWER_INSTANT])[0]
                v = f.value.uiVal if f.nvmlReturn == 0 else -1
            self.samples.append((t, v))
            k += 1
            dl = t0 + k / self.hz - time.perf_counter()
            if dl > 0:
                time.sleep(dl)


def run(exp, name, gpus, coll, label, size, R, w, check=1, plugin=True, subsys="INIT,ENV", reader=None, sizes=None):
    require_validated_protocol()
    check_free()
    d = OUT / exp / name
    if d.exists():
        raise SystemExit(f"{d} exists (append-only)")
    env = build_env(uuids(gpus), label, d, debug_subsys=subsys, plugin=plugin)
    lo, hi = sizes if sizes else (size, size)
    cmd = build_cmd(coll, len(gpus), lo, hi, R, w, check, NUMA)
    s = Sampler(gpus, period_s=0.05).start()
    time.sleep(0.2)
    if reader:
        reader.start()
    p = run_process(cmd, env, d, 600)
    if reader:
        reader.join(timeout=30)
    time.sleep(1.5)   # sampler tail so lag-shifted integration windows stay covered
    s.stop()
    s.write(d / "telemetry.csv.gz")
    regs, flags = parse_region_text((d / "region.jsonl").read_text() if (d / "region.jsonl").exists() else "")
    files, pflags = load_profiler_dir(d / "prof")
    arr = s.array()
    out = {"exp": exp, "name": name, "utc": utc_now(), "gpus": gpus, "coll": coll, "config": label, "R": R, "w": w,
           "plugin": plugin, "rc": p["rc"], "timed_out": p["timed_out"], "wall_s": p["wall_s"], "region_flags": flags,
           "stdout_rows": parse_stdout_rows((d / "stdout.txt").read_text(errors="replace")), "regions": []}
    for reg in regs:
        m = region_metrics(reg, expected_pci=[pci(g) for g in gpus], expected_R=R)
        tel = {}
        for pg in m["per_gpu"]:
            g = gpus[pg["rank"]]
            ws = {lag: window_summary(arr, g, pg["t_start_ns"] * 1e-9 + lag, pg["t_end_ns"] * 1e-9 + lag, 0.05)
                  for lag in (0.0, 0.25, 0.5, 0.75, 1.0)}
            tel[str(g)] = {"counter_J": pg["delta_J"], "int_J_by_lag": {str(k): v["energy_int_J"] for k, v in ws.items()},
                           "int_inst_J": ws[0.0].get("energy_int_inst_J"), "tail": {k: pg.get(k) for k in
                           ("aligned_delta_J", "tail_s", "p_ctx_W", "tail_J")},
                           "temp_max_c": ws[0.0].get("temp_max_c"), "clk_gr_mean": ws[0.0].get("clk_gr_mean"),
                           "clock_events_or": ws[0.0].get("clock_events_or"), "n_samples": ws[0.0].get("n_samples"),
                           "gaps": ws[0.0].get("gaps"), "power_mean_w": ws[0.0].get("power_mean_w")}
        ev = verify(files, COLL_FUNC[coll], m["count"], len(gpus), *parse_cfg_label(label), min_expected=R,
                    exact_expected=R + w + check) if plugin else None
        out["regions"].append({"metrics": m, "telemetry": tel, "evidence": ev})
    if reader:
        out["reader"] = {"kind": reader.kind, "hz": reader.hz, "n": len(reader.samples),
                         "t_first": reader.samples[0][0] if reader.samples else None,
                         "t_last": reader.samples[-1][0] if reader.samples else None}
        (d / "reader.json").write_text(json.dumps(reader.samples))
    out["profiler_flags"] = pflags
    append_jsonl(OUT / "validation.jsonl", out)
    return out


def counter_update_idle(gpu=7, hz=200, dur=4.0):
    require_validated_protocol()
    """Update period/resolution of the counter on an idle GPU (no process)."""
    check_free()
    h = N.nvmlDeviceGetHandleByIndex(gpu)
    xs = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        xs.append((time.perf_counter(), N.nvmlDeviceGetTotalEnergyConsumption(h)))
        time.sleep(1 / hz)
    return summarize_updates(xs)


def summarize_updates(xs):
    ch = [(t, v) for (t, v), (_, v0) in zip(xs[1:], xs[:-1]) if v != v0]
    dt = np.diff([t for t, _ in ch]) * 1e3 if len(ch) > 2 else np.array([])
    inc = np.diff([v for _, v in ch]) if len(ch) > 2 else np.array([])
    return {"n_reads": len(xs), "n_changes": len(ch), "median_update_ms": float(np.median(dt)) if len(dt) else None,
            "p10_update_ms": float(np.percentile(dt, 10)) if len(dt) else None,
            "p90_update_ms": float(np.percentile(dt, 90)) if len(dt) else None,
            "min_increment_mJ": int(inc.min()) if len(inc) else None,
            "median_increment_mJ": float(np.median(inc)) if len(inc) else None}


def main():
    require_validated_protocol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    globals()["OUT"] = Path(a.out).resolve()   # benchmark processes run with cwd = unit dir
    N.nvmlInit()
    OUT.mkdir(parents=True, exist_ok=True)
    snap0 = snapshot(P4)
    (OUT / f"health_pre_{a.exp}.json").write_text(redact(json.dumps(snap0, indent=1, default=str)))
    results = {}
    exps = ["basic", "idle_counter", "dose", "plugin_overhead", "linearity", "selection"] if a.exp == "all" else [a.exp]
    try:
        if "basic" in exps:
            results["basic"] = run("basic", "ar_p2_1m", P2, "all_reduce", "auto", 1 << 20, 60000, 4000)
            print("basic:", json.dumps(results["basic"]["regions"][0]["metrics"]["flags"]),
                  results["basic"]["regions"][0]["evidence"]["status"], flush=True)
        if "idle_counter" in exps:
            results["idle_counter"] = counter_update_idle()
            append_jsonl(OUT / "validation.jsonl", {"exp": "idle_counter", "utc": utc_now(), **results["idle_counter"]})
            print("idle counter:", results["idle_counter"], flush=True)
        if "dose" in exps:
            # AllReduce 64 MiB on the NVLink pair, ~10 s region; extra reads on GPU 4 between t+4 s and t+8 s
            kinds = (None, "counter", "instant") if "validation_v21" not in str(OUT) else (None, "instant")
            for rep in range(3):
                for kind in kinds:
                    r = Reader(4, kind, 100, 4.0, 4.0) if kind else None
                    o = run("dose", f"rep{rep}_{kind or 'none'}", P2, "all_reduce", "auto", 64 << 20, 6500, 20,
                            check=0, reader=r)
                    if kind == "counter":   # same run doubles as the under-load update-period measurement
                        xs = json.loads((OUT / "dose" / f"rep{rep}_counter" / "reader.json").read_text())
                        append_jsonl(OUT / "validation.jsonl", {"exp": "load_counter_updates", "rep": rep,
                                                                **summarize_updates([tuple(x) for x in xs])})
                    m = o["regions"][0]["metrics"]
                    print(f"dose rep{rep} {kind}: T={m['T_region_s']:.2f}s E4={m['per_gpu'][0]['delta_J']:.1f} "
                          f"E5={m['per_gpu'][1]['delta_J']:.1f}", flush=True)
        if "plugin_overhead" in exps:
            for rep in range(3):
                for gpus, tag in ((P2, "p2"), (P4, "p4")):
                    for size, R in ((4096, 250000 if tag == "p2" else 150000), (1 << 20, 40000 if tag == "p2" else 10000)):
                        for plug in (True, False):
                            o = run("plugin_overhead", f"rep{rep}_{tag}_{size}_{'on' if plug else 'off'}", gpus,
                                    "all_reduce", "auto", size, R, 2000, plugin=plug)
                            print(f"plugin {tag} {size} {plug}: t_gpu={o['regions'][0]['metrics']['t_gpu_s'] * 1e6:.3f} us",
                                  flush=True)
        if "linearity" in exps:
            for rep in range(2):
                for R, w in ((20000, 1000), (40000, 1000), (80000, 1000), (40000, 20000)):
                    o = run("linearity", f"rep{rep}_R{R}_w{w}", P2, "all_reduce", "auto", 1 << 20, R, w)
                    m = o["regions"][0]["metrics"]
                    print(f"linearity R={R} w={w}: E={m['E_raw_J']:.2f} J/coll={m['J_per_coll'] * 1e3:.4f} mJ "
                          f"T={m['T_region_s']:.3f}", flush=True)
        if "selection" in exps:
            for gpus, tag in ((P2, "p2"), (P4, "p4")):
                o = run("selection", f"{tag}_ar_auto_tuning", gpus, "all_reduce", "auto", None, 3, 1, check=1,
                        subsys="INIT,ENV,TUNING", sizes=(4096, 64 << 20))
                print(f"selection {tag}: {len(o['regions'])} regions", flush=True)
    except Busy as e:
        print(f"STOPPED: allocation busy: {e}", flush=True)
    snap1 = snapshot(P4)
    (OUT / f"health_post_{a.exp}.json").write_text(redact(json.dumps(snap1, indent=1, default=str)))
    print("post allocation problems:", allocation_problems(snap1, P4), flush=True)


if __name__ == "__main__":
    main()
