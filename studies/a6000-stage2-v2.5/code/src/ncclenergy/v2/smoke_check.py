"""Gate 3a smoke-test checks (CPU only; reads the smoke outputs, makes no NVML call).

    python -m ncclenergy.v2.smoke_check --s2 data/v2.2/raw/v2.2-smoke-01 --s3 data/v2.2/raw/v2.2-smoke-01-s3 \
        --s4 data/v2.2/raw/v2.2-smoke-01-s4 --audit-dir <dir with audit_*.jsonl> --after <post-run gpu json> --out <json>

Each check is PASS / FAIL with the evidence; the smoke test passes only if every check passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import REPO, read_jsonl
from .evidence import load_profiler_dir

EXPECT = {"auto": "RING/SIMPLE", "Ring/Simple": "RING/SIMPLE", "Ring/LL128": "RING/LL128"}
REGION_KEYS = ("e_start_mJ", "e_end_mJ", "e_post_mJ", "rc_start", "rc_end", "rc_post", "t_start_a_ns", "t_start_b_ns",
               "t_end_a_ns", "t_end_b_ns", "t_post_a_ns", "t_post_b_ns", "edges", "n_edges", "phase_ns", "bracket_ns",
               "t_commit_ns", "t_loop_end_ns", "event_ms", "event_rc")


def _block(camp: Path) -> Path:
    bs = sorted((camp / "blocks").glob("block_*"))
    assert len(bs) == 1, bs
    return bs[0]


def _attempts(bdir: Path) -> list[dict]:
    return read_jsonl(bdir / "attempts.jsonl")[0] if (bdir / "attempts.jsonl").exists() else []


def check(name, ok, evidence):
    return {"check": name, "result": "PASS" if ok else "FAIL", "evidence": evidence}


def s2_checks(camp: Path, audit: list[dict]) -> list[dict]:
    out = []
    b = _block(camp)
    summ = json.loads((b / "block_summary.json").read_text())
    out.append(check("S2 block complete", summ.get("status") == "COMPLETE",
                     {k: summ.get(k) for k in ("status", "stop_reason", "final_status_counts", "health_events")}))
    atts = _attempts(b)
    out.append(check("S2 three units, one attempt each, no retries", len(atts) == 3 and all(a["attempt"] == 0 for a in atts),
                     [a["unit_id"] for a in atts]))
    windows, execs = [], {}
    for a in atts:
        u = a["unit_id"]
        cfg = a["requested"]
        d = REPO / a["dir"]
        rec = json.loads((d / "region.jsonl").read_text().splitlines()[0])
        miss = [k for g in rec["gpus"] for k in REGION_KEYS if k not in g]
        rc_ok = all(g["rc_start"] == 0 and g["rc_end"] == 0 and g["rc_post"] == 0 for g in rec["gpus"])
        m = a["region"]
        out.append(check(f"S2 {cfg}: region record complete (C0/C1/Cpost, read brackets, edges, phase, commit, loop end)",
                         not miss and rc_ok and all(g["n_edges"] >= 2 for g in rec["gpus"]),
                         {"missing": miss, "rc_ok": rc_ok, "n_edges": [g["n_edges"] for g in rec["gpus"]],
                          "T_region_s": m.get("T_region_s"), "flags": m.get("flags")}))
        out.append(check(f"S2 {cfg}: v2.2 QC OK, fail-closed fields present",
                         a["qc_status"] == "OK" and a.get("qc_method") == "v2.2-fail-closed"
                         and a.get("telemetry_problems") == [],
                         {"qc_status": a["qc_status"], "qc_reasons": a.get("qc_reasons"), "qc_method": a.get("qc_method"),
                          "telemetry_problems": a.get("telemetry_problems")}))
        files, pflags = load_profiler_dir(d / "prof")
        ranks = sorted(f["rank"] for f in files)
        ex = sorted({(c["algo"], c["proto"], c["nChannels"], c["nWarps"]) for f in files for c in f["colls"]
                     if c["func"] == "AllGather"})
        execs[cfg] = ex
        ev = a.get("evidence") or {}
        out.append(check(f"S2 {cfg}: execution evidence on both ranks, expected algorithm/protocol",
                         ranks == [0, 1] and ev.get("status") == "VERIFIED" and len(ex) == 1
                         and f"{ex[0][0]}/{ex[0][1]}" == EXPECT[cfg],
                         {"ranks": ranks, "evidence": ev.get("status"), "reasons": ev.get("reasons"),
                          "executed(algo,proto,nCh,nWarps)": ex, "per_rank_n": ev.get("per_rank_n"),
                          "profiler_flags": pflags}))
        tel = a.get("telemetry") or {}
        out.append(check(f"S2 {cfg}: sampler telemetry for both GPUs, no gaps",
                         len(tel) == 2 and all((t.get("gaps") or {}).get("n_gaps") == 0 and t.get("n_samples", 0) > 10
                                               for t in tel.values()),
                         {g: {"n": t.get("n_samples"), "max_dt": (t.get("gaps") or {}).get("max_dt"),
                              "temp_max_c": t.get("temp_max_c"), "power_mean_w": t.get("power_mean_w"),
                              "inst_over_counter": (t.get("energy_int_inst_J") or 0) / t["counter_J"]
                              if t.get("counter_J") else None} for g, t in tel.items()}))
        cl = (a.get("proc") or {}).get("cleanup") or {}
        out.append(check(f"S2 {cfg}: owned process exited normally and was reaped", cl.get("ok") is True
                         and (a.get("proc") or {}).get("stop_reason") is None,
                         {"rc": (a.get("proc") or {}).get("rc"), "cleanup": cl}))
        for g in rec["gpus"]:
            windows.append((cfg, g["rank"], g["t_start_a_ns"], g["t_loop_end_ns"]))
    out.append(check("S2 identical-execution control: auto and forced Ring/Simple executed the same "
                     "algorithm/protocol/channels/warps", execs.get("auto") == execs.get("Ring/Simple")
                     and len(execs.get("auto") or []) == 1, execs))
    counter = [c for c in audit if c["fn"] == "nvmlDeviceGetTotalEnergyConsumption"]
    inside = [c for c in counter for (_, _, t0, t1) in windows if t0 <= c["t_ns"] <= t1]
    region_calls = [c for c in audit for (_, _, t0, t1) in windows if t0 <= c["t_ns"] <= t1]
    out.append(check("S2 controller NVML audit: no cumulative-counter read inside any measured region",
                     bool(audit) and not inside,
                     {"audit_calls_total": len(audit), "counter_calls_total": len(counter),
                      "counter_calls_inside_regions": len(inside),
                      "calls_inside_regions_by_thread": _count(region_calls, "thread"),
                      "calls_inside_regions_by_function": _count(region_calls, "fn")}))
    return out


def _count(calls, key):
    out = {}
    for c in calls:
        out[c[key]] = out.get(c[key], 0) + 1
    return out


def shutdown_checks(tag: str, camp: Path, expect_status: str, expect_reason: str) -> list[dict]:
    out = []
    b = _block(camp)
    summ = json.loads((b / "block_summary.json").read_text())
    out.append(check(f"{tag} block stopped as designed", summ.get("status") == expect_status
                     and expect_reason in str(summ.get("stop_reason")),
                     {k: summ.get(k) for k in ("status", "stop_reason", "final_status_counts")}))
    units = sorted((b / "units").glob("*"))
    ops = [json.loads((u / "owned_process.json").read_text()) for u in units if (u / "owned_process.json").exists()]
    cl = ops[0]["cleanup"] if ops else {}
    out.append(check(f"{tag} owned process tree terminated and reaped (bounded)",
                     len(ops) == 1 and cl.get("ok") is True and cl.get("survivors") == [],
                     {"owned_process": ops[0] if ops else None}))
    post = json.loads((b / "preflight_post.json").read_text()) if (b / "preflight_post.json").exists() else {}
    g45 = [g for g in post.get("gpus", []) if g.get("index") in (4, 5)]
    out.append(check(f"{tag} GPUs 4,5 free after the stop (no process, <= 16 MiB used)",
                     len(g45) == 2 and all(not g.get("processes") and (g.get("mem_used_mib") or 0) <= 16 for g in g45),
                     [{"index": g.get("index"), "processes": g.get("processes"), "mem_used_mib": g.get("mem_used_mib"),
                       "temp_c": g.get("temp_c")} for g in g45]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=Path, required=True)
    ap.add_argument("--s3", type=Path, required=True)
    ap.add_argument("--s4", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True, help="JSON of an nvidia-smi/NVML check made after S4")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    audit = []
    for f in sorted(a.audit_dir.glob("audit_s2*.jsonl")):
        audit += read_jsonl(f)[0]
    res = s2_checks(a.s2, audit)
    res += shutdown_checks("S3", a.s3, "STOPPED", "WALL_TIME_EXCEEDED")
    res += shutdown_checks("S4", a.s4, "INTERRUPTED", "ControllerSignal")
    after = json.loads(a.after.read_text())
    res.append(check("after all steps: no process on GPUs 4,5", not after.get("processes_on_4_5"), after))
    summary = {"passed": all(r["result"] == "PASS" for r in res), "n_checks": len(res),
               "n_failed": sum(r["result"] == "FAIL" for r in res), "checks": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=1, default=str))
    for r in res:
        print(f"{r['result']:4s}  {r['check']}")
    print(f"SMOKE {'PASSED' if summary['passed'] else 'FAILED'}: {summary['n_checks'] - summary['n_failed']}/{summary['n_checks']}")


if __name__ == "__main__":
    main()
