"""Release reproduction (CPU only, no NVML, no GPU): re-run every published analysis into a fresh directory and
compare it with the committed outputs.

    python -m ncclenergy.v2.release_repro --out <new dir> [--steps a,b] [--audit]

Each step runs as a subprocess (the documented command, with its output redirected into --out). With --audit,
scripts/open_audit/sitecustomize.py records every file each step opens; the inputs (files read, never written,
inside the repository, outside --out) are listed with SHA-256 in <out>/inputs_<step>.tsv, and any file opened
outside the repository and the Python installation is listed in <out>/outside_reads.tsv.

Comparison per output file: IDENTICAL (bytes), IDENTICAL_EXCEPT_SOURCE_PATH (only a line naming the directory it
was written from differs), NUMERIC_EQUAL (CSV/JSON equal after parsing, numbers within rel 1e-9 / abs 1e-12),
EQUAL_EXCEPT_REPO_PATH (equal after removing the repository path prefix that some outputs record),
REGENERATED (figure bytes differ; not byte-comparable), KNOWN_UNREPRODUCED (listed in KNOWN with the reason),
DIFFERENT, MISSING_OUTPUT, EXTRA_OUTPUT. The run is a pass only if no file is DIFFERENT or MISSING_OUTPUT and no frozen
file changed. phase89/phase10 are gated against the Stage 2 outputs of the same (v2.2) code; their differences from
the frozen v2.1-code outputs are reported under context_vs_frozen_v21. Frozen directories are hashed before and after.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .common import REPO

PY = sys.executable
AUDIT_DIR = REPO / "scripts/open_audit"
RA = "data/v2/reanalysis/2026-10-09_v2.2-audit"
PHASE89_FILES = ("latency_policy.csv", "nccl_tuner_P2_NVL.conf", "nccl_tuner_P2_NVL_latency.conf",
                 "nccl_tuner_P2_PCI_SHM.conf", "nccl_tuner_P2_PCI_SHM_latency.conf", "nccl_tuner_P4_MIX.conf",
                 "nccl_tuner_P4_MIX_latency.conf", "order_temperature.json", "placement_meta.json",
                 "recommender_lobo.csv", "recommender_lobo_summary.json", "robustness.csv")
# informational comparison with the frozen historical (v2.1-code) outputs; never gates the run
CONTEXT = {"v2_phase89": {f"v2/confirmatory/{n}": f"data/v2/processed/confirmatory/{n}" for n in PHASE89_FILES},
           "v2_phase10": {"v2/phase10/phase10_summary.json": "data/v2/processed/phase10/phase10_summary.json",
                          "v2/phase10/workload_execution.csv": "data/v2/processed/phase10/workload_execution.csv",
                          "v2/phase10/workload_runs.csv": "data/v2/processed/phase10/workload_runs.csv"}}
# named exceptions: (reference path) -> (status, reason)
KNOWN = {"figures/full/method_example_trace.png":
         ("KNOWN_UNREPRODUCED", "illustrative pilot figure made once by plots.example_trace() for an unrecorded launch; "
                                "not produced by `ncclenergy plot`; no reported number depends on it")}
# line-level normalization for files that embed the path they were written from
NORMALIZE = {"data/processed/full/results_summary.md": r"^Source: `.*`$"}
STEPS = {
    # name: (argv after "python -m", {output path in --out: reference path in the repo}, depends-on)
    "pilot_analyze": (["ncclenergy", "analyze", "--run", "data/raw/full", "--out", "{O}/pilot/processed_full"],
                      {"pilot/processed_full": "data/processed/full"}, None),
    "pilot_report": (["ncclenergy", "report", "--processed", "{O}/pilot/processed_full"],
                     {"pilot/processed_full": "data/processed/full"}, "pilot_analyze"),
    "pilot_plot": (["ncclenergy", "plot", "--processed", "{O}/pilot/processed_full", "--out", "{O}/pilot/figures_full"],
                   {"pilot/figures_full": "figures/full"}, "pilot_analyze"),
    "v2_exploratory": (["ncclenergy.v2.analyze_cli", "exploratory", "--campaign", "data/v2/raw/exploratory",
                        "--out", "{O}/v2/exploratory"], {"v2/exploratory": "data/v2/processed/exploratory"}, None),
    "v2_confirmatory": (["ncclenergy.v2.analyze_cli", "confirmatory", "--campaign", "data/v2/raw/confirmatory",
                         "--prereg", "docs/v2/prereg.json", "--out", "{O}/v2/confirmatory"],
                        {"v2/confirmatory": "data/v2/processed/confirmatory"}, None),
    # phase89 / phase10 with the current (v2.2) code: the reference is the Stage 2 reanalysis output made by the same
    # code (phase89_v22, phase10_v22). The frozen v2.1 outputs differ by design (scope headers, exact-size tuner rules,
    # campaign-derived scope, all-rank/numerics fields; Stage 2 ledger); that comparison is reported, not gated.
    "v2_phase89": (["ncclenergy.v2.phase89", "--campaign", "data/v2/raw/confirmatory", "--prereg", "docs/v2/prereg.json",
                    "--processed", "{O}/v2/confirmatory", "--figures", "{O}/v2/figures_confirmatory"],
                   {**{f"v2/confirmatory/{n}": f"{RA}/phase89_v22/{n}" for n in PHASE89_FILES},
                    "v2/figures_confirmatory": "figures/v2/confirmatory"}, "v2_confirmatory"),
    "v2_phase10": (["ncclenergy.v2.phase10", "analyze", "--policy-dir", "data/v2/processed/confirmatory",
                    "--out", "{O}/v2/phase10"],
                   {"v2/phase10/phase10_summary.json": f"{RA}/phase10_v22/phase10_summary.json",
                    "v2/phase10/workload_execution.csv": f"{RA}/phase10_v22/workload_execution.csv",
                    "v2/phase10/workload_runs.csv": f"{RA}/phase10_v22/workload_runs.csv",
                    "v2/phase10/figures": f"{RA}/figures/phase10_v22"}, None),
    "v22_reanalysis": (["ncclenergy.v2.reanalysis", "--out", "{O}/v22/reanalysis"],
                       {"v22/reanalysis": "data/v2/reanalysis/2026-10-09_v2.2-audit"}, None),
    "v22_addendum": (["ncclenergy.v2.reanalysis_addendum", "--out", "{O}/v22/addendum1"],
                     {"v22/addendum1": "data/v2/reanalysis/2026-10-09_v2.2-audit-addendum1"}, None),
    "gate3a_smoke_check": (["ncclenergy.v2.smoke_check", "--s2", "data/v2.2/raw/v2.2-smoke-02",
                            "--s3", "data/v2.2/raw/v2.2-smoke-02-s3", "--s4", "data/v2.2/raw/v2.2-smoke-02-s4",
                            "--audit-dir", "data/v2.2/raw/v2.2-smoke-02/nvml_audit",
                            "--after", "data/v2.2/processed/v2.2-smoke-02/after.json",
                            "--out", "{O}/v22/smoke02/smoke_check.json"],
                           {"v22/smoke02/smoke_check.json": "data/v2.2/processed/v2.2-smoke-02/smoke_check.json"}, None),
    "gate3b_measval": (["ncclenergy.v2.measval", "--campaign", "data/v2.2/raw/v2.2-measval-01",
                        "--out", "{O}/v22/measval01"],
                       {"v22/measval01/decisions.json": "data/v2.2/processed/v2.2-measval-01/decisions.json",
                        "v22/measval01/processes.csv": "data/v2.2/processed/v2.2-measval-01/processes.csv"}, None),
    "gate3b_posthoc": (["ncclenergy.v2.measval_posthoc", "--campaign", "data/v2.2/raw/v2.2-measval-01",
                        "--out", "{O}/v22/measval01/posthoc_temperature.json"],
                       {"v22/measval01/posthoc_temperature.json":
                        "data/v2.2/processed/v2.2-measval-01/posthoc_temperature.json"}, None),
    "gate3b_amendment_check": (["ncclenergy.v2.amendment_check", "--campaign", "data/v2.2/raw/v2.2-measval-01",
                                "--out", "{O}/v22/amendment_check.json"], {}, None),
    # release tables: key numbers recomputed and checked against the documents, counts, gate 3b tables and figures
    "release_tables": (["ncclenergy.v2.release_tables", "--out", "{O}/release_tables"],
                       {"release_tables": "results/release-corrections_2026-10-11"}, None),
}
FROZEN = ["data/processed", "data/v2/processed", "data/v2/reanalysis", "data/v2.2/processed", "figures"]
FIG = {".png", ".pdf", ".svg"}
REL, ABS = 1e-9, 1e-12


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(paths) -> dict[str, str]:
    out = {}
    for d in paths:
        d = REPO / d
        for f in sorted(d.rglob("*")) if d.exists() else []:
            if f.is_file():
                out[str(f.relative_to(REPO))] = sha256(f)
    return out


def _num_equal(a, b) -> tuple[bool, float]:
    """Recursive equality of parsed JSON-like values; returns (equal, max relative difference)."""
    if isinstance(a, str) and isinstance(b, str) and a[:1] in ("[", "{") and b[:1] == a[:1]:
        try:
            aa, bb = json.loads(a), json.loads(b)
        except (ValueError, TypeError):
            pass
        else:
            # CSV-serialized numerical arrays use the declared numeric tolerance.
            # Paths, boolean metadata and ordinary prose remain exact.
            return _num_equal(aa, bb)
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b, 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True, 0.0
        d = abs(a - b)
        r = d / max(abs(a), abs(b), 1e-300)
        return (d <= ABS or r <= REL), (r if d > ABS else 0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, math.inf
        res = [_num_equal(a[k], b[k]) for k in a]
        return all(x for x, _ in res), max([m for _, m in res] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, math.inf
        res = [_num_equal(x, y) for x, y in zip(a, b)]
        return all(x for x, _ in res), max([m for _, m in res] or [0.0])
    return a == b, 0.0


def _read_table(p: Path):
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    name = p.name[:-3] if p.suffix == ".gz" else p.name
    if name.endswith(".json"):
        return json.loads(raw)
    if name.endswith(".jsonl"):
        return [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), keep_default_na=False, na_values=[""])
        return {"columns": list(df.columns), "rows": json.loads(df.to_json(orient="values", double_precision=15))}
    return None


def compare_file(new: Path, ref: Path) -> dict:
    if new.read_bytes() == ref.read_bytes():
        return {"status": "IDENTICAL"}
    pat = NORMALIZE.get(str(ref.relative_to(REPO)))
    if pat:
        import re
        strip = lambda b: [l for l in b.decode().splitlines() if not re.match(pat, l)]
        if strip(new.read_bytes()) == strip(ref.read_bytes()):
            return {"status": "IDENTICAL_EXCEPT_SOURCE_PATH"}
    if new.suffix.lower() in FIG:
        return {"status": "REGENERATED"}
    try:
        a, b = _read_table(new), _read_table(ref)
    except Exception as e:  # unparsable -> plain difference
        return {"status": "DIFFERENT", "why": f"unparsable: {e}"}
    if a is None:
        return {"status": "DIFFERENT", "why": "bytes differ (not a table)"}
    eq, m = _num_equal(a, b)
    if eq:
        return {"status": "NUMERIC_EQUAL", "max_rel_diff": m}
    # outputs that record the directory they read from: equal once the repository prefix (absolute, or the
    # "<REPO>/" placeholder of the sanitized export) is removed
    strip = lambda x: _read_table_text(x.replace(str(REPO.resolve()) + "/", "").replace("<REPO>/", ""), new.name)
    try:
        eq2, m2 = _num_equal(strip(new.read_text()), strip(ref.read_text()))
    except Exception:  # noqa: BLE001
        eq2 = False
    if eq2:
        return {"status": "EQUAL_EXCEPT_REPO_PATH", "max_rel_diff": m2}
    return {"status": "DIFFERENT", "max_rel_diff": m}


def _read_table_text(text: str, name: str):
    if name.endswith(".json"):
        return json.loads(text)
    if name.endswith(".jsonl"):
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    if name.endswith(".csv"):
        df = pd.read_csv(io.StringIO(text), keep_default_na=False, na_values=[""])
        return {"columns": list(df.columns), "rows": json.loads(df.to_json(orient="values", double_precision=15))}
    return text


def compare(out: Path, mapping: dict, only_new_names: bool) -> list[dict]:
    rows = []
    for o, r in mapping.items():
        new, ref = out / o, REPO / r
        if ref.is_file() or new.is_file():
            pairs = [(new, ref)]
        else:
            names = {p.relative_to(new) for p in new.rglob("*") if p.is_file()} if new.exists() else set()
            refs = {p.relative_to(ref) for p in ref.rglob("*") if p.is_file()} if ref.exists() else set()
            keys = names if only_new_names else names | refs
            pairs = [(new / k, ref / k) for k in sorted(keys)]
        for n, rf in pairs:
            row = {"output": str(n.relative_to(out)), "reference": str(rf.relative_to(REPO))}
            if row["reference"] in KNOWN and not n.exists():
                row["status"], row["why"] = KNOWN[row["reference"]]
            elif not n.exists():
                row["status"] = "MISSING_OUTPUT"
            elif not rf.exists():
                row["status"] = "EXTRA_OUTPUT"
            else:
                row.update(compare_file(n, rf))
            rows.append(row)
    return rows


def run_step(name: str, out: Path, audit: bool) -> dict:
    argv, mapping, _ = STEPS[name]
    try:
        o = str(out.resolve().relative_to(REPO.resolve()))     # keep absolute paths out of the outputs
    except ValueError:
        o = str(out)
    args = [a.replace("{O}", o) for a in argv]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "NCCLE_OPEN_AUDIT")}
    env.update(PYTHONPATH=f"{AUDIT_DIR}:{REPO / 'src'}" if audit else str(REPO / "src"), MPLBACKEND="Agg",
               PYTHONHASHSEED="0")
    alog = out / "_audit" / name
    if audit:
        alog.parent.mkdir(parents=True, exist_ok=True)
        env["NCCLE_OPEN_AUDIT"] = str(alog)
    t0 = time.time()
    p = subprocess.run([PY, "-m", *args], cwd=REPO, env=env, capture_output=True, text=True)
    (out / "_logs").mkdir(parents=True, exist_ok=True)
    (out / "_logs" / f"{name}.out").write_text(p.stdout)
    (out / "_logs" / f"{name}.err").write_text(p.stderr)
    res = {"step": name, "cmd": "python -m " + " ".join(argv).replace("{O}", "<out>"), "rc": p.returncode,
           "seconds": round(time.time() - t0, 1)}
    # pilot_report rewrites results_summary.* in its input dir; phase89 extends the confirmatory dir: compare only the
    # files the step itself is responsible for after it has run (the whole dir for the last writer)
    res["comparison"] = compare(out, mapping, only_new_names=name in ("pilot_analyze", "v2_confirmatory",
                                                                     "release_tables"))
    if name in CONTEXT:
        res["context_vs_frozen_v21"] = compare(out, CONTEXT[name], only_new_names=False)
    if audit:
        res["audit"] = _collect_audit(name, out)
    return res


def _collect_audit(name: str, out: Path) -> dict:
    seen: dict[str, bool] = {}
    for f in (out / "_audit").glob(f"{name}.*"):
        for line in f.read_text().splitlines():
            w, p = line.split("\t", 1)
            seen[p] = seen.get(p, False) or w == "W"
    root, o = str(REPO.resolve()) + os.sep, str(out.resolve()) + os.sep
    sysroots = tuple({str(Path(x).resolve()) + os.sep for x in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
    inputs, outside = [], []
    for p, w in sorted(seen.items()):
        rp = str(Path(p).resolve()) if os.path.exists(p) else p
        if rp.startswith(o) or p.startswith(o):
            continue
        if rp.startswith(root):
            rel = rp[len(root):]
            if not w and os.path.isfile(rp) and not rel.startswith(("src/", "scripts/open_audit/")) \
                    and "__pycache__" not in rel:
                inputs.append(rel)
        elif not rp.startswith(sysroots) and not rp.startswith(("/proc/", "/dev/", "/sys/", "/etc/", "/usr/", "/tmp/")) \
                and "/.cache/matplotlib/" not in rp:     # matplotlib font cache: environment, not an input
            outside.append(f"{'W' if w else 'R'}\t{rp}")
    with open(out / f"inputs_{name}.tsv", "w") as f:
        f.write("path\tsha256\tbytes\n")
        for rel in inputs:
            f.write(f"{rel}\t{sha256(REPO / rel)}\t{(REPO / rel).stat().st_size}\n")
    return {"n_inputs": len(inputs), "outside_reads": outside}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--audit", action="store_true", help="record files opened by each step (open-audit hook)")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else Path.cwd() / a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; reproduction writes only into a new directory")
    out.mkdir(parents=True, exist_ok=True)
    steps = [s for s in a.steps.split(",") if s]
    for s in steps:
        dep = STEPS[s][2]
        if dep and dep not in steps:
            raise SystemExit(f"step {s} needs {dep}")
    before = tree_hash(FROZEN)
    results = []
    for s in steps:
        r = run_step(s, out, a.audit)
        bad = [c for c in r["comparison"] if c["status"] in ("DIFFERENT", "MISSING_OUTPUT")]
        r["pass"] = r["rc"] == 0 and not bad
        results.append(r)
        counts = {}
        for c in r["comparison"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {s:24s} rc={r['rc']} {r['seconds']:7.1f}s  {counts}", flush=True)
    after = tree_hash(FROZEN)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    summary = {"python": sys.version.split()[0], "repo": "<repo>", "steps": results,
               "frozen_files_hashed": len(before), "frozen_files_changed": changed,
               "pass": all(r["pass"] for r in results) and not changed}
    if a.audit:
        outside = sorted({x for r in results for x in r["audit"]["outside_reads"]})
        (out / "outside_reads.tsv").write_text("\n".join(outside) + ("\n" if outside else ""))
        summary["outside_reads"] = len(outside)
    (out / "reproduction_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"frozen files hashed {len(before)}, changed {len(changed)}; overall {'PASS' if summary['pass'] else 'FAIL'}")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
