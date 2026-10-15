"""Gate 3b analysis amendment AM-1 check (CPU only): the post-collection fix of measval.descriptive() changes no
decision. Runs the pre-specified analysis exactly as frozen before collection (configs/v2.2/frozen_code/
measval_f4735b1.py, git blob fb378505 of commit f4735b1) and the current measval.py on the same raw data.

    python -m ncclenergy.v2.amendment_check --campaign data/v2.2/raw/v2.2-measval-01 --out <json>

Reports: the source diff between the two files; whether the frozen main() fails, and where; and whether every
pre-specified output (acceptance, exclusions, control status, D1-D4 with all numbers) is identical.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

from . import measval as current
from .common import REPO

FROZEN = REPO / "configs/v2.2/frozen_code/measval_f4735b1.py"
FROZEN_BLOB = "fb378505edcfbdae12cc424ffab5a839ff29e0cb"


def git_blob_id(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def load_frozen():
    spec = importlib.util.spec_from_file_location("ncclenergy.v2.measval_f4735b1", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decisions(mod, campaign: Path, frozen_schedule: dict) -> dict:
    df = mod.accept(mod.load(campaign, frozen_schedule))
    acc = df[df.accepted]
    ctrl = mod.control_status(acc)
    r1, r2, r3 = mod.d1(acc), mod.d2(acc), mod.d3(acc, ctrl)
    r4 = mod.d4(acc, r1, r2, r3)
    res = {"n_processes": int(len(df)), "n_accepted": int(len(acc)),
           "exclusions": df[~df.accepted][["block", "unit_id", "exclusion"]].to_dict("records"),
           "control": ctrl, "D1": r1, "D2": r2, "D3": r3, "D4": r4}
    return json.loads(json.dumps(res, default=str)), df, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--frozen-schedule", type=Path, default=REPO / "configs/v2.2/measval01_frozen.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sched = json.loads(a.frozen_schedule.read_text())
    old = load_frozen()
    res_old, df_old, acc_old = decisions(old, a.campaign, sched)
    res_new, df_new, _ = decisions(current, a.campaign, sched)
    try:
        old.descriptive(df_old, acc_old, a.campaign)
        frozen_descriptive = "ran"
    except Exception as e:  # the defect AM-1 fixes
        frozen_descriptive = f"{type(e).__name__}: {e}"
    diff = [l for l in difflib.unified_diff(FROZEN.read_text().splitlines(),
                                            (REPO / "src/ncclenergy/v2/measval.py").read_text().splitlines(),
                                            "measval_f4735b1.py", "measval.py", lineterm="", n=0)]
    changed = [l for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    out = {"frozen_file": str(FROZEN.relative_to(REPO)), "frozen_git_blob": git_blob_id(FROZEN),
           "frozen_git_blob_expected": FROZEN_BLOB, "frozen_blob_matches": git_blob_id(FROZEN) == FROZEN_BLOB,
           "source_diff_changed_lines": changed,
           "frozen_descriptive_result": frozen_descriptive,
           "processes_table_identical": df_old.equals(df_new),
           "decisions_identical": res_old == res_new,
           "decisions": {k: (res_new[k]["decision"] if isinstance(res_new[k], dict) and "decision" in res_new[k]
                             else res_new[k]["status"] if k == "control" else None) for k in ("control", "D1", "D2",
                                                                                                 "D3", "D4")},
           "n_processes": res_new["n_processes"], "n_accepted": res_new["n_accepted"]}
    out["pass"] = bool(out["frozen_blob_matches"] and out["decisions_identical"] and out["processes_table_identical"]
                       and len(changed) == 2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("frozen_blob_matches", "source_diff_changed_lines",
                                          "frozen_descriptive_result", "decisions_identical",
                                          "processes_table_identical", "decisions", "pass")}, indent=1))


if __name__ == "__main__":
    main()
