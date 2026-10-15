"""Write a PROPOSED Stage 2 freeze manifest (schema nccle-stage2-freeze-v1) - CPU only, never a freeze record.

    python scripts/stage2_freeze_manifest.py --kind Freeze-S --out ../protocol/FREEZE_S_MANIFEST_PROPOSED.json
    python scripts/stage2_freeze_manifest.py --kind Freeze-C --out ../protocol/FREEZE_C_MANIFEST_PROPOSED.json

It binds the roles the runner requires (config, protocol, patch, binary, build record, plugin, every loaded code file,
analysis for Freeze-C) and the external NCCL library identity, all with SHA-256. It deliberately has no
approved_by_user entry (and, for Freeze-C, no parent yet), so stage2_runner.verify_freeze refuses it. A record is made
only after the user approves, from the completed config and the files as they are then.
"""
import argparse
import hashlib
import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(Path("..") / p.relative_to(REPO.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["Freeze-S", "Freeze-C"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=REPO.parent / "protocol/STAGE2_CG1_PACKET.md")
    a = ap.parse_args()
    import ncclenergy.v2.stage2_calibration  # noqa: F401
    from ncclenergy.v2 import runner as R, stage2_runner as S
    cfg = REPO / ("configs/stage2/smoke01_DRAFT.yaml" if a.kind == "Freeze-S" else "configs/stage2/calib01_DRAFT.yaml")
    binary = S.BIN_DIR_STAGE2 / f"{S.COLL}_perf"
    roles = {"config": rel(cfg), "protocol": rel(a.protocol), "patch": rel(S.PATCH_V25), "binary": rel(binary),
             "build_record": rel(binary.parent.parent / "BUILD_RECORD.txt"), "plugin": rel(R.PLUGIN),
             "code": S.loaded_code_files(),
             "analysis": list(S.MANDATORY_ANALYSIS.get(a.kind, ()))}
    rels = [roles[k] for k in ("config", "protocol", "patch", "binary", "build_record", "plugin")] + roles["code"] \
        + roles["analysis"] + sorted(rel(p) for p in (REPO / "tests/v2").glob("test_stage2_*.py")) \
        + [rel(REPO / x) for x in ("patches/nccl-tests-afd59ab-v2.1-region.patch", "scripts/build_nccl_tests_stage2.sh",
                                   "scripts/env.sh", "scripts/stage2_make_configs.py", "scripts/stage2_freeze_manifest.py",
                                   "configs/v2.2/frozen_code/measval_f4735b1.py")]
    files, missing = {}, []
    for r in dict.fromkeys(rels):
        p = REPO / r
        if p.exists():
            files[r] = {"sha256": sha(p), "bytes": p.stat().st_size}
        else:
            missing.append(r)
    nccl = S.NCCL_LIB_LINK.resolve()
    out = {"schema": S.FREEZE_SCHEMA, "kind": a.kind, "status": "PROPOSED_NOT_FROZEN",
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "repo_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                       check=True).stdout.strip(),
           "implementation_id": S.IMPLEMENTATION_ID, "method_id": S.METHOD_ID,
           "note": "proposal only: no approved_by_user entry; the runner refuses it as a freeze record",
           "roles": roles, "files": files, "missing": missing,
           "external": {"libnccl": {"path": str(nccl), "sha256": sha(nccl)}},
           "pending_before_freeze": {
               "Freeze-S": ["user confirmation of the allocated pair and access window (allocation_confirmation)",
                            "scope_expected from the confirmed pair (GPU model, driver, link, UUIDs, NUMA, power/clock "
                            "profile)", "completed smoke config replacing the DRAFT", "user approval entry"],
               "Freeze-C": ["approved protocol", "Freeze-S record (parent) and reviewed smoke results on the same pair",
                            "completed calibration config replacing the DRAFT", "allocation confirmation for the "
                            "calibration window", "user approval entry"]}[a.kind]}
    if a.kind == "Freeze-C":
        out["parent"] = {"path": None, "sha256": None, "note": "set to the approved Freeze-S record"}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{len(files)} files, {len(missing)} missing -> {a.out}")


if __name__ == "__main__":
    main()
