"""Offline replay of the disclosed public derivatives, not original Freeze-H verification.

The frozen numerical kernels are imported unchanged after all public file hashes
are checked. Identifier/path redaction changes archival bytes, so this wrapper
checks their derivative ledger rather than claiming those bytes satisfy the
original experiment's freeze. It neither acquires telemetry nor launches work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

STATUS = "PUBLIC_DERIVATIVE_REPLAY"
SESSIONS = (1, 2, 3)
GPUS = [4, 5]
IMPLEMENTATION = "stage2-cg1-2026-10-12"
METHOD = "v2.5-stage2-region"
CAMPAIGN = "v2.5-heldout"
CONFIG_ORIGINAL_SHA = {
    1: "2835623709bbc0e6feaa84560ce41986865cedcc1358fec6c065838111a115ef",
    2: "7881fe3fe1e6aba1a06434fb2e83058d5019dd1bc1134dd26bf1c2963b3a8140",
    3: "b46862dfa612d054ccf2f906c4858f72bb6837126f06dec55caf699f517a1244",
}
# Exact files used by the historical analysis, not reimplemented formulas.
KERNEL_SHA = {
    "common": "b5581311389ff8f3ca7656509ff2e1ab62b1c17bcb4aa4909a70ce4fba6e0f8a",
    "stage2_heldout": "ae8c9c6b504f45ed3039fa04abb5a53abb51dc3736daef9d5fec33c4cc29b4ee",
    "stage2_calibration": "b88ad3c110800293718a642c2351c1c04389edd43bab0bd0efe91db0a7803124",
    "stage2_decisions": "ea82214b90f9d26d54d12544ee3763c9cd028b06e5a47cace75197cd4e692b72",
    "stage2_energy": "d3317beab273fd475cc97ff4e0f97b7cb6d47208386571e5ce8798a568c61c00",
    "stage2_phase": "b399bf6cbe9de7ac92132983f176535cbbca23ecd150f30b84084bf961d837ed",
    "stage2_readiness": "2491ebe6e70e61dfdb3b525e5bb1c00fc5109083855d399869906063db423460",
    "stage2_runner": "6a9793188debab82c3c082c5a3cd6ef2007e08e052e122930f421608d3957d9d",
}
EXPECTED_PRIMARY_MISSING = {"s2-b2-p08", "s3-b2-p11"}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class PublicInputError(ValueError):
    """A public package failed its integrity, identity or numerical replay checks."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root: Path, relative: str) -> Path:
    """Refuse absolute paths, parent traversal and symlinks escaping the study."""
    root = root.resolve()
    if Path(relative).is_absolute() or "\\" in relative or ":" in relative:
        raise PublicInputError(f"unsafe public path: {relative!r}")
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise PublicInputError(f"path leaves the public study: {relative!r}")
    return target


def _cache_path(relative: str) -> bool:
    return any(p in {"__pycache__", ".pytest_cache"} for p in Path(relative).parts)


def verify_manifest(root: Path) -> dict[str, str]:
    """Verify every listed byte and reject missing or unlisted scientific files."""
    root = Path(root).resolve()
    manifest_rel = "provenance/EXPORT_MANIFEST.sha256"
    manifest = root / manifest_rel
    if not manifest.is_file():
        raise PublicInputError("public EXPORT_MANIFEST.sha256 is missing")
    listed = {}
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise PublicInputError(f"malformed manifest line {lineno}") from exc
        digest = digest.lower()
        if not SHA_RE.fullmatch(digest) or relative in listed or relative == manifest_rel:
            raise PublicInputError(f"invalid/duplicate manifest entry at line {lineno}")
        target = contained(root, relative)
        if not target.is_file() or sha256(target) != digest:
            raise PublicInputError(f"public byte hash mismatch or missing file: {relative}")
        listed[relative] = digest
    if not listed:
        raise PublicInputError("public manifest is empty")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() != manifest_rel
        and not _cache_path(p.relative_to(root).as_posix())
    }
    if actual != set(listed):
        raise PublicInputError(f"unlisted/missing public files: {sorted(actual ^ set(listed))[:8]}")
    return listed


def read_derivatives(root: Path, manifest: dict[str, str]) -> dict[str, dict]:
    path = root / "provenance/DERIVATIVES_STAGE2.csv"
    if not path.is_file():
        raise PublicInputError("DERIVATIVES_STAGE2.csv is missing")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        relative = row.get("public_path", "")
        if relative not in manifest or relative in out:
            raise PublicInputError(f"unlisted/duplicate derivative: {relative!r}")
        if row.get("public_sha256", "").lower() != manifest[relative]:
            raise PublicInputError(f"derivative public hash differs: {relative}")
        if not SHA_RE.fullmatch(row.get("original_sha256", "").lower()):
            raise PublicInputError(f"derivative lacks original digest: {relative}")
        preservation = str(row.get("numeric_preserved", ""))
        if not (preservation.lower() in {"true", "yes", "1"} or preservation == "BYTE_IDENTICAL"
                or preservation.startswith("ALL_") or preservation.startswith("TEXT_IDENTIFIER_RULES_ONLY")):
            raise PublicInputError(f"derivative numerical preservation not asserted: {relative}")
        out[relative] = row
    return out


def original_digest(relative: str, manifest: dict, derivatives: dict) -> str:
    if relative not in manifest:
        raise PublicInputError(f"required public file missing: {relative}")
    return derivatives.get(relative, {}).get("original_sha256", manifest[relative]).lower()


def verify_kernels(root: Path, manifest: dict) -> None:
    for name, expected in KERNEL_SHA.items():
        relative = f"code/src/ncclenergy/v2/{name}.py"
        if manifest.get(relative) != expected:
            raise PublicInputError(f"frozen numerical kernel changed: {relative}")
    bindings_path = root / "provenance/SOURCE_BINDINGS.json"
    if not bindings_path.is_file():
        raise PublicInputError("SOURCE_BINDINGS.json is missing")
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    bindings = bindings.get("files", bindings)
    if not isinstance(bindings, dict):
        raise PublicInputError("source bindings must be a path-to-SHA mapping")
    for relative, value in bindings.items():
        expected = value.get("sha256") if isinstance(value, dict) else value
        if manifest.get(relative) != expected:
            raise PublicInputError(f"source binding changed: {relative}")
    for relative in manifest:
        if relative.startswith("code/src/") and relative.endswith(".py") and relative not in bindings:
            raise PublicInputError(f"Python source lacks an original binding: {relative}")


def _load_kernels(root: Path):
    src = (root / "code/src").resolve()
    for name, module in list(sys.modules.items()):
        if name == "ncclenergy" or name.startswith("ncclenergy."):
            existing = getattr(module, "__file__", None)
            if existing and src not in Path(existing).resolve().parents:
                raise PublicInputError("another ncclenergy installation is already imported; use a fresh process")
    sys.path.insert(0, str(src))
    from ncclenergy.v2 import stage2_heldout as heldout
    return heldout


def read_public_inputs(root: Path, manifest: dict, derivatives: dict, heldout) -> dict:
    import yaml

    cal_rel = "prep/calib1/run/calibration_analysis.json"
    if original_digest(cal_rel, manifest, derivatives) != heldout.ANCHORS["calibration_analysis_sha256"]:
        raise PublicInputError("calibration is not a derivative of the frozen analysis")
    calibration = json.loads(contained(root, cal_rel).read_text(encoding="utf-8"))
    calr = calibration.get("result") or {}
    cal_cfg = yaml.safe_load((root / "code/configs/stage2/calib01.yaml").read_text(encoding="utf-8"))
    configs, freezes = {}, {}
    for session in SESSIONS:
        cfg_rel = f"code/configs/stage2/heldout_s{session}.yaml"
        draft_rel = f"code/configs/stage2/heldout_s{session}_DRAFT.yaml"
        freeze_rel = f"protocol/FREEZE_H_S{session}.json"
        cfg = yaml.safe_load(contained(root, cfg_rel).read_text(encoding="utf-8"))
        draft = yaml.safe_load(contained(root, draft_rel).read_text(encoding="utf-8"))
        freeze = json.loads(contained(root, freeze_rel).read_text(encoding="utf-8"))
        if original_digest(cfg_rel, manifest, derivatives) != CONFIG_ORIGINAL_SHA[session]:
            raise PublicInputError(f"session {session}: not a derivative of the executed config")
        if original_digest(draft_rel, manifest, derivatives) != heldout.ANCHORS["draft_sha256"][session]:
            raise PublicInputError(f"session {session}: schedule differs from the frozen draft")
        if (freeze.get("kind"), freeze.get("session"), freeze.get("schema")) != (
            "Freeze-H", session, heldout.FREEZE_SCHEMA
        ) or not freeze.get("approved_by_user"):
            raise PublicInputError(f"session {session}: historical freeze view lacks identity/approval")
        roles, files = freeze.get("roles") or {}, freeze.get("files") or {}
        if roles.get("config") != cfg_rel.removeprefix("code/"):
            raise PublicInputError(f"session {session}: historical config role differs")
        if (files.get(roles.get("config")) or {}).get("sha256") != CONFIG_ORIGINAL_SHA[session]:
            raise PublicInputError(f"session {session}: original freeze config digest differs")
        if (freeze.get("parent") or {}).get("sha256") != heldout.ANCHORS["parent_freeze_c_sha256"]:
            raise PublicInputError(f"session {session}: original parent digest differs")
        problems = heldout._config_problems(cfg, draft, session, calr, cal_cfg, heldout.ANCHORS)
        if problems or cfg.get("allocation") != GPUS or cfg.get("campaign") != CAMPAIGN:
            raise PublicInputError(f"session {session}: config identity/scope/count mismatch: {problems}")
        if configs and (cfg.get("counts"), cfg.get("scope_expected")) != (
            configs[1].get("counts"), configs[1].get("scope_expected")
        ):
            raise PublicInputError(f"session {session}: pair or counts differ between sessions")
        configs[session], freezes[session] = cfg, freeze
    schedule = heldout.schedule_from_configs(configs)
    return {"calibration": calr, "configs": configs, "freezes": freezes, "schedule": schedule}


def load_public_sessions(root: Path, inputs: dict, heldout) -> tuple[list[dict], dict]:
    units, scopes, provenance = [], [], {}
    for session in SESSIONS:
        cfg = inputs["configs"][session]
        directory = root / f"code/data/stage2/v2.5-heldout/session_{session}"
        loaded = heldout.load_session(
            directory, root / "code", session, inputs["schedule"][session],
            cfg["readiness_rules"], GPUS, cfg["counts"], cfg["scope_expected"],
            identity={"implementation_id": IMPLEMENTATION, "method_id": METHOD, "campaign": CAMPAIGN},
        )
        if loaded["config_sha256"] != CONFIG_ORIGINAL_SHA[session]:
            raise PublicInputError(f"session {session}: recorded original config digest differs")
        if loaded["execution_status"] != "COMPLETE" or loaded["stop_reason"] is not None:
            raise PublicInputError(f"session {session}: no complete original execution")
        if len(loaded["units"]) != 24 or any(u["status"] != "USED" for u in loaded["units"]):
            raise PublicInputError(f"session {session}: all 24 accepted original records are required")
        scopes.append(loaded["scope_observed"])
        units += loaded["units"]
        provenance[str(session)] = {k: loaded[k] for k in (
            "execution_status", "stop_reason", "config_sha256", "input_sha256"
        )}
    if any(scope != scopes[0] for scope in scopes):
        raise PublicInputError("public sessions do not retain the same pair and scope")
    missing = {u["launch_id"] for u in units if u["J"]["primary"] is None}
    if missing != EXPECTED_PRIMARY_MISSING:
        raise PublicInputError(f"primary-estimator missingness changed: {sorted(missing)}")
    return units, provenance


def compare_results(actual, expected, path="result") -> dict:
    """Compare every scientific field; skip only the two provenance containers."""
    checked, maximum = 0, 0.0

    def walk(a, b, current):
        nonlocal checked, maximum
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                raise PublicInputError(f"result keys differ at {current}")
            for key in b:
                if current == path and key in {"provenance", "provenance_status"}:
                    continue
                walk(a[key], b[key], f"{current}.{key}")
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                raise PublicInputError(f"result list differs at {current}")
            for i, (aa, bb) in enumerate(zip(a, b)):
                walk(aa, bb, f"{current}[{i}]")
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a):
                raise PublicInputError(f"nonfinite or nonnumerical result at {current}")
            delta = abs(a - b)
            if not math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11):
                raise PublicInputError(f"numerical result changed at {current}: {a!r} vs {b!r}")
            checked += 1
            maximum = max(maximum, delta)
        elif a != b:
            raise PublicInputError(f"scientific result/decision differs at {current}: {a!r} vs {b!r}")

    walk(actual, expected, path)
    return {"numeric_fields_checked": checked, "max_absolute_difference": maximum,
            "relative_tolerance": 1e-11, "absolute_tolerance": 1e-11,
            "scientific_decisions_and_missingness": "UNCHANGED"}


def replay(root: Path) -> tuple[dict, dict]:
    root = Path(root).resolve()
    manifest = verify_manifest(root)
    derivatives = read_derivatives(root, manifest)
    verify_kernels(root, manifest)
    heldout = _load_kernels(root)
    inputs = read_public_inputs(root, manifest, derivatives, heldout)
    units, sessions = load_public_sessions(root, inputs, heldout)
    provenance = {
        "checks": "public export byte hashes, original kernel bindings, derivative ledger, config and run identities",
        "original_freeze_h_verification": "NOT_REPERFORMED: published identifier/path derivatives have different bytes",
        "public_manifest_sha256": sha256(root / "provenance/EXPORT_MANIFEST.sha256"),
        "sessions": sessions,
        "external_binary_plugin_nccl": "NOT_VERIFIED_OFFLINE (original runtime binaries and external library omitted)",
    }
    result = heldout._assemble(units, GPUS, inputs["calibration"], provenance, list(SESSIONS), STATUS)
    reference = json.loads((root / "prep/heldout_final/heldout_analysis.json").read_text(encoding="utf-8"))
    comparison = compare_results(result, reference)
    return result, comparison


def write_outputs(root: Path, output: Path, result: dict, comparison: dict) -> None:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise PublicInputError("write replay outputs outside the immutable study directory")
    if output.exists() and any(output.iterdir()):
        raise PublicInputError("output directory is not empty; choose a fresh output to preserve previous results")
    writer_path = root / "prep/heldout_final/run_final_analysis.py"
    spec = importlib.util.spec_from_file_location("archival_final_writer", writer_path)
    if spec is None or spec.loader is None:
        raise PublicInputError("historical numerical table writer is unavailable")
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    writer.write_outputs(result, output)
    execution = []
    for session in SESSIONS:
        path = root / f"prep/heldout_s{session}/run/execution_qc.tsv"
        lines = path.read_text(encoding="utf-8").splitlines()
        if not execution:
            execution.append("session\t" + lines[0])
        execution.extend(f"{session}\t{line}" for line in lines[1:])
    (output / "execution.tsv").write_text("\n".join(execution) + "\n", encoding="utf-8")
    checked = {}
    for name in ("units.tsv", "exclusions.tsv", "contrasts.tsv", "thermal_control.tsv", "adequacy.tsv", "execution.tsv"):
        original = root / "prep/heldout_final" / name
        regenerated = output / name
        # Universal-newline comparison avoids a Git checkout's LF/CRLF representation difference.
        equal = original.read_text(encoding="utf-8") == regenerated.read_text(encoding="utf-8")
        if not equal:
            raise PublicInputError(f"serialized numerical table differs: {name}")
        checked[name] = "IDENTICAL_TEXT"
    (output / "replay_check.json").write_text(
        json.dumps({"provenance_status": STATUS, "comparison": comparison, "tables": checked}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="verify inputs and reproduce results without writing outputs")
    args = parser.parse_args(argv)
    try:
        result, comparison = replay(args.study)
        if not args.check:
            output = args.output or args.study.resolve().parents[1] / "reproduced/a6000-stage2-v2.5"
            write_outputs(args.study, output, result, comparison)
        print(json.dumps({"provenance_status": STATUS, "unit_status_counts": result["unit_status_counts"],
                          "comparison": comparison, "physical_adequacy": result["physical_adequacy"]["status"]}, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Public replay refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
