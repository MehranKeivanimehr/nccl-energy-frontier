"""Replay all 23 retained calibration launches using unchanged frozen kernels."""
from pathlib import Path
import argparse
import copy
import json
import math
import reproduce_public as R


def compare_calibration(actual, expected):
    """Preserve one frozen infinite-wait diagnostic, never allow nonfinite energy."""
    a, b = copy.deepcopy(actual), copy.deepcopy(expected)
    diagnostics = []
    for i, row in enumerate((b.get("q3_readiness") or {}).get("tried") or []):
        value = row.get("q90_time_to_full_gate_s")
        if isinstance(value, float) and math.isinf(value) and value > 0:
            got = a["q3_readiness"]["tried"][i]["q90_time_to_full_gate_s"]
            if got != value:
                raise R.PublicInputError("frozen infinite readiness-wait diagnostic changed")
            # Comparison-only label; original and regenerated artifacts keep Infinity.
            a["q3_readiness"]["tried"][i]["q90_time_to_full_gate_s"] = "INFINITE_WAIT_SENTINEL"
            row["q90_time_to_full_gate_s"] = "INFINITE_WAIT_SENTINEL"
            diagnostics.append(f"q3_readiness/tried/{i}/q90_time_to_full_gate_s")
    comparison = R.compare_results(a, b)
    comparison["identical_positive_infinite_readiness_diagnostics"] = diagnostics
    return comparison


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    study = Path(__file__).resolve().parents[1]
    manifest = R.verify_manifest(study)
    derivatives = R.read_derivatives(study, manifest)
    R.verify_kernels(study, manifest)
    heldout = R._load_kernels(study)
    R.read_public_inputs(study, manifest, derivatives, heldout)
    from ncclenergy.v2 import stage2_calibration as K
    result = K.analyse(study / "code/data/stage2/v2.5-calib-01/calibration", study / "code")
    reference = json.loads((study / "prep/calib1/run/calibration_analysis.json").read_text())["result"]
    comparison = compare_calibration(result, reference)
    output = (args.output or study.parents[1] / "reproduced/stage2-calibration").resolve()
    if output == study or study in output.parents or (output.exists() and any(output.iterdir())):
        raise R.PublicInputError("choose a fresh output outside the immutable study")
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_analysis.json").write_text(json.dumps({"provenance_status":R.STATUS,"result":result},indent=2))
    (output / "replay_check.json").write_text(json.dumps({"provenance_status":R.STATUS,"comparison":comparison},indent=2)+"\n")
    print(json.dumps({"n_units":result["n_units"],"n_accepted":result["n_accepted"],"comparison":comparison},indent=2))


if __name__ == "__main__":
    main()
