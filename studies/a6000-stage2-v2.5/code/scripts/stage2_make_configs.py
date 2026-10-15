"""Write the Stage 2 config DRAFTS (configs/stage2/*_DRAFT.yaml) from the seeded schedules. CPU only.

The drafts are deliberately not runnable: allocation, allocation_confirmation and freeze_record are null, so
`python -m ncclenergy.v2.stage2_runner check` refuses them. They are completed only after the user confirms the pair
and the access window and after Freeze-C / Freeze-H.
"""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ncclenergy.v2 import stage2_runner as S  # noqa: E402
from ncclenergy.v2.stage2_schedule import calibration_schedule, heldout_schedule, pause_ns  # noqa: E402

COUNTS = {"A": {"w": 95604, "R": {"2.5": 47802, "10": 191207, "20": 382413, "40": 764825}},
          "B": {"w": 95597, "R": {"2.5": 47799, "10": 191193, "20": 382385, "40": 764769}},
          "C": {"w": 98289, "R": {"2.5": 49145, "10": 196577, "20": 393154, "40": 786308}}}
COMMON = {"implementation_id": S.IMPLEMENTATION_ID, "method_id": S.METHOD_ID, "output_root": S.OUTPUT_ROOT,
          "energy_source": "counter", "max_attempts": 1, "collective": "all_gather", "size_bytes": 2097152,
          "counts": COUNTS, "power_lag_s": 0.75, "sampler_period_s": 0.05,
          "allocation": None, "placements": None, "allocation_confirmation": None, "scope_expected": None,
          "freeze_record": None}


def main():
    out = REPO / "configs/stage2"
    out.mkdir(parents=True, exist_ok=True)
    cal = dict(COMMON, campaign="v2.5-calib-01", mode="STAGE2_CALIBRATION", total_power_samples=True,
               units=calibration_schedule("v2.4-calib-01", m4d=True))
    (out / "calib01_DRAFT.yaml").write_text("# DRAFT - not runnable (no allocation, confirmation or freeze)\n" +
                                            yaml.safe_dump(cal, sort_keys=False))
    smoke_units = []
    for i, (dose, ms) in enumerate((("E0", 0), ("D100", 100), ("D500", 500))):
        lid = f"SMOKE-{i + 1:02d}"
        smoke_units.append({"part": "SMOKE", "launch_id": lid, "arm": "B", "length_s": 2.5, "dose": dose,
                            "dose_period_ms": ms, "poller_active": dose != "E0", "post_probe": True,
                            "pause_ns": pause_ns("v2.5-s2-smoke", lid)})
    sm = dict(COMMON, campaign="v2.5-s2-smoke", mode="STAGE2_SMOKE", total_power_samples=True, units=smoke_units)
    (out / "smoke01_DRAFT.yaml").write_text("# DRAFT - pre-Freeze-C software gate on real GPUs; data never analysed. "
                                            "Not runnable (no allocation, confirmation or Freeze-S)\n" +
                                            yaml.safe_dump(sm, sort_keys=False))
    plan = heldout_schedule()["plan"]
    for s in (1, 2, 3):
        units = [{k: u[k] for k in ("launch_id", "arm", "length_s", "pause_ns", "block", "pos", "triple")}
                 for u in plan if u["session"] == s]
        ho = dict(COMMON, campaign="v2.5-heldout", mode="STAGE2_HELDOUT", session=s, total_power_samples=False,
                  readiness_rules=None, units=units)
        (out / f"heldout_s{s}_DRAFT.yaml").write_text("# DRAFT - not runnable (readiness rules and pair identity are "
                                                      "set at Freeze-H)\n" + yaml.safe_dump(ho, sort_keys=False))


if __name__ == "__main__":
    main()
