"""Presentation regressions: preserve retained estimates, scope and missingness."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

STUDY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage2_readme_media", STUDY / "scripts/make_readme_media.py")
MEDIA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MEDIA)


class ReadmeMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = MEDIA.build_payload(STUDY)
        cls.analysis = json.loads((STUDY / MEDIA.INPUTS["analysis"]).read_text(encoding="utf-8"))

    def test_counts_do_not_turn_launches_into_independent_trials(self):
        self.assertEqual(self.payload["counts"], {
            "smoke": 3, "calibration": 23, "heldout": 72, "primary_available": 70, "sessions": 3})

    def test_missing_primary_remains_missing(self):
        self.assertEqual(self.payload["missing_primary"], ["s2-b2-p08", "s3-b2-p11"])

    def test_display_transform_is_not_log_percent(self):
        self.assertAlmostEqual(MEDIA.percent(math.log(1.05)), 5.0)
        self.assertEqual(MEDIA.percent(0), 0)

    def test_nonfinite_values_are_refused(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MEDIA.percent(value)

    def test_estimates_and_intervals_are_copied_not_refitted(self):
        for row in self.payload["selected"]:
            source = self.analysis["estimators"]["primary"]["summaries"][row["family"]][row["member"]]
            self.assertEqual(row["mean_pct"], 100 * math.expm1(source["mean"]))
            self.assertEqual(row["ci95_pct"], [100 * math.expm1(v) for v in source["ci95"]])
            self.assertEqual(row["n_sessions"], 3)
            self.assertEqual(row["df"], 2)

    def test_session_values_retain_order_and_model_unit(self):
        for row in self.payload["selected"]:
            summary = self.analysis["estimators"]["primary"]["summaries"][row["family"]][row["member"]]
            self.assertEqual(row["session_pct"], [100 * math.expm1(summary["session_means"][str(i)]) for i in (1, 2, 3)])

    def test_thermal_and_precision_failures_are_visible(self):
        self.assertEqual(self.payload["selected"][1]["verdict"], "THERMAL_COMPARABILITY_NOT_ESTABLISHED")
        self.assertEqual(self.payload["selected"][2]["verdict"], "INSUFFICIENT_PRECISION")
        self.assertGreater(self.payload["precision_half_width_log_pct"], self.payload["precision_target_log_pct"])

    def test_duration_panel_covers_all_arms_and_estimators(self):
        self.assertEqual(set(self.payload["duration"]), set(MEDIA.ESTIMATORS))
        for estimator in MEDIA.ESTIMATORS:
            self.assertEqual(set(self.payload["duration"][estimator]), {"A", "B", "C"})
            for arm in ("A", "B", "C"):
                rows = self.payload["duration"][estimator][arm]
                self.assertEqual([r["member"] for r in rows], [f"{arm}|2.5", f"{arm}|10", f"{arm}|20"])
                for row in rows:
                    source = self.analysis["estimators"][estimator]["summaries"]["HD1"][row["member"]]
                    self.assertEqual(row["mean_pct"], MEDIA.percent(source["mean"]))
                    self.assertEqual(row["ci95_pct"], [MEDIA.percent(v) for v in source["ci95"]])

    def test_execution_control_is_not_mislabeled(self):
        self.assertEqual(self.payload["execution"]["A"], self.payload["execution"]["B"])
        self.assertNotEqual(self.payload["execution"]["A"], self.payload["execution"]["C"])

    def test_media_ledger_matches_current_sources_and_values(self):
        retained = json.loads((STUDY / "media/media_values.json").read_text(encoding="utf-8"))
        self.assertEqual(retained, self.payload)
        for relative, digest in retained["source_sha256"].items():
            self.assertEqual(hashlib.sha256((STUDY / relative).read_bytes()).hexdigest(), digest)

    def test_new_gifs_are_not_copies_of_historical_gifs(self):
        repo = STUDY.parents[1]
        old = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo / "docs/media").glob("*.gif")}
        for name in ("heldout_comparisons", "estimator_duration"):
            digest = hashlib.sha256((STUDY / "media" / f"{name}.gif").read_bytes()).hexdigest()
            self.assertNotIn(digest, old)

    def test_gifs_have_discrete_frames_and_static_alternatives(self):
        from PIL import Image
        for name in ("heldout_comparisons", "estimator_duration"):
            with Image.open(STUDY / "media" / f"{name}.gif") as animation:
                self.assertEqual(animation.n_frames, 6)
                self.assertGreaterEqual(animation.width, 1200)
                with Image.open(STUDY / "media" / f"{name}.png") as poster:
                    self.assertEqual(animation.size, poster.size)
                for index in range(animation.n_frames):
                    animation.seek(index)
                    self.assertGreaterEqual(animation.info["duration"], 2000)


if __name__ == "__main__":
    unittest.main()
