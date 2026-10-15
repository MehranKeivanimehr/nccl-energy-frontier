"""CPU-only checks of public packaging guards; frozen inference rules are not changed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce_public.py"
spec = importlib.util.spec_from_file_location("public_replay", SCRIPT)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class PublicReplayGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "provenance").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, files):
        (self.root / "provenance/EXPORT_MANIFEST.sha256").write_text(
            "".join(f"{R.sha256(self.root / name)}  {name}\n" for name in files), encoding="utf-8"
        )

    def test_manifest_missing_refused(self):
        with self.assertRaisesRegex(R.PublicInputError, "missing"):
            R.verify_manifest(self.root)

    def test_manifest_detects_tampering(self):
        (self.root / "record.json").write_text("{}")
        self.manifest(["record.json"])
        (self.root / "record.json").write_text('{"value":1}')
        with self.assertRaisesRegex(R.PublicInputError, "mismatch"):
            R.verify_manifest(self.root)

    def test_manifest_detects_missing_record(self):
        (self.root / "record.json").write_text("{}")
        self.manifest(["record.json"])
        (self.root / "record.json").unlink()
        with self.assertRaisesRegex(R.PublicInputError, "missing"):
            R.verify_manifest(self.root)

    def test_unlisted_file_refused(self):
        (self.root / "record.json").write_text("{}")
        self.manifest(["record.json"])
        (self.root / "replacement.json").write_text("{}")
        with self.assertRaisesRegex(R.PublicInputError, "unlisted"):
            R.verify_manifest(self.root)

    def test_cache_not_scientific_record(self):
        (self.root / "record.json").write_text("{}")
        self.manifest(["record.json"])
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__/temporary.pyc").write_bytes(b"cache")
        self.assertEqual(set(R.verify_manifest(self.root)), {"record.json"})

    def test_path_traversal_refused(self):
        for value in ("../outside", "C:/private", "/absolute", "..\\outside"):
            with self.subTest(value=value), self.assertRaises(R.PublicInputError):
                R.contained(self.root, value)

    def test_duplicate_manifest_refused(self):
        (self.root / "record.json").write_text("{}")
        self.manifest(["record.json", "record.json"])
        with self.assertRaisesRegex(R.PublicInputError, "duplicate"):
            R.verify_manifest(self.root)

    def test_changed_kernel_refused_even_with_rebuilt_manifest(self):
        with self.assertRaisesRegex(R.PublicInputError, "kernel changed"):
            R.verify_kernels(self.root, {"code/src/ncclenergy/v2/stage2_heldout.py": "0" * 64})

    def test_provenance_is_not_falsely_identical(self):
        actual = {"provenance_status": R.STATUS, "provenance": {"public": True}, "J": 1.5, "verdict": "INCONCLUSIVE"}
        expected = {"provenance_status": "VERIFIED_OFFLINE", "provenance": {"original": True}, "J": 1.5, "verdict": "INCONCLUSIVE"}
        result = R.compare_results(actual, expected)
        self.assertEqual(result["numeric_fields_checked"], 1)
        self.assertEqual(actual["provenance_status"], "PUBLIC_DERIVATIVE_REPLAY")

    def test_scientific_decision_cannot_change(self):
        with self.assertRaisesRegex(R.PublicInputError, "decision differs"):
            R.compare_results({"verdict": "EQUIVALENT"}, {"verdict": "INCONCLUSIVE"})

    def test_numerical_change_refused(self):
        with self.assertRaisesRegex(R.PublicInputError, "numerical result changed"):
            R.compare_results({"J": 1.012}, {"J": 1.013})

    def test_missingness_cannot_be_filled_in(self):
        with self.assertRaises(R.PublicInputError):
            R.compare_results({"J": 1.01}, {"J": None})

    def test_nonfinite_value_refused(self):
        with self.assertRaisesRegex(R.PublicInputError, "nonfinite"):
            R.compare_results({"J": float("nan")}, {"J": 1.0})

    def calibration_helper(self):
        sys.modules["reproduce_public"] = R
        spec = importlib.util.spec_from_file_location("public_calibration", SCRIPT.parent / "reproduce_calibration.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.compare_calibration

    def test_frozen_infinite_wait_diagnostic_is_preserved(self):
        obj = {"q3_readiness": {"tried": [{"q90_time_to_full_gate_s": float("inf")}]}, "energy_J": 2.0}
        result = self.calibration_helper()(obj, obj)
        self.assertEqual(len(result["identical_positive_infinite_readiness_diagnostics"]), 1)
        self.assertEqual(obj["q3_readiness"]["tried"][0]["q90_time_to_full_gate_s"], float("inf"))

    def test_changed_infinite_wait_diagnostic_is_refused(self):
        expected = {"q3_readiness": {"tried": [{"q90_time_to_full_gate_s": float("inf")}]}}
        actual = {"q3_readiness": {"tried": [{"q90_time_to_full_gate_s": 0.0}]}}
        with self.assertRaises(R.PublicInputError):
            self.calibration_helper()(actual, expected)

    def test_nonfinite_energy_is_not_readiness_diagnostic(self):
        with self.assertRaises(R.PublicInputError):
            self.calibration_helper()({"energy_J": float("inf")}, {"energy_J": float("inf")})

    def test_output_cannot_overwrite_immutable_inputs(self):
        with self.assertRaisesRegex(R.PublicInputError, "immutable"):
            R.write_outputs(self.root, self.root / "results/new", {}, {})

    def test_nonempty_output_refused(self):
        other = self.root.parent / (self.root.name + "_output")
        other.mkdir()
        (other / "prior.txt").write_text("preserved")
        try:
            with self.assertRaisesRegex(R.PublicInputError, "not empty"):
                R.write_outputs(self.root, other, {}, {})
        finally:
            (other / "prior.txt").unlink()
            other.rmdir()


if __name__ == "__main__":
    unittest.main()
