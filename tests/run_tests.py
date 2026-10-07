"""Minimal test runner (no pytest dependency): python tests/run_tests.py"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_parse  # noqa: E402

failed = 0
for name in sorted(n for n in dir(test_parse) if n.startswith("test_")):
    try:
        getattr(test_parse, name)()
        print(f"PASS {name}")
    except Exception:  # noqa: BLE001
        failed += 1
        print(f"FAIL {name}")
        traceback.print_exc()
print(f"{failed} failed")
sys.exit(1 if failed else 0)
