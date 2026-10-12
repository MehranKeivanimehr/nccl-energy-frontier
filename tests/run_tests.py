"""Minimal test runner (no pytest dependency): python tests/run_tests.py

Runs every test_* function in tests/test_*.py and tests/v2/test_*.py.
The tests are pytest-compatible as well.
"""
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

failed = total = 0
for path in sorted(list(HERE.glob("test_*.py")) + list((HERE / "v2").glob("test_*.py"))):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        total += 1
        try:
            getattr(mod, name)()
            print(f"PASS {path.relative_to(HERE)}::{name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {path.relative_to(HERE)}::{name}")
            traceback.print_exc()
print(f"{total - failed}/{total} passed, {failed} failed")
sys.exit(1 if failed else 0)
