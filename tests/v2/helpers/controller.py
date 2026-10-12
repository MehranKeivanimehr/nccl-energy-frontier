"""Controller used by the signal tests: runs proc_tree.py as an OwnedProcess and prints the cleanup report.

    python controller.py <pidfile> <mode> <reportfile>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from ncclenergy.v2.procctl import OwnedProcess  # noqa: E402

pidfile, mode, report = sys.argv[1:4]
op = OwnedProcess([sys.executable, os.path.join(os.path.dirname(__file__), "proc_tree.py"), pidfile, mode],
                  timeout_s=60, grace_s=1.5, kill_wait_s=3.0)
try:
    op.run()
    out = {"result": "returned"}
except BaseException as e:  # noqa: BLE001
    out = {"result": type(e).__name__, "signum": getattr(e, "signum", None)}
out["tracked"] = sorted(op.tracked)
out["cleanup"] = (getattr(op, "result", None) or {}).get("cleanup")
out["signals_during_cleanup"] = (getattr(op, "result", None) or {}).get("signals_during_cleanup")
with open(report, "w") as f:
    json.dump(out, f)
