"""Test-only NVML call audit (not used by production runs). Put this directory first on PYTHONPATH and set
NCCLE_NVML_AUDIT_LOG=<path prefix>: every pynvml.nvml* call made by this Python process is recorded as
(monotonic ns, function, thread) and written to <prefix>.<pid>.jsonl at interpreter exit. Read-only: it changes no
NVML behaviour and adds no NVML call of its own. Monotonic ns is the clock of the nccl-tests region records.
"""
import os

if os.environ.get("NCCLE_NVML_AUDIT_LOG"):
    import atexit
    import json
    import threading
    import time

    import pynvml as _N

    _calls = []

    def _wrap(name, fn):
        def w(*a, **k):
            _calls.append((time.monotonic_ns(), name, threading.current_thread().name))
            return fn(*a, **k)
        w.__name__ = name
        return w

    for _n in dir(_N):
        if _n.startswith("nvml") and callable(getattr(_N, _n)):
            setattr(_N, _n, _wrap(_n, getattr(_N, _n)))

    def _dump():
        with open(f"{os.environ['NCCLE_NVML_AUDIT_LOG']}.{os.getpid()}.jsonl", "w") as f:
            for t, n, th in _calls:
                f.write(json.dumps({"t_ns": t, "fn": n, "thread": th}) + "\n")

    atexit.register(_dump)
