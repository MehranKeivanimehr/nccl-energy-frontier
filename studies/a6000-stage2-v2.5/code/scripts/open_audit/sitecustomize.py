"""Release reproduction only: record every file a Python analysis process opens (sys.addaudithook).

Active only when NCCLE_OPEN_AUDIT is set; each process writes <NCCLE_OPEN_AUDIT>.<pid> at exit with one
"R|W<TAB>absolute path" line per file (W if it was ever opened for writing). Changes no analysis behaviour.
"""

import atexit
import os
import sys

_LOG = os.environ.get("NCCLE_OPEN_AUDIT")

if _LOG:
    _seen: dict[str, bool] = {}
    _WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def _hook(event, args):
        if event != "open":
            return
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if path is None or isinstance(path, int):
            return
        try:
            p = os.path.abspath(os.fsdecode(path))
        except Exception:
            return
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
            (mode is None and isinstance(flags, int) and bool(flags & _WFLAGS))
        _seen[p] = _seen.get(p, False) or w

    sys.addaudithook(_hook)

    def _dump():
        out = f"{_LOG}.{os.getpid()}"
        _seen.pop(os.path.abspath(out), None)
        with open(out, "w") as f:
            for p, w in sorted(_seen.items()):
                f.write(f"{'W' if w else 'R'}\t{p}\n")

    atexit.register(_dump)
