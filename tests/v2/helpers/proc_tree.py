"""CPU-only process tree for cleanup tests (no GPU, no NVML). Mimics torchrun's layout:
a grandchild in the launcher's process group and a grandchild in its own session (start_new_session).

    python proc_tree.py <pidfile> <mode>
mode: sleep   -> launcher sleeps (killed by the controller)
      orphan  -> launcher exits at once, leaving both grandchildren running (re-parented)
      stubborn-> grandchildren ignore SIGTERM (needs SIGKILL)
Writes "launcher gc_same_group gc_new_session" pids to <pidfile>.
"""
import os
import signal
import subprocess
import sys
import time

pidfile, mode = sys.argv[1], sys.argv[2]
code = "import signal,time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if mode == "stubborn" else "") \
       + "time.sleep(120)\n"
a = subprocess.Popen([sys.executable, "-c", code])
b = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
time.sleep(0.3)
with open(pidfile + ".tmp", "w") as f:
    f.write(f"{os.getpid()} {a.pid} {b.pid}\n")
os.replace(pidfile + ".tmp", pidfile)
if mode == "orphan":
    os._exit(0)
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(120)
