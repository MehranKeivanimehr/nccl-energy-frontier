"""CPU-only tests of owned-process cleanup (procctl.OwnedProcess). Only processes created by these tests are
signalled; no GPU, no NVML. Covers normal exit with orphaned grandchildren, timeout, watchdog trip, controller
exception, SIGTERM/SIGINT to the controller, a second signal during cleanup, a failing cleanup step,
SIGTERM-resistant grandchildren, and protection of unrelated processes."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ncclenergy.v2.procctl import OwnedProcess, _stat

HELP = Path(__file__).resolve().parent / "helpers"
PY = sys.executable


def _alive(pid):
    st = _stat(pid)
    return st is not None and st[3] != "Z"


def _pids(pidfile, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        if os.path.exists(pidfile):
            return [int(x) for x in open(pidfile).read().split()]
        time.sleep(0.05)
    raise AssertionError("helper did not write pids")


def _tree(mode, tmp, **kw):
    pf = os.path.join(tmp, f"pids_{mode}")
    op = OwnedProcess([PY, str(HELP / "proc_tree.py"), pf, mode], grace_s=kw.pop("grace_s", 1.5),
                      kill_wait_s=3.0, **kw)
    return op, pf


def test_normal_exit_reaps_orphaned_grandchildren_including_new_session():
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("orphan", tmp, timeout_s=30)
        res = op.run()
        pids = _pids(pf)
        assert res["rc"] == 0 and res["cleanup"]["ok"] and res["cleanup"]["sigterm"]
        assert not any(_alive(p) for p in pids), pids               # new-session grandchild included


def test_timeout_kills_whole_tree():
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("sleep", tmp, timeout_s=1.5)
        res = op.run()
        pids = _pids(pf)
        assert res["timed_out"] and res["stop_reason"] == "TIMEOUT" and res["cleanup"]["ok"]
        assert not any(_alive(p) for p in pids)


def test_sigterm_resistant_tree_is_killed_after_grace():
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("stubborn", tmp, timeout_s=1.0, grace_s=0.8)
        res = op.run()
        pids = _pids(pf)
        assert res["cleanup"]["sigkill"] and res["cleanup"]["ok"]
        assert not any(_alive(p) for p in pids)


class _TripLater:
    def __init__(self, after_s, reason="TEMPERATURE:GPU4:81C>=80C"):
        self.tripped = None
        threading.Timer(after_s, lambda: setattr(self, "tripped", reason)).start()


def test_watchdog_trip_cancels_and_cleans_up():
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("sleep", tmp, timeout_s=30, watchdog=_TripLater(1.0))
        res = op.run()
        pids = _pids(pf)
        assert res["stop_reason"].startswith("WATCHDOG:TEMPERATURE") and res["cleanup"]["ok"]
        assert not any(_alive(p) for p in pids)


class _Exploding:
    """A watchdog whose check raises inside the controller loop."""
    def __init__(self, after_s):
        self.t0, self.after_s = time.time(), after_s

    @property
    def tripped(self):
        if time.time() - self.t0 > self.after_s:
            raise RuntimeError("controller bug")
        return None


def test_controller_exception_still_cleans_up_and_propagates():
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("sleep", tmp, timeout_s=30, watchdog=_Exploding(1.0))
        try:
            op.run()
            raise AssertionError("exception was swallowed")
        except RuntimeError:
            pass
        pids = _pids(pf)
        assert op.result["cleanup"]["ok"] and not any(_alive(p) for p in pids)


def test_failing_cleanup_step_does_not_skip_later_steps():
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("sleep", tmp, timeout_s=1.0)
        orig = op._signal_all

        def broken(sig):
            if sig == signal.SIGTERM:
                raise OSError("simulated failure in the SIGTERM step")
            return orig(sig)

        op._signal_all = broken
        res = op.run()
        pids = _pids(pf)
        assert any("term" in e for e in res["cleanup"]["errors"])
        assert res["cleanup"]["sigkill"] and not any(_alive(p) for p in pids)
        assert res["cleanup"]["ok"] is False          # v2.2 release: a failed cleanup step is never reported ok


def _run_controller(sig, mode, tmp, second=False):
    pf, rep = os.path.join(tmp, "pids"), os.path.join(tmp, "report.json")
    c = subprocess.Popen([PY, str(HELP / "controller.py"), pf, mode, rep])
    pids = _pids(pf)
    time.sleep(0.8)                                   # let the controller snapshot the tree
    c.send_signal(sig)
    if second:
        time.sleep(0.4)                                # during the grace period of the cleanup
        c.send_signal(signal.SIGTERM)
    c.wait(timeout=30)
    return json.load(open(rep)), pids, c.returncode


def test_controller_sigterm_and_ctrl_c_clean_up_owned_tree():
    for sig in (signal.SIGTERM, signal.SIGINT):
        with tempfile.TemporaryDirectory() as tmp:
            out, pids, rc = _run_controller(sig, "sleep", tmp)
            assert out["result"] == "ControllerSignal" and out["signum"] == int(sig), out
            assert out["cleanup"]["ok"] and not any(_alive(p) for p in pids), (sig, pids)


def test_second_signal_during_cleanup_does_not_abort_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        out, pids, rc = _run_controller(signal.SIGTERM, "stubborn", tmp, second=True)
        assert out["result"] == "ControllerSignal" and out["cleanup"]["ok"] and out["cleanup"]["sigkill"]
        assert out["signals_during_cleanup"] == [int(signal.SIGTERM)]
        assert not any(_alive(p) for p in pids)


def test_unrelated_and_recycled_pids_are_never_signalled():
    unrelated = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])   # exists before the owned run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            op, pf = _tree("orphan", tmp, timeout_s=30)
            res = op.run()
            assert res["cleanup"]["ok"] and unrelated.poll() is None and unrelated.pid not in op.tracked
            # a tracked pid whose start time no longer matches (recycled pid) is not signalled
            op.tracked = {unrelated.pid: -1}
            op.p = None
            op._signal_all(signal.SIGKILL)
            time.sleep(0.2)
            assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def test_cleanup_finds_child_when_popen_never_returned():
    """Audit 1A: a controller signal between fork and exec leaves self.p unset; the cleanup snapshot must still find
    and stop the child (a new descendant of the controller)."""
    op = OwnedProcess([PY, "-c", "pass"], grace_s=0.5, kill_wait_s=3.0)
    child = subprocess.Popen(["sleep", "30"])          # stands in for the forked-but-unrecorded child
    try:
        assert op.p is None
        rep = op.cleanup()                              # also reaps it (our direct child)
        assert rep["tracked"] >= 1 and rep["sigterm"] and rep["ok"] and not _alive(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_wait_failure_still_escalates_to_sigkill_and_fails_closed():
    """Audit 1B: if waiting after SIGTERM raises, survivors are re-derived, SIGKILL is sent, and ok is False."""
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("stubborn", tmp, timeout_s=1.0, grace_s=0.5)
        orig, calls = op._wait_gone, []

        def broken(limit):
            calls.append(limit)
            if len(calls) == 1:
                raise RuntimeError("dictionary changed size during iteration (simulated)")
            return orig(limit)

        op._wait_gone = broken
        res = op.run()
        pids = _pids(pf)
        assert res["cleanup"]["sigkill"] and not any(_alive(p) for p in pids)
        assert res["cleanup"]["ok"] is False and any(e.startswith("wait_term") for e in res["cleanup"]["errors"])


def test_concurrent_snapshot_and_iteration_do_not_race():
    """Audit 1B (cause): the watchdog thread snapshots while the main thread iterates the owned set."""
    with tempfile.TemporaryDirectory() as tmp:
        op, pf = _tree("orphan", tmp, timeout_s=None)
        stop, errors = threading.Event(), []

        def snapper():
            while not stop.is_set():
                try:
                    op.snapshot()
                    op.ever_owned_pids()
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))

        th = threading.Thread(target=snapper)
        th.start()
        try:
            for i in range(300):
                with op._lock:
                    op.tracked[10_000_000 + i] = 0          # grow the dict while the other thread iterates it
                op.owned_pids()
        finally:
            stop.set()
            th.join()
        assert not errors, errors[:3]
