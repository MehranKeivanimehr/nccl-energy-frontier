"""Owned-process control for experiment launches (methodology v2.2).

Every experiment command is started in its own session, so its process group id
equals its pid. That is not enough on its own: torchrun starts each worker in a
new session as well (torch/distributed/elastic/.../subprocess_handler.py), so a
signal to the launcher's group never reaches the GPU workers. This module
therefore tracks every descendant of the controller by (pid, start time, uid).
The controller is made a child subreaper when the kernel allows it, so orphaned
descendants are re-parented to the controller instead of init and stay visible.

Guarantees (tested with CPU-only processes in tests/v2/test_v2_procctl.py):
* On normal exit, timeout, watchdog trip, SIGINT, SIGTERM, SIGHUP or any
  exception, all owned processes get SIGTERM, then SIGKILL after a bounded grace
  period, and are reaped.
* Only owned processes are signalled. A pid is signalled only if it was
  recorded as a descendant of this controller and its start time and uid still
  match the record, so a recycled pid is never touched. The process-group
  signal is only sent while the group leader (our direct child) is unreaped.
* A failure inside one cleanup step does not skip the later steps.
* Survivors after cleanup are reported (cleanup["survivors"]); callers must treat
  that, a watchdog trip and a controller signal as a campaign stop, never retry.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PR_SET_CHILD_SUBREAPER = 36
_CLK_TCK = os.sysconf("SC_CLK_TCK")


class ControllerSignal(BaseException):
    """Raised in the controller's main thread when it receives SIGINT/SIGTERM/SIGHUP."""

    def __init__(self, signum: int):
        super().__init__(f"controller received {signal.Signals(signum).name}")
        self.signum = signum


class SafetyStop(Exception):
    """A safety condition (watchdog trip, cleanup failure) that must stop the campaign."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


def set_child_subreaper() -> bool:
    """Make this process the reaper of its orphaned descendants (Linux >= 3.4). Unprivileged."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    except Exception:  # noqa: BLE001
        return False


def _stat(pid: int):
    """(ppid, pgid, starttime_ticks, state) from /proc/<pid>/stat, or None if gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            s = f.read().decode(errors="replace")
    except OSError:
        return None
    rest = s[s.rfind(")") + 2:].split()
    # fields after comm: state(0) ppid(1) pgrp(2) session(3) ... starttime is field 22 overall -> index 19 here
    return int(rest[1]), int(rest[2]), int(rest[19]), rest[0]


def _uid(pid: int):
    try:
        with open(f"/proc/{pid}/status") as f:
            for ln in f:
                if ln.startswith("Uid:"):
                    return int(ln.split()[1])
    except OSError:
        return None
    return None


def process_table() -> dict[int, tuple]:
    out = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = _stat(int(name))
            if st is not None:
                out[int(name)] = st
    return out


def descendants(root: int, table: dict[int, tuple] | None = None) -> dict[int, int]:
    """{pid: starttime} of every process below root in the current process tree."""
    table = process_table() if table is None else table
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _pg, _st, _s) in table.items():
        kids.setdefault(ppid, []).append(pid)
    out, todo = {}, list(kids.get(root, []))
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out[p] = table[p][2]
        todo += kids.get(p, [])
    return out


def _alive_same(pid: int, starttime: int) -> bool:
    st = _stat(pid)
    return st is not None and st[2] == starttime and st[3] != "Z"


@contextmanager
def controller_signals(signals=(signal.SIGINT, signal.SIGTERM, signal.SIGHUP)):
    """Turn controller signals into ControllerSignal so cleanup runs. Main thread only; no-op elsewhere."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    prev = {}

    def handler(signum, _frame):
        raise ControllerSignal(signum)

    for s in signals:
        prev[s] = signal.signal(s, handler)
    try:
        yield
    finally:
        for s, h in prev.items():
            signal.signal(s, h)


def _no_core():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class OwnedProcess:
    """Run one command as an owned process tree with timeout, watchdog and guaranteed cleanup."""

    def __init__(self, cmd, env=None, cwd=None, stdout=None, stderr=None, timeout_s: float | None = None,
                 watchdog=None, grace_s: float = 10.0, kill_wait_s: float = 5.0, poll_s: float = 0.1,
                 snapshot_s: float = 0.5):
        self.cmd, self.env, self.cwd = cmd, env, cwd
        self.stdout, self.stderr = stdout, stderr
        self.timeout_s, self.watchdog = timeout_s, watchdog
        self.grace_s, self.kill_wait_s = grace_s, kill_wait_s
        self.poll_s, self.snapshot_s = poll_s, snapshot_s
        self.uid = os.getuid()
        self.tracked: dict[int, int] = {}       # pid -> starttime, cumulative over the run
        self._lock = threading.RLock()            # tracked is updated by the watchdog thread too (snapshot)
        self.p: subprocess.Popen | None = None
        self.subreaper = set_child_subreaper()
        # processes the controller already had before this run are not ours to clean up
        self.preexisting: dict[int, int] = descendants(os.getpid())

    # -- ownership -------------------------------------------------------------------------------
    def snapshot(self) -> None:
        """Record every current descendant of the controller (and of the child)."""
        table = process_table()
        excluded = {p for p, st in self.preexisting.items() if table.get(p, (0, 0, None))[2] == st}
        for p in list(excluded):
            excluded |= set(descendants(p, table))
        found = {}
        for root in (os.getpid(), self.p.pid if self.p else None):
            if root is None:
                continue
            for pid, st in descendants(root, table).items():
                if pid != os.getpid() and pid not in excluded and _uid(pid) == self.uid:
                    found.setdefault(pid, st)
        with self._lock:
            for pid, st in found.items():
                self.tracked.setdefault(pid, st)

    def _tracked_items(self) -> list[tuple[int, int]]:
        with self._lock:
            return list(self.tracked.items())

    def owned_pids(self) -> set[int]:
        """Owned processes that are still running (used for cleanup)."""
        return {p for p, st in self._tracked_items() if _alive_same(p, st)}

    def ever_owned_pids(self) -> set[int]:
        """Every pid this run has owned, including ones that already exited (or are zombies), unless the pid now
        belongs to a different process (start time changed). Used for foreign-process checks: NVML's process list
        can still show an owned process for up to a sampling period after it exits (gate 3a smoke, 2026-10-09)."""
        out = set()
        for p, st in self._tracked_items():
            cur = _stat(p)
            if cur is None or cur[2] == st:
                out.add(p)
        return out

    # -- signals ---------------------------------------------------------------------------------
    def _install_signals(self):
        """Controller signals raise ControllerSignal, except during cleanup, where they are only recorded
        (a second Ctrl-C or SIGTERM must not interrupt the cleanup itself)."""
        self._prev_handlers = {}
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum, _frame):
            if self._in_cleanup:
                self.pending_signals.append(signum)
                return
            raise ControllerSignal(signum)

        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._prev_handlers[s] = signal.signal(s, handler)

    def _restore_signals(self):
        for s, h in self._prev_handlers.items():
            try:
                signal.signal(s, h)
            except Exception:  # noqa: BLE001
                pass

    # -- run -------------------------------------------------------------------------------------
    def run(self) -> dict:
        t0w, t0 = time.time(), time.perf_counter()
        res = {"rc": None, "timed_out": False, "stop_reason": None, "exception": None}
        exc: BaseException | None = None
        self._in_cleanup, self.pending_signals = False, []
        self._install_signals()
        try:
            try:
                self.p = subprocess.Popen(self.cmd, env=self.env, cwd=self.cwd, stdout=self.stdout,
                                          stderr=self.stderr, start_new_session=True, preexec_fn=_no_core)
                next_snap = 0.0
                while True:
                    now = time.perf_counter()
                    if now >= next_snap:
                        self.snapshot()
                        next_snap = now + self.snapshot_s
                    rc = self.p.poll()
                    if rc is not None:
                        res["rc"] = rc
                        break
                    if self.timeout_s is not None and now - t0 > self.timeout_s:
                        res.update(timed_out=True, stop_reason="TIMEOUT")
                        break
                    if self.watchdog is not None and self.watchdog.tripped:
                        res["stop_reason"] = f"WATCHDOG:{self.watchdog.tripped}"
                        break
                    time.sleep(self.poll_s)
            except BaseException as e:  # noqa: BLE001 - includes ControllerSignal and KeyboardInterrupt
                exc = e
                res["stop_reason"] = f"CONTROLLER:{type(e).__name__}:{e}"
                res["exception"] = repr(e)
            finally:
                self._in_cleanup = True
                res["cleanup"] = self.cleanup()
                if res["rc"] is None and self.p is not None:
                    res["rc"] = self.p.returncode
        finally:
            self._restore_signals()
        res.update(pid=self.p.pid if self.p else None, utc_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0w)),
                   mono_start=t0, mono_end=time.perf_counter(), wall_s=time.perf_counter() - t0,
                   subreaper=self.subreaper, signals_during_cleanup=list(self.pending_signals))
        self.result = res                      # available to callers even when an exception propagates
        if exc is not None:
            raise exc
        if self.pending_signals:
            raise ControllerSignal(self.pending_signals[0])
        return res

    # -- cleanup ---------------------------------------------------------------------------------
    def _signal_all(self, sig) -> list[str]:
        errs = []
        if self.p is not None and self.p.returncode is None:   # leader unreaped: its pgid cannot be recycled
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                pass
            except Exception as e:  # noqa: BLE001
                errs.append(f"killpg {sig}: {e!r}")
        for pid, st in self._tracked_items():
            try:
                if _alive_same(pid, st) and _uid(pid) == self.uid:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except Exception as e:  # noqa: BLE001
                errs.append(f"kill {pid} {sig}: {e!r}")
        return errs

    def _reap(self) -> None:
        if self.p is not None:
            try:
                self.p.poll()
            except Exception:  # noqa: BLE001
                pass
        for pid, _ in self._tracked_items():
            if self.p is not None and pid == self.p.pid:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)        # only succeeds for our own (re-parented) children
            except ChildProcessError:
                pass
            except Exception:  # noqa: BLE001
                pass

    def _wait_gone(self, limit_s: float) -> set[int]:
        t_end = time.perf_counter() + limit_s
        while True:
            self._reap()
            alive = self.owned_pids()
            leader_alive = self.p is not None and self.p.returncode is None
            if not alive and not leader_alive:
                return set()
            if time.perf_counter() >= t_end:
                return alive | ({self.p.pid} if leader_alive else set())
            time.sleep(0.05)

    def cleanup(self) -> dict:
        """Bounded cleanup of the owned tree. ok only if nothing owned survives AND no step failed (fail closed).
        The snapshot runs even if Popen never returned (a signal between fork and exec): the child is then found as
        a new descendant of the controller. If waiting after SIGTERM fails, SIGKILL is still sent to anything owned
        that is alive."""
        rep = {"errors": [], "sigterm": False, "sigkill": False, "survivors": [], "tracked": 0}
        failed = set()
        for step in ("snapshot", "term", "wait_term", "kill", "wait_kill"):
            try:
                if step == "snapshot":
                    self.snapshot()
                    rep["tracked"] = len(self._tracked_items())
                elif step == "term":
                    if self.owned_pids() or (self.p is not None and self.p.poll() is None):
                        rep["errors"] += self._signal_all(signal.SIGTERM)
                        rep["sigterm"] = True
                elif step == "wait_term":
                    rep["survivors"] = sorted(self._wait_gone(self.grace_s if rep["sigterm"] else 0.0))
                elif step == "kill":
                    if "wait_term" in failed:      # survivors unknown: decide from the live process table
                        rep["survivors"] = sorted(self.owned_pids() | (
                            {self.p.pid} if self.p is not None and self.p.poll() is None else set()))
                    if rep["survivors"]:
                        self.snapshot()   # anything spawned during the grace period is owned too
                        rep["errors"] += self._signal_all(signal.SIGKILL)
                        rep["sigkill"] = True
                elif step == "wait_kill":
                    if rep["sigkill"]:
                        rep["survivors"] = sorted(self._wait_gone(self.kill_wait_s))
            except BaseException as e:  # noqa: BLE001 - a failing step must not skip the remaining ones
                failed.add(step)
                rep["errors"].append(f"{step}: {e!r}")
        rep["ok"] = not rep["survivors"] and not rep["errors"]
        return rep


def run_owned(cmd, env, d: Path, timeout_s: float | None, watchdog=None, grace_s: float = 10.0) -> dict:
    """Run cmd in directory d with stdout/stderr files; returns the OwnedProcess result dict."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "stdout.txt", "wb") as so, open(d / "stderr.txt", "wb") as se:
        return OwnedProcess(cmd, env=env, cwd=d, stdout=so, stderr=se, timeout_s=timeout_s, watchdog=watchdog,
                            grace_s=grace_s).run()
