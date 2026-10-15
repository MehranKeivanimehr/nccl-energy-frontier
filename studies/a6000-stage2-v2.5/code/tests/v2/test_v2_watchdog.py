"""CPU-only tests of the active watchdog (no NVML: a stand-in sampler object provides the telemetry)."""

import os
import sys
import tempfile
import time
from pathlib import Path

from ncclenergy.v2.procctl import OwnedProcess, _stat
from ncclenergy.v2.watchdog import DiskFloorCheck, SamplerHealthCheck, WallClockCheck, Watchdog

HELP = Path(__file__).resolve().parent / "helpers"
(Path(__file__).resolve().parents[2] / "logs").mkdir(exist_ok=True)   # git-ignored; absent in a fresh checkout


class FakeSampler:
    """Same attributes the watchdog reads from telemetry.Sampler. Row layout = telemetry.FIELDS."""

    def __init__(self, gpus, temp=40, ce=0, pids=None, period=0.05, stale=False):
        self.rows, self.proc_events, self.errors, self.truncated, self.proc_period_s = [], [], [], False, 1.0
        now = time.perf_counter() - (10.0 if stale else 0.0)
        for g in gpus:
            self.rows.append((now, g, 100.0, temp, 1900, 8000, 50, 10, 0, ce, 300000, 1.0, 0.3, 100.0))
            self.proc_events.append((now, g, list((pids or {}).get(g, []))))


def _check(s, gpus, owned=frozenset({1234}), **kw):
    return SamplerHealthCheck(s, gpus, lambda: set(owned), temp_stop_c=80, startup_s=0.0, **kw)


def test_healthy_sampler_does_not_trip():
    assert _check(FakeSampler([4, 5], pids={4: [1234]}), [4, 5])() is None


def test_each_condition_trips_with_a_reason():
    assert _check(FakeSampler([4], temp=80), [4])().startswith("TEMPERATURE:GPU4")
    assert _check(FakeSampler([4], ce=0x40), [4])().startswith("HEALTH_CLOCK_EVENT:GPU4:0x40")      # HW thermal
    assert _check(FakeSampler([4], ce=0x80), [4])().startswith("HEALTH_CLOCK_EVENT")                  # power brake
    assert _check(FakeSampler([4], ce=0x4), [4])() is None             # SW power cap alone is not a health event
    assert _check(FakeSampler([4], pids={4: [1234, 999]}), [4])() == "FOREIGN_PROCESS:GPU4:[999]"
    assert _check(FakeSampler([4], stale=True), [4])().startswith("TELEMETRY_STALE")
    assert _check(FakeSampler([4, 5]), [4, 5, 6])().startswith("TELEMETRY_STALE:GPU6")             # GPU never sampled
    s = FakeSampler([4])
    c = _check(s, [4])
    s.errors += [(0, 4, "x")] * 3
    assert c().startswith("TELEMETRY_ERRORS")
    s2 = FakeSampler([4])
    s2.truncated = True
    assert _check(s2, [4])() == "TELEMETRY_TRUNCATED"


def test_disk_and_wallclock_checks():
    assert DiskFloorCheck(Path("/"), 1e9)().startswith("LOW_DISK")
    assert DiskFloorCheck(Path("/"), 0.0)() is None
    w = WallClockCheck(0.0)
    time.sleep(0.01)
    assert w() == "WALL_TIME_EXCEEDED"


def test_failing_check_trips_fail_closed():
    def broken():
        raise RuntimeError("telemetry library crashed")
    wd = Watchdog([broken], period_s=0.05).start()
    time.sleep(0.3)
    wd.stop()
    assert wd.tripped.startswith("CHECK_FAILED")


def test_watchdog_foreign_process_cancels_owned_run():
    import threading
    s = FakeSampler([4])
    with tempfile.TemporaryDirectory() as tmp:
        pf = f"{tmp}/pids"
        op = OwnedProcess([sys.executable, str(HELP / "proc_tree.py"), pf, "sleep"], timeout_s=30, grace_s=1.0)
        wd = op.watchdog = Watchdog([SamplerHealthCheck(s, [4], lambda: op.owned_pids(), 80, startup_s=0.0,
                                                        stale_s=60)], period_s=0.2)
        wd.start()
        # a pid we do not own appears on the allocated GPU after the owned tree is running
        threading.Timer(1.2, lambda: s.proc_events.append((time.perf_counter(), 4, [424242]))).start()
        res = op.run()
        wd.stop()
        pids = [int(x) for x in open(pf).read().split()]
        assert res["stop_reason"].startswith("WATCHDOG:FOREIGN_PROCESS") and res["cleanup"]["ok"]
        assert all(_stat(p) is None or _stat(p)[3] == "Z" for p in pids)


def test_runner_run_process_wall_limit_and_persisted_cleanup_report():
    """runner.run_process with a per-unit wall-clock limit (gate 3a S3 option): our own watchdog cancels the run,
    the owned tree is cleaned and the report is written to owned_process.json (CPU only, fake sampler)."""
    import json
    from ncclenergy.v2.runner import REPO, run_process, safety_problem
    s = FakeSampler([4])
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as tmp:
        d = Path(tmp) / "unit"
        res = run_process([sys.executable, str(HELP / "proc_tree.py"), str(Path(tmp) / "pids"), "sleep"], None, d, 60,
                          watch={"gpus": [4], "sampler": s, "temp_stop_c": 80, "disk_floor_gb": None,
                                 "unit_wall_limit_s": 0.5})
        rep = json.loads((d / "owned_process.json").read_text())
        pids = [int(x) for x in open(Path(tmp) / "pids").read().split()]
        assert res["stop_reason"] == "WATCHDOG:WALL_TIME_EXCEEDED" and safety_problem(res).startswith("WATCHDOG")
        assert rep["cleanup"]["ok"] and rep["watchdog_trip"] == "WALL_TIME_EXCEEDED"
        assert all(_stat(p) is None or _stat(p)[3] == "Z" for p in pids)


def test_exited_owned_process_still_listed_by_nvml_is_not_foreign():
    """Regression for the gate 3a smoke stop (2026-10-09): NVML's process list still showed our benchmark pid for up
    to a sampling period after it exited; the ownership check counted only live processes and tripped
    FOREIGN_PROCESS on our own pid. A pid stays ours after exit unless it is recycled by another process."""
    with tempfile.TemporaryDirectory() as tmp:
        op = OwnedProcess([sys.executable, "-c", "import time; time.sleep(0.6)"], timeout_s=30)
        res = op.run()                                         # child has exited and been reaped
        child = res["pid"]
        assert child in op.tracked and child not in op.owned_pids()
        s = FakeSampler([5], pids={5: [child]})                 # stale NVML list still shows it
        chk = SamplerHealthCheck(s, [5], lambda: op.ever_owned_pids(), 80, startup_s=0.0, stale_s=60)
        assert chk() is None
        op.tracked[424242] = -1                                 # a recycled pid (start time differs) is not ours
        s.proc_events.append((time.perf_counter(), 5, [424242]))
        if _stat(424242) is not None:
            assert chk() == "FOREIGN_PROCESS:GPU5:[424242]"


def test_trip_after_exit_is_a_safety_problem():
    from ncclenergy.v2.runner import safety_problem
    ok = {"stop_reason": None, "cleanup": {"ok": True}, "watchdog_trip": None}
    assert safety_problem(ok) is None
    late = dict(ok, watchdog_trip="FOREIGN_PROCESS:GPU5:[999]")
    assert safety_problem(late) == "WATCHDOG_AFTER_EXIT:FOREIGN_PROCESS:GPU5:[999]"


def test_runner_watch_uses_ever_owned_for_foreign_check():
    """run_process with a fake sampler whose process list keeps showing the child after it exits: no trip."""
    import json
    from ncclenergy.v2.runner import REPO, run_process, safety_problem

    class EchoSampler(FakeSampler):
        """Lists whatever pid the test registers, like NVML listing our benchmark."""

    s = EchoSampler([5])
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as tmp:
        d = Path(tmp) / "unit"
        import threading
        stop = threading.Event()

        def mirror():                                           # NVML-like: list the child pid, keep it after exit
            while not stop.is_set():
                pids = []
                for p in Path("/proc").iterdir():          # only our own child carrying the marker
                    try:
                        if (p.name.isdigit() and _stat(int(p.name)) and _stat(int(p.name))[0] == os.getpid()
                                and b"nccle-mirror-marker" in (p / "cmdline").read_bytes()):
                            pids.append(int(p.name))
                    except OSError:
                        pass
                if pids:
                    mirror.seen = pids
                if getattr(mirror, "seen", None):
                    s.proc_events.append((time.perf_counter(), 5, list(mirror.seen)))
                s.rows.append((time.perf_counter(), 5, 100.0, 40, 1900, 8000, 50, 10, 0, 0, 300000, 1.0, 0.3, 100.0))
                time.sleep(0.05)

        th = threading.Thread(target=mirror, daemon=True)
        th.start()
        res = run_process([sys.executable, "-c", "import time; time.sleep(2.5)", "nccle-mirror-marker"], None, d, 60,
                          watch={"gpus": [5], "sampler": s, "temp_stop_c": 80, "disk_floor_gb": None})
        time.sleep(1.2)                                         # the stale listing persists after exit
        stop.set()
        th.join()
        rep = json.loads((d / "owned_process.json").read_text())
        assert getattr(mirror, "seen", None), "mirror never saw the child"
        assert res["stop_reason"] is None and rep["watchdog_trip"] is None and safety_problem(res) is None, rep


def test_unknown_process_list_trips():
    """v2.2 release audit: a failed process-list query (recorded as None) must not read as "no foreign process"."""
    s = FakeSampler([4, 5], pids={4: [1234]})
    s.proc_events.append((time.perf_counter(), 5, None))
    assert _check(s, [4, 5])() == "PROCESS_LIST_UNAVAILABLE:GPU5"
