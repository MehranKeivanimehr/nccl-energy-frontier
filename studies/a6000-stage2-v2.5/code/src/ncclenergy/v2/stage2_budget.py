"""Stage 2 wall-time planning, the unit-start gate and the reservation deadline (CPU only).

Three different numbers, never confused:
  * expected: per-launch components are gate 3b MEDIANS (not bounds) plus design values;
  * cap-based planning estimate: the same medians with every wait at its cap. It is NOT an upper bound, because the
    per-launch components are medians;
  * enforced bound: the approved deadline. A unit (readiness wait included) starts only if its enforced unit bound
    (wait cap + per-unit wall limit + cleanup bound) plus the reserved session tail fits before the deadline. The
    wait cap is enforced by the readiness loop, the wall limit by the watchdog (cancel + owned-tree cleanup), the
    cleanup by procctl (SIGTERM, 10 s, SIGKILL, 5 s, reap). The session tail always reserves cleanup + postflight
    verification; the end baseline runs only if it also fits.
A process that survives SIGKILL is outside our control: it is reported (CLEANUP_FAILED) and the session stops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

MEASURED_MEDIANS = {"init_and_probe_s": 1.30,   # launch -> C0 minus the 0.2 s historical warmup (median 1.50 - 0.20)
                    "end_reads_s": 0.19,        # loop end -> post read done
                    "exit_cleanup_s": 0.65,     # post read -> process reaped (data check, output, exit, reap)
                    "controller_s": 0.70}       # reaped -> next launch (bookkeeping, 0.4 s sampler tail)


@dataclass
class Design:
    prep_target_s: float = 5.0           # target only: w = ceil(5 s / t_op); actual duration is measured
    pause_mean_s: float = 0.05           # M4c requested U[0, 100) ms
    pause_max_s: float = 0.10
    post_probe_s: float = 0.0            # M4d bound per launch (4 nominal periods), calibration only
    post_probe_max_s: float = 0.41
    cal_fixed_wait_s: float = 120.0
    ho_wait_floor_s: float = 30.0        # the 30 s stability window is the minimum wait
    ho_wait_plan_s: float = 120.0        # planning value only; actual known after calibration
    ho_wait_cap_s: float = 240.0         # W_max (safety/availability ceiling, not a scientific criterion)
    p8_wait_expected_s: float = 5.0
    p8_wait_cap_s: float = 60.0
    baseline_settle_s: float = 10.0
    baseline_measure_s: float = 30.0
    preflight_s: float = 120.0
    postflight_verify_s: float = 60.0    # owned-process verification, allocation re-check, post snapshot
    cleanup_bound_s: float = 15.0        # SIGTERM grace 10 s + SIGKILL wait 5 s (only when a run must be stopped)
    calib_regions_s: list = field(default_factory=lambda: [2.5, 2.5] + [2.5, 10, 20, 40] * 3 + [10] * 9)
    ho_regions_s: list = field(default_factory=lambda: [2.5, 10, 20, 40] * 6)       # 2 blocks x 3 arms


def calibration_design(m4d: bool = True) -> Design:
    return Design(post_probe_s=0.4 if m4d else 0.0)


def per_launch_overhead_s(d: Design) -> float:
    m = MEASURED_MEDIANS
    return m["init_and_probe_s"] + d.prep_target_s + d.pause_mean_s + m["end_reads_s"] + d.post_probe_s + \
        m["exit_cleanup_s"] + m["controller_s"]


def unit_wall_limit_s(region_s: float, d: Design) -> float:
    """Per-unit wall limit enforced by the watchdog: 3 x planned + 60 s, planned = init/probe + prep + max pause +
    region + end reads + post-probe bound + exit (medians and design values; the limit itself is enforced)."""
    m = MEASURED_MEDIANS
    planned = m["init_and_probe_s"] + d.prep_target_s + d.pause_max_s + region_s + m["end_reads_s"] + \
        (d.post_probe_max_s if d.post_probe_s else 0.0) + m["exit_cleanup_s"]
    return 3.0 * planned + 60.0


def session_tail_reserved_s(d: Design) -> float:
    """Always reserved: one owned-tree cleanup + postflight verification."""
    return d.cleanup_bound_s + d.postflight_verify_s


def end_baseline_bound_s(d: Design) -> float:
    return d.p8_wait_cap_s + d.baseline_settle_s + d.baseline_measure_s


def unit_bound_s(region_s: float, d: Design, held_out: bool) -> float:
    wait = d.ho_wait_cap_s if held_out else d.cal_fixed_wait_s
    return wait + unit_wall_limit_s(region_s, d) + d.cleanup_bound_s


def can_start_unit(remaining_s: float, region_s: float, d: Design, held_out: bool) -> bool:
    """The unit, a possible cleanup, the end baseline and the reserved tail must all fit."""
    return remaining_s >= unit_bound_s(region_s, d, held_out) + end_baseline_bound_s(d) + session_tail_reserved_s(d)


class Deadline:
    """Reservation deadline on CLOCK_MONOTONIC, fixed when the session starts from the approved end of the window."""

    def __init__(self, remaining_at_start_s: float, clock=time.monotonic):
        if not remaining_at_start_s > 0:
            raise ValueError("the approved window has already ended")
        self._clock = clock
        self._end = clock() + float(remaining_at_start_s)

    def remaining_s(self) -> float:
        return self._end - self._clock()

    def can_start_unit(self, region_s: float, d: Design, held_out: bool) -> bool:
        return can_start_unit(self.remaining_s(), region_s, d, held_out)

    def can_run_end_baseline(self, d: Design) -> bool:
        return self.remaining_s() >= end_baseline_bound_s(d) + session_tail_reserved_s(d)

    def unit_wall_limit_s(self, region_s: float, d: Design) -> float:
        """The watchdog limit for a unit that passed the gate: the rule, never beyond the time left for cleanup and
        verification."""
        return min(unit_wall_limit_s(region_s, d), self.remaining_s() - session_tail_reserved_s(d))


def components(regions: list, d: Design, wait_each_s: float, p8_wait_s: float) -> dict:
    m, n = MEASURED_MEDIANS, len(regions)
    return {"timed_regions_s": float(sum(regions)),
            "init_and_probe_s": n * m["init_and_probe_s"],
            "preparation_target_s": n * d.prep_target_s,
            "start_pause_s": n * d.pause_mean_s,
            "end_reads_s": n * m["end_reads_s"],
            "post_probe_s": n * d.post_probe_s,
            "data_check_exit_cleanup_s": n * m["exit_cleanup_s"],
            "telemetry_tail_bookkeeping_s": n * m["controller_s"],
            "cooldown_or_readiness_s": n * wait_each_s,
            "idle_baselines_s": 2 * (p8_wait_s + d.baseline_settle_s + d.baseline_measure_s),
            "preflight_postflight_s": d.preflight_s + d.postflight_verify_s}


def budget_table(d_cal: Design | None = None, d_ho: Design | None = None) -> dict:
    d_cal = d_cal or calibration_design(m4d=True)
    d_ho = d_ho or Design()
    if d_ho.post_probe_s:
        raise ValueError("M4d is calibration-only")
    cal = components(d_cal.calib_regions_s, d_cal, d_cal.cal_fixed_wait_s, d_cal.p8_wait_expected_s)
    cal_caps = components(d_cal.calib_regions_s, d_cal, d_cal.cal_fixed_wait_s, d_cal.p8_wait_cap_s)
    ho = components(d_ho.ho_regions_s, d_ho, d_ho.ho_wait_plan_s, d_ho.p8_wait_expected_s)
    ho_floor = components(d_ho.ho_regions_s, d_ho, d_ho.ho_wait_floor_s, d_ho.p8_wait_expected_s)
    ho_caps = components(d_ho.ho_regions_s, d_ho, d_ho.ho_wait_cap_s, d_ho.p8_wait_cap_s)
    tot = lambda c: sum(c.values())  # noqa: E731
    return {"calibration": {"launches": len(d_cal.calib_regions_s), "m4d": bool(d_cal.post_probe_s),
                            "components": cal, "expected_s": tot(cal), "cap_based_planning_estimate_s": tot(cal_caps)},
            "heldout_session": {"launches": len(d_ho.ho_regions_s), "components": ho, "expected_s": tot(ho),
                                "floor_s": tot(ho_floor), "cap_based_planning_estimate_s": tot(ho_caps)},
            "unit_bounds_s": {"calib_40s": unit_bound_s(40, d_cal, False), "heldout_40s": unit_bound_s(40, d_ho, True),
                              "end_baseline": end_baseline_bound_s(d_ho), "tail_reserved": session_tail_reserved_s(d_ho)},
            "note": "expected and cap-based values use medians for per-launch components and are not upper bounds; "
                    "the only enforced bound is the approved deadline (unit-start gate + watchdog wall limit)"}
