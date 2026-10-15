# Stage 2 calibration (calib-1): protocol

**Status.** Text for Freeze-C, written 2026-10-13. It is bound only after the user authorizes the pair and a fresh window. It restates the existing predetermined rules:
* `APPROVAL_PACKET_v3.md` (K1, F, L, N, M), as amended by `STAGE2_CG1_PACKET.md`;
* the smoke-1 corrections A–C, F and G (`prep/smoke1/CHANGE_LEDGER.md`).

It adds no rule, unit, retry or matrix, and no code was changed for it.

**Software.** It is the software frozen in `FREEZE_S_SMOKE1.json`. Freeze-C uses that record as its parent with identical shared roles (patch, binary, build record, plugin, code). A dry run of the runner's own parent check passed.

## 1. Endpoint

* **Measured quantity:** the **NVML-reported** board energy of the participating GPUs (cumulative counter; estimators E_aligned(P̂) primary, E_counter_raw and E_aligned(P_nom) as sensitivity, E_samp,trap,inst as the sampled channel).
* **What it supports:** conditional comparisons on **this pair and host** (GPUs 4 and 5, RTX A6000, driver 580.178.04, NV4), under this software.
* **Adequacy:** physical systematic adequacy (sensor accuracy, context-power model, load extrapolation, start-phase and thermal imbalance) is **NOT_ESTABLISHED**, and no calibration outcome relabels it as passed. A CONDITIONAL timing contribution, an association or an observed maximum is not a bound.

## 2. Matrix: exactly the existing 23 launches

All launches: AllGather 2 MiB, arm B (forced Ring/Simple), frozen counts w = 95,597 and R = 47,799 / 191,193 / 382,385 / 764,769 for 2.5 / 10 / 20 / 40 s, M4d on, one fresh process each, frozen M4c pauses. Source: `configs/stage2/calib01_DRAFT.yaml`, completed only with the allocation, window, scope and freeze record.

| order | launch | part | length | dose |
|---|---|---|---|---|
| 1 | S0a-01 | S0a | 2.5 s | E0 |
| 2 | S0b-02 | S0b | 2.5 s | D100 |
| 3–6 | C1r1-03…06 | C1 round 1 | 20, 2.5, 40, 10 s | E0 |
| 7–9 | C2r1-07…09 | C2 round 1 | 10 s | E0, D100, D500 |
| 10–13 | C1r2-10…13 | C1 round 2 | 20, 2.5, 10, 40 s | E0 |
| 14–16 | C2r2-14…16 | C2 round 2 | 10 s | D100, D500, E0 |
| 17–20 | C1r3-17…20 | C1 round 3 | 2.5, 20, 40, 10 s | E0 |
| 21–23 | C2r3-21…23 | C2 round 3 | 10 s | D500, E0, D100 |

* **No retries.** `max_attempts` is 1.
* **No replacement units, optional stopping or added launches.**
* **A launch that is not accepted** is recorded with its reasons and excluded from the analyses that need it; it is never imputed.
* **S0a gate.** A non-accepted S0a stops the session. Other non-accepted units do not, unless a stop rule fires (§5).

## 3. Calibration decisions (existing rules, `stage2_calibration.py` / `stage2_decisions.py`)

* **Q1, read dose (C2 rounds 1–3, accepted units only).**
  * Per dose D ∈ {D100, D500}: d = ln(J_D / J_E0) on the counter (E_aligned(P̂); E_counter_raw also reported), s on the sampled channel, and DiD = d − s.
  * Statistics: t on 3 rounds (df 2). Difference and minimum-effect tests: two-sided, **Holm-adjusted** over the 2 doses. Equivalence: TOST at ±ln 1.003 per dose; the **individual TOST evidence is unadjusted**. The joint rule (adequacy needs d BOUNDED for **both** doses) and all margins and counts are unchanged.
  * Classes: DETECTED, BOUNDED_WITHIN_MARGIN, DETECTED_AND_BOUNDED, INCONCLUSIVE, INSUFFICIENT_ROUNDS, DEGENERATE.
  * **Every class is a legitimate calibration finding. None is a calibration failure.** The margin is not widened and no loss model is fitted.
  * E0 already contains the ~250-read pre-region probe and the C_prep0/C0/C1/Cpost reads. **Q1 tests only the additional in-region reads**: D100 = 100 reads per 10 s per GPU (10 Hz), D500 = 20 per 10 s (2 Hz). It does not test all measurement perturbation.
  * Historical Phase 4 (100 Hz on GPU 4 only, AllReduce 64 MiB) motivates the test. It does not predict its outcome.
  * Q1 enters adequacy only as "d BOUNDED for both doses".
* **Q2, channels:** descriptive only.
* **Q3, readiness derivation:** derives T_launch and the R2 bands from calibration temperatures, clocks and states only, never energies.
  * It replays the full R1 row rule (including G) on the recorded fixed-wait traces.
  * It needs ≥ 15 accepted C1/C2 launches and ≥ 15 wait traces.
  * Status OK is the only usable result. NOT_DERIVABLE or NOT_ACHIEVABLE means Freeze-H needs an amendment.
* **Q4, H_delay:** the secondary delay-shifted estimator is added only if the 95 % interval of d_eff over ≥ 3 accepted E0 launches excludes 0.
* **M4c:** RETAINED unless more than 1 launch has |d_act − d_req| > 2 ms, or the state association exceeds 0.5 °C / 15 MHz per 100 ms. Otherwise AMENDMENT_PROPOSED.
  * M4c randomizes the start phase relative to the counter's updates and permits an assessment of phase sensitivity.
  * It does **not** establish an unbiased physical energy, and the pause may itself affect GPU state.
* **End alignment (M4d):** reported as CONDITIONAL only if every accepted launch is ALIGNED_UNDER_MODEL, with at least 18 per GPU. The value is the observed maximum of model-conditional intervals, **never a bound**. Otherwise ALIGNMENT_NOT_ESTABLISHED.
* **Thermal slope:** ASSOCIATION_ONLY, or NOT_ESTIMABLE / CONFOUNDED. Never a bound.
* **t_op:** B's observed t_op is compared with the historical value. A deviation > 3 % gives an amendment proposal. R and w are never re-frozen.
* **Exclusion of smoke data:** `stage2_calibration.analyse` refuses smoke directories. Smoke-1 data never enter calibration inference.

## 4. Clock-event policy (existing code; no runtime change)

| layer | where | rule | consequence |
|---|---|---|---|
| recording | every sampler: baselines, fixed waits, units | raw `nvmlDeviceGetCurrentClocksEventReasons` bitmask kept; per-bit shares reported for baselines and waits (G) | none |
| idle gates | session baselines (P8 entry and validity); held-out R1; Q3 replay | P8 rows: only 0x1 and 0x4 tolerated (G); any other bit, including 0x20, invalidates the row; rows that are not P8 fail | start baseline invalid: session stops before any launch. End baseline invalid: recorded as FAILED. R1 not ready: the gate keeps waiting up to W_max, then READINESS_NOT_ACHIEVED stops the session |
| fixed wait (calibration launch readiness) | 120 s before each calibration unit | a cooldown, not a gate. Clock-event masks are recorded only and **neither delay nor stop**. It stops only on a foreign or unknown process (after the first poll) or temperature ≥ 80 °C | a transient mask such as the smoke-1 0x24 is recorded and the unit still starts after 120 s. No all-phase stop on clock events exists or is claimed |
| unit lifetime (watchdog) | every 50 ms sample from process start to exit, polled once per second | bits 0x8, 0x20, 0x40, 0x80 at any P-state (0x1 and 0x4 are not watchdog bits) | unit cancelled (SAFETY_STOP), session stops, no end baseline |
| measured-region QC | OR of the mask over **[C0, work end]**, 0x1 masked. The preceding 250 ms is used only for required-channel coverage and the recorded pre-region state; held-out R2 gates that pre-window | any of 0x4, 0x8, 0x20, 0x40, 0x80 → THROTTLED (`stop_block`) | unit not accepted and the session stops |
| held-out R2 | pre-region window [C0 − 250 ms, C0) | clock events present and exactly 0 | unit not accepted |

## 5. Stop rules and safety (unchanged)

The session stops on any of:
* a foreign or unknown process;
* the watchdog (temperature ≥ 80 °C, health clock event, telemetry stale or invalid, wall limit, disk);
* QC `stop_block` (FOREIGN_PROCESS, THROTTLED, TOO_HOT);
* a timeout or errors in the logs;
* an Xid event (polled between units);
* a power-limit or clock-policy change between units;
* disk below 20 GB;
* a scope mismatch or busy allocation at start;
* a start-baseline failure;
* the unit-start budget gate;
* S0a not accepted.

Owned processes only are terminated and reaped. There are no administrator actions and no global clock, power or driver changes.

## 6. Deadline

The enforced bound is the **absolute end of the window the user confirmed**: for calib-1, **2026-10-13T06:23:30Z** (window 05:08:30Z–06:23:30Z). It is held on CLOCK_MONOTONIC from session start.
* There is no separate automatic 75-minute cap. 75 min is the window length the user chose.
* The deadline is **not extended** if the run starts late.
* The 30-minute ceiling applies to the smoke only.

| estimate (`stage2_budget.budget_table`, medians) | minutes |
|---|---|
| expected | 58.9 |
| cap-based planning estimate (not a bound) | 60.7 |

A unit starts only if the remaining time covers its wall limit, cleanup, the end-baseline bound (100 s) and the reserved tail (75 s): 513.0 s for a 40 s unit. The end baseline runs only if it fits.

## 7. After calibration

Stop for the user's calibration decision. Calibration does not start held-out collection (§8 of the gate sheet lists the checks).
