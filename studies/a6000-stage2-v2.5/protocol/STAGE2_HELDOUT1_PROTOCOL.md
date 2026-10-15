# Stage 2 held-out protocol (heldout-1)

**Status: proposed at gate heldout-1 (2026-10-13), revised r1 the same day** (production provenance boundary, disclosure corrections; `prep/heldout1_r1/CHANGE_LEDGER_r1.md`). It becomes binding only through Freeze-H records that carry the user's approval and a fresh pair/window authorization: one record per session, each with parent Freeze-C. This document alone authorizes nothing.

## 1. What is fixed by earlier records (unchanged)

| item | record | sha256 |
|---|---|---|
| parent freeze | `protocol/FREEZE_C_CALIB1.json` (Freeze-C; its parent is Freeze-S smoke-1) | `45b127f3f29a65f64552f00525e13b26670226dac6ccd765cd78ae6aa390278c` |
| calibration analysis (Q1, Q3 inputs) | `prep/calib1/run/calibration_analysis.json` | `152dd0a5a8cac1e8f587941eef1ac490c8095de073b2e0422580045fc59d89f4` |
| session 1 schedule and counts | `code/configs/stage2/heldout_s1_DRAFT.yaml` | `2200fad1ff3ed25d9e3c0ed53e8eb3144b4f1bcdc579c9d221365e60078b00d3` |
| session 2 schedule and counts | `code/configs/stage2/heldout_s2_DRAFT.yaml` | `28fbbb5a1d6d3cbb359ee9825a7ede227333b396181c461f8f4b4d1bfbeaddfb` |
| session 3 schedule and counts | `code/configs/stage2/heldout_s3_DRAFT.yaml` | `e4656b69fbaf2b704bd5c7ae4b767c823b2569dfb359032bbdaae6bcbe10ae3d` |

* The three schedule files are byte-identical to the cg1 implementation record (`prep/cg1/review/implementation_files_cg1.sha256`).
* **Unchanged:** the shared Freeze-C roles (native patch v2.5, binary, build record, profiler plugin, the 22 code files), acquisition, NVML use, acceptance, watchdog, cleanup and stop rules (v2 §L, v3 §L), and the decision helpers.
* Held-out mode has no dose and no M4d. The runner refuses either before anything runs.

## 2. Design

* **3 sessions × 2 blocks × 12 launches = 72 launches.** Each block has one launch per arm × length:
  * arms: A (NCCL default), B (RING/SIMPLE forced), C (RING/LL128 forced);
  * lengths: 2.5, 10, 20 and 40 s.
* The order and the requested pre-region pauses are as in the schedule files.
* **Counts** R (timed operations) and w (preparation iterations) are exact integers, the same in every session:

| arm | w | R at 2.5 s | R at 10 s | R at 20 s | R at 40 s |
|---|---|---|---|---|---|
| A | 95604 | 47802 | 191207 | 382413 | 764825 |
| B | 95597 | 47799 | 191193 | 382385 | 764769 |
| C | 98289 | 49145 | 196577 | 393154 | 786308 |

* `max_attempts` = 1. There are no retries, no replacement launches or sessions, and no optional stopping.
* A session ends early on any stop rule or on the unit-start budget gate. Its data are kept and labelled incomplete (v2 §N).

## 3. Readiness (derived by the frozen calibration rule; not re-derived)

**R1 (before every launch).** The runner's live gate requires, on both allocated GPUs:
* P8 and no foreign process;
* temperature ≤ **T_launch = 26 °C** and stable within 1 °C over the last 30 s;
* known process state and no telemetry gap > 2 s;
* at most the idle clock-event bits 0x1/0x4 (fix G) while in P8.

The wait cap is W_max = 240 s. **If the wait cap expires, nothing is launched and the session stops: an expiring wait is never readiness.** For calibration Q3, the q90 time to the full gate at 26 °C was 102 s (21 traces).

**R2 (pre-region state, per GPU).** Each value must lie inside the calibration band of its own GPU:

| GPU | T_region | graphics clock | P-state |
|---|---|---|---|
| 4 | 28.0–31.2 °C | 1950–1980 MHz | P2 |
| 5 | 29.0–32.0 °C | 1920–1950 MHz | P2 |

The clock-event OR over [C0 − 250 ms, C0) must be exactly 0, and the last POWER_INSTANT row must be ≤ 1 s old.

**What the offline analysis re-checks:**
* R1: only the runner's **recorded** status, READY. It does **not** replay the readiness trace.
* R2: recomputed from the recorded pre-region state with these frozen bands.

**R3 (analysis).** Per session, length and pair (B−A, C−A), and **per GPU**, the mean pre-region temperatures of the two arms must differ by ≤ 0.5 °C. A GPU-mean balance is not enough.

## 4. Analysis

`src/ncclenergy/v2/stage2_heldout.py` runs offline only; the runner never imports it. It is bound in the Freeze-H `analysis` role together with the unchanged helpers `stage2_decisions`, `stage2_calibration`, `stage2_energy`, `stage2_phase` and `stage2_readiness`. Rules where the earlier protocol was silent are listed in `prep/heldout1/CHANGE_LEDGER.md` (N1–N18, N11a) and `prep/heldout1_r1/CHANGE_LEDGER_r1.md` (P1–P7, D1–D7).

**Production boundary.** The only production entry point is `analyse_heldout(session_dirs, freeze_records)`. It runs portable offline checks only: no NVML, no GPU, and not the runner's live `verify_freeze`. It requires:
* exactly one approved Freeze-H per supplied session. A frozen session without data is reported NOT_STARTED; data without its Freeze-H is refused;
* every bound file present with its hash, and the analysis-role hashes equal to the code actually running;
* the parent equal to the frozen Freeze-C, whose shared roles must match;
* each config read from its bound bytes, equal to the frozen DRAFT except the finalization keys (calibration-derived rules, the calibration pair) and equal to the run's recorded `config_sha256`;
* the calibration analysis equal to its frozen bytes;
* the permitted implementation/method identity, mode, campaign and GPU order in the session summary and the executed unit records;
* the exact R and w of every record.

The compiled binary/plugin and the external NCCL library cannot always be verified offline. They are then listed as NOT_VERIFIED_OFFLINE rather than skipped. The low-level interfaces (`analyse_sessions_unverified`, `load_session`, `analyse_units`) are for synthetic tests; their output is labelled UNVERIFIED.

* **Unit quantity:** J = E ÷ R, with R the exact frozen integer.
  * Primary: E = E_aligned(P̂), summed over the participating GPUs.
  * Sensitivities, kept separate and never promoted: E_counter_raw, E_aligned(P_nom) and E_samp,trap,inst.
  * An unavailable primary stays unavailable with its reason. Nothing is imputed, substituted or trimmed.
* **Contrasts** (per block):

| decision | contrast | margin |
|---|---|---|
| H-D1 | ln(J_L ÷ J_40), for L = 2.5/10/20 s × arms A/B/C (9 contrasts) | ln 1.003 |
| H-D2 | the coefficient of 1{C} in OLS ln(E_samp,trap,inst ÷ E) ~ 1 + 1/T + 1{C}, with T the measured region duration; needs ≥ 4 units including C and non-C, rank 3 | ln 1.003 |
| H-D3 | ln(J_B ÷ J_A) at 10/20/40 s, over control-valid blocks | ln 1.005 |
| H-D4 | ln(J_C ÷ J_A) at 40 s | precision: half-width ≤ ln 1.004 |

* **Aggregation:**
  * block contrast → session mean of the available blocks → t on the 3 session means (df 2);
  * **S = 3 is required**; otherwise the result is INSUFFICIENT_SESSIONS;
  * a conditional interval over all block contrasts is reported as descriptive.
* **Multiplicity:**
  * Holm within H-D1 (9) and H-D3 (3), applied to both the difference test and the minimum-effect test;
  * 95 % intervals are labelled "descriptive (unadjusted)";
  * each TOST p is unadjusted;
  * a joint equivalence claim exists only through the intersection-union adequacy rule.
* **Verdict order** (v3 K3):
  1. INVALID_CONTROL_OR_MEASUREMENT (H-D3, when any A/B block has different verified algorithm/protocol/channels);
  2. THERMAL_COMPARABILITY_NOT_ESTABLISHED (R3 evaluated and failed; H-D3 at L, and H-D4);
  3. INSUFFICIENT_SESSIONS / DEGENERATE_ZERO_VARIANCE;
  4. RESOLVED_MEANINGFUL_DIFFERENCE;
  5. COMPATIBLE_WITH_EQUIVALENCE / DETECTED_WITHIN_MARGIN;
  6. DIFFERENCE_MAGNITUDE_UNRESOLVED;
  7. INSUFFICIENT_PRECISION.
* **Adequacy** (scopes BROAD, SAME_EXEC_L10/L20/L40 and CROSS_PROTOCOL_40) uses the primary estimator only.
  * Q1 comes from calibration: the primary is INSUFFICIENT_ROUNDS, so it is not bounded.
  * **Every aggregate scope requires a bounded Q1.** So with the actual frozen Q1, all five scopes have `conditional_statistical` = NOT_MET, and physical adequacy is NOT_ESTABLISHED, **even if every held-out criterion succeeds**. The output reports this as `aggregate_adequacy_ceiling`.
  * No hard-bound rule for any systematic component is frozen either: every component is UNKNOWN.
  * Q1 is not relaxed, no sensitivity is promoted, and calibration is not rerun.
  * The individual H-D1–H-D4 verdicts can still be informative as conditional NVML-reported comparisons.
* **Not computed.** These are accepted prospective scope reductions, so this stage does **not** complete every original protocol output. They also appear as NOT_COMPUTED in the machine-readable output:
  * the K2 historical D1–D4 replay. The frozen `measval_f4735b1.py` expects the historical campaign layout, and no approved adapter exists. Raw data are kept;
  * the K3 secondary covariate analysis (no resampling count or seed is frozen);
  * E_samp,hold,inst. The frozen helper `stage2_energy.sampled_energy_hold_J` exists, but its wiring into `unit_energies` is missing and is not added in this stage;
  * any delay-shifted sampled estimator (refused by calibration Q4: NO_SECONDARY_ESTIMATOR);
  * systematic bounds (no hard-bound rule is frozen).

## 5. Scope of any result

* What a result is: a conditional comparison of **NVML-reported** participating-GPU board energy of the tested arms, on this pair and host, in 3 sessions (a pilot).
* What it is not: global optima, externally calibrated physical accuracy, or a training or real-model claim.
* Three sessions are fewer than the five recommended at the highest level (Kalibera & Jones, ISMM 2013), so cross-session conclusions are a pilot.
* **H-D1:** a length dependence of J/op does not by itself identify a sensor-only mechanism. Thermal and clock changes over the region can also change board energy.
* **H-D4:** one selected C/A comparison (AllGather 2 MiB, 40 s, this pair). It is not evidence that global latency-optimal and energy-optimal configurations differ.

## 6. Windows and deadlines (per session)

Each session needs its own fresh authorization: the same physical pair (NVML GPUs 4 and 5) and a **fixed absolute window of at most 120 min**, with time zone. The deadline is enforced on CLOCK_MONOTONIC from session start:
* **Unit-start gate:** a unit, including its readiness wait, starts only if the time remaining ≥ wait cap + per-unit wall limit + 15 s cleanup + end-baseline bound (100 s) + reserved tail (75 s). For a 40 s unit that is 631.7 s.
* **Planning values** (frozen budget module):
  * expected: 62.9 min;
  * floor: 26.9 min;
  * cap-based estimate: 112.7 min (not an upper bound);
  * proposed ceiling and hard stop: 120 min per session.
* **These are median-overhead planning estimates, never guaranteed completion.** That includes the per-session 114.6–115.8 min "window needed if every wait hits its cap" and the 116-min minimum. Per-session values are in `prep/heldout1_r1/gate/budget.json`.
* A deadline or readiness failure may end a session partial. Its data are kept, and there are no retries and no replacement sessions.
