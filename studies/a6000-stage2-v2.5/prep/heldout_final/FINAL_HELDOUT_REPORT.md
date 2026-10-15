# Stage 2 held-out result (3 sessions, 72 launches): final report

**2026-10-14.** Produced by the single strict production analysis: `stage2_heldout.analyse_heldout`, run once over the three authentic sessions with their approved Freeze-H records and the default trust anchors.
* Provenance status: **VERIFIED_OFFLINE (production; see provenance)**.
* Machine-readable output: `heldout_analysis.json` (sha256 `a6126dd5628035eb98dea115f6a8680e56374442e4891e84a29623347ebde9c3`).
* What a result is: a **conditional comparison of NVML-reported participating-GPU board energy** of the tested arms on the calibration pair (NVML GPUs 4/5, RTX A6000, NVLink) and this host. It is a pilot with 3 sessions.
* What it is not: global optima, externally calibrated physical energy, or a training or real-model claim.

## Results first

* **Execution: 72/72 launches accepted, 0 excluded, 0 not run.**
  * All three sessions are COMPLETE, with no stop, retry or replacement.
  * The primary estimator is unavailable for 2 of the 72 units (the frozen period guard), so 3 block contrasts are missing. Nothing was imputed.
* **H-D3, identical execution (B forced RING/SIMPLE vs A default; A also ran RING/SIMPLE/4 in every block):**
  * **10 s: COMPATIBLE_WITH_EQUIVALENCE.**
    * The difference is -0.19 %, with 95 % CI [-0.45, +0.07] % (descriptive) and 90 % CI [-0.37, -0.01] % inside ±0.5 %.
    * TOST p = 0.018 (unadjusted), 95 % half-width 0.26 % (log scale; required ≤ 0.40 %).
    * The primary and nominal-period estimators agree. The raw counter is also within the margin, with a small detected offset (−0.40 %).
  * **20 s: THERMAL_COMPARABILITY_NOT_ESTABLISHED.** The per-GPU pre-region temperature differed by up to 1.0 °C between B and A in sessions 2 and 3.
  * **40 s: INSUFFICIENT_PRECISION.** +0.17 % [-0.95, +1.31]; half-width 1.13 %.
* **H-D4 (C = RING/LL128 vs A at 40 s): RESOLVED_MEANINGFUL_DIFFERENCE beyond ±0.4 %.**
  * C uses **+1.65 % more NVML-reported energy per operation**, 95 % CI [+1.00, +2.29] %.
  * Every session is positive (+1.46 / +1.94 / +1.54 %), and every sensitivity estimator agrees (+1.62 to +1.65 %).
  * **The pre-specified H-D4 precision criterion is not met:** the 95 % half-width is 0.63 % against ≤ 0.40 %.
* **H-D1 (length dependence of J/op, primary):**
  * No length effect is resolved. Eight of nine contrasts are INSUFFICIENT_PRECISION.
  * B 2.5 s vs 40 s is DIFFERENCE_MAGNITUDE_UNRESOLVED: −0.37 % [−0.42, −0.31] %. It is detected after Holm (p 0.011), but the Holm-adjusted minimum-effect test is not significant (p 0.295), so it is not resolved beyond 0.3 %.
* **H-D2** (protocol dependence of the sampled-vs-counter disagreement): INSUFFICIENT_PRECISION, +0.20 % [−0.47, +0.88].
* **Aggregate adequacy: every scope NOT_ESTABLISHED, `conditional_statistical` NOT_MET; physical adequacy NOT_ESTABLISHED.**
  * This was fixed before any held-out data: every scope requires Q1 bounded, and the frozen calibration Q1 primary is INSUFFICIENT_ROUNDS.
  * Independently of Q1, the held-out data alone also fail other criteria in every scope.
  * Nothing was relaxed, promoted or rerun.

## Primary estimator E_aligned(P̂): all contrasts

* Percent is 100·expm1(mean log contrast), i.e. the geometric-mean ratio − 1.
* 95 % CIs are descriptive (unadjusted). Decisions use Holm-adjusted p within each family (H-D1: 9 contrasts, H-D3: 3).
* TOST p is per-contrast and unadjusted.
* Margins: H-D1 and H-D2 ln 1.003; H-D3 ln 1.005; H-D4 ln 1.004 (precision criterion).

| contrast | session 1 / 2 / 3 (%) | mean % [95 % CI, descriptive] | p Holm | p beyond margin, Holm | TOST p (unadj.) | verdict |
|---|---|---|---|---|---|---|
| H-D1 A: 2.5 s vs 40 s | -0.60 / +1.31 / -0.47 | +0.08 [-2.54, +2.76] | 1.000 | 1.000 | 0.375 | INSUFFICIENT_PRECISION |
| H-D1 A: 10 s vs 40 s | -1.11 / -0.62 / -0.83 | -0.85 [-1.46, -0.24] | 0.188 | 0.423 | 0.970 | INSUFFICIENT_PRECISION |
| H-D1 A: 20 s vs 40 s | -1.07 / -0.10 / -0.62 | -0.60 [-1.80, +0.62] | 0.676 | 1.000 | 0.799 | INSUFFICIENT_PRECISION |
| H-D1 B: 2.5 s vs 40 s | -0.36 / -0.35 / -0.39 | -0.37 [-0.42, -0.31] | 0.011 | 0.295 | 0.984 | DIFFERENCE_MAGNITUDE_UNRESOLVED |
| H-D1 B: 10 s vs 40 s | -1.09 / -1.56 / -0.98 | -1.21 [-1.97, -0.44] | 0.169 | 0.295 | 0.982 | INSUFFICIENT_PRECISION |
| H-D1 B: 20 s vs 40 s | -0.56 / -0.92 / +0.07 | -0.47 [-1.71, +0.78] | 0.738 | 1.000 | 0.693 | INSUFFICIENT_PRECISION |
| H-D1 C: 2.5 s vs 40 s | -0.04 / +0.16 / -0.41 | -0.10 [-0.82, +0.63] | 1.000 | 1.000 | 0.176 | INSUFFICIENT_PRECISION |
| H-D1 C: 10 s vs 40 s | -0.43 / -0.79 / -0.50 | -0.58 [-1.05, -0.10] | 0.211 | 0.781 | 0.935 | INSUFFICIENT_PRECISION |
| H-D1 C: 20 s vs 40 s | -0.25 / -0.68 / -0.49 | -0.47 [-1.01, +0.07] | 0.319 | 1.000 | 0.850 | INSUFFICIENT_PRECISION |
| H-D2 C protocol term | -0.10 / +0.28 / +0.43 | +0.20 [-0.47, +0.88] | 0.329 | 1.000 | 0.297 | INSUFFICIENT_PRECISION |
| H-D3 B/A at 10 s | -0.07 / -0.26 / -0.24 | -0.19 [-0.45, +0.07] | 0.261 | 1.000 | 0.018 | COMPATIBLE_WITH_EQUIVALENCE |
| H-D3 B/A at 20 s | +0.43 / -0.13 / +0.36 | +0.22 [-0.54, +0.99] | 0.689 | 1.000 | 0.127 | THERMAL_COMPARABILITY_NOT_ESTABLISHED |
| H-D3 B/A at 40 s | -0.09 / +0.70 / -0.10 | +0.17 [-0.95, +1.31] | 0.689 | 1.000 | 0.168 | INSUFFICIENT_PRECISION |
| H-D4 C/A at 40 s | +1.46 / +1.94 / +1.54 | +1.65 [+1.00, +2.29] | 0.008 | 0.014 | 0.993 | RESOLVED_MEANINGFUL_DIFFERENCE |

**Session variability.**
* The H-D3(10) session means range over -0.26 to -0.07 %.
* The H-D4 session means range over +1.46 to +1.94 %.
* The full per-member session means, SDs, conditional block-level intervals (6 blocks, conditional on these sessions) and all p-values are in `contrasts.tsv`.

## Sensitivity estimators (separate; never promoted; no decisions)

Mean % (verdict under the same rules):

| contrast | primary | raw_counter | nominal_period | sampled_instant |
|---|---|---|---|---|
| H-D1 A: 2.5 s vs 40 s | +0.08 (INSUFF_PREC) | +1.45 (INSUFF_PREC) | +0.08 (INSUFF_PREC) | -5.00 (RESOLVED_DIFF) |
| H-D1 A: 10 s vs 40 s | -0.85 (INSUFF_PREC) | -0.49 (INSUFF_PREC) | -0.85 (INSUFF_PREC) | -1.56 (RESOLVED_DIFF) |
| H-D1 A: 20 s vs 40 s | -0.60 (INSUFF_PREC) | -0.51 (INSUFF_PREC) | -0.60 (INSUFF_PREC) | -0.75 (INSUFF_PREC) |
| H-D1 B: 2.5 s vs 40 s | -0.37 (DIFF_MAG_UNRES) | +0.90 (INSUFF_PREC) | -0.54 (INSUFF_PREC) | -5.00 (RESOLVED_DIFF) |
| H-D1 B: 10 s vs 40 s | -1.21 (INSUFF_PREC) | -1.02 (INSUFF_PREC) | -1.21 (INSUFF_PREC) | -2.00 (DIFF_MAG_UNRES) |
| H-D1 B: 20 s vs 40 s | -0.47 (INSUFF_PREC) | -0.46 (INSUFF_PREC) | -0.63 (INSUFF_PREC) | -0.79 (INSUFF_PREC) |
| H-D1 C: 2.5 s vs 40 s | -0.10 (INSUFF_PREC) | +1.35 (RESOLVED_DIFF) | -0.09 (INSUFF_PREC) | -3.77 (RESOLVED_DIFF) |
| H-D1 C: 10 s vs 40 s | -0.58 (INSUFF_PREC) | -0.34 (INSUFF_PREC) | -0.57 (INSUFF_PREC) | -1.64 (RESOLVED_DIFF) |
| H-D1 C: 20 s vs 40 s | -0.47 (INSUFF_PREC) | -0.41 (INSUFF_PREC) | -0.47 (INSUFF_PREC) | -0.80 (INSUFF_PREC) |
| H-D2 C protocol term | +0.20 (INSUFF_PREC) | +0.17 (INSUFF_PREC) | +0.14 (INSUFF_PREC) | n/a |
| H-D3 B/A at 10 s | -0.19 (EQUIV) | -0.40 (DET_WITHIN) | -0.19 (EQUIV) | -0.25 (INSUFF_PREC) |
| H-D3 B/A at 20 s | +0.22 (R3_NOT_EST) | +0.19 (R3_NOT_EST) | +0.13 (R3_NOT_EST) | +0.16 (R3_NOT_EST) |
| H-D3 B/A at 40 s | +0.17 (INSUFF_PREC) | +0.14 (INSUFF_PREC) | +0.17 (INSUFF_PREC) | +0.20 (INSUFF_PREC) |
| H-D4 C/A at 40 s | +1.65 (RESOLVED_DIFF) | +1.62 (RESOLVED_DIFF) | +1.65 (RESOLVED_DIFF) | +1.65 (RESOLVED_DIFF) |

* The counter-based estimators (primary, nominal period, raw counter) agree on H-D3 and H-D4.
* The **sampled POWER_INSTANT estimator shows a strong length dependence** (−3.8 to −5.0 % at 2.5 s vs 40 s) that the counter-based estimators do not. This is consistent with a roughly constant per-region offset of the sampled channel, which the per-operation ratio spreads over fewer operations at short lengths. The mechanism is not established here.
* The raw counter shows +1.35 % for C at 2.5 s, which the aligned estimators remove.

## Missing data (nothing imputed, substituted or replaced)

* **Units (status):** 72 used; 0 not accepted; 0 excluded; 0 not run.
* **Primary estimator unavailable** (frozen period guard, no clipping):
  * `s2-b2-p08` (B, 2.5 s), both GPUs: P̂ 98.56 / 101.42 ms outside the hard feasible intervals;
  * `s3-b2-p11` (B, 20 s), GPU 4: P̂ 98.80 ms outside [98.84, 100.66] ms.
* The other three estimators are available for both units.
* **Missing primary block contrasts:** H-D1(B, 2.5 s) s2b2; H-D1(B, 20 s) s3b2; H-D3(20 s) s3b2. The session mean uses the other block (`exclusions.tsv`).

## Thermal (R3, per GPU, ≤ 0.5 °C) and control checks

* **Control validity:** valid in all 6 blocks. The A and B execution keys were identical (RING/SIMPLE/4) in every launch.
* **R3 failures:** 7 of 24 (session, length, pair) checks, listed below. The full table is in `thermal_control.tsv`.

  * 1|2.5|B-A: {"4": -0.5000000000000036, "5": -0.1999999999999993}
  * 2|10|C-A: {"4": -1.1000000000000014, "5": -0.6000000000000014}
  * 2|20|B-A: {"4": -0.6999999999999993, "5": 0.0}
  * 2|20|C-A: {"4": -0.8999999999999986, "5": -0.3999999999999986}
  * 3|10|C-A: {"4": -0.20000000000000284, "5": -0.6000000000000014}
  * 3|20|B-A: {"4": 1.0, "5": 0.5}
  * 3|20|C-A: {"4": 0.6000000000000014, "5": 0.5}

**Effect on verdicts:** only H-D3(20) is overridden, by the B−A failures at 20 s in sessions 2 and 3.
* C−A at 40 s balanced in all sessions, so H-D4 is not overridden.
* The C−A failures at 10/20 s have no C/A contrast at those lengths. They enter only `r3_ok`, which N11 defines over both pairs and which therefore also marks SAME_EXEC_L10 adequacy unmet.

**Disclosed numerical artifact in the frozen R3 rule (not changed after seeing data).** Session 1, 2.5 s, B−A: GPU 4's arm-mean difference is −0.5000000000000036 °C, which is exactly 0.5 °C up to floating-point rounding of the means. The frozen `thermal_balance` therefore flags it.
* It affects no H-D3 or H-D4 verdict, because no H-D3/H-D4 contrast uses 2.5 s.
* It only sets `r3_ok[2.5]` false, which is used solely in BROAD adequacy. BROAD is NOT_MET for many other reasons.

## Attained precision (primary, 95 % half-width in log %)

| contrast | half-width | required | met |
|---|---|---|---|
| H-D3 10 s | 0.26 % | ≤ 0.40 % (scope criterion) | True |
| H-D3 20 s | 0.76 % | ≤ 0.40 % | False |
| H-D3 40 s | 1.13 % | ≤ 0.40 % | False |
| H-D4 40 s | 0.63 % | ≤ 0.40 % | False |

With df = 2 (3 sessions), the t multiplier is 4.30. A session-mean SD of about 0.1 % already gives about 0.25 % half-width.

## Adequacy (primary only; `adequacy.tsv`)

| scope | status | conditional_statistical | unmet statistical criteria |
|---|---|---|---|
| BROAD | NOT_ESTABLISHED | NOT_MET | H-D1(A, 2.5 s vs 40 s) equivalence; H-D1(A, 10 s vs 40 s) equivalence; H-D1(A, 20 s vs 40 s) equivalence; H-D1(B, 2.5 s vs 40 s) equivalence; H-D1(B, 10 s vs 40 s) equivalence; H-D1(B, 20 s vs 40 s) equivalence; H-D1(C, 2.5 s vs 40 s) equivalence; H-D1(C, 10 s vs 40 s) equivalence; H-D1(C, 20 s vs 40 s) equivalence; H-D2 equivalence; H-D3(20 s) equivalence; H-D3(40 s) equivalence; H-D4 precision; Q1 bounded (D100 and D500); R3 per-GPU balance at 2.5 s; R3 per-GPU balance at 10 s; R3 per-GPU balance at 20 s |
| SAME_EXEC_L10 | NOT_ESTABLISHED | NOT_MET | Q1 bounded (D100 and D500); R3 per-GPU balance at 10 s |
| SAME_EXEC_L20 | NOT_ESTABLISHED | NOT_MET | H-D3(20 s) equivalence; H-D3(20 s) precision; Q1 bounded (D100 and D500); R3 per-GPU balance at 20 s |
| SAME_EXEC_L40 | NOT_ESTABLISHED | NOT_MET | H-D3(40 s) equivalence; H-D3(40 s) precision; Q1 bounded (D100 and D500) |
| CROSS_PROTOCOL_40 | NOT_ESTABLISHED | NOT_MET | H-D3(40 s) equivalence; H-D3(40 s) precision; Q1 bounded (D100 and D500); H-D4 precision; H-D2 equivalence |

* **Systematic part:** every component is UNKNOWN in every scope, because no hard-bound rule was frozen.
* **`aggregate_adequacy_ceiling`:** NOT_MET for every aggregate scope (Q1 bounded is required by every scope; the frozen calibration Q1 primary is not bounded), whatever the held-out results.
* **Physical adequacy:** NOT_ESTABLISHED (Q1 not bounded; every systematic component UNKNOWN).

## Not computed (accepted prospective scope reductions; also in `heldout_analysis.json`)

* `K2_historical_replay`: NOT_COMPUTED (accepted prospective scope reduction): the frozen measval_f4735b1.py reads the historical campaign layout; no approved adapter to the Stage 2 session layout exists
* `secondary_covariate_analysis`: NOT_COMPUTED (accepted prospective scope reduction): the protocol does not freeze the block-resampling count or seed
* `E_samp_hold_inst`: NOT_COMPUTED (accepted prospective scope reduction): the frozen helper stage2_energy.sampled_energy_hold_J exists, but its wiring into stage2_calibration.unit_energies is missing and is not added in this stage
* `delay_shifted_secondary_estimator`: NOT_COMPUTED: refused by calibration Q4 (NO_SECONDARY_ESTIMATOR)
* `systematic_bounds`: NOT_COMPUTED: no hard-bound rule was frozen; every systematic component is UNKNOWN

## Interpretation limits

* **scope:** conditional comparison of NVML-reported participating-GPU board energy of the tested arms on this pair and host (pilot, 3 sessions); not global optima, not externally calibrated physical accuracy, no training or real-model claim
* **HD1:** length dependence of J/op does not by itself identify a sensor-only mechanism: thermal and clock changes over the region can also change board energy
* **HD4:** one selected C/A comparison (AllGather 2 MiB, 40 s, this pair); not evidence that global latency and energy optima differ
* **R1:** offline R1 verifies the runner's recorded READY status only (no readiness-trace replay); R2 is recomputed from the recorded pre-region state with the frozen bands
* **adequacy:** with the actual frozen calibration Q1 (primary not bounded) no aggregate scope can meet its statistical criteria and physical adequacy cannot be established; individual H-D1..H-D4 results can still be informative
* **Pilot:** three sessions are fewer than the five recommended at the highest level (Kalibera & Jones, ISMM 2013).
* **Host:** another user's jobs ran on GPUs 0–2 during sessions 1–3, never on the pair. Host 1-min load was 26–41.

## Provenance and evidence

* **Freeze-H records:**
  * S1 `bc2c2935ec5b1753bb77dffbaeda2ac667150b392108d2d9990309a07655d6a6`
  * S2 `f58300377ceb27e919a8440d2d5f0826f93ac7f50eb5b519d1e3c46081d40e4b`
  * S3 `a95104a92f2477df0aa448b6c44b4ffab0071b5be9306e3b0bbbbc0d1409d6c1`
  * All are approved, with parent Freeze-C `45b127f3…`. They agree on protocol, analysis, calibration analysis and shared roles.
* **Analysis code** (equal to the Freeze-H analysis role): `stage2_heldout.py` `ae8c9c6b504f45ed3039fa04abb5a53abb51dc3736daef9d5fec33c4cc29b4ee`; the helpers are unchanged from Freeze-C.
* **NOT_VERIFIED_OFFLINE on this host:** none.
* **Raw evidence:**
  * Each session's raw files match its `session_files.sha256` (270 files each).
  * Evidence ZIPs: `delivery/heldout_s1/` `c4f56305…`, `delivery/heldout_s2/` `bfc4c399…`, `delivery/heldout_s3/` `e02ac928…`.
  * Details: `evidence_manifests.tsv`.
* **Tables:** `units.tsv` (every launch, status, R, J per estimator, unavailability reasons), `exclusions.tsv`, `execution.tsv` (per-launch execution and QC for all 72 launches), `contrasts.tsv` (all estimators and families), `thermal_control.tsv`, `adequacy.tsv`.
* **Older-method data** (pilot, v2, gate 3b, smoke, calibration launches) are not pooled. Only the frozen calibration analysis is read, for Q1 and Q3, as the protocol specifies.

## End of this stage

* This bounded stage ends here, with the answer at the precision the evidence supports.
* No further sessions, calibration rerun, added rounds, training or Pascal campaign are proposed or run.
* Nothing is committed, pushed or published.
