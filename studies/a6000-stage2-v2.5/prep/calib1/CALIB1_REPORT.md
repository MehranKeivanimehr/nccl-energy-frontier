# Stage 2 calibration (calib-1): report

Run: 2026-10-13 05:09:50Z–06:05:38Z, GPUs 4 and 5. Executed once under Freeze-C.
* No retries, replacement units, optional stopping or added launches.
* Nothing is committed or pushed. Held-out collection is not started.
* Smoke-1 data are excluded: `analyse()` refuses smoke directories.

## 0. Outcome

| level | result |
|---|---|
| execution | **COMPLETE**: 23 of 23 launches executed. No stop, no in-unit watchdog trip, deadline not reached (1,072 s left) |
| QC / acceptance | **23 of 23 OK and accepted**, both session baselines OK |
| predeclared analyses | run unchanged (module hashes = Freeze-C); results in §4 |
| **Q1, primary** (E_aligned(P̂)) | **INSUFFICIENT_ROUNDS** (counter and DiD, both doses): 2 of 3 rounds available. Sampled: INCONCLUSIVE |
| Q1, predeclared raw-counter sensitivity | **DETECTED** at both doses: D100 −10.86 %, D500 −2.14 %. COUNTER_SPECIFIC_DOSE_DEPENDENCE, mechanism not established |
| Q3 readiness | **OK**: T_launch 26 °C, R2 bands derived |
| M4c | **RETAINED** |
| end alignment (M4d) | **ALIGNMENT_NOT_ESTABLISHED**: 5 of 46 GPU-launches AMBIGUOUS_CYCLE, so every length is UNKNOWN |
| Q4 | NO_SECONDARY_ESTIMATOR |
| thermal slope | ASSOCIATION_ONLY, +0.41 %/°C |
| t_op | within 3 % (max \|dev\| 1.27 %): no amendment |
| adequacy (physical systematic) | **NOT_ESTABLISHED** |
| held-out allowed? | **No, not yet**: your review, Freeze-H (`stage2_heldout.py` missing) and fresh windows are outstanding (§6) |

## 1. Authorization, freeze, preflight

* **Authorization.** Your text is verbatim in `allocation/user_authorization.json`. Window **05:08:30Z–06:23:30Z**, an absolute deadline, taken from "start: Now" plus "exactly 75 minutes". GPUs 4 and 5.
* **Documents-only clarifications made before the freeze.** The originals are retained.
  * The QC window is [C0, work end].
  * The window end is the only deadline.
  * Q1 TOST evidence is unadjusted; Holm applies to the difference and minimum-effect tests.
  * Smoke packet r2 fixes the overview rows.
* **Read-only scope** (05:09:12Z):
  * UUID and PCI match your values exactly: rank 0 = GPU 4, GPU-4a2a69dd…, 81:00.0; rank 1 = GPU 5, GPU-0ce81f5d…, A1:00.0;
  * RTX A6000 × 2, driver 580.178.04, NV4, NUMA 1, 300 W, 2100 MHz;
  * P8, no processes. GPUs 1–3 were busy with other users' jobs and were not touched.
* **Freeze-C** `protocol/FREEZE_C_CALIB1.json`, sha256 `45b127f3…`:
  * parent Freeze-S `22cfae14…`, shared roles identical;
  * 41 files;
  * config `calib01.yaml` `10f18500…`, protocol `STAGE2_CALIB1_PROTOCOL.md` `738b56ad…`, analysis `stage2_calibration` `b88ad3c1…` and `stage2_decisions` `ea82214b…`;
  * libnccl `1d9c1032…`;
  * `verify_freeze`: no problems. `stage2_runner check`: no refusals.
  * `LD_PRELOAD` empty; `ldd` resolves the frozen libnccl.
* **Command** (`run/command.txt`): `python -m ncclenergy.v2.stage2_runner calibration --campaign configs/stage2/calib01.yaml`. Started 05:09:49Z. Deadline binding: reservation end 06:23:30Z.

## 2. Execution (all 23 launches)

* **All launches:** exit 0, QC OK, accepted. Evidence VERIFIED **RING/SIMPLE/4** with an op count exactly R + w + 1, and an exact counter-call inventory on both GPUs.
* **Region length:** 2.5 s → 2.472–2.481 s; 10 s → 9.874–9.938 s; 20 s → 19.819–19.872 s; 40 s → 39.588–39.726 s. That is 0.6–1.3 % short, consistent with t_op 0.9 % below the historical value (§4).
* **Warm-up:** 4.953–4.988 s. |pause deviation| ≤ 0.090 ms. Post-probe: 4 brackets on every GPU-launch.
* **Dose (S5 rule):**
  * D100 (C2, 10 s): 98 of 98 eligible delivered on both GPUs, in all 3 rounds;
  * D100 (S0b, 2.5 s): 24 of 24;
  * D500: 19 of 19;
  * order work_end ≤ stop ≤ join held in all 8 dose launches;
  * C2r2-14, rank 0: one read was in flight at the stop request and straddled the work end. It was logged and not delivered, and it is not a failure (correction A in use).
* **In-region clock events** (QC window [C0, work end], 0x1 masked): 0 for every launch. Peak temperature 37 °C (40 s units). Sampler errors 0, overruns 0–2 per launch.
* **Fixed waits:** GPU 4 had **one P8 row with mask 0x24 at 25 °C** in each of 3 of the 23 waits: S0b-02 at 70 s, C1r1-05 at 97 s, C2r3-22 at 95 s. They were recorded only, by policy, and fixed waits do not stop on masks. No such bit appeared in any QC window, and the watchdog never tripped.
* **Baselines:**
  * start: GPU 4 28.21 W, GPU 5 21.24 W;
  * end: 27.95 W and 21.51 W;
  * only 0x1/0x4 at P8;
  * raw e0/e1 readings saved.
* **Safety:** no foreign processes in the sampled process lists. Xid event sets were registered, with none reported between units. Cleanup normal. Postflight: no owned process left. Host load averaged 47–51 from other users (effect not assessed).

## 3. Guard B in real data

3 of 46 GPU-launches had a least-squares P̂ outside the hard feasible period interval:

| launch | rank | P̂ (ms) |
|---|---|---|
| S0b-02 | 0 | 101.30 |
| C1r3-17 | 0 | 101.43 |
| C2r2-16 | 1 | 101.17 (interval ends at 101.166) |

For these, E_aligned(P̂) is **unavailable** with that reason, and raw and nominal-period estimates are retained. Nothing was clipped or imputed. Because C2r2-16 is the E0 unit of C2 round 2, the predeclared rule ("a missing unit or estimator removes that round") removes round 2 from the primary Q1 at both doses. End alignment recorded 2 explicit nominal-period denominator substitutions.

## 4. Predeclared analyses (unchanged; `run/calibration_analysis.json`)

**Q1 read dose** (C2, 3 rounds × E0/D100/D500 at 10 s). Margin ±ln 1.003. Holm over the 2 doses on the difference and minimum-effect tests; TOST unadjusted.

| estimator | dose | counter d | sampled s | DiD d − s |
|---|---|---|---|---|
| primary E_aligned(P̂) | D100 | INSUFFICIENT_ROUNDS (2 rounds: −10.95, −9.97 %) | INCONCLUSIVE: +0.25 % [−1.64, +2.14] | INSUFFICIENT_ROUNDS |
| primary E_aligned(P̂) | D500 | INSUFFICIENT_ROUNDS (2 rounds: −2.39, −1.96 %) | INCONCLUSIVE: +0.72 % [−0.17, +1.62] | INSUFFICIENT_ROUNDS |
| raw counter (sensitivity) | D100 | **DETECTED** −10.86 % [−13.24, −8.47]; Holm p 0.0052, beyond-margin Holm p 0.0055; TOST p 0.999 | INCONCLUSIVE (as above) | **DETECTED** −11.11 % [−12.55, −9.66]; Holm p 0.0018 |
| raw counter (sensitivity) | D500 | **DETECTED** −2.14 % [−3.04, −1.24]; Holm p 0.0095, beyond-margin Holm p 0.0127; TOST p 0.994 | INCONCLUSIVE (as above) | **DETECTED** −2.86 % [−3.23, −2.49]; Holm p 0.0018 |

Intervals are 95 %, descriptive (unadjusted).

* **Interpretation** (predeclared, from the classes only):
  * primary: UNRESOLVED;
  * raw counter: **COUNTER_SPECIFIC_DOSE_DEPENDENCE**. The reported counter energy is lower with in-region reads, while the sampled channel shows no detectable change. **The mechanism is not established.**
* **Adequacy:** `bounded_for_adequacy = False` for both doses.
* **Scope:** this tests only the additional in-region reads (10 Hz and 2 Hz on both GPUs). E0 itself contains the probe and boundary reads.
* The two available primary rounds are shown descriptively. They are not a decision, and no round was added or imputed.

**Other decisions:**
* **Q3 readiness: OK.**
  * T_launch **26 °C**: 25 °C gave q90 = ∞, so it was raised by 1; at 26 °C q90 time-to-gate = 102 s. 21 launches, 21 traces.
  * R2 bands: GPU 4 T_region 28.0–31.2 °C, clock 1950–1980 MHz, P2. GPU 5 29.0–32.0 °C, 1920–1950 MHz, P2.
* **M4c: RETAINED.** 0 launches with |dev| > 2 ms. Temperature association −0.13 °C per 100 ms (limit 0.5); clock +0.0002 MHz (limit 15).
* **End alignment: ALIGNMENT_NOT_ESTABLISHED.** 5 AMBIGUOUS_CYCLE (S0b-02 r0, C2r1-07 r0, C1r3-19 r0, C2r3-21 r1, C2r3-23 r0). All lengths UNKNOWN. Never a bound.
* **Thermal slope: ASSOCIATION_ONLY.** +0.41 %/°C [0.22, 0.60], n 11, dof 4, correlation with launch order −0.56, VIF 1.8. Not causal, not a bound.
* **t_op:** mean −0.92 %, max |dev| 1.27 % → action NONE.
* **Q4: NO_SECONDARY_ESTIMATOR** (n 14, d_eff 95 % CI [−2.6, 52.9] s).
  * Observation, not a re-analysis: d_eff values range 0.7–154 s. The predeclared denominator P_region − P_before takes P_before from [C0 − 1 s, C0), which in Stage 2 follows the 5 s warm-up. The denominator can therefore be small, which makes d_eff unstable.
* **Q2:** descriptive output in the JSON.
* **Adequacy** (physical systematic: sensor/counter accuracy, context-power model, load extrapolation, start-phase and thermal imbalance): **NOT_ESTABLISHED**. No component has a hard bound.

## 5. What is and is not claimed

**Claimed:**
* The calibration executed as frozen.
* On this pair, host and software, the predeclared raw-counter sensitivity shows a **read-dependent reduction of the NVML-reported counter energy**:
  * ≈ −10.9 % at 10 Hz in-region reads;
  * ≈ −2.1 % at 2 Hz;
  * with no detectable change in the sampled channel.
* The primary Q1 is INSUFFICIENT_ROUNDS.

**Not claimed:**
* a physical energy loss or its mechanism;
* sub-percent physical accuracy;
* a bound on any systematic component;
* resolved latency/energy optima;
* training savings.

## 6. Checks that decide whether held-out is allowed (existing rules)

| check | status |
|---|---|
| calibration executed under Freeze-C, S0a accepted | **met** |
| Q3 OK (T_launch and R2 bands) | **met** |
| M4c RETAINED | **met** |
| t_op within 3 % | **met** |
| your calibration review and decision | **pending** |
| Freeze-H: `stage2_heldout.py` (missing; new analysis code with CPU validation), parent Freeze-C, readiness rules from Q3, your approval | **not met** |
| a fresh allocation confirmation per held-out session window | **not given** |

Q1, Q2, Q4, end alignment and the thermal slope are reported findings, not held-out gates. Nothing was started.

## 7. Files

`delivery/calib1_run/calib1_evidence_2026-10-13.zip` + `MANIFEST.sha256` + `verify_package.py`. It contains:
* the session (all 23 unit directories with `region.jsonl`, `s2.jsonl`, `prof/`, `telemetry.csv.gz`, TOTAL_POWER samples, `nccl.log`, stdout/stderr, `unit.json`, owned-process records);
* the waits, baselines (with raw e0/e1), `attempts.jsonl`, `session_summary.json`;
* Freeze-C, Freeze-S, config (final and draft), protocols, the authorization, the scope read, the runner log, the analysis JSON, the gate sheet and this report.

Private identifiers (hostnames, UUIDs, paths) are present. It is not for publication.
