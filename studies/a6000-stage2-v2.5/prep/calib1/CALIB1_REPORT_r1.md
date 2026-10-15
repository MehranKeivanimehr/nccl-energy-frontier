# Stage 2 calibration (calib-1): report, revision r1

> **r1 (2026-10-13), documents only.** The original `CALIB1_REPORT.md` (sha256 `bcecb9603664ef1f8ed5f73782fbaa954cb5374ae0d99ccdd9e364541c38f43c`) is unchanged and is in `calib1_evidence_2026-10-13.zip`. Data, Freeze-C, config, protocol and the analysis JSON are unchanged. Nothing was rerun. **No outcome class changes.**
>
> **Correction ledger:**
>
> | # | r0 | r1 |
> |---|---|---|
> | 1 | Percentages were 100 × mean(log ratio) without saying so | Ordinary percentages are now 100 × expm1(x) for point and CI endpoints, with the geometric ratio exp(mean log) named. The log-scale values are kept and labelled. DiD is a ratio of ratios, exp(d − s) − 1, not a difference of percentages |
> | 2 | Raw counter D100 −10.86 % [−13.24, −8.47], D500 −2.14 % [−3.04, −1.24] | D100 **−10.2874 %** [−12.4011, −8.1228]; D500 **−2.1154 %** [−2.9935, −1.2293]. Holm p-values unchanged |
> | 3 | "all 8 dose launches" | **7** active-poller launches (S0b-02 + 3 D100 + 3 D500) |
> | 4 | clocks not reported | in-region graphics clock varied: GPU 4 **1935–1965 MHz**, GPU 5 **1920–1935 MHz** (sample-level extremes), with zero measured-region event masks. Not constant; no thermal causality inferred |
> | 5 | "Host load averaged 47–51" | per-unit 1-minute load snapshot at each unit start: mean **47.064**, range **37.86–61.61** (23 launches). Session start/end snapshots 47.4 / 50.6. Effect not assessed |
> | 6 | Q4 d_eff "values" | these are unstable **candidate ratios, not measured physical delays** |
> | 7 | thermal slope "+0.41 %/°C" | log-scale 0.4099 per °C (100 × slope); ordinary **+0.4107 %/°C** [0.2171, 0.6047]. Association only |

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
| Q1, predeclared raw-counter sensitivity | **DETECTED** at both doses: D100 **−10.2874 %** (geometric ratio 0.89713), D500 **−2.1154 %** (0.97885). COUNTER_SPECIFIC_DOSE_DEPENDENCE, mechanism not established. Raw counter is a sensitivity estimator, not the primary |
| Q3 readiness | **OK**: T_launch 26 °C, R2 bands derived |
| M4c | **RETAINED** |
| end alignment (M4d) | **ALIGNMENT_NOT_ESTABLISHED**: 5 of 46 GPU-launches AMBIGUOUS_CYCLE, so every length is UNKNOWN |
| Q4 | NO_SECONDARY_ESTIMATOR |
| thermal slope | ASSOCIATION_ONLY, +0.4107 %/°C (ordinary; log-scale 0.4099) |
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
  * order work_end ≤ stop ≤ join held in all **7** active-poller launches (r1: r0 said 8);
  * C2r2-14, rank 0: one read was in flight at the stop request and straddled the work end. It was logged and not delivered, and it is not a failure (correction A in use).
* **In-region clock events** (QC window [C0, work end], 0x1 masked): 0 for every launch. Peak temperature 37 °C (40 s units). Sampler errors 0, overruns 0–2 per launch.
* **In-region graphics clock** (r1): not constant despite zero event masks. Sample-level extremes: GPU 4 **1935–1965 MHz**, GPU 5 **1920–1935 MHz**; per-launch means 1947.9–1965 and 1922.2–1935 MHz. Pre-region last clock 1965 / 1935 MHz in every launch. No thermal or causal interpretation.
* **Fixed waits:** GPU 4 had **one P8 row with mask 0x24 at 25 °C** in each of 3 of the 23 waits: S0b-02 at 70 s, C1r1-05 at 97 s, C2r3-22 at 95 s. They were recorded only, by policy, and fixed waits do not stop on masks. No such bit appeared in any QC window, and the watchdog never tripped.
* **Baselines:**
  * start: GPU 4 28.21 W, GPU 5 21.24 W;
  * end: 27.95 W and 21.51 W;
  * only 0x1/0x4 at P8;
  * raw e0/e1 readings saved.
* **Safety:** no foreign processes in the sampled process lists. Xid event sets were registered, with none reported between units. Cleanup normal. Postflight: no owned process left. Host load (r1): per-unit 1-minute load snapshot at each unit start, mean **47.064**, range **37.86–61.61** (23 launches); session start/end snapshots 47.4 / 50.6. Effect not assessed.

## 3. Guard B in real data

3 of 46 GPU-launches had a least-squares P̂ outside the hard feasible period interval:

| launch | rank | P̂ (ms) |
|---|---|---|
| S0b-02 | 0 | 101.30 |
| C1r3-17 | 0 | 101.43 |
| C2r2-16 | 1 | 101.17 (interval ends at 101.166) |

For these, E_aligned(P̂) is **unavailable** with that reason, and raw and nominal-period estimates are retained. Nothing was clipped or imputed. Because C2r2-16 is the E0 unit of C2 round 2, the predeclared rule ("a missing unit or estimator removes that round") removes round 2 from the primary Q1 at both doses. End alignment recorded 2 explicit nominal-period denominator substitutions.

## 4. Predeclared analyses (unchanged; `run/calibration_analysis.json`)

**Q1 read dose** (C2, 3 rounds × E0/D100/D500 at 10 s). Margin ±ln 1.003. Holm over the 2 doses on the difference and minimum-effect tests; TOST evidence unadjusted per dose; the joint both-dose BOUNDED rule for adequacy is unchanged.

**Scales (r1).** The analysis works on x = mean of per-round ln(J_D / J_E0).
* **ordinary %** = 100 × expm1(x), applied to the point and to each CI endpoint; geometric ratio = exp(x).
* **log-%** = 100 × x (the r0 numbers).
* DiD x = d − s is a ratio of ratios: exp(d − s) − 1. It is not a subtraction of ordinary percentages.

Intervals are 95 %, descriptive (unadjusted).

| estimator | dose | channel | class | ordinary % [95 % CI] | geometric ratio | log-% (r0) | Holm p (difference / beyond margin) | TOST p (unadjusted) |
|---|---|---|---|---|---|---|---|---|
| raw counter (sensitivity) | D100 | counter d | **DETECTED** | **−10.2874 [−12.4011, −8.1228]** | 0.89713 | −10.856 [−13.240, −8.472] | 0.0052 / 0.0055 | 0.999 |
| raw counter (sensitivity) | D100 | DiD | **DETECTED** | −10.5109 [−11.7915, −9.2117] | 0.89489 | −11.105 [−12.547, −9.664] | 0.0018 / 0.0019 | 0.9995 |
| raw counter (sensitivity) | D500 | counter d | **DETECTED** | **−2.1154 [−2.9935, −1.2293]** | 0.97885 | −2.138 [−3.039, −1.237] | 0.0095 / 0.0127 | 0.994 |
| raw counter (sensitivity) | D500 | DiD | **DETECTED** | −2.8204 [−3.1776, −2.4620] | 0.97180 | −2.861 [−3.229, −2.493] | 0.0018 / 0.0019 | 0.9994 |
| sampled (same in both rows) | D100 | sampled s | INCONCLUSIVE | +0.2497 [−1.6282, +2.1635] | 1.00250 | +0.249 [−1.642, +2.140] | 0.628 / 1.0 | 0.460 |
| sampled (same in both rows) | D500 | sampled s | INCONCLUSIVE | +0.7255 [−0.1705, +1.6296] | 1.00726 | +0.723 [−0.171, +1.617] | 0.147 / 0.357 | 0.911 |
| **primary E_aligned(P̂)** | D100 | counter d, DiD | **INSUFFICIENT_ROUNDS** (2 rounds) | rounds, descriptive only: d −10.3714, −9.4883; DiD −9.9543, −10.4342 | — | — | — | — |
| **primary E_aligned(P̂)** | D500 | counter d, DiD | **INSUFFICIENT_ROUNDS** (2 rounds) | rounds, descriptive only: d −2.3632, −1.9392; DiD −2.6631, −2.8432 | — | — | — | — |

The raw-counter rows are the predeclared **sensitivity** estimator. They are **not promoted to primary**. The primary Q1 remains INSUFFICIENT_ROUNDS; no round was added, imputed or trimmed.

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
* **Thermal slope: ASSOCIATION_ONLY.** +0.4107 %/°C ordinary [0.2171, 0.6047] (log-scale 0.4099 [0.2169, 0.6029]), n 11, dof 4, correlation with launch order −0.56, VIF 1.8. Not causal, not a bound.
* **t_op:** mean −0.92 %, max |dev| 1.27 % → action NONE.
* **Q4: NO_SECONDARY_ESTIMATOR** (n 14, 95 % CI of the candidate ratio [−2.6, 52.9] s).
  * Observation, not a re-analysis: the per-launch values (0.7–154 s) are **unstable candidate ratios, not measured physical delays**. Nothing is trimmed. The predeclared denominator P_region − P_before takes P_before from [C0 − 1 s, C0), which in Stage 2 follows the 5 s warm-up. The denominator can therefore be small, which makes d_eff unstable.
* **Q2:** descriptive output in the JSON.
* **Adequacy** (physical systematic: sensor/counter accuracy, context-power model, load extrapolation, start-phase and thermal imbalance): **NOT_ESTABLISHED**. No component has a hard bound.

## 5. What is and is not claimed

**Claimed:**
* The calibration executed as frozen.
* On this pair, host and software, the predeclared raw-counter sensitivity shows a **read-dependent reduction of the NVML-reported counter energy**:
  * −10.29 % (geometric ratio 0.897) at 10 Hz in-region reads;
  * −2.12 % (0.979) at 2 Hz;
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
