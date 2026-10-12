# Gate 3b — measurement validation `v2.2-measval-01`: result

## Pre-specified result: NOT SUFFICIENT

| decision | outcome |
|---|---|
| D1 | INCONCLUSIVE |
| D2 | INCONCLUSIVE |
| D3 | SYSTEMATIC_OFFSET |
| D4 | NOT_SUFFICIENT |

These are the decisions of the analysis frozen before collection (§2),
reported exactly as its rules produce them. The identical-execution
control was established in all 6 blocks.

**Pre-specified consequence** (plan §7):
* This method, as designed, on this host's board sensors, does **not**
  resolve a ≈ 1 % energy difference for AllGather 2 MiB.
* Claim **C7 stays preliminary and unresolved**.
* No confirmation or workload campaign is proposed on this basis.

**Post-hoc exploratory analyses** (§5) are labeled as such and change
none of the above:
* a statistical association between GPU temperature and energy per
  operation;
* a matched-window comparison of the counter and sampled power.

| | |
|---|---|
| Label | `MEASUREMENT_VALIDATION`; method IDs `v2.2-fail-closed`, `v2.2-aligned-meter-logged`; not pooled with v1 or v2.1 data |
| Allocation | GPUs 4,5 (one NVLink pair), approved by the user on 2026-10-09 |
| Plan | [`stage3_plan_rev1.md`](stage3_plan_rev1.md) §7 |
| Analysis | [`measval.py`](../../../src/ncclenergy/v2/measval.py) as frozen in `f4735b1`; amendment record [`analysis_amendments.md`](analysis_amendments.md) |

## 1. Run record

**Timing and occupancy**:
* 2026-10-09, 19:31–20:05 UTC, about 34 min on GPUs 4,5;
* one benchmark job at a time;
* sequential blocks, with the cool-down below 42 °C before each block.

**Processes**:
* 6 of 6 blocks `COMPLETE`, 72 processes, one attempt each;
* no reruns and no extra blocks;
* 70 of 72 processes `OK` and `VERIFIED` (accepted).

**Exclusions** (pre-specified acceptance, not rerun):

| block | unit | reason |
|---|---|---|
| 1 | Ring/Simple 10 s | `MEASUREMENT_INVALID`: sampler gap 0.18 s on both GPUs, 1 sampler overrun |
| 2 | Ring/LL128 40 s | `MEASUREMENT_INVALID`: sampler gap 0.47 s, 3 overruns |

The host was heavily loaded by other users' jobs during blocks 1–2.

**Health**:
* no health events and no watchdog trip;
* maximum GPU temperature 50 °C;
* all 72 owned process trees exited with code 0 and were reaped, with no
  survivors and no foreign processes;
* GPUs 4,5 afterwards: P8, 5 MiB, no process.

**Execution evidence**: every accepted process ran exactly one (algorithm,
protocol, channels, warps) on both ranks. auto and forced Ring/Simple
matched in every block, so the §3 control was established in 6 of 6.

**After collection** (details and demonstrations in
[`analysis_amendments.md`](analysis_amendments.md)):

| id | change | effect |
|---|---|---|
| AM-1 | the pre-specified analysis crashed in its descriptive section (`df.flags` is pandas' `DataFrame.flags`); one-line fix | the frozen code yields identical decisions on the same data |
| AM-2 | the campaign script exited 1 after a fully successful campaign; fixed | runner invocations are identical, so no measurement is affected |
| — | raw data | committed unmodified (`e963e12`) before any analysis output |

## 2. Pre-specified decisions (counter estimator unless stated)

**Method**:
* values are % = exp(mean of within-block log ratios) − 1, with a 95 %
  t-interval over blocks (df = n − 1);
* blocks ran back to back in one session on one host. They are
  replicates within that session, not independent days or hosts;
* thresholds: ±0.3 % for D1 and D2, ±0.5 % for D3, half-width ≤ 0.4 %
  for D4.

### D1 — length dependence of J/op (2.5 s vs 40 s)

This needs all 4 lengths in a block. The exclusions remove block 1 for B
and block 2 for C.

| config | n | J/op 2.5 s ÷ 40 s | decision |
|---|---|---|---|
| A auto | 6 | +1.80 % [−0.22, +3.85] | INCONCLUSIVE |
| B Ring/Simple | 5 | +0.60 % [−1.01, +2.23] | INCONCLUSIVE |
| C Ring/LL128 | 5 | +0.66 % [−0.90, +2.25] | INCONCLUSIVE |

**Regression of region energy on R**:
* slopes 0.0140, 0.0142 and 0.0143 J/op;
* per-block intercepts from −66 to +90 J, uninformative at this noise
  level;
* overall D1 is **INCONCLUSIVE**.

### D2 — sampled estimators vs counter

The model is d = ln(E_est / E_counter) ~ 1 + 1/T + [LL128], over 6
blocks.

| estimator | intercept | 1/T term as J per region | LL128 term | decision |
|---|---|---|---|---|
| instant (decides) | +0.02 % [−0.11, +0.16] | −27.1 J [−36.3, −18.0] | −0.12 % [−0.48, +0.25] | INCONCLUSIVE |
| lag PowerUsage (secondary) | +0.11 % [+0.06, +0.17] | −29.1 J [−30.4, −27.8] | −0.17 % [−0.28, −0.06] | (secondary; within ±0.3 %) |

The deciding estimator's protocol-term interval reaches −0.48 %, so D2 is
**INCONCLUSIVE**.

The 1/T term is a per-region energy difference between the estimators,
independent of length. In ratio terms: instant ÷ counter = 0.959 at
2.5 s and 0.998 at 40 s (§3). §5.2 examines it over matched windows; it
is **not explained**.

The gate 3a smoke ratios (0.98 / 0.96 / 0.94, single 2.5 s runs) are of
the same size as this per-region difference. They are not evidence of a
protocol dependence.

### D3 — identical-execution control, B ÷ A

B and A are auto and forced Ring/Simple, verified to execute the same
algorithm, protocol, channels and warps.

| length | n | counter | instant | lag |
|---|---|---|---|---|
| 2.5 s | 6 | −0.11 % [−0.72, +0.51] | +0.45 % [−1.72, +2.66] | +0.09 % [−0.61, +0.79] |
| 10 s | 5 | +1.26 % [−1.08, +3.65] | +0.96 % [−1.26, +3.22] | +1.19 % [−1.16, +3.60] |
| 20 s | 6 | **−2.44 % [−4.80, −0.01]** | −2.43 % [−4.77, −0.03] | −2.42 % [−4.76, −0.03] |
| 40 s | 6 | +1.44 % [−0.83, +3.76] | +1.42 % [−0.80, +3.69] | +1.47 % [−0.82, +3.81] |

**Decision (pre-specified rule): SYSTEMATIC_OFFSET.** The 20 s counter
interval excludes 0. The decision stands as defined.

**What the data alone show**:
* Per-block B ÷ A ranges from −5.2 % to +5.2 % at 20 and 40 s.
* The estimators' per-pair B ÷ A values differ from the counter's by at
  most:

  | estimator | 40 s | 20 s | 10 s |
  |---|---|---|---|
  | lag PowerUsage | 0.06 pp | 0.05 pp | 0.19 pp |
  | instant | 0.17 pp | 0.39 pp | 1.16 pp |

* Those differences are small next to the ±5 % spread. The B ÷ A spread
  is therefore present in every estimator; it is not specific to the
  counter path.
* This does **not** show which, if any, of them is accurate (§4).

**Competing explanations for the 20 s result**, none established by
these data:
1. **Sampling variability.** Three unadjusted 95 % intervals were tested.
   One excludes 0 by 0.01 percentage points, and the sign changes across
   10, 20 and 40 s. This is compatible with chance but does not prove it.
2. **A real difference in board energy between the two configurations**
   that the execution evidence does not capture, for example
   initialization or code-path differences from setting `NCCL_ALGO` and
   `NCCL_PROTO`. The evidence covers algorithm, protocol, channels and
   warps only.
3. **Imbalance in an uncontrolled covariate.** Execution order was
   randomized within blocks, but thermal state was not controlled. At
   20 s, B ran at a lower mean GPU temperature than A in 5 of 6 blocks
   (post-hoc observation, §5.1).
4. **Per-process measurement effects.** At 20 s, the boundary and tail
   components are ≤ 0.2 % (§3), too small on their own to produce
   2.4 %.

### D4 — sufficiency

| quantity | value |
|---|---|
| C ÷ A at 40 s (counter), n = 5 | +2.02 % [−1.96, +6.17] |
| half-width | **3.98 %** (needs ≤ 0.4 %) |
| C ÷ A at 2.5 s minus 40 s | −1.40 % [−4.87, +2.19] |

**Decision: NOT_SUFFICIENT.** C ÷ A is descriptive measurement-validation
output, not a test of the AllGather 2 MiB effect.

## 3. Pre-specified descriptive outputs (medians over blocks)

| config | length | tail correction ÷ energy | timing components A1+A2+B1 | instant ÷ counter | lag ÷ counter | update period (probe edges) |
|---|---|---|---|---|---|---|
| A | 2.5 / 10 / 20 / 40 s | 1.53 / 0.33 / 0.16 / 0.08 % | 0.060 / 0.015 / 0.008 / 0.003 % | 0.959 / 0.990 / 0.995 / 0.998 | 0.960 / 0.990 / 0.996 / 0.998 | 99.9–100.0 ms |
| B | same | 1.99 / 0.13 / 0.16 / 0.07 % | 0.065 / 0.014 / 0.008 / 0.004 % | 0.967 / 0.989 / 0.995 / 0.997 | 0.961 / 0.990 / 0.996 / 0.998 | 99.9–100.0 ms |
| C | same | 1.61 / 0.38 / 0.16 / 0.07 % | 0.059 / 0.017 / 0.008 / 0.003 % | 0.958 / 0.991 / 0.994 / 0.997 | 0.957 / 0.990 / 0.995 / 0.998 | 100.0–100.1 ms |

**Other observations**:
* No `END_READ_BEFORE_COMMIT`, phase or period flags.
* **Idle baselines** (24 GPU-readings, 21–30 W):
  * PowerUsage ÷ counter = 1.0029 (range 1.0005–1.0061);
  * instant ÷ counter = 1.0028.

## 4. What gate 3b establishes and what it does not

| property | status in this campaign |
|---|---|
| **Identical executed configuration** (A vs B) | **Established** in 6 of 6 blocks, for the parameters the NCCL profiler records (algorithm, protocol, channels, warps, on both ranks). Not established for NCCL internals outside that record |
| **Measurement agreement between identically executed configurations** (B ÷ A) | **Not established** at ±0.5 %. Process pairs differ by up to ±5 %; formal D3 = SYSTEMATIC_OFFSET |
| **Repeatability** (same configuration and length, different processes) | Residual SD of ln(J/op) under arm × length fixed effects: 1.44 % over all lengths, 1.62 % at ≥ 10 s. This holds under this design: one session, randomized order, no thermal control |
| **Estimator consistency** (counter vs sampled board power) | Steady-state power agrees within about 0.06 % per GPU (§5.2). Whole-region energy differs by about 13 J per GPU per region (≈ 27 J per two-GPU region), which is **unexplained**. Both read the same board's sensors, so agreement is consistency, not accuracy |
| **Absolute accuracy** | **Not assessed.** There is no external power reference. Board-sensor calibration (Stage 2 uncertainty component E) remains uncharacterized |

**Statistical vs systematic uncertainty**:
* The intervals above are **statistical** (block-to-block variation).
* They do **not** include:
  * the sensor calibration;
  * the unexplained per-region estimator gap, which shifts any 2.5 s
    comparison between estimators by about 4 % and a 40 s one by about
    0.26 %;
  * the tail-model bound (Stage 2 addendum component C2, up to about
    1.4 % on historical 2.5 s regions).
* These are **systematic** and are reported, not propagated.

**Energy scope**: all energies are **GPU-board energy** of the two
participating GPUs while the collective runs. That is not the energy of
communication alone (it includes the boards' static and idle power), and
not whole-system energy (no CPU, memory, NVLink bridge, fans outside the
board, or PSU losses).

## 5. Post-hoc exploratory analyses (not pre-specified; change no decision)

**Code and output**:
* [`measval_posthoc.py`](../../../src/ncclenergy/v2/measval_posthoc.py)
  → `data/v2.2/processed/v2.2-measval-01/posthoc_temperature.json`;
* reproduced by `release_repro` step `gate3b_posthoc`.

These analyses were chosen after seeing the data. They generate
hypotheses for future designs and are not evidence for any claim.

### 5.1 Temperature association

Temperature is each process's mean GPU temperature during the region,
averaged over both GPUs.

**Association**:
* In this campaign, ln(J/op) is **associated** with temperature
  (r = 0.93) and with execution position (r = 0.76).
* It is negatively associated with graphics clock (r = −0.88; the clock
  falls from about 1935 to 1897 MHz as the GPU warms).

**Model**:
* ln(J/op) ~ arm × length fixed effects + temperature.
* Temperature slope **0.43 %/°C**; per-block slopes 0.34–0.47, mean 0.43
  [0.38, 0.48].
* Adding temperature reduces the residual SD of ln(J/op) from 1.62 % to
  0.25 % at ≥ 10 s (n = 52), and from 1.44 % to 0.30 % over all lengths
  (n = 70).
* Under that model, temperature accounts for most of the between-process
  variation.

**Warm-up within each block**: mean temperature rises from 36.4 °C at
execution position 1 to 47.6 °C at position 12.

**Earlier data**: the Stage 2 reanalysis of the historical v2 blocks
found an association of the same sign and size (0.37 %/°C,
block-clustered CI 0.22–0.52, on a different measurand).

**What this does not establish**:
* that temperature is the cause, or the sole cause, of the variation.
  Temperature was not randomized: it follows execution position and is
  partly driven by each configuration's own power. Candidate mechanisms
  (leakage, board fan power) are not separated;
* anything about physical measurement accuracy.

**Temperature-adjusted contrasts**:
* ln(J/op) − slope × (T − 40 °C), with the slope estimated leaving the
  contrast's own block out;
* hypothesis-generating only;
* the adjustment can remove part of a real configuration effect.

| contrast | n | raw | adjusted |
|---|---|---|---|
| B ÷ A 2.5 / 10 / 20 / 40 s | 6 / 5 / 6 / 6 | −0.11 / +1.26 / −2.44 / +1.44 % | −0.20 [−0.87, +0.48] / +0.26 [−0.22, +0.75] / −0.17 [−0.60, +0.25] / +0.08 [−0.29, +0.46] % |
| J/op 2.5 ÷ 40 s, A / B / C | 6 / 6 / 5 | +1.80 / +0.25 / +0.66 % | +0.77 [+0.24, +1.31] / +0.49 [+0.06, +0.92] / +0.52 [−0.14, +1.17] % |
| C ÷ A 40 s | 5 | +2.02 % | +1.17 [+0.86, +1.49] % |

**Reading the adjusted contrasts**:
* Even adjusted, the D3 rule would not pass: the intervals at 10 and 20 s
  exceed ±0.5 %.
* The adjusted C ÷ A is of the same size as the ≈ 1.2 % candidate. It
  **does not change C7**, because:
  * it is post hoc;
  * it adjusts for a non-randomized covariate;
  * it would need its own pre-registered design to count as evidence.

### 5.2 Counter vs sampled energy over matched windows

**Windows and units**:
* Per GPU and region, both estimators cover the same window: from the
  midpoint of the start read to the loop end, on CLOCK_MONOTONIC, in
  joules per GPU.
* The counter estimate is C1 − C0 minus context power × tail.
* The sampled estimates integrate POWER_INSTANT, or lag-shifted
  PowerUsage, over that window.
* 140 GPU-regions, mean 136.5 W per GPU.
* Values below are block means with 95 % t-intervals over 6 blocks.

| quantity | instant | lag PowerUsage |
|---|---|---|
| **steady-state power**, counter − sampled (per-second coefficient ÷ mean power) | −0.016 % [−0.062, +0.030] | −0.031 % [−0.043, −0.019] |
| **whole-region energy gap**, counter − sampled, J per GPU per region | +13.5 [12.2, 14.8] | +13.1 [12.9, 13.3] |
| median gap by length 2.5 / 10 / 20 / 40 s (J per GPU) | 14.4 / 14.8 / 15.2 / 12.6 | 13.7 / 13.5 / 12.8 / 11.9 |

**Test of one candidate explanation** ("the counter delta includes the
partial update window before the start read"):
* It predicts a coefficient of 1 on region power × time since the last
  counter commit, and an intercept of 0.
* **Observed**: coefficient 0.26 [0.07, 0.46] (instant) and 0.32
  [0.22, 0.43] (lag); intercept 13.6 [11.7, 15.5] and 12.9
  [11.5, 14.3] J.
* The start phases cluster at a few values, so the direct test is per
  cluster:

  | time since last update | n | gap lag / instant | candidate predicts |
  |---|---|---|---|
  | < 5 ms | 64 | 11.3 / 11.6 J | 0.2 J |
  | 5–50 ms | 50 | 14.2 / 14.8 J | 3.3 J |
  | 50–90 ms | 20 | 16.3 / 15.5 J | 10.9 J |
  | ≥ 90 ms | 6 | 12.7 / 15.2 J | 13.1 J |

  (`window_gap.by_start_phase`.)
* The candidate is therefore **not supported as the explanation**. There
  is a partial association with the counter's phase at the start read.

The gap is about 0.1 s of region power per GPU, close to one counter
update period. That is an observation, not an explanation.

**Which estimator is right about whole-region energy is unknown**:
* The counter is roughly length-invariant: raw D1 +0.6 to +1.8 %, not
  significant.
* The sampled estimators are not: −2.1 to −3.6 % at 2.5 s vs 40 s.
* Resolving the question needs an external reference or a
  controlled-length design.

## 6. Interpretation (bounded)

**What this campaign shows**:
* As designed (one session, randomized order, no thermal control), the
  method does not resolve differences of 0.3–0.5 % between single
  processes of 2.5–40 s on this host.
* The pre-specified decisions record that.

**What the post-hoc analyses add**:
* Between-process variation is strongly associated with GPU thermal
  state.
* The counter and sampled power agree in steady-state power but differ
  by an unexplained, roughly constant amount per region.

**What it does not show**:
* that the method is accurate in absolute terms;
* that temperature is the sole cause of the variation;
* that the 1.2 % LL128 energy penalty for AllGather 2 MiB exists or does
  not exist.

## 7. Future experiments (outside this pilot release; each needs its own plan and approval)

1. A thermally controlled design:
   * preconditioning to a fixed temperature band, or temperature-matched
     interleaving;
   * temperature as a pre-specified covariate;
   * block count from the measured residual SD at a fixed precision
     target.
2. A dedicated test of the per-region estimator gap: varied start phase,
   explicit pre- and post-roll windows, idle bracketing.
3. An external power reference. This needs hardware or administrator
   action.
4. Repetition on other days, hosts, GPU models and driver versions.

## 8. Files

| file | content |
|---|---|
| `data/v2.2/raw/v2.2-measval-01/` | raw evidence (`e963e12`, unmodified). Per-process directories, NCCL logs and stdout stay on disk under `.gitignore`; the sanitized release export carries the files the analysis reads |
| `data/v2.2/processed/v2.2-measval-01/processes.csv` | one row per process, with acceptance and exclusion reason |
| `data/v2.2/processed/v2.2-measval-01/decisions.json` | pre-specified D1–D4, control status and descriptive outputs |
| `data/v2.2/processed/v2.2-measval-01/posthoc_temperature.json` | §5.1 and §5.2 (`window_gap`), POST-HOC |
| `configs/v2.2/frozen_code/` | the analysis and campaign script exactly as used at collection |
| `logs/v2.2/measval01/` | campaign logs (local only) |
