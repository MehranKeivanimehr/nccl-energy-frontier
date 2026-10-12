# v2.2 audit — clarifications to Stage 2 and revised Stage 3 plan (rev 1)

**Status:**
* Stage 2 is acknowledged.
* Stage 3 GPU execution is **not approved**, and GPU allocation is
  **not confirmed**.
* Nothing in this document has been run on a GPU.
* The evidence cited comes from CPU-only tests and reanalysis of existing
  data:
  * `tests/v2/test_v2_no_counter_reads.py`;
  * `data/v2/reanalysis/2026-10-09_v2.2-audit-addendum1/`.

## 1. The watchdog and sampler add no cumulative-counter reads inside measured regions

**Static inventory** (`grep` and an AST test over `src/ncclenergy/v2`). Calls to
`nvmlDeviceGetTotalEnergyConsumption`, and when each one happens:

| call site | when it runs | inside a measured region? |
|---|---|---|
| C patch `nccleProbePhase` (patch v2.1, unchanged) | after warm-up sync, before the start read; at most 4 update periods | no: before the start read |
| C patch start read `e0` | between barriers, before the start CUDA event and the timed loop | region boundary (defines the start) |
| C patch end read `e1` | after the loop, stream sync and barrier; 8 ms after the predicted update | no: after the loop end |
| C patch post read | one update period after the end read | no |
| `aligned_meter.AlignedMeter._read` | workload rank 0: probe before the region, then start and stop | boundary only |
| `runner.idle_baseline` (counter mode only) | before and after a block, GPUs idle, no experiment process | no |
| `capabilities.py --probe-counter` | only on allowed GPUs with no process | not used by campaigns |
| `validate.py` | Phase 4 tool, including the deliberate dose experiment | not used by campaigns |

**Code with no counter call**:
* `telemetry.Sampler` (the only NVML field it reads is
  `NVML_FI_DEV_POWER_INSTANT`);
* `watchdog.py` (makes no NVML calls at all; it reads the sampler's data);
* `procctl.py` (reads `/proc` only);
* `preflight.py`, `qc.py`, `phase10.py`, `workload.py`.

`test_static_counter_call_sites_are_allow_listed` enforces this.

**Dynamic audit** (`tests/v2/helpers/nvml_call_audit.py`, output in the
review package):
* **Method**: every NVML function was wrapped with a timestamped logger;
  the real `runner.idle_baseline` and `runner.run_unit` then ran in
  counter mode, with a fake benchmark that writes a valid region record.
* **Idle baseline**: 4 counter reads (2 GPUs × start and end), all
  outside any region.
* **`run_unit` overall**: 0 counter reads, with the counter mock set to
  raise if called.
* **Measured 2.5 s region** (between the start read and the loop end):
  * 1,008 NVML calls, all from the sampler thread;
  * 0 counter reads;
  * functions called: PowerUsage, temperature, clocks, utilization,
    P-state, clock-event reasons, enforced limit, memory info, the
    `POWER_INSTANT` field, and process lists.

`test_no_counter_reads_by_controller_during_a_unit` enforces this.

**Not covered by this demonstration**:
* **The benchmark's own boundary reads.** They are by design, at the
  positions in the table above.
* **`POWER_INSTANT` reads.** These are not counter reads. The Phase 4 dose
  test on this driver found them non-destructive, a local empirical
  result.
* **NVML use by other processes** (for example monitoring tools elsewhere
  on the host). We can neither observe nor control it. A foreign
  *process on the allocated GPUs* stops a run, but a remote NVML query
  does not show up as one.
* **Real hardware.** The smoke test (§6) will repeat the controller-side
  audit on the real GPUs using an audit wrapper loaded only for that test.

## 2. What the reported 0.10 % bounds (and what it does not)

The "≤ 0.10 % of region energy" in `stage2_report.md` §4.1 is the
**worst-case sum of read-timestamp and commit-phase half-widths** only:

**Conditions**:
* It assumes the local interpretation of the counter: a read discards
  energy accumulated since the last update, and updates sit on a grid
  measured by the probe.
* It is **not** total physical measurement accuracy.

Decomposition, over 984 confirmatory units (2,784 GPU-regions). Fractions
are of each unit's region energy, with worst-case addition over the
unit's GPUs (`uncertainty_components.json`):

| component | what it is | median | p95 | max |
|---|---|---|---|---|
| **A1** start-read timestamp | reset instant anywhere inside the start-read call (median call 1.4 ms) | 0.017 % | 0.035 % | 0.056 % |
| **A2** end-read timestamp | changes the post window used for idle-with-context power, hence the tail correction | 0.012 % | 0.017 % | 0.021 % |
| A1 + A2 | read-timestamp uncertainty | 0.028 % | 0.047 % | 0.070 % |
| **B1** commit phase | predicted update time inside the probe bracket (about 2.3 ms) | 0.029 % | 0.032 % | 0.034 % |
| **B2** period model | predicted-update drift over 32–65 periods, assuming the load-measured period stability (0.002 ms per period) | 0.0016 % | 0.0018 % | 0.0019 % |
| **B3** update semantics | which time interval a committed value covers | not identifiable from these data | | |
| **C1** tail correction magnitude | idle tail removed (loop end → update) | 1.34 % | 1.96 % | 2.41 % |
| **C2** tail-model sensitivity (one-sided) | over-estimate of region energy if the true tail power were the region's mean power instead of idle-with-context power | 0.41 % | 1.23 % | 1.39 % |
| **D** quantization | 1 mJ per reading | 0.001 % | | 0.001 % |
| **E** sensor accuracy | board sensor vs true energy | **not characterized**: no external reference; all estimators share the sensor | | |

The earlier 0.10 % is A1 + A2 + B1 (maximum 0.10 %). It bounds boundary
timing and phase only.

**Update-period evidence** (`period_evidence` in the same file):
* **Under load (Phase 4)**: 100.000 ms, p10–p90 99.998–100.002 ms, 3 runs.
* **At P8 idle (Phase 4)**: one run, median 101.3 ms, p10–p90
  93.5–101.5 ms. The period was not resolved at that read spacing.

  **The Phase 4 report's "100.00 ms idle and under load" overstated the
  idle evidence.** That report now carries a correction note.
* **At idle-with-context, from the confirmatory probe edges**: median
  99.95 ms per period, p95 100.49 ms, max 105.9 ms. Each edge is bracketed
  to about 2 ms, so this resolves the period only to about ±1 ms. The
  post-loop tail and post window are in this state.

**For the AllGather 2 MiB candidate (≈ 1.2 % counter-based penalty)**:
* **Timing terms are small**: A + B1 ≈ 0.05 % per arm.
* **The tail correction and its model sensitivity are not small**:

  | arm | tail correction | C2 bound |
  |---|---|---|
  | auto | 1.66 % | 0.94 % |
  | LL128 | 1.36 % | 0.79 % |

  The tail fraction differs between arms by 0.25 percentage points
  (median paired).
* **The tail model therefore matters at the historical 3 s region
  length.** The tail shrinks roughly as 1/length, so a 40 s region
  reduces it about 13-fold. That is the reason for the length series in
  §7.
* **Sensor accuracy (E)** can't be bounded with on-board sensors.

## 3. Identical-execution control (A = automatic, B = forced Ring/Simple)

Historically, automatic executed RING/SIMPLE/4 at P2_NVL AllGather 2 MiB,
and forced Ring/Simple executed RING/SIMPLE/4 too.

**Rule, fixed before collection.** In each block, B is an
identical-execution control only if the profiler records of **every
rank** of both processes show:
* the same algorithm, protocol, channel count and warp count;
* the same exact operation counts;
* a single configuration per process.

If they differ in a block:
* That block's B/A contrast is **excluded from the negative-control
  analysis** and reported with both executions.
* A and C stay in the length and estimator analyses, labelled with what
  they executed.
* Nothing is rerun, and no configuration is substituted.

If they differ in **2 or more of the 6 blocks**, the negative control is
`CONTROL_NOT_ESTABLISHED` and decision D3 (§7) is reported as
inconclusive.

The C/A contrast is interpreted as "LL128 vs Simple at equal channels"
only if C and A executed the same channel count. Otherwise it is reported
as differing in protocol and channels.

## 4. Blocks, not 72 replicates

The design is **6 blocks × 12 within-block conditions** (3 configurations ×
4 region lengths), in a seeded random order within each block.

* **Unit of inference**: the block. Every contrast and slope is computed
  within a block first, then summarized over 6 blocks with t-intervals
  (df = 5).
* **Not independent**: the 72 processes are never treated as 72
  independent replicates, and LOBO-style overlapping scores are not
  used.
* **One session**: all 6 blocks fall in one session, so they are
  within-session replicates.

## 5. Placement if GPUs (6,7) are allocated instead of (4,5)

* **Distinct identity**: the placement gets the id `P2_NVL_67`, with its
  own concrete fingerprint. It is never relabelled `P2_NVL`.
* **Compatibility checks**: preflight (read-only) verifies and records:
  * the link pattern NV4;
  * active NVLink link counts;
  * NUMA node;
  * the canonical fingerprint against `e65ef2bdbf86ae93`;
  * GPU model, driver, VBIOS version, PCIe generation and width, enforced
    power limit, persistence and ECC mode;
  * idle power over 30 s.
* **Comparisons with historical data**: with GPUs 4,5 these are
  descriptive only, never pooled.
* **If the canonical fingerprint or any recorded property differs**: the
  pair is "not equivalent". Measurement validation can still run there,
  because it validates the method, not the historical cell.
* **Tuner**: the historical P2_NVL tuner profile does not apply, and the
  guarded plugin rejects it by PCI bus ids.

## 6. Gate 3a — minimal real-GPU smoke test (separate approval)

**Purpose**: on real hardware, confirm that the following work, before any
measurement-validation data are collected:
* the actual v2.2 runner, with owned processes, watchdog and fail-closed
  QC;
* NVML logging;
* execution evidence;
* bounded owned-process shutdown.

| item | plan |
|---|---|
| campaign id | `v2.2-smoke-01` (label `SMOKE`; never used for inference); data under `data/v2.2/raw/v2.2-smoke-01/` |
| GPUs | the NVLink pair you allocate (2 GPUs) |
| S1 preflight (read-only) | topology, idle and no-process check, versions, disk, temperature; NVML counter availability read only on the two idle allocated GPUs |
| S2 three short units | AllGather 2 MiB, A, B and C, one 2.5 s region each, through `runner block` with full logging and 30 s idle baselines. Check: region record fields (start, end and post counter values, read brackets, edges, phase, commit, loop end), profiler evidence on both ranks, sampler telemetry, v2.2 QC fields, and the controller-side NVML audit (0 counter reads during regions) |
| S3 bounded shutdown 1 (self-initiated) | one AllGather 2 MiB unit planned at 20 s with a smoke-only per-unit wall-clock limit of 5 s. Our own watchdog trips; SIGTERM, then SIGKILL after ≤ 10 s, then reap. Verify no process of ours on the GPUs within 15 s, memory back to ≤ 16 MiB, unit `SAFETY_STOP`, campaign stopped |
| S4 bounded shutdown 2 (controller signal) | a 20 s unit; after 5 s, SIGTERM to **our own** controller. Verify the cleanup and the `INTERRUPTED` status |
| not done | no overheating, no load or contention generation, no fault injection, no signal to any process not ours |
| pass criteria | every check passes. Any failure stops the smoke test, the evidence is kept and reported, and no measurement validation follows |
| code needed first (CPU-tested before the smoke) | the per-unit wall-clock limit option; the controller NVML audit wrapper as a test-only shim (no change to production code paths); `v2.2-smoke-01.yaml` with an explicit allocation |
| resources | 1 benchmark process; under 0.5 GiB GPU memory per GPU; under 100 MB disk; GPU busy for about 2 min; wall time about 10 min |

## 7. Gate 3b — measurement validation (revised; requires a separate approval after the smoke review)

**Campaign**: `v2.2-measval-01` (method `v2.2`), data under
`data/v2.2/raw/v2.2-measval-01/`, never pooled with v2.1.

**Conditions**: P2_NVL (or `P2_NVL_67`) AllGather 2 MiB FP32,
out-of-place.
* **Configurations**: A automatic; B forced Ring/Simple; C forced
  Ring/LL128.
* **Region lengths**: 2.5, 10, 20 and 40 s.
* **Operation counts**: R = ceil(T / t) from the historical per-op times
  (52.3 µs for A and B, 50.8 µs for C), frozen in a schedule file before
  collection.
* **Warm-up**: 0.2 s.
* **Structure**: 6 blocks of 12 conditions, plus 30 s idle baselines
  before and after each block (the low-power regime).
* **Budget**: about 40 min; hard stop at 60 min.

**Health**:
* **Stop conditions**: any HW slowdown, HW thermal or power-brake bit;
  SW thermal; SW power cap at ≥ 90 % of the limit; GPU temperature
  ≥ 80 °C; a foreign process on the allocated GPUs; telemetry stale
  > 2 s or ≥ 3 sampler errors; disk < 20 GB free; per-process timeout of
  3 × planned + 60 s.
* **On a trip**: the owned tree gets SIGTERM, then SIGKILL after 10 s, and
  is reaped. The reason is recorded and the campaign stops.
* **No retries**: `max_attempts: 1`.

**Acceptance (fixed now)**:
* v2.2 fail-closed QC;
* execution evidence on every rank;
* exact operation counts;
* region length within ±10 % of target;
* the §3 rules for the control.

An excluded process is reported, not rerun.

### Analysis and decision rules (fixed before collection)

**Per-process quantities**:
* energy per operation from the counter, integrated `POWER_INSTANT` and
  lag-adjusted PowerUsage;
* tail-correction fraction;
* the components A1, A2 and B1;
* the update period from that process's probe edges;
* executed configuration.

All summaries are over 6 blocks (df = 5).

| id | question | analysis | decision |
|---|---|---|---|
| D1 | Does energy per operation depend on region length? | Per block and configuration: regress region energy on R over the 4 lengths (slope = J/op, intercept = boundary offset). Report the block-mean slope and intercept with t-CIs, and ln(J/op at 2.5 s ÷ J/op at 40 s) | **no material length effect** if the 95 % CI of that log ratio lies within ±0.3 %; **length effect** if it lies entirely outside ±0.3 %; otherwise **inconclusive** |
| D2 | How does estimator disagreement change with duration and protocol? | d = ln(E_instant / E_counter), and the same for lag-PowerUsage, per process. Per block: OLS of d on 1/T and a protocol indicator (C vs A, B); average the coefficients over blocks | **protocol-independent disagreement** at the needed precision if the 95 % CI of the protocol coefficient lies within ±0.3 %; **protocol-dependent** if entirely outside; otherwise **inconclusive**. The 1/T coefficient is reported as a fixed per-region offset in joules |
| D3 | Noise floor and identical-execution control | ln(E_B / E_A) per block, at each length, under each estimator (only blocks passing §3) | **control passes** if, at 10, 20 and 40 s, the counter CI lies within ±0.5 % and contains 0; **systematic offset** if a CI excludes 0; otherwise **inconclusive**. `CONTROL_NOT_ESTABLISHED` per §3 means inconclusive |
| D4 | Is the uncertainty characterized well enough to investigate the ≈ 1.2 % candidate? | C/A at 40 s (tail correction ≈ 0.1 %) under the counter: CI half-width; C/A at 2.5 s vs 40 s (does the protocol difference depend on the tail?) | **sufficient** if D1, D2 and D3 each pass and the C/A 40 s half-width is ≤ 0.4 %. Then, and only then, a separate confirmation plan may be proposed: new blocks, block count from the measured SD at a fixed precision target, its own approval. **Not sufficient** otherwise |

Also reported, without decisions:
* tail fraction against 1/T;
* A1, A2, B1 by length and configuration;
* the update period from the probe edges;
* any `END_READ_BEFORE_COMMIT` or phase flags;
* counter versus estimator agreement in the idle baselines (low-power
  regime).

**What Stage 3 does not do**:
* The C/A values it produces are **descriptive measurement-validation
  output, not a confirmatory test** of the AllGather 2 MiB penalty. They
  are not used to decide anything about that effect beyond D4.
* There is no significance-based stopping, no retry, and no extension of
  the block count.

**If calibration remains inconclusive or not sufficient**:
* The report states which of D1–D4 failed or was inconclusive, with the
  numbers.
* It concludes that this method on this host's board sensors does not
  resolve a ≈ 1 % energy difference for AllGather 2 MiB.
* Claim C7 stays **preliminary**, worded as estimator-sensitive and
  unresolved.
* The report lists what would be needed, without running any of it:
  * an external power reference (needs hardware or administrator
    action);
  * a different region design;
  * repetition on other days or hosts.
* No confirmation or workload campaign is proposed on that basis.

## 8. What I need from you before any GPU work

1. **Review** of this plan and of the review package.
2. **Explicit allocation** of one NVLink pair, and who or what confirms
   it. I won't assume (4,5) or (6,7).
3. **Approval of gate 3a (smoke) only.** Gate 3b needs a separate
   approval after the smoke results are reviewed.
