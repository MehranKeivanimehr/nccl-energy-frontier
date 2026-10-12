# Methodology v2 — measurement protocol

Status: **Phase 1 design** (2026-10-08). Items marked *(Phase 6)* are filled in
and hashed when the confirmatory protocol is frozen
([`prereg.md`](prereg.md)). The pilot (methodology v1, driver 550) is frozen
evidence and is never pooled with v2 data.

## 1. System under test and allocation

* Host `<host>`, 8× RTX A6000, driver 580.178.04, NCCL 2.27.5
  (the library PyTorch 2.9.1 loads), nccl-tests v2.21.1 `afd59ab` plus the
  v2 region patch (§4), CUDA 12.4 toolkit.
* **Allocation.** There is no scheduler. Per the user's instruction
  (2026-10-08), only GPUs that are idle are used, never more than 6 at once,
  default ≤ 4. v2 uses physical GPUs **4, 5, 6, 7** (NUMA node 1). Before
  every block, and before every launch, all four must have no process and
  ≤ 16 MiB used memory. Otherwise the block stops (§9). GPUs are addressed
  by UUID with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Records store the NVML index,
  PCI bus id, and a stable alias `A6000-<first 8 hex of SHA-256(UUID)>`.
* Experiment processes run under `nice -n 10` and
  `numactl --cpunodebind=1 --membind=1`. Only one experiment process runs at
  a time.

## 2. Topology placements (not GPU-count scaling)

| id | GPUs | path | role |
|---|---|---|---|
| `P2_NVL` | 4,5 | one NVLink bridge (NV4) | 2-GPU NVLink pair |
| `P2_PCI` | 5,6 | PCIe within one socket (`NODE`), no NVLink | 2-GPU cross-pair control |
| `P4_MIX` | 4,5,6,7 | two NVLink pairs joined over PCIe/host | 4-GPU deployment |
| `P6_*` | — | optional, not planned | — |

Each placement has a **topology fingerprint**: a canonical string of the
participating GPUs' pairwise `nvidia-smi topo -m` link types, NVLink link
counts, NUMA nodes and PCI bus ids, plus its SHA-256. Results are only ever
compared or reused within a fingerprint.

## 3. Conditions

* Collectives: AllReduce (`sum`), AllGather, ReduceScatter (`sum`); `float`.
* Sizes: exploratory 1 KiB … 1 GiB, ×2 progression (21 sizes);
  confirmatory set frozen in Phase 6. A size runs only if
  `3 × max(sendbytes, recvbytes) + 2 GiB` fits in the free memory measured at
  preflight (else `SKIPPED_MEMORY`).
* Configurations (requested):
  * `auto` — neither `NCCL_ALGO` nor `NCCL_PROTO` set;
  * `{Ring, Tree} × {Simple, LL, LL128}` — both variables set.

  Tree is requested only where NCCL implements it (AllReduce). LL128 is
  requested only where NCCL enables it by default for that placement, read
  from NCCL's own tuning table at init (LL128 bandwidth > 0 with
  `NCCL_PROTO` unset). Otherwise the condition is recorded as
  `NOT_ENABLED_BY_NCCL` and not forced. Half-forced configurations
  (e.g. `Ring/auto`) are not run in v2.
* Channel/CTA counts, power limits and clocks are not varied.

## 4. Measurement region (one region per fresh process)

The pinned nccl-tests is built with
`patches/nccl-tests-afd59ab-v2-region.patch`. With the patch, and with
`NCCLE_REGION_FILE` set, each benchmark of one size and placement runs as
follows:

1. Initialization, buffer allocation and data initialization (outside).
2. `w` warm-up collectives, stream sync, thread barrier (outside).
3. Explicit stream sync on every GPU; barrier. Each thread reads, for its
   GPU, `CLOCK_MONOTONIC` and `nvmlDeviceGetTotalEnergyConsumption`, and
   records the NVML return code and the read duration; barrier.
4. CUDA start event on every GPU's stream.
5. **R** timed collectives (`R = iters × agg_iters`; CUDA graphs, blocking
   mode and per-iteration events are refused when the patch is active).
6. CUDA end event on every stream; stream sync (nccl-tests' own
   `completeColl`); host timer stops.
7. Stream sync; barrier; per-GPU counter and monotonic-clock read; barrier.
8. Data check (`-c 1`) runs **after** the region (correctness, outside).
9. One JSON record per region is appended to `NCCLE_REGION_FILE`. It holds
   per-GPU counter start/end values, return codes, read timestamps, event
   elapsed time, and the run parameters.

Only the out-of-place placement is measured (`NCCLE_PLACEMENT=oop`). In-place
is not run.

**Definitions** (per region, participating GPUs `g` only):

```
E_raw          = Σ_g (E_end[g] − E_start[g])                     [J, counters in mJ]
J_per_coll     = E_raw / R
T_region       = max_g t_end[g] − min_g t_start[g]               [CLOCK_MONOTONIC]
t_gpu          = max_g eventElapsed[g] / R                       [GPU-valid latency per collective]
t_host         = nccl-tests host timer / R                       [kept for comparison]
E_idle_adj     = E_raw − Σ_g P_idle[g] · T_region                (secondary)
EDP            = J_per_coll × t_gpu
bytes/J        = size_bytes / J_per_coll                          (size = nccl-tests size column)
```

Setup, warm-up and teardown durations are recorded separately: process
start → first region read, region end → process exit.

## 5. Actual NCCL configuration

* **Requested**: the `NCCL_ALGO` / `NCCL_PROTO` values set.
* **Allowed**: NCCL's enabled Func/Proto/Algo matrix and its init tuning
  table. These are recorded by untimed probe processes run with
  `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING`; that setting also logs
  every call, so it is never used in timed processes.
* **Actual**: a user-space NCCL profiler plugin
  (`native/nccl_profiler_algo`, API v4, `NCCL_PROFILER_PLUGIN=<abs path>`)
  counts every collective by `(func, algo, proto, nChannels, nWarps,
  count, datatype)` and writes one JSON file per rank at communicator
  finalize.
  * A region is **VERIFIED** if, for every rank, all collectives of the
    measured size (warm-up + `R` + data check) carry one single
    `(algo, proto)`, and that pair equals the requested one when forced.
  * Otherwise the region is **UNVERIFIED** and excluded from
    configuration-specific claims.
  * The plugin's overhead is measured in Phase 4. If it changes small-message
    latency by more than 2 %, it is not used in timed runs and this section
    is revised before any campaign.

Timed processes use `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV`, written to
a per-process log file. That records the environment NCCL accepted, the
transports, and the plugin-load lines.

## 6. Energy measurement

* **Primary**: sum of `nvmlDeviceGetTotalEnergyConsumption` deltas over
  participating GPUs, read only at the two region boundaries (§4).
* **Cross-check**: a 50 ms NVML sampler (separate Python process) samples,
  for each participating GPU:
  * power (`nvmlDeviceGetPowerUsage`);
  * temperature, graphics and memory clocks, utilization, P-state;
  * clock-event (throttle) reasons and the enforced power limit;
  * process list (every 1 s).

  Integration: trapezoidal over the region boundaries with linear
  interpolation at the ends. It is computed both on the raw window and on the
  window shifted by the measured averaging lag of `PowerUsage`, determined in
  Phase 4. The sampler never calls the energy counter or
  `NVML_FI_DEV_POWER_INSTANT` (on driver 550 both perturbed the counter).
  Phase 4 re-tests this on 580.
* **Retained per region**: counter start/end and deltas, return codes,
  monotonic timestamps, actual sample intervals, gaps > 3× the period,
  integration bounds, and counter-vs-integration ratio.
* **Validity checks** (never repaired or clamped):
  * unsupported or permission error → `COUNTER_UNAVAILABLE`;
  * delta ≤ 0 → `COUNTER_NONPOSITIVE` (reversal or wrap suspected);
  * `t_end ≤ t_start` → `TIME_REVERSAL`;
  * implied average power per GPU outside [2 W, 320 W] →
    `COUNTER_IMPLAUSIBLE`;
  * sampler gaps → `SAMPLES_MISSING` (flag only).
* **Calibration (Phase 4)** establishes, on driver 580:
  * counter resolution and update period (idle and load);
  * whether counter reads perturb the counter (dose test);
  * `PowerUsage` update period and averaging lag;
  * the minimum region duration such that the counter's update granularity
    contributes ≤ 1 % (`T_min`) and the target duration `T*`.

## 7. Idle baselines

* Before and after every block, after all allocated GPUs report P8 and 10 s
  more, each GPU is measured for 30 s: counter delta / duration, plus sampled
  power.
* Both baselines are kept; the per-block baseline used for idle adjustment
  is their mean, and the pre–post difference is reported.
* Idle-adjusted energy is secondary. It is not communication-only energy:
  P8 idle excludes the active P-state uplift. Negative values are flagged,
  not clamped.

## 8. Calibration of R (separate from trials)

* Calibration processes (labelled `CALIBRATION`, never analysed as trials)
  run each condition over all sizes. Sizes ≤ 1 MiB use 2,000 iterations;
  larger sizes use 10 iterations after 2 warm-ups. They give `t_gpu,cal`.
* `R = max(10, ceil(T* / t_gpu,cal))` and `w = max(5, ceil(0.2 s / t_gpu,cal))`.
* R and w are frozen in `calibration_frozen.json` (hashed) before any
  exploratory or confirmatory block, and the same schedule is used in every
  block.
* There are no outcome-dependent re-runs. A region shorter than `T_min` is
  `QC_SHORT` and is excluded, and the count is reported.

## 9. Blocks, ordering, retries, stop conditions

* **Block** = one pass over all units of the campaign, each unit a fresh
  process.
  * Order is shuffled with `random.Random("<campaign>-<seed>-<block>")`, and
    the planned order and the wall-clock order are both stored.
  * Pre-block: allocation check, health snapshot, idle baseline.
  * Post-block: idle baseline, health snapshot, summary.
* **Stability rule**: a block starts only when all allocated GPUs are in P8
  and below 45 °C. There is no cool-down between units; randomization spreads
  thermal drift, and temperature is recorded per region.
* **Retry rule** (outcome-blind, predefined): an attempt that fails for an
  infrastructure reason gets exactly one retry at the end of the same block.
  Infrastructure reasons are non-zero exit, timeout, missing region record,
  or a foreign process detected and then gone. All attempts are kept with
  their relationships, and QC failures are never retried.
* **Per-process timeout**: `max(120 s, 10 × expected duration)`.
* **Stop the block** (all artifacts kept; no automatic resumption):
  * a foreign process or memory > 16 MiB on any allocated GPU;
  * timeout;
  * Xid/CUDA/NCCL errors in accessible logs;
  * temperature ≥ slowdown − 15 °C (i.e. ≥ 80 °C);
  * active clock-event reason in {HW slowdown, HW thermal, HW power brake,
    SW thermal} during a region, or SW power cap;
  * enforced power limit ≠ preflight value;
  * free disk < 20 GB;
  * a counter validity failure in two consecutive units;
  * three consecutive failed units.

## 10. Endpoints and statistics (pre-registration skeleton)

Frozen research questions:

1. **RQ1** Does the topology-aware Energy-SLO policy reduce raw
   participating-GPU board energy per collective relative to NCCL `auto`,
   while meeting the latency SLO?
2. **RQ2** Do latency-optimal and raw-energy-optimal configurations differ
   after accounting for block-level uncertainty?
3. **RQ3** Does the best microbenchmark policy change energy per step of a
   controlled synthetic PyTorch workload?
4. **RQ4** How sensitive are recommendations to placement (three
   fingerprints), temperature and clocks? NCCL-version sensitivity is out of
   scope for the confirmatory campaign (one NCCL version); it is listed as
   untested.

Definitions:

* **Unit of analysis**: block. Per cell `c` (placement × collective × size)
  and configuration `k`, the block value is `Y = ln(J_per_coll)` and
  `L = ln(t_gpu)`.
* **Policies**, fixed from exploratory data before confirmation:
  * `auto`;
  * `lat-opt` (minimum exploratory mean `t_gpu`);
  * `Energy-SLO`: minimum exploratory mean `J_per_coll` subject to
    exploratory mean `t_gpu ≤ 1.05 × t_gpu(auto)`.
* **Meaningful effect**: ≥ 5 % lower raw energy per collective.
  **SLO**: `t_gpu(policy) ≤ 1.05 × t_gpu(auto)` in the same block.
* **RQ1 test**: cells where the policy's verified configuration differs from
  `auto`'s.
  * Paired block differences `D = Y_policy − Y_auto`.
  * Mean and t-based 95 % CI over blocks; Holm–Bonferroni across those cells
    (one-sided α = 0.025).
  * A saving is **confirmed** if the Holm-adjusted upper bound of `exp(mean D)`
    is < 1, and the latency ratio's 95 % upper bound is ≤ 1.05.
  * The saving is **meaningful** if, in addition, the point estimate is
    ≤ 0.95.
* **RQ2**: for cells where the exploratory latency- and energy-optimal
  configurations differ.
  * Paired test of both orderings (energy of lat-opt > energy of en-opt, and
    latency of en-opt > latency of lat-opt), Holm across cells.
  * Where they coincide, TOST equivalence of the best two distinct
    configurations with a ±3 % margin is reported.
  * A non-significant result is reported as "inconclusive", never as
    "agree".
* **Robustness, pre-specified**:
  * Hodges–Lehmann / Wilcoxon signed-rank;
  * block bootstrap (resampling blocks);
  * idle-adjusted energy;
  * counter vs integration;
  * order and temperature associations — regression of the paired
    difference on position-in-block difference and mean temperature
    difference.
* **Outliers**: none removed. **Missing**: complete pairs per cell only; the
  missing count is reported per cell.
* **Block count** *(Phase 6)*: smallest `N ≤ 12` such that the predicted
  95 % CI half-width of `mean D` is ≤ 2.5 % in ≥ 80 % of RQ1 cells. The
  prediction uses the larger of the exploratory-v2 and pilot-v1 SD estimates.
  If `N = 12` does not reach this, the under-powered cells are listed before
  the campaign.

## 11. Provenance per run

Each unit record carries:
* UTC and monotonic timestamps;
* git HEAD plus a SHA-256 of `git diff HEAD` (empty if the tree is clean);
* config and frozen-calibration hashes;
* the seed;
* software versions (driver, NVML, CUDA, NCCL, nccl-tests commit and patch
  hash, plugin hash);
* GPU NVML indices, PCI bus ids and aliases, and the `CUDA_VISIBLE_DEVICES`
  mapping;
* topology fingerprint;
* NCCL environment;
* requested, allowed and actual configuration;
* `w`, `R`, region duration and counters;
* paths of telemetry, logs and profiler files;
* exit status, QC status, attempt id / parent attempt;
* observability gaps (`UNAVAILABLE_DUE_TO_PERMISSIONS`).

v2 raw data go to `data/v2/raw/<campaign>/` and are append-only.

## 12. Amendments after Phases 3–4 (2026-10-08, before any exploratory or confirmatory block)

Each amendment below is based on measurements in
[`phase3_preflight.md`](phase3_preflight.md) and
[`phase4_validation.md`](phase4_validation.md).

1. **Commit-aligned end read (patch v2.1).** The counter commits every
   100 ms, and a read discards the energy accumulated in the current window.
   §4 step 7 becomes: wait for the GPU's next commit after the loop end, read
   8 ms later, measure one more window of idle-with-context power, and
   subtract `P_ctx × (t_commit − t_loop_end)`. `T_region` runs from the start
   read to the loop end. Missing, inconsistent or imprecise phase,
   end-read-before-commit, or implausible tail power make the region
   `MEASUREMENT_INVALID`.
2. **Sampler.** It adds `NVML_FI_DEV_POWER_INSTANT`, shown non-destructive
   on driver 580. The cross-check reports ∫instant power and ∫PowerUsage
   (shifted 0.75 s) against the counter.
3. **Host-path placement.** It is `P2_PCI_SHM` = GPUs 5,6 with
   `NCCL_P2P_DISABLE=1`, because P2P/direct-pointer over PCIe hangs on this
   host. Its configurations are relative to that environment.
4. **SW power cap.** This driver reports it at idle far below the limit. It
   is a stop condition only when region power is ≥ 90 % of the enforced
   limit; otherwise it is recorded as `SW_POWER_CAP_FLAG_BELOW_LIMIT`.
   HW/SW thermal and HW power brake still stop the block. Cumulative
   throttle counters are diffed per block.
5. **Allocation check.** It uses the NVML v2 `used` memory, which excludes
   601 MiB of driver reservation.
6. **Region length.** `T* = 2.5 s`, `T_min = 2.0 s`.
7. **Stop on hang.** Probe and calibration commands stop all GPU work on
   the first timeout.
