# Methodology

This document describes the implemented measurement and analysis pipeline. Known validity limits are listed in [limitations.md](limitations.md).

## Experimental question and comparison unit

The study asks whether NCCL configurations that minimize collective latency also minimize participating-GPU board energy on the measured server.

An analysis cell is one collective × GPU set × message size × buffer placement. Within a cell, requested configurations combine algorithm `{auto, Ring, Tree}` and protocol `{auto, Simple, LL, LL128}`. `auto` means the corresponding NCCL environment variable is unset.

One timed nccl-tests process measures both an out-of-place and an in-place region. A region contains `w` warmup operations followed by `n` timed operations. The nccl-tests latency row describes the `n` timed operations; region energy covers and is divided by all `n + w` operations.

## System and topology

The recorded environment used eight NVIDIA RTX A6000 GPUs, driver 550.144.03, CUDA driver API 12.4, NCCL 2.27.5, nccl-tests 2.21.1, and two AMD EPYC 7713 CPUs. System metadata was captured before and after each run. Public copies have machine names, user paths, and device serial identifiers redacted; measurement values and topology fields are unchanged.

The completed measurements used:

- `2gpu_nvlink`: GPUs `[4,5]`, one NVLink-connected pair;
- `4gpu`: GPUs `[4,5,6,7]`, two NVLink pairs connected across pairs through a host/PCIe path.

The planned 8-GPU set was skipped because a target GPU was occupied at run start. No 8-GPU measurements exist. The 2-GPU and 4-GPU conditions differ in topology as well as GPU count.

## nccl-tests invocation

nccl-tests was built for `sm_86` at commit `afd59ab` and linked against NCCL 2.27.5. Timed launches use one process with one host thread per GPU (`-t N -g 1`), `float` data, sum reduction, and one message size per process. GPUs are selected by UUID and `CUDA_DEVICE_ORDER=PCI_BUS_ID`; GPUs 4–7 are bound to NUMA node 1.

`NCCL_RUNTIME_CONNECT=0` connects channels during communicator initialization so that connection setup is kept outside the measured collective regions. Other inherited `NCCL_*` variables are removed before a launch.

## Requested and observed NCCL configuration

Before the timed sweep, each requested condition was run with NCCL TUNING logging and data checking. The parser records the algorithm, protocol, and channel range selected for each size, checks whether forced settings were honored, and records nccl-tests `#wrong` counts. Unsupported requests are retained as explicit `not_run_probe_unsupported` final records and are not timed. NCCL 2.27.5 rejected Tree for AllGather and ReduceScatter in this matrix.

The timed processes did not emit TUNING logs. Their effective configuration is therefore inferred by joining each timed condition to the initial probe result. A closing probe was planned but did not run. The smoke-test probe was repeated successfully, but that does not verify selection stability during the full sweep.

## Calibration, regions, and retries

Calibration estimates latency for every supported condition and size. The timed iteration count is then

```text
n = max(min_iters, ceil(target_region_s / estimated_latency_s))
w = max(min_warmup_iters, ceil(warmup_s / estimated_latency_s))
```

The process runs under unbuffered stdout. nccl-tests constructs each result row with separate writes: a preamble before the out-of-place benchmark, the out-of-place result, and the in-place result. The orchestrator reads stdout and takes a monotonic `time.perf_counter()` timestamp plus an energy-counter snapshot when each chunk arrives.

```text
out-of-place region = preamble boundary -> out-of-place result boundary
in-place region     = out-of-place result boundary -> in-place result boundary
```

A final region is accepted only when `n × reported_latency >= 2.0 s` for both placements and other validity checks pass. A short or failed launch may be retried once. The full run retained 406 nonfinal short attempts; only final attempts enter the processed tables. Because a retry immediately follows an initial attempt, retry heating or state carryover remains a possible bias.

## Energy and telemetry

### Cumulative counter

The primary measurement calls `nvmlDeviceGetTotalEnergyConsumption` for every participating GPU at each region boundary. The API returns cumulative millijoules since driver load. For GPU `g`:

```text
E_g [J] = (counter_end_g [mJ] - counter_start_g [mJ]) / 1000
E_region [J] = sum_g E_g
E_raw/op [J] = E_region / (n + w)
```

All GPUs in the active set, and only those GPUs, are included. Counter support was observed on all measured RTX A6000 devices in this environment. The implementation rejects regions with missing boundaries or missing counter values. It does not check for counter decreases or nonpositive deltas and does not attempt wraparound correction; every region in the full processed dataset has positive counter energy.

The reads are synchronous with receipt of stdout chunks, not with direct CUDA events. Pipe delivery, parsing, and the serial per-GPU NVML reads can shift the boundary relative to GPU execution. Region durations of roughly 2.1–3.8 s reduce the relative effect but do not remove it.

### Sampled power

A background thread requests, for each participating GPU, power, utilization, temperature, graphics clock, memory clock, and performance state every 50 ms using the same monotonic clock. GPU process lists are sampled every second. Actual cadence statistics are saved in `qc.json`; for the full run the median interval was 0.0500 s and the maximum was 0.1528 s.

Power is integrated with linear interpolation at the region boundaries and trapezoidal integration between samples:

```text
E_integrated [J] = sum_g integral(power_g(t) [W] dt [s])
```

A second cross-check shifts the integration window by +0.5 s to account empirically for lag in the power reading. Sampled-power integration is not the reported primary energy. In the full data, shifted-window integrated energy divided by counter energy had median 0.9926 over 6,767 covered regions; the unshifted ratio had median 0.9787 over 7,056 regions.

### Idle subtraction

At trial boundaries and periodically during a trial, the implementation waits for every participating GPU to report P8, waits an additional five seconds, and measures a ten-second idle window. Per-trial, per-GPU idle power is the median of that trial's valid idle windows.

```text
E_idle [J] = region_duration_s * sum_g P_idle,g [W]
E_idle_subtracted/op [J] = (E_region - E_idle) / (n + w)
```

Negative idle-subtracted values are retained and flagged rather than clipped. None occurred in the full processed regions. This baseline is a P8 board-idle reference, not a measurement of communication-only dynamic energy.

## Timing and derived metrics

Latency is read directly from nccl-tests, which synchronizes its repeated collective loop and reports average time per operation. The orchestrator does not independently replace this timer. Setup and communicator initialization occur before the first parsed row boundary, although the out-of-place region can include first-call work after that boundary.

For message size `S` from the nccl-tests `size` column and reported time `t_op`:

| metric | implemented definition |
|---|---|
| latency | `t_op` in µs, from nccl-tests |
| algorithm bandwidth | nccl-tests `S / t_op` |
| bus bandwidth | nccl-tests collective-specific scaling of algorithm bandwidth |
| raw energy per op | `E_region / (n + w)` |
| idle-subtracted energy per op | `(E_region - T × sum(P_idle,g)) / (n + w)` |
| J/byte | `E_op / S` |
| GB/J | `S / E_op / 1e9` |
| energy-delay product | `E_op × t_op × 1e-6` in J·s |

`S` is the nccl-tests message buffer size, not physical bytes crossing each link or aggregate fabric traffic. J/byte and GB/J must therefore be read as message-normalized metrics.

## Trial blocks and statistics

Five trial blocks were configured and three completed sequentially over about 7.4 hours of timed launches. Every supported unit starts a fresh process. Unit order is independently shuffled within each block with a deterministic seed. Randomization spreads order effects but does not make sequential blocks statistically independent of shared-host load, thermal history, or time drift.

Processed metrics are first reduced to one value per trial, condition, size, and placement. Reported means and percentile bootstrap intervals resample these three trial-level values, not individual nccl-tests iterations. With only three blocks, the intervals and `P(best)` values are descriptive and should not be interpreted as publication-quality uncertainty estimates.

An optimum is the requested configuration with the lowest three-trial mean among configurations with at least three valid trials. Effective disagreement is counted only when the initial probe indicates different algorithm/protocol/channel selections. Pareto fronts are based on mean latency and mean energy.

## Interference checks

Before each launch, the active GPUs must have no foreign compute process. During a launch, processes on active GPUs are sampled and flagged. Host load and jobs on nonparticipating GPUs are recorded but not controlled. CPU, memory, filesystem, scheduler, and other shared-host interference therefore remain possible, especially for the 4-GPU host-mediated path.

## Public data boundary

The public tree retains processed tables, generated figures, run-level configuration/order/event records, sanitized machine metadata, and representative sanitized parser fixtures. Per-launch raw artifacts and `launches.jsonl` are omitted because they contain private host/network identifiers and create tens of thousands of small files. See [data/raw/README.md](../data/raw/README.md). A public checkout can regenerate reports and figures from processed tables, but cannot rerun raw-to-processed analysis without the private raw archive.
