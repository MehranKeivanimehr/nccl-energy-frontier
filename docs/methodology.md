# Methodology

> Pilot (methodology v1) document, superseded by methodology v2.
>
> **v2.2 pilot release (2026-10-09).** The current claim status, limitations and reproduction instructions are in
> [`release/claims.md`](release/claims.md), [`release/limitations.md`](release/limitations.md) and
> [`release/reproduction.md`](release/reproduction.md). This document is kept as the record of its stage.


This document describes how every number in this repository is produced. The
code paths named here are in `src/ncclenergy/`. Limitations are collected in
[limitations.md](limitations.md).

## 1. Question and unit of analysis

**Question.** On a single multi-GPU server, do the NCCL configurations that
minimize collective latency also minimize the energy of the collective?

**Unit of analysis (a "cell").** One collective (AllReduce, AllGather,
ReduceScatter) × one GPU set × one message size × one buffer placement
(out-of-place or in-place). Within a cell we compare **requested
configurations**: NCCL algorithm ∈ {auto, Ring, Tree} × protocol ∈ {auto,
Simple, LL, LL128}. "auto" means the corresponding environment variable is
*unset* and NCCL's tuner chooses.

## 2. System under test

Captured automatically at the start and end of every run in
`data/raw/<run>/metadata_{start,end}/` (raw command output plus a parsed
`machine.json`): `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi -q`,
`nvidia-smi nvlink -s`, `nvcc --version`, `lscpu`, `numactl -H`, `uname -a`,
hostname, `/etc/os-release`, git commit and dirty state of this repository,
nccl-tests commit, NCCL library version (via `ncclGetVersion`) and SHA-256,
SHA-256 of every nccl-tests binary, PyTorch version and its NCCL version,
`pip list`, per-GPU NVML data (name, UUID, serial, PCI bus id, VBIOS, power
limits, max clocks, persistence/compute/ECC mode, NVLink state and peers,
energy-counter support, processes present).

For the reported runs: 8× NVIDIA RTX A6000 (GA102, 48 GB, 300 W board limit),
driver 550.144.03 (CUDA 12.4 driver API), 2× AMD EPYC 7713, Ubuntu 24.04,
kernel 6.11. GPUs are bridged in NVLink pairs (0–1, 2–3, 4–5, 6–7, `NV4`);
all other GPU pairs communicate over PCIe (`NODE` within a socket, `SYS`
across sockets).

## 3. Benchmark

* **nccl-tests** v2.21.1, commit `afd59ab`, built by
  `scripts/build_nccl_tests.sh` for `sm_86` with CUDA 12.4 `nvcc`, linked
  against **NCCL 2.27.5** from the `nvidia-nccl-cu12` wheel (the NCCL that
  PyTorch 2.9.1+cu128 loads). nccl-tests is used unmodified.
* One process, one host thread per GPU (`-t N -g 1`), `float` data, `sum`
  reduction, message sizes from the YAML (`-b S -e S`: one size per process
  for timed launches).
* GPUs are selected by UUID (`CUDA_VISIBLE_DEVICES=<uuids>`,
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`); the NCCL log's `nvmlDev` field is parsed to
  confirm the mapping.
* Processes on GPUs 4–7 are bound to their NUMA node
  (`numactl --cpunodebind=1 --membind=1`).
* `NCCL_RUNTIME_CONNECT=0`: channels are connected during communicator init.
  With NCCL's default lazy connection the first out-of-place region of a 4-GPU
  launch absorbed ~257 ms of SHM connection set-up in a pilot launch (see §10);
  with `NCCL_RUNTIME_CONNECT=0` the residual first-region overhead was 16–20 ms.
* All other `NCCL_*` variables from the caller's environment are removed so
  that only the configuration under test is set.

## 4. Verifying the configuration NCCL actually ran ("probe" pass)

Requested and executed configurations are not assumed to coincide. Before the
timed sweep (`probe`) and again after it (`probe_end`), every condition is run
once over the whole size range with

```
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING  <coll>_perf -n 1 -w 0 -c 1 ...
```

In the initial TUNING-enabled probe, NCCL 2.27.5 records selections of the form
`AllReduce: 1048576 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..7}`.
Timed v1 processes did not record per-process TUNING selection. The initial
probe was used to infer their effective algorithm/protocol; the closing probe
was not obtained. These logs do not verify every timed invocation.
For AllGather/ReduceScatter the byte count is per rank; the parser maps it via
the element count nccl-tests reports. From the probe we record, per size:

* the algorithm, protocol and channel range NCCL selected (`selection.csv`);
* `forced_honoured`: whether a forced algorithm/protocol is what ran;
* `#wrong` from nccl-tests' data check for out-of-place and in-place;
* whether the request was rejected. NCCL rejects e.g. `NCCL_ALGO=Tree` for
  AllGather with `NCCL WARN Error : no algorithm/protocol available for
  function AllGather ... NCCL_ALGO was set to Tree.` and nccl-tests aborts.
  Such a size is recorded as `unsupported` with NCCL's own message, and the
  probe continues with the next size in a new process, so every size gets its
  own verdict.

Sizes that are unsupported, failed, or returned wrong data are **not timed**;
each such unit still gets an explicit record (`not_run_probe_<status>`) in
every trial. The selection from `probe` and `probe_end` is compared
(`selection_stable`). *In the reported sweep `probe_end` was not run (the
sweep was stopped early, §9); selection stability was verified only in the
smoke run (108/108 identical).*

Because a probe executes one operation per size with per-call logging, its
timings are not used.

## 5. Calibration

For each supported condition, two short sweeps (`-c 0`) estimate the latency
per size: sizes ≤ 1 MiB with 200 iterations (20 warm-up), larger sizes with
5 iterations (1 warm-up). From the estimate *t̂* the timed launch uses

* `n = max(5, ceil(target_region_s / t̂))` timed iterations, and
* `w = max(1, ceil(warmup_s / t̂))` warm-up iterations.

## 6. Timed launches and measurement regions

Each timed launch runs one condition at one size. nccl-tests prints a result
row with three `printf` calls: the preamble (size, count, type, op, root)
just before the out-of-place benchmark, the out-of-place result after it, and
the in-place result after the in-place benchmark. The process runs under
`stdbuf -o0` (unbuffered stdout) and the orchestrator reads the pipe in a
dedicated loop. Each arriving chunk is time-stamped (`time.perf_counter`) and
the NVML energy counter of every participating GPU is read immediately.
Measurement regions are therefore:

| region | starts when this text arrives | ends when this text arrives | contains |
|---|---|---|---|
| out-of-place | row preamble | out-of-place result | `w` warm-up + `n` timed ops |
| in-place | out-of-place result | in-place result | `w` warm-up + `n` timed ops |

Timed launches use `-c 0` so that no data check runs inside a region.

**Region length.** The requirement is that every measured region lasts ≥ 2 s.
We enforce the stricter condition that the *timed iterations alone*
(`n × t_op`) last ≥ `min_region_s` = 2.0 s in both regions. If a launch falls
short (calibration error), it is repeated once with `n` scaled to the target;
the short attempt is kept in `launches.jsonl` with `final: false`.
Launches that fail (non-zero exit, timeout, unparsable output, missing region
boundary) are also retried once and the failed attempt is kept.

**Per-operation quantities.**

* Latency `t_op`: the time nccl-tests reports for the region (host wall clock
  over the `n` timed iterations, averaged over ranks).
* Energy per operation: `E_op = E_region / (n + w)`, where `E_region` is the
  sum over participating GPUs of the counter difference across the region.
  Warm-up operations are the same operation as the timed ones and lie inside
  the region, so they are counted. QC compares `region_duration / (n + w)`
  with `t_op` (`wall_vs_reported` flag at > 10 % difference).

## 7. Energy measurement

**Primary: NVML energy counter.** `nvmlDeviceGetTotalEnergyConsumption`
(mJ since driver load) is supported on all eight GPUs. It is read **only at
region boundaries** (one read per GPU per stdout chunk, ≤ 4 chunks per
launch during the run). On this driver/GPU we had observed in earlier,
unrelated work on the same machine that frequent energy-counter reads — and
reads of `NVML_FI_DEV_POWER_INSTANT` — reduce the counter's reported energy;
the sampler therefore never calls either API, and the boundary reads are
too infrequent for that effect to matter beyond about one internal
accumulation window per region (see limitations).

**Secondary: sampled power.** A background thread (`nvml_sampler.py`)
samples every 50 ms, per participating GPU:

| field | NVML call |
|---|---|
| timestamp | `time.perf_counter()` (same clock as region boundaries) |
| power (W) | `nvmlDeviceGetPowerUsage` |
| GPU utilization, memory utilization (%) | `nvmlDeviceGetUtilizationRates` |
| temperature (°C) | `nvmlDeviceGetTemperature(GPU)` |
| graphics clock, memory clock (MHz) | `nvmlDeviceGetClockInfo(GRAPHICS / MEM)` |
| P-state | `nvmlDeviceGetPerformanceState` |
| query time | wall time of the calls above |

and every 1 s the list of processes on each GPU. The 50 ms period was chosen
from measurements on this machine: a full sample costs ≈2–2.6 ms of NVML time
per GPU, and `nvmlDeviceGetPowerUsage` on this GPU is a ~1 s average whose
value changes every ~100 ms under load (every ~1 s at idle), so sampling
faster adds no information. Trapezoidal integration of the sampled power over
each region is stored next to the counter value (`energy_integrated_j`), and
also with the window shifted by +0.5 s to compensate for the averaging lag
(`energy_integrated_lag05_j`). These integrals are a cross-check only; the
counter is the primary measurement (the integral of a 1-s moving average
smears energy across neighbouring regions).

**Idle power.** At the start and end of each trial and after every
`every_n_launches` timed launches, with no process on any GPU of the active
sets, the orchestrator waits until every GPU reports P-state P8, waits 5 s
more, then measures 10 s with the counter (two reads) and the sampler.
Idle power per GPU and trial is the median of that trial's idle windows.

## 8. Derived metrics

For a region with `n + w` operations, duration `T`, counter energy `E`
(summed over the `G` participating GPUs), idle power `P_idle,g`, message size
`S` bytes (the nccl-tests `size` column: the full buffer of the collective):

| metric | definition |
|---|---|
| latency | `t_op` (µs), nccl-tests |
| algorithm bandwidth | `algbw = S / t_op` (nccl-tests) |
| bus bandwidth | `busbw` (nccl-tests: `algbw × 2(G−1)/G` for AllReduce, `× (G−1)/G` for AllGather/ReduceScatter) |
| raw energy per op | `E_op = E / (n + w)` |
| idle-subtracted energy per op | `E_dyn = (E − Σ_g P_idle,g · T) / (n + w)` |
| energy per byte | `E_op / S` and `E_dyn / S` (J/B) |
| GB per joule | `S / E_op / 1e9` and `S / E_dyn / 1e9` |
| energy-delay product | `EDP = E_op × t_op` (J·s), also `E_dyn × t_op` |

## 9. Trials, ordering, interference control

* **Trials.** 5 repeated trial blocks were configured. Independence across
  the sequential blocks on this shared host is not established. **The reported sweep
  completed 3**: it was stopped with SIGTERM right after trial 3's closing
  idle measurement, at the operator's request to release the shared GPUs
  (logged in `data/raw/full/events.jsonl`; no trial-4 launch was started;
  `order_trial4.json` was written before the stop and is unused). Statistics
  rank only configurations with ≥ 3 valid trials. In each trial, every
  (collective, GPU set, algorithm, protocol, size) unit is launched once as a
  fresh process; the unit order is shuffled with `random.Random(f"{seed}-trial-{k}")`
  and saved to `order_trial{k}.json` before the trial runs.
* **Competing processes.** Before each launch, NVML process lists of the
  target GPUs must be empty (the orchestrator itself never opens a CUDA
  context); otherwise the launch waits (polling every 30 s, up to 30 min), is
  deferred to the end of the trial, and is finally recorded as
  `skipped_foreign_process`. During a launch the sampler records processes on
  the target GPUs once per second; any process other than the benchmark marks
  the region `foreign_process_during`, and such regions are excluded from the
  statistics (listed in `exclusions.csv`). Processes on non-target GPUs and the
  host load average are recorded per launch. A GPU set with a process on any
  of its GPUs at the start of a run is skipped as a whole and the reason is
  stored in `gpu_set_status.json`.
* **No silent drops.** Every planned unit has exactly one final record in
  `launches.jsonl`, whatever its status; the smoke check verifies this.
  stdout, stderr, the NCCL log, the stdout chunk timeline with counter reads,
  and the power trace of every launch are kept under
  `data/raw/<run>/launches/<launch_id>__a<attempt>/`.

## 10. Statistics

* Per condition, metrics are averaged over trials; 95 % intervals are
  percentile bootstrap intervals of the mean over trials (10 000 resamples,
  fixed seed).
* **Optimal configuration** per cell and objective (latency, raw energy,
  idle-subtracted energy, EDP): the requested configuration with the lowest
  trial mean, among configurations with ≥ 3 valid trials. `P(best)` is the
  fraction of bootstrap resamples (each configuration's trials resampled
  independently) in which it has the lowest mean.
* **Disagreement.** Latency- and energy-optimal configurations "disagree" in a
  cell if NCCL ran a different algorithm/protocol/channel range for them
  (from the probe). Requested labels can differ while the executed
  configuration is the same (e.g. `auto/auto` and `Ring/LL` both run
  `RING/LL` for small AllReduce); those pairs are counted as agreeing, and
  such duplicate pairs act as a built-in noise check.
* **Energy penalty of the latency-optimal choice:**
  `mean E(lat-opt) / mean E(energy-opt) − 1`, with a bootstrap interval; a
  disagreement is called *resolved* when that interval excludes 0. The
  matching latency penalty of the energy-optimal choice is also reported.
* **Pareto front**: configurations whose (mean latency, mean energy) is not
  dominated within the cell.

## 11. Validation (smoke test)

`configs/smoke.yaml` (3 sizes, 2 GPU sets, 3 algorithms × 2 protocols,
2 trials) exercises every code path. `python -m ncclenergy smoke-check`
writes `data/processed/smoke/SMOKE_REPORT.md`; the full sweep was started
only after it passed. The checks and the values obtained are listed in
[smoke_test.md](smoke_test.md).

## 12. Reproduction

See the *Reproduce* section of the [README](../README.md).
