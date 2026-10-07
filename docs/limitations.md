# Limitations

These limits apply to every result in the repository.

## Scope and topology

- The data come from one RTX A6000 server, one NCCL/nccl-tests stack, and isolated repeated collectives. They do not establish behavior on other GPUs, NCCL versions, NVSwitch systems, multi-node systems, or training workloads that overlap communication with compute.
- The 2-GPU condition uses one NVLink pair, GPUs `[4,5]`. The 4-GPU condition uses two NVLink pairs, `[4,5]` and `[6,7]`, with host/PCIe communication between pairs. GPU-count comparisons are confounded by topology.
- No 8-GPU measurement ran. A foreign process occupied a required GPU at startup, so the configured 8-GPU set was skipped.
- Only powers-of-two message sizes from 1 KiB to 1 GiB, `float` data, and sum reduction were tested.
- nccl-tests used one process with one thread per GPU. Production multi-process jobs may have different launch and transport behavior.

## Energy attribution

- Energy is participating-GPU board energy from NVML. CPU, DRAM, PCIe switch, NIC, fan, and power-supply energy are not included. This is especially important for the host-mediated links in the 4-GPU condition.
- Counter boundaries are triggered by receipt and parsing of stdout fragments. They are not CUDA events or direct GPU execution markers. Buffering is reduced with unbuffered stdout, but scheduling, pipe delivery, parsing, and sequential NVML calls can shift a boundary.
- `nvmlDeviceGetTotalEnergyConsumption` reports cumulative millijoules. The repository data do not establish the hardware counter's internal update frequency or effective precision beyond the API unit. No wraparound occurred in the retained regions, and the implementation has no wraparound correction.
- The sampled `nvmlDeviceGetPowerUsage` integral is a cross-check, not an independent sensor. It is another NVML-derived view of the same board power domain and exhibits temporal smoothing/lag.
- Energy regions include `w` warmup plus `n` timed operations and divide by `n + w`. nccl-tests latency covers only the `n` timed operations. This is internally consistent for energy per operation but prevents exact alignment between the two estimands.
- The out-of-place region always occurs first and may include first-call work after its stdout boundary. Buffer placement order is not randomized.
- P8 idle subtraction is not pure communication-only energy. It also includes the board-power increase associated with active clocks and performance state.

## Repetition and uncertainty

- Five trial blocks were planned; three completed. They ran sequentially on the same shared host. Fresh process creation and randomized launch order reduce some carryover and order bias, but do not guarantee independent errors.
- Percentile bootstrap intervals resample three trial-level observations. They are descriptive and can substantially understate uncertainty. Individual nccl-tests iterations are not treated as independent replicates.
- The full run retained 406 short nonfinal attempts before accepted retries. Only final attempts enter the reported statistics, but the short attempt could warm GPUs or alter clock state before the retry. Retry frequency is condition-dependent, so accepted retries may not be fully comparable with first-attempt successes.
- Selecting the minimum among as many as 12 noisy configuration means creates winner's-curse bias. Small optimum differences require confirmation with more independent blocks.
- Requested configurations that the initial probe mapped to the same effective algorithm/protocol/channel selection provide a noise check. Observed differences among such duplicates reached several percent on 2 GPUs and were substantially larger in parts of the 4-GPU data. The 2–5% candidate disagreements are within this pilot uncertainty scale.

## Configuration verification

- Requested and effective configurations can differ. The analysis uses the initial TUNING probe to infer the effective algorithm, protocol, and channel range.
- Timed nccl-tests processes did not emit TUNING records, so their exact selections were not observed directly.
- The planned closing probe did not run. Selection stability was checked in the smoke test, not across the full timed sweep.
- Forcing an algorithm or protocol does not fix channel count, chunk size, or thread count. Those remain NCCL choices.
- NCCL 2.27.5 rejected Tree for AllGather and ReduceScatter in this matrix. These cells are recorded as unsupported rather than measured.

## Thermal, clocks, and interference

- GPU clocks, fan speed, and power limits were not locked. Telemetry showed P2 during retained regions, but temperature and graphics clocks varied.
- The server was shared. The implementation excludes detected foreign processes on participating GPUs, but jobs on other GPUs and CPU, memory, filesystem, or scheduler contention can still affect results.
- The 4-GPU data show much greater trial-to-trial variability than the 2-GPU NVLink pair. Shared-host load and the host-mediated path are plausible contributors, but the existing data do not identify a cause.

## Dataset boundary

- The public repository retains processed tables and representative sanitized logs, but not the complete per-launch raw archive or `launches.jsonl`. Raw-to-processed reproduction therefore requires the separate private archive.
- The retained data contain no Full Trial 4 measurements, no Trial 5 measurements, no 8-GPU measurements, and no closing full-run probe. `order_trial4.json` is only a generated plan.

The dataset is suitable for smoke-test evidence, pilot characterization, and identifying conditions for a confirmatory rerun. It is not sufficient for publication-quality inference about a general latency-energy optimum.
