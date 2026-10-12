# Limitations

> Pilot (methodology v1) document, superseded by methodology v2.
>
> **v2.2 pilot release (2026-10-09).** The current claim status, limitations and reproduction instructions are in
> [`release/claims.md`](release/claims.md), [`release/limitations.md`](release/limitations.md) and
> [`release/reproduction.md`](release/reproduction.md). This document is kept as the record of its stage.


Read these before using any number from this repository.

## Scope

* **One machine, one GPU model, one software stack.** 8× RTX A6000 (GA102,
  PCIe Gen4, pairwise NVLink bridges, no NVSwitch), driver 550.144.03,
  NCCL 2.27.5, nccl-tests v2.21.1. NCCL's tuning model, protocol
  availability and kernels change between versions; GPUs with NVSwitch,
  HBM, or different power management may behave differently. Nothing here
  should be read as a statement about other systems.
* **GPU sets.** 2 GPUs = one NVLink-bridged pair (GPUs 4, 5). 4 GPUs =
  GPUs 4–7, i.e. two NVLink pairs joined over PCIe through host shared memory
  (NCCL `SHM/direct/direct` transport). The 8-GPU set needs GPUs 0–3, which
  are used by other people on this shared machine; the run skips it
  automatically when any of those GPUs has a process (see
  `data/raw/full/gpu_set_status.json` for what happened in the reported run).
* **Message sizes** are powers of two from 1 KiB to 1 GiB, `float` data, `sum`
  reduction only.
* **Isolated, back-to-back collectives.** nccl-tests issues the same
  collective repeatedly with nothing else on the GPU. In training, collectives
  overlap with compute kernels, share SMs and memory bandwidth, and run at
  whatever clocks the compute load sets. Energy attribution under overlap is a
  different problem and is not addressed.
* **Single process, one thread per GPU.** Production jobs typically use one
  process per GPU. In a single process NCCL uses `P2P/direct pointer`
  instead of `P2P/IPC` or `P2P/CUMEM`; the data path over NVLink is the same,
  but launch overhead and small-message latency may differ.

## Energy measurement

* **GPU board energy only.** The NVML counter covers the GPU board. CPU
  package, DRAM, PCIe switches, NICs, fans and PSU losses are not measured
  (no RAPL access without root). This matters for the 4-GPU set, where half
  of the ring hops go through host memory (SHM transport): host-side energy of
  those copies is invisible here, so 4-GPU energies are lower bounds of system
  energy and comparisons between transports are GPU-side comparisons.
* **Counter behaviour.** `nvmlDeviceGetTotalEnergyConsumption` has mJ
  resolution and is updated on an internal accumulation window of tens of
  milliseconds. In earlier, separate testing on this same machine and driver
  we observed that each read of the counter (and each read of
  `NVML_FI_DEV_POWER_INSTANT`) can drop the energy of the window in progress.
  The counter is therefore read only at region boundaries (≤ 4 reads per
  launch) and never by the sampler. With ≥ 2 s regions the expected loss is
  about one window per region, i.e. on the order of 1 % and nearly the same
  for every configuration because region lengths are similar. We did not
  correct for it.
* **Sampled power is a ~1 s average.** `nvmlDeviceGetPowerUsage` on GA102
  returns a ~1 s average updated every ~100 ms under load. Its integral over a
  2–3 s region smears energy across neighbouring regions; it is reported as a
  cross-check only (agreement statistics are in the smoke-test report and
  `qc.json`).
* **Idle baseline.** Idle power is measured with no process on the GPU
  (P-state P8, ~21–29 W per GPU). During a collective the GPU sits in P0/P2
  with raised clocks, so "idle-subtracted" energy includes the static power
  increase of the active P-state, not only the dynamic energy of moving data.
  Idle power also depends on temperature; idle windows follow hot launches
  only after the GPUs have dropped back to P8.
* **Regions include warm-up and a small one-time overhead.** Warm-up
  iterations (≈0.1 s worth) are inside each measured region and counted as
  operations. The first (out-of-place) region of every process additionally
  contains ≈16–20 ms of first-call overhead (measured, §3 of the methodology),
  i.e. ≲1 % of a ≥2.2 s region; the in-place region does not. Both placements
  are reported.
* **Thermal and clock state are not controlled.** Without root we cannot lock
  clocks, set fan speed or enable persistence mode. Temperatures and clocks are
  recorded for every region; randomized order spreads thermal drift across
  configurations rather than removing it.

## Statistics

* **Three trials (five planned).** The reported sweep was stopped after
  trial 3 to free the shared GPUs. Percentile bootstrap intervals from three
  values per condition under-cover nominal 95 % substantially; read them as
  descriptive. `P(best)` is coarse (median 0.68–0.78 for the winners).
* **Noise floor.** Requested configurations that NCCL executed identically
  differ in mean energy per operation by up to 4.1 % (2 GPUs; 95th percentile
  2.6 %) and up to 20.7 % (4 GPUs; 95th percentile 9.1 %). Differences below
  these levels, including all three latency/energy disagreements found, are
  not distinguishable from run-to-run variation with this design.
* **4-GPU variability.** Trial-to-trial variation in the 4-GPU set
  (NVLink pairs joined through host memory) is an order of magnitude larger
  than on the NVLink pair, for latency as well as energy. Its cause was not
  identified; CPU/memory load from other users of the shared host is a
  plausible contributor (host load average was recorded per launch).
* **Winner's curse.** The energy-optimal configuration is the minimum of up
  to 12 noisy means, so its mean is biased low and the reported "energy
  penalty of the latency-optimal choice" is biased high, most noticeably
  where configurations are within noise of each other. Cells where
  requested-but-identical configurations (same executed algorithm/protocol)
  differ give a direct view of the noise floor.
* **Fixed ordering inside a process.** Order is randomized across launches,
  but within a launch the out-of-place region always precedes the in-place
  region.

## Verification gaps in the reported sweep

* The closing probe pass (re-recording NCCL's selection after the sweep) was
  not run. NCCL's selection is a deterministic function of topology, size and
  environment, and it was stable in the smoke run, but this was not
  re-checked for the full sweep.

## Configurations

* Forcing `NCCL_ALGO`/`NCCL_PROTO` restricts NCCL's choice; it does not
  control channel counts, chunk sizes or thread counts, which NCCL still
  chooses per size. The executed algorithm/protocol/channel range is recorded
  from NCCL's log, not assumed.
* `LL128` is forced on paths where NCCL may not enable it by default. The data
  check in the probe (`-c 1`, one operation per size, out-of-place and
  in-place) passed wherever it is reported as supported; that is evidence,
  not a proof of correctness under all conditions.
* `Tree` is not implemented for AllGather and ReduceScatter in NCCL 2.27.5;
  NCCL rejects it and those conditions are reported as unsupported.

## Interference

The machine is shared. Other users' jobs ran on other GPUs (e.g. GPU 1) and
on the CPUs during the runs. Processes on the target GPUs are detected and
excluded, processes on other GPUs and the host load average are recorded per
launch, but CPU contention can still affect host-driven small-message latency.
