# Limitations, untested paths and future experiments — v2.2 pilot release

## 1. Scope

* **One host, one session per campaign.**
  * One RTX A6000 server; driver 580.178.04 (550 for the pilot);
    NCCL 2.27.5.
  * 2–4 GPUs, single node; FP32 sum.
  * The blocks of each campaign ran back to back in one session. They are
    **within-session replicates**: no between-day, between-host or
    between-GPU-unit variance is estimated.
* **GPU-board energy only.**
  * It covers the participating GPUs while the collective runs.
  * It is not communication-only energy: static and idle board power are
    included. Idle-adjusted variants exist only as sensitivity analyses.
  * It is not whole-system energy: CPU, DRAM, PCIe switches, host memory
    traffic (which matters for P4_MIX) and PSU losses are excluded.
* **No external power reference.** Counter and sampled power come from the
  same boards. Their agreement is consistency, not accuracy.
* **Counter semantics.** That the 100 ms update and the read discarding
  uncommitted energy follow a local interpretation of the Phase 4
  experiments on driver 580 is not an NVML guarantee.

## 2. By evidence category

### Historical pilot (v1)

* Uncorrected boundary counter reads on driver 550.
* 3 of 5 trials; 4-GPU run-to-run variation up to about 20 % of
  unidentified origin.
* Superseded by v2. Kept unchanged.

### v2 confirmatory

* Each cell compares one selected pick with automatic. The best
  configuration overall was not established.
* The ≥ 5 % count depends on the rule: 25 cells by point estimate, 21 by
  the threshold test.
* C7, the AllGather 2 MiB energy penalty of about 1 %, is
  estimator-sensitive and unresolved.
* NCCL's choices on P4_MIX are explained only by a hypothesis.

### v2 Phase 10 (synthetic workload)

* Synthetic FP32 MLP; 5 within-session blocks.
* The `no_sync` reference used a different power/overlap regime. Its
  difference from DDP is not an isolated communication-time or energy
  fraction, and the synthetic result is not a demonstrated upper bound on
  training benefits. Batch 2048 ran at the power limit (Phase 10 run records).
* Numerical equivalence of the policies was not recorded.
* Every rank used the same batch, so a skipped or local-only gradient
  AllReduce would not have been detected. `--data per_rank`, the default
  since this release, makes it detectable; a test demonstrates this.
* The v2.1 workload meter recorded energy terms but no QC flags. A post-hoc
  plausibility check with the v2.2 rules flagged none of the 200 GPU
  readings.
* The example tuner (two fixes, no scope guard) was deployed, not the
  guarded tuner.
* The `SW_POWER_CAP` clock event is masked unconditionally in Phase 10's
  execution checks.

### v2.2 measurement validation (gate 3b)

* Not sufficient to resolve differences of 0.3–0.5 % between single 2.5–40 s
  processes. D1 and D2 are INCONCLUSIVE, D3 SYSTEMATIC_OFFSET and D4
  NOT_SUFFICIENT.
* The gate covered one pair of GPUs, one collective and size, and one
  session.
* Thermal state was not controlled. Execution position and temperature are
  confounded within each block.

### Post-hoc analyses

* Chosen after seeing the data.
* The temperature model associates; it does not establish cause.
* The per-region counter-vs-sampled gap (about 13 J per GPU) is
  unexplained.

## 3. Software limits that remain

| area | limit |
|---|---|
| process control | cleanup cannot survive a `kill -9` of the controller. The subreaper status is recorded, not enforced. There is a small check-then-kill race (no pidfd) |
| watchdog | the process list is polled once a second, so very short foreign processes can be missed. Watchdog liveness itself is not monitored |
| sampler | POWER_INSTANT is sampled every 50 ms on any driver. Phase 4 found that it perturbs the counter on driver 550, so a replication on other drivers must re-run the Phase 4 dose test first |
| counter reads | the historical v2.1 native patch reads the counter even in sampled campaigns. A new v2.3 patch adds a sampled gate, CPU-tested only; native sampled collection stays disabled until real CUDA/Pascal validation. `capabilities.py` reads the counter once on an apparently idle allowed GPU, which is a time-of-check race |
| guarded tuner | validated with a C harness and mock NVML only. No production path loads it (Phase 10 used the example tuner). NCCL's v4 tuner interface does not pass dtype, reduction op or in-place flags, so those are declared assumptions. Parsing and re-entrancy defects are fixed in this candidate, not retroactive validation of historical execution |
| scope guards | the canonical topology fingerprint excludes PCI ids. NCCL environment variables beyond the checked list, and power or clock settings, are not checked |
| numerics | tolerances (loss 1e-3, parameter norm 1e-4) are declared, not calibrated on repeat runs. FSDP has no replica check (`NOT_APPLICABLE_SHARDED`) |
| execution evidence | covers algorithm, protocol, channels and warps, and the operation count only |
| Xid and resource changes | Xid events and power/clock-policy changes are checked between units, not continuously; the active watchdog cannot prevent every hardware or host failure |

## 4. Fixed in this release (future runs only; no collected data or published output changed)

Each fix has a regression test that fails on the previous code.

* **Process cleanup**:
  * The cleanup now runs when a signal arrives between fork and exec.
  * If waiting after SIGTERM fails, SIGKILL is still sent.
  * Cleanup is reported OK only with no survivors and no failed step.
  * A thread race on the ownership set is fixed.
* **Process list**: a failed process-list query is now unknown, not
  "no process". The watchdog trips and QC fails.
* **QC**: these are now hard failures:
  * a missing exit status;
  * region-parse and schema flags;
  * an unknown power fraction together with the SW power cap;
  * a non-finite lag integral;
  * errors in the logs, which give EXECUTION_ERROR.
* **Sampling gaps**: gaps at the window edges are detected.
* **Meters**:
  * a flagged aligned-meter reading has no energy value;
  * the sampled meter has no energy value on telemetry problems;
  * Phase 10 analysis excludes flagged v2.2 meter records.
* **Workload**: per-rank data with a check that DDP replicas stay equal.
* **Guarded tuner**: a malformed rule rejects the whole file.

## 5. Untested on real hardware

The CPU test suite uses fake NVML, mocks and the gloo backend. It does not
validate real CUDA, NVML or NCCL behaviour.

**Exercised on real hardware** (gate 3a/3b, GPUs 4,5, driver 580):
* the `runner block` path with patched nccl-tests;
* counter-mode idle baselines;
* the sampler with real NVML, including POWER_INSTANT;
* profiler evidence and fail-closed QC on OK units;
* a watchdog `WALL_TIME_EXCEEDED` trip with SIGTERM cleanup;
* controller-SIGTERM cleanup.

**Never exercised on real hardware**:
* the SIGKILL escalation path with a real GPU process;
* real `torchrun` trees under owned-process control;
* the v2.2 workload, AlignedMeter and SampledMeter on CUDA and NVML;
* Phase 10 verify, overhead and workload commands under v2.2;
* the guarded tuner inside real NCCL;
* sampled energy mode and the counter-not-supported path;
* the code changes of this release (§4) that touch GPU-run paths;
* watchdog trips other than wall time (foreign process, temperature, clock
  events, stale or failed telemetry, process list unavailable, low disk);
* other GPU models and drivers.

## 6. Future experiments (optional; not required for this release)

1. **Thermally controlled measurement validation**:
   * a fixed temperature band before every process, or temperature-matched
     interleaving;
   * temperature as a pre-specified covariate;
   * block count from the measured residual SD.
2. **Per-region estimator gap**: a dedicated test with varied start phase,
   pre- and post-roll windows and idle bracketing.
3. **External power reference**: needs hardware or administrator action.
4. **Repetition** on other days and hosts, and on other GPU models and
   drivers. Pascal boards cannot run the PyTorch 2.9.1 workload (no sm_61),
   though nccl-tests can.
5. **Real-model transfer**:
   * BF16 AllReduce profiling;
   * DDP with a real model and auto vs Energy-SLO;
   * vectors and numerical correctness recorded with per-rank data;
   * repetition determined by a separate pilot and precision/power plan.
6. **A newer NCCL version** on the 17 P4_MIX RQ1 cells.
7. **Guarded tuner inside real NCCL**, within the controlled harness.

The prospective designs above are proposals, not permission to collect or
guarantees of correctness. In particular, an arbitrary block count is not a
power/precision calculation. See [scientific_resolution.md](scientific_resolution.md)
for numerical-validation and simultaneous-optimum inference gates.

## 7. Release-candidate corrections (2026-10-11)

See [corrections_2026-10-11.md](corrections_2026-10-11.md). New malformed-input,
allocation, telemetry-order and native-parser tests do not validate a GPU
measurement. Historical launchers, Phase4 GPU probes, CUDA workloads and native
sampled collection are disabled. Counter-mode v2 collection requires a newly
approved campaign identity; changing a guard is not permission to run it.
Changed forward accounting/QC has v2.3 method identifiers and must not be pooled
silently with retained v2.1/v2.2 data. The 50 historical workload runs fail the
optional strict numerical-sanity gate; their default descriptive analysis is
retained without promoting it to training equivalence.
