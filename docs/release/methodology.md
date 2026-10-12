# Methodology by evidence category — v2.2 pilot release

**Rules that hold for every category**:
* The categories are never pooled.
* Each has its own method ID and data directory.
* Detailed protocols are linked from each section.

**Common setup**:

| item | value |
|---|---|
| host | one 8× RTX A6000 server (300 W limit), no scheduler |
| placement | GPUs 4–7 for v2, 4,5 for v2.2; process/idle checks were recorded. These alone do not demonstrate an exclusive reservation |
| concurrency | one experiment process at a time; no privileged operation (no clock, power-limit, persistence or driver change) |
| software | NCCL 2.27.5 (the library PyTorch 2.9.1 loads); nccl-tests v2.21.1 `afd59ab` |

## 1. Historical pilot (method `v1`, 2026-10-06; driver 550.144.03)

* **Design**: 12 requested configurations (auto/Ring/Tree × auto/Simple/LL/LL128), 3 collectives, 2 and 4 GPUs, 1 KiB–1 GiB, randomized order within trials. **3 of 5** planned trials were completed.
* **Energy**: the NVML cumulative counter, read at stdout-chunk boundaries **without** the commit alignment that methodology v2 introduced. The counter-commit behaviour found later on driver 580 was not re-tested on 550.
* **Statistics**: trial-level percentile bootstrap. Bootstrap intervals from 3 values per condition are descriptive.
* **Status**: frozen and superseded. Details: [`../pilot_v1/README_pilot_v1.md`](../pilot_v1/README_pilot_v1.md), [`../methodology.md`](../methodology.md), [`../limitations.md`](../limitations.md).

## 2. v2 exploratory and confirmatory (method `v2.1`, 2026-10-08/09; driver 580.178.04)

### Measurement region

* One region per fresh process, built with the nccl-tests region patch
  `patches/nccl-tests-afd59ab-v2.1-region.patch`.
* The patch:
  * reads the counter (C0) at the start;
  * runs R timed operations;
  * reads it (C1) just after the next counter update following the loop
    end;
  * reads it once more (Cpost) one period later.
* Region energy = (C1 − C0) − p_ctx × tail, where:
  * the tail is the time from loop end to that update;
  * p_ctx (the post-loop power) is estimated from Cpost − C1.

### Counter interpretation (local, not an NVML guarantee)

The local Phase 4 experiments support this reading on driver 580:
* the counter updates about every 100 ms;
* a read discards the not-yet-committed energy.

Therefore the controller never reads the counter inside a region.

### Execution evidence and QC

* An NCCL profiler plugin records, on every rank, the executed algorithm,
  protocol, channels and warps and the exact operation count.
* A 50 ms sampler records PowerUsage, POWER_INSTANT, temperature, clocks,
  clock-event reasons and processes.

### Design

* 2 exploratory blocks selected a matrix and one pick per cell. These were
  frozen in a pre-registration ([`../v2/prereg.md`](../v2/prereg.md)).
* 12 confirmatory blocks followed: 984 units, all OK and verified.
* **Statistics**: blocks are the unit of inference; block-paired t-tests,
  Holm correction, TOST for equivalence.
  The original nominal ±3% equivalence rule uses log bounds ±ln(1.03),
  hence ratio bounds [1/1.03, 1.03]; it is not exact equality. TOST's interval
  criterion differs from an ordinary 95% difference interval.
* The blocks ran back to back in one session, so they are within-session
  replicates.

### Phase 10 (synthetic workload)

* 4-GPU DDP/FSDP MLP, FP32.
* The tuner-deployed policy (NCCL example tuner with two fixes) was
  compared with automatic, over 5 within-session blocks.
* Energy came from the v2.1 workload meter, which recorded energy terms but
  no QC flags. The same synthetic batch was used on every rank.

### Detailed protocols

* [`../v2/protocol.md`](../v2/protocol.md), [`../v2/phase4_validation.md`](../v2/phase4_validation.md);
* [`../v2/phase8_results.md`](../v2/phase8_results.md), [`../v2/phase10_results.md`](../v2/phase10_results.md);
* the Stage 2 corrections in [`../v2/v22_audit/stage2_report.md`](../v2/v22_audit/stage2_report.md).

## 3. v2.2 measurement validation (gate 3b, 2026-10-09; methods `v2.2-fail-closed`, `v2.2-aligned-meter-logged`)

* **Changed from v2.1** (Stage 2): owned-process control, an active
  watchdog, fail-closed QC, and logging of every boundary value. The logged
  boundary values are raw counter values, NVML status codes, read-call
  brackets, probe edges, phase and commit times.
* **Energy accounting**: identical to v2.1. Tested on 200 random inputs, and
  the historical acceptance is identical under the new QC.
* **Design** (frozen before collection, `f4735b1`):
  * P2_NVL (GPUs 4,5), AllGather 2 MiB;
  * A = auto, B = forced Ring/Simple (an identical-execution control for
    A), C = forced Ring/LL128;
  * region lengths 2.5, 10, 20 and 40 s;
  * 6 blocks of 12 fresh processes in seeded random order, with idle
    baselines;
  * one attempt per process, no reruns, no significance-based stopping.
* **Pre-specified analysis**: [`../../src/ncclenergy/v2/measval.py`](../../src/ncclenergy/v2/measval.py)
  * D1: J/op 2.5 s vs 40 s, within ±0.3 %;
  * D2: protocol term of the estimator disagreement, within ±0.3 %;
  * D3: the B ÷ A control at 10–40 s, within ±0.5 % and containing 0;
  * D4: sufficient only if D1–D3 pass and the 40 s C ÷ A half-width is
    ≤ 0.4 %.
* **Post-collection changes**: two code fixes, neither touching a decision
  rule ([`../v2/v22_audit/analysis_amendments.md`](../v2/v22_audit/analysis_amendments.md)).
* **Report**: [`../v2/v22_audit/gate3b_measval_report.md`](../v2/v22_audit/gate3b_measval_report.md).

## 4. Post-hoc exploratory analysis

**Stage 2 reanalysis of the v2 data**
(`data/v2/reanalysis/2026-10-09_v2.2-audit*`). It covers:
* the threshold test for ≥ 5 % (C2);
* RQ2 under four estimators (C7, C8);
* clustered and wild-bootstrap order and temperature models (C10, C11);
* read-latency and timing uncertainty components;
* QC reclassification;
* the recommender with exact-size rules.

**Gate 3b post hoc** (`measval_posthoc.py`):
* a temperature-association model, with leave-one-block-out adjustment;
* a counter-vs-sampled comparison over matched windows: the same
  [start-read midpoint, loop end] per GPU, in J per GPU.

These analyses were chosen after the data were seen. They are labeled
`POST-HOC EXPLORATORY` in code, outputs and reports, and they change no
pre-specified decision.

## 5. Future experiments

None are part of this release. See [`limitations.md`](limitations.md) §6.
Each would need its own plan, allocation and approval.

## Uncertainty: statistical vs systematic

**Statistical**: the block-to-block t-intervals. They are conditional on
one session and one host.

**Systematic** (reported, not propagated into the intervals):
* board-sensor calibration (not characterized);
* the tail-model bound: up to about 1.4 % on 2.5 s regions (Stage 2
  addendum component C2);
* timing and phase components (≤ 0.1 %);
* the unexplained per-region gap between counter and sampled estimators
  (gate 3b §5.2).
