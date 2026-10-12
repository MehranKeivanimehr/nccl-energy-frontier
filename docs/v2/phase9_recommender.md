# Phase 9 — Energy-SLO recommender and its validation

> **v2.2 audit corrections (2026-10-09).**
> * **Scope guards**: queries must now match the profile's recorded scope
>   exactly: GPU model, driver, CUDA driver version, NCCL version, dtype,
>   reduction op, in-place flag, ranks/nodes, fingerprint and placement
>   environment, all derived from raw evidence. Anything missing or
>   different falls back to NCCL automatic. The v2.1 recommender ignored
>   the CUDA version and had no dtype/op scope.
> * **No interpolation**: interpolation between sizes is disabled. The
>   v2.1 rule below copied the endpoint's ratio and CI to the intermediate
>   size, and the v2.1 tuner files covered unmeasured sizes with range
>   rules. v2.2 exports one rule per measured size.
> * **Native deployment**: the guarded plugin is only for a controlled
>   harness. dtype and op are declared, not verified.
> * **LOBO**: the 492 LOBO rows share training blocks and cells. They are
>   descriptive, not 492 independent replicates.
> * **Unchanged**: LOBO and robustness outputs are identical under v2.2.
> * Details: [`v22_audit/stage2_report.md`](v22_audit/stage2_report.md).

## What it is

`src/ncclenergy/v2/recommender.py` is a transparent, table-based recommender
built only from verified, QC-passed block-level measurements. There is no
learned model.

### Query

A query specifies:
* collective and message size;
* ranks and nodes;
* the topology fingerprint (`topology.py`);
* placement-level NCCL environment, for example `NCCL_P2P_DISABLE=1`;
* NCCL version and CUDA version;
* latency budget (default 1.05 × NCCL automatic);
* objective: `min_energy_slo`, `min_latency`, `min_edp`,
  `max_bytes_per_joule` or `compare_auto`.

### Evidence rule

A query is answered only when fingerprint, environment, NCCL version,
ranks, nodes and collective all match a measured profile exactly.

Inside a measured cell:
* Each candidate configuration is compared with NCCL automatic using the
  block-paired log ratios.
* `min_energy_slo` recommends the lowest-energy configuration that
  satisfies both:
  * the 95 % upper bound of the energy ratio is < 1;
  * the 95 % upper bound of the latency ratio is ≤ the budget.
* If the winner executes the same ALGO/PROTO/nChannels as automatic, the
  answer is `FALLBACK` to automatic.

### No extrapolation

Outside the measured size range, or for an unmeasured topology,
environment or version, the answer is `INSUFFICIENT_EVIDENCE` with NCCL
automatic as the fallback.

Between measured sizes, a recommendation is given only when:
* the two nearest measured sizes are adjacent in the ×2 progression; and
* both have the same recommendation.

Otherwise the answer is `FALLBACK`. Unit test:
`test_no_interpolation_or_merging_across_unmeasured_gaps`.

### Outputs

Predicted energy and time ratios carry their CIs and block counts. **They
are predictions from past blocks, not measurements of the new run.**

## Validation 1 — out of sample in time: exploratory → confirmatory (CONFIRMATORY)

* **Setup**: the Energy-SLO picks were made from 2 exploratory blocks
  (Phase 5), frozen in `prereg.json`, then tested on 12 new blocks
  (Phase 7).
* **Choice step**: 36/39 picks confirmed as saving energy within the SLO;
  25 of them by ≥ 5 %. Two picks were neutral (≈ 0.6 %). One reversed:
  P2_PCI_SHM AllReduce 256 MiB, +1.4 % energy.
* **Magnitude**: comparing the predicted energy ratio (exploratory) with
  the measured ratio (confirmatory) over the 39 cells:
  * median error +0.0003;
  * median absolute error 0.005;
  * maximum absolute error 0.081 (P4_MIX ReduceScatter 512 KiB, predicted
    0.599, measured 0.529);
  * Pearson correlation of the log ratios 0.987.
* **Calibration caveat**: the confirmatory 95 % CI contains the
  exploratory point in 26/39 cells. Two exploratory blocks give point
  predictions that are good on average but too confident cell by cell.
  The confirmatory intervals are the ones to quote.

## Validation 2 — leave-one-block-out within the confirmatory campaign (SECONDARY)

* **Procedure**: for each of the 12 blocks, the recommender is built from
  the other 11 and queried for each of the 41 confirmatory cells. Every
  pick is scored on the held-out block's measurements: 492 rows, in
  `recommender_lobo.csv` and `recommender_lobo_summary.json`.
* **Scope**: the confirmatory matrix holds only automatic and one
  alternative per cell, and its cells were selected because the
  exploratory screen suggested automatic was suboptimal. LOBO therefore
  tests whether the choice between those two is stable across blocks. It
  does not test performance over the full configuration space or over
  the 150 cells where automatic was already best.

| metric (492 held-out cell-blocks) | value |
|---|---|
| status | 436 `RECOMMEND`, 56 `FALLBACK` |
| regret vs best SLO-feasible measured option in the held-out block | median 0, 90th percentile 0, max 3.4 % |
| regret of NCCL automatic in the same rows | median 11.5 % |
| measured saving vs automatic, all rows | median 10.3 % |
| measured saving vs automatic, `RECOMMEND` rows | median 15.2 % |
| predicted − measured energy ratio, `RECOMMEND` rows | median −0.00007 |
| SLO violations (held-out latency > 1.05 × automatic) | 1 / 492 |

`FALLBACK` rows by cell:

| cell | fallback blocks | reason |
|---|---|---|
| P2_NVL AllGather 2 MiB | 12 / 12 | LL128 is faster but uses more energy (RQ2 `DIFFER`) |
| P2_NVL AllReduce 1 GiB | 12 / 12 | energy-equivalent (RQ2) |
| P2_PCI_SHM AllReduce 256 MiB | 12 / 12 | reversal |
| P4_MIX AllReduce 1 KiB | 11 / 12 | ≈ 0.6 % effect not resolved |
| P2_PCI_SHM AllReduce 4 MiB | 9 / 12 | ≈ 0.6 % effect, resolved in only 3 folds |

Held-out failures:

* **Recommended pick used more energy than automatic** in 9 of 436
  `RECOMMEND` rows, by 0.1–3.4 %.
  * Seven of the nine are in cells whose 12-block effect is ≤ 2 %:
    P2_PCI_SHM AllReduce 2/4/16 MiB, and P4_MIX AllReduce 1 KiB.
  * The other two are P4_MIX AllReduce 64 KiB and ReduceScatter 128 KiB.
    Their 12-block effects are 8 % and 6.5 %, with high block-to-block
    variation.
* **The one SLO violation** is P4_MIX ReduceScatter 128 KiB in block 1.
  Ring/Simple was predicted at 0.927 but measured 3.4 % more energy, and
  it was more than 5 % slower than automatic's Ring/LL in that block.

## Tuner export (Phase 10 input)

`phase89.py` writes `nccl_tuner_<placement>.conf` (Energy-SLO) and
`nccl_tuner_<placement>_latency.conf` (min-latency). They are CSV files for
NCCL's example tuner plugin. Only `RECOMMEND` cells are written, and
ranges merge only adjacent ×2 sizes with the same pick. Every other size
stays at NCCL defaults.

Three defects were found and fixed before any workload run:
1. **Byte semantics.** NCCL passes the tuner `nBytes = nRanks × count ×
   elemsize` for AllGather and ReduceScatter: the total buffer, which is
   the nccl-tests size. The first export divided by nRanks.
2. **Example plugin channels.** The upstream example sets `*nChannels = 1`
   on every call, forcing one channel everywhere, even when no rule
   matches.
3. **Example plugin indexing.** The upstream example indexes NCCL's
   contiguous cost table as `float**`. Rules have no effect, and with
   TUNING logging on the process segfaulted.

Fixes 2 and 3: `patches/nccl-2.27.5-example-tuner-fixes.patch`.

Verification (`data/v2/raw/phase10/verify_tuner_r3/`):
* full 1 KiB–1 GiB sweeps on all three placements;
* **36/36 ruled sizes execute the rule**;
* **189/189 sizes execute exactly the ALGO/PROTO/nChannels measured for
  that configuration** in the measurement campaigns;
* ruled sizes match the forced-configuration runs and unruled sizes match
  NCCL automatic, so the tuner reproduces the measured executions.

Two failed attempts are kept with READMEs:
* `verify_tuner/`: stock plugin;
* `verify_tuner_r2/`: a relative config path that NCCL could not open.

## Limits

* **Coverage**: profiles exist only for this host's three placements,
  NCCL 2.27.5, CUDA 12.4 toolkit / 13.0 driver, FP32 sum, 1 KiB–1 GiB.
* **Unprofiled topologies**: any other fingerprint gets
  `INSUFFICIENT_EVIDENCE`. There is no transfer model across topologies.
* **Confirmatory-built recommender**: the recommender and the tuner
  files are built from the 12 confirmatory blocks, so they hold
  alternatives only for the 41 confirmatory cells.
  * Every other size returns `FALLBACK` or `INSUFFICIENT_EVIDENCE`, which
    means NCCL automatic.
  * In the exploratory screen (2 blocks), automatic was already the
    energy-optimal choice in 150 of 189 cells.
* **Workload transfer**: whether collective-level savings carry over to
  a training step is Phase 10, measured separately.
