# Phase 10 — tuner deployment, overhead, and end-to-end workload A/B

> **v2.2 audit corrections (2026-10-09).** Numbers are unchanged; the
> reanalysis reproduces every contrast and finds all-rank execution
> consistent in 50 / 50 runs. Corrections:
> * "Energy" is the **board energy of GPUs 4–7**, not communication-only or
>   whole-system energy.
> * The `no_sync` difference is **not** a measurement of communication
>   energy. The sentence calling it "communication-attributable" is
>   withdrawn.
> * The 5 blocks ran in one session: within-session replicates.
> * These runs recorded no losses, so numerical equivalence of the
>   policies was not checked. v2.2 adds those checks for future runs.
> * The v2.1 workload meter did not log raw counter values or read
>   timestamps. Its energies cannot be recomputed from raw readings; v2.2
>   logs them.
> * Details: [`v22_audit/stage2_report.md`](v22_audit/stage2_report.md).

**Design**: pre-specified in [`phase10_plan.md`](phase10_plan.md)
(commit bf866a7). Amendment A1 (commit 4bb5704) was made after smoke runs
and before workload block 1.

**GPU work**:
* GPUs 4–7 only (subsets 4,5 and 5,6 for the 2-GPU placements);
* ≤ 4 concurrent, one process at a time, idle check before every launch;
* 2026-10-09 03:29–04:08 UTC.

**Raw data**: `data/v2/raw/phase10/`. **Processed**:
`data/v2/processed/phase10/`. **Figure**:
`figures/v2/phase10/workload_blocks.png`.

## 1. Tuner deployment check

**Tuner files**: `data/v2/processed/confirmatory/nccl_tuner_<placement>.conf`
(Energy-SLO) and `nccl_tuner_<placement>_latency.conf` (min-latency). They
were generated from the 12 confirmatory blocks by `phase89.py`. The plugin is
NCCL 2.27.5 `ext-tuner/example` with two fixes
(`patches/nccl-2.27.5-example-tuner-fixes.patch`).

| attempt | dir | result | cause |
|---|---|---|---|
| 1 | `verify_tuner/` | 0/28 rules applied; every collective ran with 1 channel | stock example plugin: `*nChannels = 1` on every call, and cost table indexed as `float**`; our confs also divided AllGather/ReduceScatter sizes by nRanks, while NCCL passes total bytes |
| diag | `tuner_diag/` | stock plugin segfaulted with TUNING logging (rc −11); fixed plugin plus absolute path: rules applied | — |
| 2 | `verify_tuner_r2/` | 0/36 rules; 153/153 unruled sizes now match automatic incl. nChannels | my error: relative `--policy-dir`; NCCL could not open the config and the plugin silently used defaults |
| **3** | **`verify_tuner_r3/`** | **36/36 ruled sizes execute the rule; 189/189 sizes execute exactly the ALGO/PROTO/nChannels measured for that configuration in the campaigns** | — |

Attempt 3 ran full 1 KiB–1 GiB sweeps on all three placements:
* P4_MIX: 16 ruled sizes, 47 unruled;
* P2_NVL: 12 ruled, 51 unruled;
* P2_PCI_SHM: 8 ruled, 55 unruled.

Every ruled size matched the forced-configuration run of the same
configuration, and every unruled size matched NCCL automatic, including the
channel count.

## 2. Tuner overhead

* **Setup**: P4_MIX AllReduce, 3 interleaved ABAB repetitions. The
  comparison arm runs the same execution without the plugin:
  * 4 KiB: no rule matches, comparison is automatic;
  * 64 KiB: rule Tree/LL, comparison is forced with
    `NCCL_ALGO=Tree NCCL_PROTO=LL`.
* **Execution check**: identical in both arms, from the profiler.

| size | executed | t per op, tuner / plain | time ratio [95 % CI] | energy ratio [95 % CI] |
|---|---|---|---|---|
| 4 KiB | RING/LL/1 | 12.194 / 12.202 µs | 0.999 [0.984, 1.015] | 0.999 [0.982, 1.017] |
| 64 KiB | TREE/LL/8 | 32.82 / 33.28 µs | 0.986 [0.906, 1.073] | 0.983 [0.895, 1.079] |

No overhead is detectable. At 4 KiB, where per-call host cost would show
most, the bound is ±1.6 %. The 64 KiB interval is wide (3 reps).

## 3. Workload A/B (RQ3)

**Setup**:
* synthetic MLP, 8 × Linear(4096, 4096) + GELU, FP32, P4_MIX (GPUs 4–7);
* 5 blocks, 10 units each, in seeded random order;
* 100 measured steps (1000 for `ddp_nosync` at batch 128) after 10
  warm-up steps;
* energy = raw board energy of GPUs 4–7 over the measured steps
  (commit-aligned counter, read by rank 0 at start and stop only);
* time = CUDA-event step time (max over ranks).

**Completion**: 50/50 runs, rc 0. The stop rule did not fire: there was
no HW slowdown, HW thermal, power brake, sync boost or SW thermal event.
Maximum temperature 63 °C. No foreign process.

### Executed collectives (profiler, rank 0, all blocks)

| arm | 64 MiB AllReduce (DDP buckets) | 64 MiB AllGather / ReduceScatter (FSDP) |
|---|---|---|
| `ddp_auto`, `ddpb2048_auto` | RING/SIMPLE/8 | — |
| `ddp_energy_slo`, `ddp_lat_opt`, `ddpb2048_energy_slo` | **TREE/SIMPLE/8** (tuner rule) | — |
| `fsdp_auto`, `fsdp_energy_slo`, `fsdp_lat_opt` | — | RING/SIMPLE/8 in all three (no rule at 64 MiB) |

These executions are what the plan predicted. The DDP first-iteration
512 MiB AllReduce runs TREE/SIMPLE/8 in every arm, and the 256 MiB
parameter broadcast runs RING/SIMPLE/8; both happen during warm-up, outside
the measured region.

### Results (5 blocks; 95 % t-CI, df = 4)

| family | contrast | energy/step ratio [95 % CI] | per-block energy ratios | step-time ratio [95 % CI] |
|---|---|---|---|---|
| **DDP, batch 128** (primary) | `energy_slo / auto` | **0.791 [0.784, 0.799]**, p = 2.4e-7 | 0.797, 0.790, 0.787, 0.784, 0.797 | **0.706 [0.696, 0.716]** |
| DDP, batch 128 | `lat_opt / auto` | 0.787 [0.778, 0.797] | 0.791, 0.791, 0.794, 0.775, 0.786 | 0.703 [0.699, 0.708] |
| DDP, batch 128 | `lat_opt / energy_slo` | 0.995 [0.983, 1.007]; TOST ±3 % p = 0.002 | — | 0.996 [0.978, 1.014] |
| **FSDP, batch 128** (primary, negative control) | `energy_slo / auto` | **1.001 [0.973, 1.030]**; TOST ±3 % p = 0.025 | 1.043, 0.990, 0.987, 0.994, 0.993 | 0.996 [0.985, 1.007] |
| FSDP | `lat_opt / auto` | 1.000 [0.984, 1.016]; TOST p = 0.004 | — | 0.991 [0.983, 0.998] (unadjusted p = 0.028) |
| DDP, batch 2048 (secondary) | `energy_slo / auto` | **0.885 [0.880, 0.889]**, p = 3e-7 | 0.886, 0.889, 0.887, 0.882, 0.880 | 0.823 [0.818, 0.828] |

Absolute values (mean ± SD over 5 blocks):

| arm | energy per step (J, 4 GPUs) | step time (ms) |
|---|---|---|
| `ddp_auto` | 51.04 ± 0.93 | 83.41 ± 0.51 |
| `ddp_energy_slo` | 40.38 ± 0.51 | 58.89 ± 0.58 |
| `ddp_lat_opt` | 40.18 ± 0.56 | 58.64 ± 0.36 |
| `ddp_nosync_auto` (compute only) | 9.60 ± 0.01 | 8.04 ± 0.02 |
| `ddpb2048_auto` | 124.07 ± 0.32 | 121.00 ± 0.61 |
| `ddpb2048_energy_slo` | 109.76 ± 0.35 | 99.58 ± 0.59 |
| `ddpb2048_nosync_auto` (compute only) | 94.11 ± 0.18 | 80.19 ± 0.14 |
| `fsdp_auto` | 72.22 ± 1.28 | 123.43 ± 0.36 |
| `fsdp_energy_slo` | 72.29 ± 0.41 | 122.91 ± 1.01 |
| `fsdp_lat_opt` | 72.22 ± 0.89 | 122.29 ± 0.63 |

Two notes on the statistics:
* **Wilcoxon**: the plan named an exact two-sided Wilcoxon test. With
  5 blocks its smallest attainable p is 0.0625, which the DDP contrasts
  reach (all 5 blocks in the same direction). It cannot fall below 0.05
  at this n, so the t-CI carries the inference.
* **FSDP outlier**: block 1 of FSDP `energy_slo/auto` was 1.043. The
  other four blocks are 0.987–0.994.

### Power regime (from the 50 ms sampler during each measured region)

| arms | mean board power per GPU | SW power cap active | mean graphics clock |
|---|---|---|---|
| DDP batch 128, FSDP (all policies) | 145–168 W | 0 % of samples | 1910–1914 MHz |
| `ddp_nosync_auto`, all `ddpb2048_*` | 251–290 W (peaks to 360 W) | **96–98 % of samples** | 1484–1621 MHz |

* **Capped runs**: the compute-heavy runs are held at the 300 W board
  limit. SW power cap is the only clock-event bit present. It was not a
  stop condition in the plan (`phase10_plan.md`), and the board limit was
  not changed.
* **Consequence for batch 2048**: that contrast measures the policy effect
  under power capping. In this regime energy ≈ capped power × time, so the
  energy saving follows the step-time saving.
* **Consequence for batch 128**: the DDP batch-128 contrast is not
  affected (no capping). However, its compute-only reference
  (`ddp_nosync`) is capped, so a subtraction from it compares runs in
  different power regimes.

### Against the predictions in the plan

| prediction | outcome |
|---|---|
| DDP batch 128: the original rule compared energy/time intervals and policy execution | **Descriptive synthetic result reproduced**: 0.791 [0.784, 0.799]; TREE/SIMPLE/8 executed. Equivalent training and real-model transfer were not checked |
| Earlier heuristic using `no_sync` as a communication share | Withdrawn as a scientific decomposition/prediction: power regime, overlap and execution differ. The historical 20.9% board-energy contrast remains descriptive; its numerical arithmetic does not validate the heuristic |
| FSDP: ratio = 1, identical executions | **met**: identical executions; 1.001 [0.973, 1.030] |
| DDP `lat_opt` ≡ `energy_slo` (same executions) | **met**: 0.995 [0.983, 1.007], TOST ±3 % p = 0.002 |
| batch 2048: smaller relative board-energy contrast than batch 128 | **Descriptive result**: 0.885 vs 0.791. The recorded wall-time difference from `no_sync` fell from 40.8 ms to 19.4 ms; it is not an isolated communication duration. Changes in overlap are a possible explanation, not measured causal attribution |

The secondary FSDP `lat_opt/auto` step-time difference (−0.9 %,
unadjusted p = 0.028) occurs with identical executions. Step time was
tested in 7 contrasts. Under Holm correction this p ranks 4th, with
threshold 0.05/4 = 0.0125, so it does not survive. It is not interpreted.

## 4. What this supports

**Supported**, on this host (P4_MIX, NCCL 2.27.5, FP32, synthetic MLP, 5
within-session blocks): deploying the collective-level Energy-SLO policy
through NCCL's tuner interface reduces the **board energy of the four GPUs
per training step**. Numerical equivalence of the policies' training was
not checked in these runs.
* DDP batch 128: by 21 %, with 29 % shorter steps.
* DDP batch 2048, power-capped: by 11.5 %, with 18 % shorter steps.
* FSDP: the policy correctly leaves NCCL's choices unchanged, and energy
  is unchanged.

**Not supported / not tested**:
* real models, mixed precision, other parallelism layouts, other
  placements (P2_NVL, P2_PCI_SHM), other GPU counts;
* any setting where the DDP buckets fall outside the profiled range;
* whole-system energy, time to quality, or industry-scale training.

The historical `no_sync` subtraction yielded the previously reported
90/86 % (batch 128) and 34/19 % (batch 2048) time fractions. Those are
differences between runs, not isolated communication fractions: disabling
synchronization changes overlap and power/clock regime. They do not
establish a communication-energy decomposition or an upper bound on
real-training benefit. Numerical training equivalence was not recorded;
the valid retained claim is a synthetic workload timing/board-energy
observation (`data/v2/raw/phase10/`, processed `workload_runs.csv`).
