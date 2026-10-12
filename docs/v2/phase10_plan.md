# Phase 10 — analysis plan for the workload A/B (written before any workload run)

This plan was committed before the first workload run. The commit hash is in
`git log`. Anything not listed here is labelled exploratory in the Phase 10
report.

## Workload

* **Model**: `src/ncclenergy/v2/workload.py`, a synthetic MLP:
  * 8 × (`Linear(4096, 4096, bias=False)` + GELU);
  * batch 128, FP32, TF32 off, SGD;
  * synthetic data only.
* **Hardware**: one process per GPU on P4_MIX, GPUs 4–7. CPU and memory are
  bound to NUMA node 1.
* **Software**: PyTorch 2.9.1 with NCCL 2.27.5 from the wheel.
* **Modes**:
  * `ddp`: DistributedDataParallel, `bucket_cap_mb=25`. Each layer's
    gradient (64 MiB) exceeds the cap, so it gets its own bucket. Expected
    traffic is 8 AllReduces of 64 MiB per step.
  * `fsdp`: FullyShardedDataParallel, one unit per Block. Expected traffic
    per step is 8 AllGathers forward + 8 AllGathers backward + 8
    ReduceScatters, each 64 MiB total.
  * `ddp_nosync`: DDP under `no_sync()`. No gradient AllReduce; this is the
    compute-only reference.
* **Run length**: each run does 10 warm-up steps, then 40 measured steps.
* **Energy**: raw board energy of GPUs 4–7, from the commit-aligned NVML
  counter, read by rank 0 at start and stop only.
* **Time**: CUDA-event step time, taking the maximum over ranks.

## Policies

| arm | how | NCCL tuner file |
|---|---|---|
| `auto` | NCCL defaults, no tuner plugin | — |
| `lat_opt` | example tuner (fixed, `patches/nccl-2.27.5-example-tuner-fixes.patch`) | `data/v2/processed/confirmatory/nccl_tuner_P4_MIX_latency.conf` |
| `energy_slo` | same plugin | `data/v2/processed/confirmatory/nccl_tuner_P4_MIX.conf` |

Rules relevant to the expected collective sizes:

* **AllReduce, 256 KiB–64 MiB → Tree/Simple**, in both files. The DDP
  buckets (64 MiB) fall inside this range.
* **AllGather and ReduceScatter: rules only at 128/256 KiB–512 KiB.** FSDP's
  64 MiB collectives fall outside, so NCCL's defaults should run.

FSDP is therefore a **negative control**: the policies should execute
exactly what `auto` executes. In DDP, `lat_opt` and `energy_slo` should
execute the same configuration.

## Design

* 5 blocks, fixed in advance; no optional stopping.
* Each block runs 7 units (`ddp`/`fsdp` × 3 arms, plus `ddp_nosync`) in a
  seeded random order (`phase10-workload-<block>`).
* One torchrun at a time; GPU allocation checked idle before every run.
* **Stop on any of**:
  * non-zero return code;
  * timeout;
  * HW-slowdown, HW-thermal, HW-power-brake, sync-boost or SW-thermal
    clock event during a run;
  * any foreign process on GPUs 4–7.
* The execution evidence for every run is the profiler plugin output
  (`prof/`): for every collective and size, the executed
  ALGO/PROTO/nChannels.

## Endpoints and statistics

* **Primary endpoint**:
  * `energy_slo` vs `auto`, per mode (`ddp`, `fsdp`);
  * per block, ln(energy-per-step ratio);
  * mean over blocks with a 95 % t-CI (df = 4);
  * exact Wilcoxon signed-rank test (two-sided);
  * every block value is listed.
* **Secondary endpoints**:
  * step-time ratio, same method;
  * `lat_opt` vs `auto`;
  * `lat_opt` vs `energy_slo`.
* **Equivalence**: FSDP and the DDP `lat_opt`–`energy_slo` pair are judged
  with TOST at ±3 %, which with 5 blocks is descriptive.
* **Execution check**: a contrast is interpreted only if the profiler
  evidence shows that the arms executed the configurations listed above.
  Any deviation is reported, and that contrast is not interpreted as a
  policy effect.

## Predictions (recorded before the runs)

* **DDP**:
  * In the confirmatory campaign, P4_MIX AllReduce at 64 MiB with
    Tree/Simple vs auto (Ring/Simple, 8 channels) had energy ratio 0.654
    [0.643, 0.664] and time ratio 0.651 (12 blocks, `rq1.csv`).
  * End to end, the saving is diluted by compute. Predicted direction:
    energy-per-step ratio < 1 and step-time ratio < 1.
  * Rough magnitude estimate, not a measurement: if all of the
    communication energy in `E_ddp_auto − E_ddp_nosync` scaled like the
    isolated collective, the saving would be about
    0.35 × (E_ddp_auto − E_ddp_nosync) / E_ddp_auto. Overlap with compute
    and the different power profile of concurrent compute make this an
    estimate only.
* **FSDP**: ratio = 1 within noise. The executed configurations should be
  identical across arms.
* **Transfer is supported** if the DDP `energy_slo`/`auto` 95 % CI lies
  entirely below 1, and the profiler shows the policy configuration
  executed.

## Amendment A1 (2026-10-09 ~03:51 UTC, after smoke runs, before workload block 1)

The smoke runs are in `data/v2/raw/phase10/workload_smoke/` and are not
analysed. They showed three things:

* DDP at 40 steps measured a 2.4 s region, below the protocol's
  T* = 2.5 s.
* `ddp_nosync` takes 8 ms per step, so 40 steps would give a 0.3 s region.
* DDP at batch 128 is about 87 % exposed communication (step 59 ms;
  compute-only step 8 ms). That is the most communication-bound case
  possible, not a typical one.

Changes, all made before any block:

1. **Measured steps** (warm-up stays at 10):
   * `ddp` and `fsdp`: 100;
   * `ddp_nosync` (batch 128): 1000;
   * every measured region is now ≥ 6 s.
2. **New family `ddpb2048`**: DDP at batch 2048, with arms `auto` and
   `energy_slo` (100 steps), plus the reference `ddpb2048_nosync_auto`
   (100 steps).
   * Smoke: compute-only step 80 ms; `energy_slo` step 100 ms.
   * Its contrast `energy_slo/auto` is a **secondary endpoint**, analysed
     the same way as the others.
   * Prediction: the relative saving is smaller than at batch 128, because
     the same communication saving is diluted by about 10× more compute
     energy.
3. **Block size**: 10 units; order seeded as before.

The primary endpoints are unchanged:

* DDP batch 128 `energy_slo/auto`;
* FSDP `energy_slo/auto` (negative control).

Both smoke runs confirmed the planned executions:

* DDP `energy_slo` runs its 64 MiB AllReduces as TREE/SIMPLE/8;
* FSDP `auto` runs RING/SIMPLE/8 for both its 64 MiB AllGathers and its
  64 MiB ReduceScatters.
