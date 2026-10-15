# Replication on a 6× GTX 1080 Ti + 3× Titan cluster (methodology v2)

> **Archived proposal (2026-10-14).** This replication was not performed and
> is no longer part of the current project scope. The completed release is
> A6000-only. The design and limitations below are retained as historical
> material, not an instruction or authorization to run this campaign.

> **Release-candidate correction (2026-10-11).** Python sampled paths pass
> CPU no-counter tests, but this does not validate the native executable.
> The historical v2.1 patch is unchanged. The new
> `nccl-tests-afd59ab-v2.3-sampled-gate.patch` avoids native counter reads in
> sampled mode and passes CPU helper tests; a full CUDA build and real Pascal
> measurement validation remain outstanding. Native sampled collection and
> CUDA training are disabled until those gates pass. This guide is a proposed
> replication design, not a ready-to-run validated workflow.

> **v2.2 audit note (2026-10-09).**
> * **Fallback was incomplete before v2.2**: the v2.1 "sampled-power
>   fallback" was not complete. `runner.idle_baseline` always read the
>   cumulative counter, which would stop a block on a board without it.
> * **Fixed in v2.2**: the sampled idle baseline, the `SampledMeter` and
>   the fail-closed QC never call the counter. This was tested end to end
>   with a mock that fails if the counter is called. The C region patch
>   still attempts counter reads; on such boards they return NOT_SUPPORTED
>   and become flags.
> * **Blocker: PyTorch wheel has no Pascal code**: PyTorch 2.9.1+cu128 is
>   compiled for sm_70–sm_120 only, so the Phase 10 workload cannot run on
>   Pascal (GTX 1080 Ti, Titan X Pascal, Titan Xp) with this wheel.
> * **NCCL wheel does**: the NCCL 2.27.5 wheel contains sm_61 code, so the
>   nccl-tests microbenchmarks remain possible.
> * **Blocker: unknown Titan models**: the exact Titan models are unknown.
> * Use `requirements-analysis.txt` / `requirements-gpu.txt`, and declare
>   the allocation explicitly.

This package prepares, but does not run, a replication of methodology v2 on a
cluster with six GTX 1080 Ti and three Titan GPUs. Nothing here assumes the
A6000 findings transfer. Replication data are stored, analysed and reported
separately and are never pooled with A6000 data.

## 0. Rules carried over unchanged

* No administrator actions, only idle GPUs in your allocation, ≤ 6 GPUs at
  once, one experiment process at a time.
* No external/unspecified counter queries during a measurement. The validated
  A6000 sampler intentionally uses POWER_INSTANT; validate its behavior on the
  replication hardware. Local dose tests are not universal API guarantees;
  see `phase4_validation.md`.
* Phases, approval gates, QC and statistics are those of `protocol.md`, with
  the replication-specific differences below.

## 1. User-space environment

```bash
git clone <this repository> nccl-energy-frontier && cd nccl-energy-frontier
```

```bash
python3 -m venv .venv-repl && . .venv-repl/bin/activate
```

```bash
pip install numpy pandas matplotlib PyYAML scipy nvidia-ml-py
```

PyTorch and NCCL must support the GPUs' compute capability: GTX 1080 Ti is
`sm_61`, and the Titans are `sm_61` (X/Xp), `sm_70` (V) or `sm_75` (RTX).
Install a PyTorch wheel whose `torch.cuda.get_arch_list()` contains it, then
check (step 2). The NCCL used by nccl-tests must be the same library PyTorch
loads (`nvidia-nccl-cu12` from that environment), as in the A6000 study.

## 2. Hardware and counter capability detection

List the GPUs you are allowed to use in `--gpus`. Counter checks are made only
on those, and only when they have no process.

```bash
PYTHONPATH=src python -m ncclenergy.v2.capabilities --gpus 0,1,2,3,4,5 --out data/v2_replication/<cluster>/caps.json
```

On idle allocated GPUs that report a counter, also time its commit period:

```bash
PYTHONPATH=src python -m ncclenergy.v2.capabilities --gpus 0,1 --probe-counter 0,1 --out data/v2_replication/<cluster>/caps_probe.json
```

Expected, to be confirmed by the script:

| GPU | arch | `nvmlDeviceGetTotalEnergyConsumption` | measurement mode |
|---|---|---|---|
| GTX 1080 Ti | Pascal, sm_61 | NVML documents it for Volta and newer: expect *not supported* | `SAMPLED_POWER_FALLBACK` |
| Titan X / Xp | Pascal, sm_61 | expect *not supported* | `SAMPLED_POWER_FALLBACK` |
| Titan V | Volta, sm_70 | expect supported | `COUNTER_ALIGNED` if a commit period is detectable |
| Titan RTX | Turing, sm_75 | expect supported | `COUNTER_ALIGNED` if a commit period is detectable |

The capability file also records `power_usage`, `power_instant_field`,
NVLink links, P2P matrix (`nvidia-smi topo -p2p r`), and homogeneous groups.

## 3. Build (user space)

```bash
bash scripts/build_nccl_tests.sh
```

Then apply the v2.1 region patch in a separate checkout, as on the A6000.
Set `NVCC_GENCODE` to the target architecture:

```bash
git clone third_party/nccl-tests third_party/nccl-tests-v2 && git -C third_party/nccl-tests-v2 checkout afd59abab774a6a4b492eb344bd0c845a0b7bb4d && git -C third_party/nccl-tests-v2 apply ../../patches/nccl-tests-afd59ab-v2.1-region.patch
```

```bash
make -C third_party/nccl-tests-v2 -j16 CUDA_HOME=$CUDA_HOME NCCL_HOME=$PWD/third_party/nccl_home NVCC_GENCODE="-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75" BUILDDIR=$PWD/third_party/nccl-tests-v2/build src.build
```

Build the NCCL profiler plugin (`native/nccl_profiler_algo`, API v4) against
the headers of the same NCCL version:

```bash
make -C native/nccl_profiler_algo NCCL_SRC=<NCCL source of the same version>
```

**If NCCL is older than the v4 profiler API**, the plugin will not load
(`PROFILER/Plugin: failed to find ncclProfiler_v4` in `nccl.log`). Every timed
process is then `UNVERIFIED`. In that case either keep NCCL ≥ 2.27, or
document a TUNING-log-based verification run per condition before claiming
configuration-specific results.

## 4. Topology-specific configuration

Copy `configs/v2/replication/template.yaml` once per homogeneous group and
fill in:
* `allocation` and `placements` from `caps.json` / `topo_m`: names of your
  choosing, with their link categories, e.g. 1080 Ti PCIe pairs within one
  switch and across switches;
* the 4- and 6-GPU sets;
* Titan pairs, bridged or not.

If a placement's default transport hangs, as PCIe P2P did on the A6000 host,
the probe stops at the first timeout. Do not retry it; use a host-path control
(`env: {NCCL_P2P_DISABLE: "1"}`) and record why.

**Order of work**:
1. Homogeneous 1080 Ti subsets (fallback mode).
2. Homogeneous Titan subsets (counter mode if supported).
3. Mixed 1080 Ti + Titan only as `EXPLORATORY_HETEROGENEOUS`, never in
   confirmatory claims.

## 5. Sampled-power fallback (no cumulative counter)

Set `energy_source: sampled` in the campaign YAML. Then:

* Region energy per GPU = trapezoidal integral of sampled power over the
  synchronized region [start read, loop end] from the region record. It uses
  `POWER_INSTANT` if the field is supported, else `PowerUsage` shifted by its
  measured averaging lag (estimate the lag as in Phase 4 from regions with
  sharp load edges).
* The patched benchmark still writes region records. Their counter fields
  carry NVML's `NOT_SUPPORTED` codes, which become flags. They are not
  repaired, and QC accepts them only because `energy_source: sampled` is
  declared.
* Validation required before any campaign: R-linearity (energy ∝ R, small
  intercept), repeated-run scatter, and idle stability. There is no counter
  cross-check, so report this as a weaker method. Use a longer `T*` (≥ 5 s is
  advisable) because sampling-edge errors are a fixed offset per region.
* Results are labelled `energy_source=sampled` in every table and are never
  compared numerically with counter-based A6000 data.

## 6. Smoke and full runs (per homogeneous group)

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner preflight --alloc <ids> --out data/v2_replication/<cluster>/preflight.json
```

```bash
PYTHONPATH=src python -m ncclenergy.v2.validate --exp all --out data/v2_replication/<cluster>/validation
```

(edit `P2`/`P4` in `validate.py` for your GPU ids first, or call the
experiment functions directly)

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner probe --campaign configs/v2/replication/<group>_setup.yaml --setup data/v2_replication/<cluster>/setup --timeout 60
```

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner calibrate --campaign configs/v2/replication/<group>_setup.yaml --setup data/v2_replication/<cluster>/setup
```

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner freeze --setup data/v2_replication/<cluster>/setup --T-star 2.5
```

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner block --campaign configs/v2/replication/<group>_exploratory.yaml --setup data/v2_replication/<cluster>/setup
```

```bash
PYTHONPATH=src python -m ncclenergy.v2.analyze_cli exploratory --campaign data/v2_replication/<cluster>/<group>_exploratory --out data/v2_replication/<cluster>/processed/<group>_exploratory
```

Confirmatory runs follow Phase 6 of `protocol.md`: freeze a new `prereg`
for the replication; do not reuse A6000 candidates or block counts.

## 7. Schema and version checks

Before analysis, verify for every block:
* `block_summary.json → versions`: driver, NVML, NCCL library hash, patch
  hash (`patches/nccl-tests-afd59ab-v2.1-region.patch`), plugin hash. All
  blocks of a campaign must agree; otherwise split the campaign.
* Region records have `"schema": "nccle-region-v2"`, profiler files
  `"schema": "nccle-profiler-v1"`.
* `energy_source` is identical across a campaign.

## 8. Strict separation

| item | A6000 study | replication |
|---|---|---|
| raw data | `data/v2/raw/` | `data/v2_replication/<cluster>/` |
| processed | `data/v2/processed/` | `data/v2_replication/<cluster>/processed/` |
| recommender profiles | keyed by A6000 fingerprints | keyed by the replication fingerprints; the recommender returns `INSUFFICIENT_EVIDENCE` across clusters by construction |
