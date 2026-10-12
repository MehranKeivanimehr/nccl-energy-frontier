# nccl-energy-frontier

A **pilot characterization study** of latency and GPU-board energy for NCCL
collectives (AllReduce, AllGather, ReduceScatter) across NCCL algorithms and
protocols. It was measured on one server: 8× RTX A6000, driver 580, NCCL
2.27.5, 2–4 GPUs.

It is **not** a general deployment claim, and **not** a completed proof of a
latency–energy tradeoff.

> **Release status: v2.2 pilot release (2026-10-09, branch
> `ext/v2.2-audit`).**
> * The experimental scope is frozen. No further GPU campaign belongs to this
>   release.
> * The current claim status is in [`docs/release/claims.md`](docs/release/claims.md).
> * The final review report is in
>   [`docs/release/final_review_report.md`](docs/release/final_review_report.md).

> **Release-candidate corrections (2026-10-11).** This is a sanitized-content
> candidate, not a new experiment or a Git commit. Historical data, protocols,
> pre-registration and frozen numerical outputs are unchanged. Current claim
> limits and literature support are in
> [`docs/release/scientific_resolution.md`](docs/release/scientific_resolution.md);
> presentation/code corrections are listed in
> [`docs/release/corrections_2026-10-11.md`](docs/release/corrections_2026-10-11.md).

## Main findings

The category tags refer to the evidence categories in the next section.

1. **[v2 confirmatory] Tested settings can substantially outperform NCCL's
   automatic choice on this measured deployment.**
   * A pre-registered pick used less GPU-board energy than automatic in
     36 of 39 cells.
   * By the pre-registered point-estimate rule, the saving was ≥ 5 % in
     25 cells. A separately identified threshold test supports ≥ 5 % in
     21 cells.
   * The largest effect is 4-GPU AllReduce at 512 KiB, Tree/Simple vs
     automatic's Ring/LL: energy ratio 0.312 [0.308, 0.317].
   * Scope: each cell compares one selected pick with automatic only. The
     best configuration overall was not established.
   * The 12 blocks ran back to back in one session. They are within-session
     replicates, not independent days or hosts.
2. **[v2 Phase 10] Transfer to a synthetic training workload shows promising
   improvements, with stated limitations.**
   * Synthetic FP32 MLP, DDP, 4 GPUs, 5 within-session blocks, tuner-deployed
     policy vs automatic:
     * batch 128: energy per step 0.791 [0.784, 0.799], step time 0.706;
     * batch 2048: 0.885 [0.880, 0.889], power-capped;
     * FSDP negative control: 1.001 [0.973, 1.030].
   * Numerical equivalence was not recorded in these runs.
   * Every rank used the same synthetic batch. Under that setting a skipped
     gradient AllReduce would not have been detected.
   * The meter of that time recorded no QC flags. A post-hoc plausibility
     check of all 200 GPU readings found none implausible.
   * This says nothing about real models, BF16, time to quality or
     whole-system energy.
3. **[v2 + v2.2] The small latency-versus-energy optimum difference was not
   resolved.**
   * On the NVLink pair, AllGather 2 MiB with LL128 has 2.8 % lower latency
     than auto (equivalently, auto is 2.9 % slower than LL128).
   * Its energy penalty is 1.012 [1.003, 1.022] with the energy counter. It
     is not robust across estimators.
   * The v2.2 measurement-validation campaign: 70 of 72 processes accepted.
     Its pre-specified decisions were D1 INCONCLUSIVE, D2 INCONCLUSIVE, D3
     SYSTEMATIC_OFFSET and D4 NOT_SUFFICIENT. The 40 s LL128 ÷ auto interval
     half-width was 3.98 %, against the ≤ 0.4 % needed.
   * The claim (C7) stays **preliminary and unresolved**.
4. **[pilot v1, superseded]** The first pilot (driver 550, uncorrected
   boundary counter reads, 3 of 5 trials) found the latency- and
   energy-optimal executed configuration identical in 123 of 126 cells.

Post-hoc exploratory analyses of the validation campaign (category 4) suggest
two things to address in future designs:
* GPU thermal state is strongly associated with between-process energy
  variation;
* the counter and sampled board power differ by an unexplained, roughly
  constant amount per measured region.

Neither changes any decision above.

## Evidence categories (never pooled)

| # | category | method ID | data | documents |
|---|---|---|---|---|
| 1 | **Historical pilot** (2026-10-06): driver 550, uncorrected boundary counter reads, 3 trials | `v1` | `data/raw/`, `data/processed/`, `figures/full/` | [`docs/pilot_v1/README_pilot_v1.md`](docs/pilot_v1/README_pilot_v1.md), [`docs/methodology.md`](docs/methodology.md), [`docs/limitations.md`](docs/limitations.md) |
| 2 | **v2 exploratory and confirmatory** (2026-10-08/09): driver 580, commit-aligned counter region, 2 exploratory and 12 pre-registered confirmatory blocks, Phase 10 workload | `v2.1` | `data/v2/raw/`, `data/v2/processed/`, `figures/v2/` | [`docs/v2/`](docs/v2/), with Stage 2 corrections |
| 3 | **v2.2 measurement validation** (gate 3b, 2026-10-09): 6 blocks on GPUs 4,5, pre-specified D1–D4 | `v2.2-fail-closed`, `v2.2-aligned-meter-logged` | `data/v2.2/` | [`gate3b_measval_report.md`](docs/v2/v22_audit/gate3b_measval_report.md) §1–4 |
| 4 | **Post-hoc exploratory analysis**: reanalysis of the v2 data (Stage 2) and of the gate 3b data | — | `data/v2/reanalysis/`, `data/v2.2/processed/*/posthoc_temperature.json` | [`stage2_report.md`](docs/v2/v22_audit/stage2_report.md), gate 3b report §5 |
| 5 | **Future experiments**: not run, not needed for this release | — | — | [`docs/release/limitations.md`](docs/release/limitations.md) §6 |

**What "energy" means here**:
* NVML board energy of the **participating GPUs** while the collective
  runs.
* It is **not** communication-only energy: it includes the boards' static
  and idle power.
* It is **not** whole-system energy: CPU, DRAM, PCIe switches, NVLink
  bridges outside the board, and PSU losses are not measured.
* No external power reference was used. Absolute accuracy is not
  characterized.

## Reproduce

**Analysis only** (CPU, no GPU, no NVML, no PyTorch), Python 3.11:

Use Linux for the supplied command-line tools: they import Linux process
control even in some analysis paths. Windows requires a separate adapter;
native Windows reproduction is not currently supported.

```bash
python -m pip install -r requirements-analysis.txt
```

```bash
PYTHONPATH=src python -m ncclenergy.v2.release_repro --out logs/release_repro/my_run
```

* `release_repro` re-runs every published analysis from the retained
  evidence into the new directory. It compares each output file with the
  committed one and verifies that no frozen output changed.
* The full CPU test suite additionally needs PyTorch's CPU wheel, the
  `nvidia-ml-py` Python bindings (mocked; no NVIDIA driver required), GCC and
  Make. See the separate test installation commands below.
* Full commands, expected results and the sanitized-export check are in
  [`docs/release/reproduction.md`](docs/release/reproduction.md).

**Full CPU tests** (Linux):

```bash
python -m pip install -r requirements-tests.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
python tests/run_tests.py
```

**Future GPU experiments** (not part of this release):
* Experimental launch entry points that have not passed the current safety
  and numerical-validation gates are disabled in this release candidate.
* They need an explicitly allocated, idle-checked set of GPUs, at most four
  by default, one benchmark job at a time, with the watchdog and safety
  limits.
* See [`docs/release/reproduction.md`](docs/release/reproduction.md) §3.
  Enabled v2 runner commands require explicit allocation. Historical v1,
  Phase4 validation and CUDA workload launchers are disabled; guarded
  workloads and native sampled fallback still need real-hardware validation.

## Documents

| document | content |
|---|---|
| [`docs/release/claims.md`](docs/release/claims.md) | claim-status table with key numbers and sources |
| [`docs/release/methodology.md`](docs/release/methodology.md) | methods by evidence category |
| [`docs/release/limitations.md`](docs/release/limitations.md) | limitations, untested code paths, future experiments |
| [`docs/release/reproduction.md`](docs/release/reproduction.md) | analysis-only reproduction; future GPU commands with allocation and safety requirements |
| [`docs/release/change_ledger.md`](docs/release/change_ledger.md) | every number or statement that changed from an earlier report |
| [`docs/release/data_inventory.md`](docs/release/data_inventory.md) | data inventory and checksums |
| [`docs/release/integration_review.md`](docs/release/integration_review.md) | content-only public branch review; never merge private history |
| [`docs/release/validation_2026-10-11.md`](docs/release/validation_2026-10-11.md) | current local CPU tests and analysis reproduction, with validation boundaries |
| [`docs/release/final_review_report.md`](docs/release/final_review_report.md) | final review report of the extension |
| [`docs/v2/v22_audit/`](docs/v2/v22_audit/) | v2.2 audit: Stage 2 report, gate 3a and 3b reports, analysis-amendment record |

## Layout

```
src/ncclenergy/          pilot (v1) pipeline
src/ncclenergy/v2/       v2/v2.2: runner, QC, telemetry, meters, process control, watchdog, analyses, release tools
native/                  NCCL profiler plugin, example tuner (patched), guarded tuner
patches/                 nccl-tests region patch, tuner fixes
configs/                 campaign configs, frozen schedules, frozen analysis code (configs/v2.2/frozen_code)
data/, figures/          raw evidence (per-process directories on disk, see data inventory) and outputs
results/                 versioned release tables and figures
tests/                   CPU test suite (python tests/run_tests.py)
docs/                    reports (pilot_v1, v2, v2.2 audit, release)
```

## License

Project code is MIT; see [LICENSE](LICENSE). Bundled NVIDIA headers retain
their own copyright and license notices.
