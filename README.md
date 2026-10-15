# NCCL Energy Frontier

**How much GPU-board energy does an NCCL collective cost, and does NCCL's automatic
choice of algorithm and protocol leave energy on the table?**

![study: pilot characterization](https://img.shields.io/badge/study-pilot%20characterization-orange)
![measured: 2-4 RTX A6000 GPUs](https://img.shields.io/badge/measured-2%E2%80%934%20RTX%20A6000-76B900)
![NCCL 2.27.5](https://img.shields.io/badge/NCCL-2.27.5-76B900)
![python 3.11](https://img.shields.io/badge/python-3.11-3776AB)
![license MIT](https://img.shields.io/badge/license-MIT-blue)

A **pilot characterization study** of latency and GPU-board energy for NCCL
collectives (AllReduce, AllGather, ReduceScatter) across NCCL algorithms and
protocols. It was measured on one server: 8× RTX A6000, driver 580, NCCL
2.27.5, 2–4 GPUs.

It is **not** a general deployment claim, and **not** a completed proof of a
latency–energy tradeoff.

Energy values are NVML-reported estimates of participating-GPU board energy.
No external physical-energy reference or absolute error bound was established.

![A6000 study overview: 3 smoke launches, 23 calibration launches and 72 held-out launches; 70 primary estimates, one NVLink pair, no global-optimum claim](studies/a6000-stage2-v2.5/media/study_overview.png)

[Latest study](studies/a6000-stage2-v2.5/README.md) ·
[Results and limits](studies/a6000-stage2-v2.5/docs/results.md) ·
[Measurement method](studies/a6000-stage2-v2.5/docs/methodology.md) ·
[Persian educational guide](docs/teaching/NCCL_Energy_Frontier_FA.pdf) ·
[Reproduce](#reproduce)

## Latest extension: A6000 measurement characterization

The [Stage 2 study](studies/a6000-stage2-v2.5/README.md) adds 3 smoke launches,
23 calibration launches and 72 accepted held-out launches on the same NVLink
pair, GPUs 4 and 5. The primary estimator is available for 70 of the 72
held-out launches. These are new, separately reported campaigns; they do not
replace or pool with the older results below. [Stage 2 (2026/10),
[execution](studies/a6000-stage2-v2.5/prep/heldout_final/execution.tsv),
[unit estimates](studies/a6000-stage2-v2.5/prep/heldout_final/units.tsv)]

For AllGather 2 MiB, LL128 versus automatic at 40 s used **+1.6454%**
NVML-reported board energy (descriptive 95% CI **[+1.0047%, +2.2901%]**).
The same-execution B/A control at 10 s was compatible with its prespecified
approximately ±0.5% mean-ratio equivalence margin. The precision target was
missed, every aggregate adequacy scope remains unmet/unestablished, and
global latency-versus-energy optima remain unresolved. The three sessions
ran on one host during the same evening, not independent days or machines.
[Stage 2 (2026/10), [results](studies/a6000-stage2-v2.5/docs/results.md),
[contrasts](studies/a6000-stage2-v2.5/prep/heldout_final/contrasts.tsv)]

![Held-out energy comparisons: session means and descriptive 95% intervals; B/A at 20 s fails thermal comparability, B/A at 40 s lacks precision, and C/A misses the precision target](studies/a6000-stage2-v2.5/media/heldout_comparisons.gif)

*New-study animation. A is automatic, B is Ring/Simple, C is Ring/LL128;
A and B both executed Ring/Simple with four channels. The animation highlights
retained comparisons; it does not add observations or update an inference.
[Static alternative](studies/a6000-stage2-v2.5/media/heldout_comparisons.png) ·
[Values and source hashes](studies/a6000-stage2-v2.5/media/media_values.json).*

### Why measurement duration matters

Sampled-power and counter estimators show different short-region effects.
That is observed estimator disagreement, not proof of its physical cause.
The sampled result remains a sensitivity analysis, not a replacement primary.
[Stage 2 (2026/10), [duration results](studies/a6000-stage2-v2.5/docs/results.md#outcomes-that-remain-unresolved)]

![Duration sensitivity: all three arms and four estimators, with three-session descriptive 95% intervals; sampled power differs most at 2.5 s, mechanism unresolved](studies/a6000-stage2-v2.5/media/estimator_duration.gif)

*New-study sensitivity animation. Only emphasis changes; all estimates use
three session means, and all plotted durations are retained observations.
The estimators share NVML infrastructure, not independent physical references.
[Static alternative](studies/a6000-stage2-v2.5/media/estimator_duration.png) ·
[Media methods and limitations](studies/a6000-stage2-v2.5/docs/media.md).*

Read the [results and limitations](studies/a6000-stage2-v2.5/docs/results.md),
[method and evidence](studies/a6000-stage2-v2.5/docs/methodology.md), or the
[50-page Persian educational guide](docs/teaching/NCCL_Energy_Frontier_FA.pdf).
This closes the bounded A6000 pilot; no Ti/Titan replication was performed
or is required for this release.

## Project stages: separate evidence, one repository

| Stage | What it contributes | Start here |
|---|---|---|
| Original pilot | Three of five planned trials; historical latency/energy characterization | [Pilot record](docs/pilot_v1/README_pilot_v1.md) |
| First extension | Selected configurations versus automatic, synthetic-workload observations, earlier measurement-validation limits | [Historical claim table](docs/release/claims.md) |
| **Latest A6000 extension** | Smoke, calibration and three held-out sessions; measurement sensitivity and a selected protocol comparison | [Current study](studies/a6000-stage2-v2.5/README.md) |

The sections below preserve the first extension's results. Their GIFs are
historical, not visualizations of the new held-out campaign. A selected
configuration comparison is not a search for the global latency or energy
optimum. [Release (2026/10), [claim boundaries](docs/release/claims.md),
[Stage 2 boundaries](studies/a6000-stage2-v2.5/docs/claims.md)]

<details>
<summary>Earlier release status and corrections</summary>

The earlier v2.2 pilot release used the private branch name `ext/v2.2-audit`;
the public first-extension branch is `extension-pilot-reviewed`. Its
[final review](docs/release/final_review_report.md),
[scientific resolutions](docs/release/scientific_resolution.md) and
[2026-10-11 correction ledger](docs/release/corrections_2026-10-11.md) remain
historical records. The later A6000 study is a separate evidence category,
not a retroactive change to that campaign or its frozen decisions.

</details>

<details>
<summary><b>Historical first-extension results, GIFs and claim scoreboard</b></summary>

## First extension: historical results at a glance

| | |
|---|---|
| 🔬 **What was measured** | NVML board energy and latency of the participating GPUs for AllReduce, AllGather and ReduceScatter, 1 KiB–1 GiB, the tested automatic/Ring/Tree × automatic/Simple/LL/LL128 matrix where supported |
| 🖥️ **Where** | One host: 8× RTX A6000, NCCL 2.27.5, 2–4 GPUs, single node, FP32; sum where applicable |
| ✅ **Strongest result** | A pre-registered pick used less board energy than automatic in 36 of 39 cells |
| 📉 **Largest effect** | 4-GPU AllReduce 512 KiB, Tree/Simple vs automatic's Ring/LL: energy ratio 0.312 [0.308, 0.317] |
| ⚖️ **Open question** | Do latency and energy optima differ? **Unresolved**: the measurement-validation precision gate was not met |
| 🚫 **Not claimed** | Global optimum, real-model training benefit, whole-system energy, absolute sensor accuracy |

## First extension: findings and limitations

The category tags refer to the [evidence categories](#evidence-categories-never-pooled) below.

### 1 · Tested settings beat NCCL's automatic choice on this host `[v2 confirmatory]`

**Tested settings can substantially outperform NCCL's automatic choice on this measured deployment.**

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

<details>
<summary>Historical first-extension animation: 39 selected-pick comparisons</summary>

![Historical first-extension animation, not the new held-out campaign: energy and latency ratios across 39 pre-registered cells](docs/media/rq1_cells.gif)

*Unchanged historical GIF: pick/automatic energy ratios with 95% intervals
over 12 within-session blocks; diamonds show latency ratios. Source:
[`data/v2/processed/confirmatory/rq1.csv`](data/v2/processed/confirmatory/rq1.csv).
Generated by `scripts/make_readme_media.py`, not the new-study renderer.*

</details>

<details>
<summary>📊 <b>Show the per-block view and where automatic leaves energy on the table</b></summary>

<br>

![RQ1 per block: pick vs auto for each confirmatory cell, within-session blocks in grey](results/release-corrections_2026-10-11/figures/v2_confirmatory/rq1_per_block.png)

*Every grey dot is one within-session block; blue is the geometric mean and 95 % CI.*

![Exploratory auto-energy regret heatmap across collectives, placements and message sizes](results/release-corrections_2026-10-11/figures/v2_exploratory/default_regret.png)

*Exploratory (two within-session blocks, before pre-registration): where the Energy-SLO pick differs from
automatic. `=` means the pick executes the same algorithm/protocol/channels as automatic.*

</details>

### 2 · Transfer to a synthetic training workload `[v2 Phase 10]`

**Transfer to a synthetic training workload shows promising improvements, with stated limitations.**

| Synthetic FP32 MLP, 4 GPUs, 5 within-session blocks | energy per step (policy ÷ auto) | note |
|---|---|---|
| DDP, batch 128 | 0.791 [0.784, 0.799] | step time 0.706 |
| DDP, batch 2048 | 0.885 [0.880, 0.889] | power-capped |
| FSDP (negative control) | 1.001 [0.973, 1.030] | within the original ±3 % energy-equivalence criterion |

> [!CAUTION]
> * Numerical equivalence was not recorded in these runs.
> * Every rank used the same synthetic batch. Under that setting a skipped
>   gradient AllReduce would not have been detected.
> * The meter of that time recorded no QC flags. A post-hoc plausibility
>   check of all 200 GPU readings found none implausible.
> * This says nothing about real models, BF16, time to quality or
>   whole-system energy.

<details>
<summary>📊 <b>Show every block of the workload runs</b></summary>

<br>

![Historical synthetic Phase 10 workload: energy per step and step time for auto, lat_opt and energy_slo, one line per block](results/release-corrections_2026-10-11/figures/historical_phase10/workload_blocks.png)

*One line per within-session block. Descriptive historical observation; training equivalence unverified.*

</details>

### 3 · Do the latency optimum and the energy optimum differ? Unresolved `[v2 + v2.2]`

**The small latency-versus-energy optimum difference was not resolved.**

* On the NVLink pair, AllGather 2 MiB with LL128 has 2.8 % lower latency
  than auto (equivalently, auto is 2.9 % slower than LL128).
* Its energy penalty is 1.012 [1.003, 1.022] with the energy counter. It
  is not robust across estimators.
* The v2.2 measurement-validation campaign: 70 of 72 processes accepted.
  Its pre-specified decisions were:

  | decision | pre-specified question | outcome |
  |---|---|---|
  | D1 | Is energy per op independent of region length (2.5 s vs 40 s, ±0.3 %)? | INCONCLUSIVE |
  | D2 | Do sampled-power estimators agree with the counter regardless of protocol (LL128 term, ±0.3 %)? | INCONCLUSIVE |
  | D3 | Do two identically executing configurations read the same (B ÷ A, ±0.5 %)? | SYSTEMATIC_OFFSET |
  | D4 | Is the 40 s interval precise enough (half-width ≤ 0.4 %)? | NOT_SUFFICIENT |

  The 40 s LL128 ÷ auto interval half-width was 3.98 %, against the ≤ 0.4 %
  needed.
* The claim (C7) stays **preliminary and unresolved**.

<details>
<summary>📊 <b>Show the estimator sensitivity and the identical-execution control</b></summary>

<br>

![RQ2 estimator sensitivity: four energy estimators give different intervals for the same comparison](results/release-corrections_2026-10-11/figures/rq2_estimator_sensitivity.png)

*Four estimators over the same shared board sensors disagree on whether the AllGather 2 MiB interval excludes 1.*

![Gate 3b D3 identical-execution control: per-block B ÷ A energy at 2.5, 10, 20 and 40 s regions against a ±0.5 % band](results/release-corrections_2026-10-11/figures/gate3b_D3_identical_execution_control.png)

*Two configurations that execute identically should read the same. Their intervals are not contained in the
±0.5 % band (D3 = SYSTEMATIC_OFFSET, a frozen decision label, not proof of a persistent physical bias), so
measurement agreement at ±0.5 % is not established.*

</details>

### 4 · Historical pilot `[pilot v1, superseded]`

The first pilot (driver 550, uncorrected boundary counter reads, 3 of 5
trials) found the latency- and energy-optimal executed configuration
identical in 123 of 126 cells.

<p align="center">
  <img src="docs/media/pilot_frontier_sweep.gif" width="760"
       alt="Animation over message sizes 1 KiB to 1 GiB: each NCCL setting plotted as latency and board energy relative to automatic, for 4-GPU AllReduce and 2-GPU NVLink AllGather">
  <br>
  <sub><b>Historical pilot, swept from 1 KiB to 1 GiB (superseded, descriptive).</b> Each marker is one NCCL
  setting placed by latency ÷ automatic and board energy ÷ automatic; ○ marks the lowest-latency and □ the
  lowest-energy setting at that size. Generated from <code>data/processed/full/summary.csv</code>.</sub>
</p>

### Post-hoc exploration `[category 4]`

Post-hoc exploratory analyses of the validation campaign (category 4) suggest
two things to address in future designs:
* GPU thermal state is strongly associated with between-process energy
  variation;
* the counter and sampled board power differ by an unexplained, roughly
  constant amount per measured region.

Neither changes any decision above.

<details>
<summary>📊 <b>Show the post-hoc temperature association and the counter/sample gap</b></summary>

<br>

![Gate 3b post-hoc: energy per op against GPU temperature in partial-regression coordinates](results/release-corrections_2026-10-11/figures/gate3b_POSTHOC_temperature_association.png)

![Gate 3b post-hoc: counter vs sampled power over matched windows](results/release-corrections_2026-10-11/figures/gate3b_POSTHOC_matched_window_gap.png)

*Chosen after seeing the data; these generate hypotheses and support no claim.*

</details>

## Historical first-extension claim scoreboard

Full wording, numbers and sources: [`docs/release/claims.md`](docs/release/claims.md).

| status | claims |
|---|---|
| 🟢 **supported** | C1 pick < auto in 36/39 cells · C2 ≥ 5 % saving (21 cells) · C3 largest effect · C6 LL128 latency · C8 AllReduce 1 GiB Tree energy-equivalent · C9 PCIe 256 MiB reversal · C13 tuner reproduces executions · M1 pipeline smoke test · M2 no counter read inside a region · M3 Gate 3b ran as specified · M5 identical execution |
| 🔵 **supported, descriptive / association** | C4 picks are also faster · C11 energy–temperature association (not randomized) · C12 counter vs sampled power differ · C14–C16 synthetic workload ratios |
| 🟡 **preliminary / unresolved** | C2 (4 of the 25 cells) · **C7 latency vs energy optimum differ** · C10 order effects |
| ⚪ **not established / not assessed** | M6 measurement agreement ±0.5 % · M7 absolute accuracy · **M8 resolving a ≈ 1 % difference** · C18 real models, BF16, time to quality |
| 🔴 **unsupported** | C17 `no_sync` isolates communication energy · C19 tuner safe to deploy generally |
| 🧪 **exploratory (post hoc)** | H1 temperature association · H2 unexplained counter/sample gap · H3 temperature-adjusted ratio |
| ⚫ **superseded** | P1, P2 historical pilot |

</details>

## Evidence categories (never pooled)

```mermaid
flowchart LR
    P1["1 · Historical pilot<br/>v1 · driver 550"]:::old
    E["2 · v2 exploratory<br/>2 blocks"] --> C["2 · v2 confirmatory<br/>12 pre-registered blocks"]
    C --> W["2 · Phase 10<br/>synthetic workload"]
    G["3 · v2.2 Gate 3b<br/>measurement validation"]
    H["4 · Post-hoc<br/>exploratory"]:::explore
    S["5 · A6000 Stage 2<br/>72 held-out launches<br/>70 primary estimates"]
    C --> R1(["pick vs auto<br/>supported"]):::ok
    W --> R2(["workload ratios<br/>descriptive only"]):::desc
    C --> R3(["latency vs energy optimum<br/>unresolved"]):::open
    G -- "D4 not sufficient" --> R3
    S -- "aggregate adequacy not met" --> R3
    H -. "changes no claim" .-> R3
    P1 -. "superseded" .-> X(["historical record"]):::old
    classDef ok fill:#d8f3e3,stroke:#2ea44f
    classDef desc fill:#dbe9fb,stroke:#2a6fdb
    classDef open fill:#fff3c4,stroke:#c99a00
    classDef old fill:#eeeeee,stroke:#888888,color:#555555
    classDef explore fill:#f3e8fd,stroke:#8250df
```

| # | category | method ID | data | documents |
|---|---|---|---|---|
| 1 | **Historical pilot** (2026-10-06): driver 550, uncorrected boundary counter reads, 3 trials | `v1` | `data/raw/`, `data/processed/`, `figures/full/` | [`docs/pilot_v1/README_pilot_v1.md`](docs/pilot_v1/README_pilot_v1.md), [`docs/methodology.md`](docs/methodology.md), [`docs/limitations.md`](docs/limitations.md) |
| 2 | **v2 exploratory and confirmatory** (2026-10-08/09): driver 580, commit-aligned counter region, 2 exploratory and 12 pre-registered confirmatory blocks, Phase 10 workload | `v2.1` | `data/v2/raw/`, `data/v2/processed/`, `figures/v2/` | [`docs/v2/`](docs/v2/), with Stage 2 corrections |
| 3 | **v2.2 measurement validation** (gate 3b, 2026-10-09): 6 blocks on GPUs 4,5, pre-specified D1–D4 | `v2.2-fail-closed`, `v2.2-aligned-meter-logged` | `data/v2.2/` | [`gate3b_measval_report.md`](docs/v2/v22_audit/gate3b_measval_report.md) §1–4 |
| 4 | **Post-hoc exploratory analysis**: reanalysis of the v2 data (Stage 2) and of the gate 3b data | — | `data/v2/reanalysis/`, `data/v2.2/processed/*/posthoc_temperature.json` | [`stage2_report.md`](docs/v2/v22_audit/stage2_report.md), gate 3b report §5 |
| 5 | **A6000 Stage 2**: 3 smoke, 23 calibration, 72 held-out launches; primary available for 70; three same-evening sessions | `v2.5-stage2-region` | [`studies/a6000-stage2-v2.5/code/data/stage2/`](studies/a6000-stage2-v2.5/code/data/stage2/) | [Current results](studies/a6000-stage2-v2.5/docs/results.md), [protocol](studies/a6000-stage2-v2.5/protocol/STAGE2_HELDOUT1_PROTOCOL.md) |
| 6 | **Future experiments**: not run, not needed for this release | — | — | [`docs/release/limitations.md`](docs/release/limitations.md) §6 |

> [!WARNING]
> **What "energy" means here**:
> * NVML board energy of the **participating GPUs** while the collective
>   runs.
> * It is **not** communication-only energy: it includes the boards' static
>   and idle power.
> * It is **not** whole-system energy: CPU, DRAM, PCIe switches, NVLink
>   bridges outside the board, and PSU losses are not measured.
> * No external power reference was used. Absolute accuracy is not
>   characterized.

<details>
<summary>🖼️ <b>Figure gallery: historical pilot (v1, superseded)</b></summary>

<br>

| | |
|---|---|
| ![Pilot latency vs message size](figures/full/fig1_latency_vs_size.png) | ![Pilot energy per op vs message size](figures/full/fig2a_energy_per_op_vs_size.png) |
| ![Pilot Pareto fronts, 4 GPUs](figures/full/fig5a_pareto_4gpu_raw.png) | ![Pilot disagreement heatmap](figures/full/fig8_heatmap_disagreement.png) |

Frozen historical figures; every pilot figure is in [`figures/full/`](figures/full/).
Bootstrap intervals from three trials are descriptive at best.

</details>

The new [Stage 2 category](studies/a6000-stage2-v2.5/README.md) has its own
source, raw records, protocols and public-derivative replay. It is never
pooled with categories 1–4 above.

Historical v1 figure/table labels such as “ran” describe the initial-probe
inference, not TUNING recorded in every timed process. Those historical
artifacts remain unchanged; the newer profiler-verified execution evidence
is a different measurement category.

## Reproduce

For the completed Stage 2 study, after installing the analysis dependencies:

```bash
python studies/a6000-stage2-v2.5/scripts/reproduce_public.py
python -m unittest discover -s studies/a6000-stage2-v2.5/tests -p 'test_public_replay.py' -v
```

The replay verifies the public derivative manifest and frozen numerical
kernels before recomputing the raw-to-result analysis. It is not a claim
that privacy-transformed files match the original private freeze bytes.
See [publication status](studies/a6000-stage2-v2.5/docs/publication_status.md).
The historical commands below still apply only to their original categories.

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

* `release_repro` re-runs the historical categories 1–4 from the retained
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

**New-study README media** (CPU only, no new statistics; writes a fresh directory):

```bash
python studies/a6000-stage2-v2.5/scripts/make_readme_media.py
python -m unittest discover -s studies/a6000-stage2-v2.5/tests -p 'test_readme_media.py' -v
```

GIF rendering requires Pillow, already a Matplotlib dependency. The generator
refuses to overwrite an existing output directory. The animation changes only
visual emphasis; numerical values, intervals and verdicts come from the
retained analysis. See the [media source notes](studies/a6000-stage2-v2.5/docs/media.md).

**Historical README animations** (presentation only; writes `docs/media/`):

```bash
python scripts/make_readme_media.py
```

<details>
<summary>⚠️ <b>Future GPU experiments (not part of this release)</b></summary>

<br>

* Experimental launch entry points that have not passed the current safety
  and numerical-validation gates are disabled in this release candidate.
* They need an explicitly allocated, idle-checked set of GPUs, at most four
  by default, one benchmark job at a time, with the watchdog and safety
  limits.
* See [`docs/release/reproduction.md`](docs/release/reproduction.md) §3.
  Enabled v2 runner commands require explicit allocation. Historical v1,
  Phase4 validation and CUDA workload launchers are disabled; guarded
  workloads and native sampled fallback still need real-hardware validation.

</details>

## Documents

| document | content |
|---|---|
| [A6000 Stage 2 results](studies/a6000-stage2-v2.5/docs/results.md) | new-study outcomes, missing estimates, thermal and precision limitations |
| [A6000 media provenance](studies/a6000-stage2-v2.5/docs/media.md) | new GIFs, static alternatives, sources and reproduction |
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
docs/media/              README animations (scripts/make_readme_media.py)
studies/a6000-stage2-v2.5/ new A6000 study: data, code, protocols, analysis, figures and media
docs/teaching/           Persian educational guide (PDF and editable source)
tests/                   CPU test suite (python tests/run_tests.py)
docs/                    reports (pilot_v1, v2, v2.2 audit, release)
```

## License

Project code is MIT; see [LICENSE](LICENSE). Bundled NVIDIA headers retain
their own copyright and license notices.
