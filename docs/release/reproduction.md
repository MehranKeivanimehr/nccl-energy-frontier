# Reproduction — historical pilot and A6000 Stage 2

## Completed A6000 Stage 2 addition

This subsection is separate from the historical 14-step replay below. Use
Python with `requirements-analysis.txt` on Linux/WSL. No GPU, live NVML or
administrator access is needed by the analysis adapter.

```bash
python studies/a6000-stage2-v2.5/scripts/reproduce_public.py
python -m unittest discover -s studies/a6000-stage2-v2.5/tests -p 'test_public_replay.py' -v
```

The adapter checks public hashes and original source identities, recomputes
72 unit records and the 55 estimator/contrast entries, and compares all
scientific fields and six tables against the retained analysis. It writes
to a fresh `reproduced/a6000-stage2-v2.5/` directory; choose `--output` for a
later replay. `--check` writes no result files. Status is
`PUBLIC_DERIVATIVE_REPLAY`, not original private Freeze-H verification.
The original final writer generates JSON/TSV, not the explanatory Markdown
report. See the [provenance and limitations](../../studies/a6000-stage2-v2.5/docs/publication_status.md).

The frozen Stage 2 source/tests are retained under the study's `code/`.
Their historical full suite had one stale draft-identity test; this release
does not change that frozen test after seeing the data. Native CPU tests
additionally require GCC/G++ and Make. New public-export tests are separate
from hardware validation; absent compiler tests must not be called passed.

The current-study README media have their own CPU-only renderer and tests:

```bash
python studies/a6000-stage2-v2.5/scripts/make_readme_media.py
python -m unittest discover -s studies/a6000-stage2-v2.5/tests -p 'test_readme_media.py' -v
```

Rendering requires Pillow (a Matplotlib dependency) and writes to a fresh
`reproduced/stage2-readme-media/` directory by default. No data are acquired
and no interval or decision is refitted. See [media sources and scope](../../studies/a6000-stage2-v2.5/docs/media.md).
The older `scripts/make_readme_media.py` generates only historical GIFs.

There are two separate paths:
* **§1–2, analysis only**: CPU, no GPU access, no NVML, no administrator
  rights. It reproduces every published analysis from the retained
  evidence.
* **§3, future GPU experiments**: not part of this release. They need an
  explicit GPU allocation and keep every safety requirement.

## 1. Analysis-only environment

Original analysis used Python 3.11; pins in `requirements-analysis.txt`: numpy 1.26.1,
pandas 2.3.3, scipy 1.11.3, matplotlib 3.10.7, PyYAML 6.0.1.

```bash
python -m pip install -r requirements-analysis.txt
```

The full CPU test suite additionally needs:
* PyTorch (`torch==2.9.1`; the CPU build is enough), for the gloo workload
  tests;
* `gcc` with its C++ frontend and `make`, for native helper/tuner-guard tests. These use the NCCL
  ext-tuner example headers in
  `third_party/nccl-src/ext-tuner/example/nccl/`.
* Linux/WSL and `nvidia-ml-py` bindings for legacy imports; mocked tests
  do not need a working NVML driver. Use `requirements-tests.txt`.

```bash
python -m pip install -r requirements-tests.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
```

## 2. Reproduce the published analyses

All commands run from the repository root, or from the root of the
sanitized export.

**Unit and regression tests** (CPU only; fake NVML, mocks, gloo):

```bash
python tests/run_tests.py
```

The historical expectation was an all-green suite on its complete original
environment. This public package does not claim that result on every local
installation: missing compiler/private-fixture prerequisites and a frozen
stale test are disclosed in the [release validation report](a6000_stage2_release_notes.md#local-verification).
The public replay and its separate guard tests are the supported checks of
the retained public data; neither validates a new GPU acquisition path.

**Re-run every published analysis** into a new directory and compare each
output file with the committed one:

```bash
PYTHONPATH=src python -m ncclenergy.v2.release_repro --audit --out logs/release_repro/check1
```

**Steps**:

| step | what it reproduces |
|---|---|
| `pilot_analyze`, `pilot_report`, `pilot_plot` | pilot tables, summary and figures from the pilot raw launches |
| `v2_exploratory`, `v2_confirmatory` | v2 pre-registered analysis (RQ1/RQ2) |
| `v2_phase89` | robustness, LOBO and tuner files with the v2.2 code; compared with the Stage 2 outputs of the same code, and informationally with the frozen v2.1 outputs |
| `v2_phase10` | Phase 10 workload analysis |
| `v22_reanalysis`, `v22_addendum` | Stage 2 reanalysis and the uncertainty addendum |
| `gate3a_smoke_check` | the gate 3a smoke check |
| `gate3b_measval` | the pre-specified gate 3b analysis |
| `gate3b_posthoc` | the post-hoc gate 3b analysis |
| `gate3b_amendment_check` | the frozen pre-collection analysis vs the current one: identical decisions |
| `release_tables` | current corrected numerical presentations and annotated figures in `results/release-corrections_2026-10-11/` |

The frozen numerical references must remain unchanged. Text byte identity may
differ across platforms, and metadata about byte identity is not numerical
equality. Inspect any `DIFFERENT` entry rather than treating a successful
process exit as complete reproduction. Per-file statuses include:

| status | meaning |
|---|---|
| `IDENTICAL` | byte-identical |
| `NUMERIC_EQUAL` | parsed tables equal within relative 1e-9 / absolute 1e-12 tolerance; includes gzip headers and serialized numerical arrays |
| `IDENTICAL_EXCEPT_SOURCE_PATH` | the pilot `results_summary.md` names the directory it was written from |
| `EQUAL_EXCEPT_REPO_PATH` | `placement_meta.json` records the campaign path |
| `KNOWN_UNREPRODUCED` | `figures/full/method_example_trace.png`, an illustrative figure made once for an unrecorded launch; no number depends on it |

**Input audit**:
* With `--audit`, `inputs_<step>.tsv` lists every input file each step
  read, with its SHA-256.
* Inspect `outside_reads.tsv` for undeclared data inputs. System/font-cache
  reads depend on the local environment; explain any nonempty entries.

**Release tables, figures and consistency checks**:

```bash
PYTHONPATH=src python -m ncclenergy.v2.release_tables --out logs/new_release_tables
```

* This writes `key_numbers.csv`, `gate3b_decisions.csv`,
  `gate3b_posthoc.csv`, `counts.json`, `doc_check.json` and `figures/`.
* It exits non-zero if:
  * an accepted or excluded count disagrees with the raw records;
  * the gate 3a counter-read check finds a read inside a region;
  * a document does not quote a recomputed key number.

**Gate 3b amendment check on its own**:

```bash
PYTHONPATH=src python -m ncclenergy.v2.amendment_check --campaign data/v2.2/raw/v2.2-measval-01 --out logs/amendment_check.json
```

## 3. Future GPU experiments (not part of this release)

Historical v1/Phase4 launchers and all Phase10 CUDA collection are disabled.
Do not remove these guards merely to resume a historical command. Future
workload collection requires `scientific_resolution.md` and real guarded-plugin
validation. Enabled v2 microbenchmarks still require explicit allocation,
approval and a new method/campaign identity for changed measurement code.
New counter-mode campaigns must declare
`implementation_id: release-corrections-2026-10-11`. Native sampled collection
remains disabled; the new v2.3 native patch has CPU tests but no full CUDA or
Pascal validation. Do not bypass this gate to run the replication template.

Nothing here is needed to reproduce the published analyses. A future
experiment needs its own plan and approval, and all of these requirements:

* **Allocation**:
  * Enabled v2 GPU commands take explicit allocation: `--alloc` for preflight
    and the campaign YAML's `allocation:` list for
    blocks.
  * The runner refuses more than 6 GPUs. The default is ≤ 4; use 2 for
    calibration.
  * An idle GPU is not proof that it is yours. Respect any scheduler or
    reservation.
* **Concurrency**: one benchmark job at a time; no parallel sweeps.
* **No privileged operations**: no sudo, driver changes, GPU resets, clock
  locking, power-limit or persistence-mode changes, or package
  installation.
* **Safety**:
  * The watchdog stops a unit on:
    * a foreign process;
    * temperature at or above `stop.temp_c`;
    * hardware slowdown, thermal or power-brake events, or software
      thermal throttling;
    * stale, failed or truncated telemetry;
    * an unavailable process list;
    * low disk;
    * the per-unit wall limit.
  * After a stop, the owned process tree is terminated and reaped, and the
    campaign stops. Never signal processes you did not start.
* **No silent retries**: `max_attempts: 1` for validation campaigns, and no
  significance-driven stopping.
* **Never read the cumulative energy counter on a GPU with a running
  experiment.** Read-frequency sensitivity was observed locally on driver
  580; discarding an open 100 ms window is a hypothesis, not an NVML contract
  or a physically identified mechanism.
* **Data**: a new campaign ID and a new raw directory, never pooled with
  earlier method IDs.

**Commands** (example allocation 4,5; replace with your own confirmed
allocation):

Read-only preflight (topology, idle and process check, versions):

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner preflight --alloc 4,5 --out logs/preflight_new.json
```

One block of a campaign. Allocation, units, limits and the frozen schedule
come from the YAML:

```bash
PYTHONPATH=src python -m ncclenergy.v2.runner block --campaign configs/v2.2/<new_campaign>.yaml
```

A sequential campaign with a wall budget. It stops at the first incomplete
block and forwards signals to the runner:

```bash
CAMPAIGN=configs/v2.2/<new_campaign>.yaml LOGDIR=logs/v2.2/<new_campaign> WALL_BUDGET_S=3600 BLOCK_EST_S=480 bash scripts/run_confirmatory_v2.sh
```

Building the patched nccl-tests and the plugins, and the replication notes
for other hardware: [`../v2/replication_1080ti_titan.md`](../v2/replication_1080ti_titan.md)
§3, and [`../v2/final_report.md`](../v2/final_report.md) §13.
