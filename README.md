# NCCL Energy Frontier

NCCL Energy Frontier is a pilot characterization study of collective-communication latency and participating-GPU board energy on one NVIDIA RTX A6000 server. It records requested and observed NCCL configurations, measures multi-second benchmark regions, and compares latency and energy objectives without treating the current dataset as confirmatory evidence.

## Research Question

Do NCCL configurations that minimize collective communication latency also minimize participating-GPU board energy?

The present data provide preliminary observations on one hardware, software, and topology combination. They do not establish a general answer.

## What Was Measured

The study used NCCL 2.27.5 and nccl-tests 2.21.1 on RTX A6000 GPUs. The matrix covered:

- AllReduce, AllGather, and ReduceScatter;
- message sizes from 1 KiB through 1 GiB in powers of two;
- NCCL automatic selection and requested Ring or Tree algorithms where supported;
- automatic, Simple, LL, and LL128 protocol requests;
- a 2-GPU deployment on GPUs `[4,5]`, one NVLink-connected pair; and
- a 4-GPU deployment on GPUs `[4,5,6,7]`.

The 4-GPU deployment contains two NVLink pairs, `[4,5]` and `[6,7]`. Communication between the pairs uses a slower host/PCIe path. A comparison between the 2-GPU and 4-GPU results therefore changes both GPU count and topology; it is not a topology-independent scaling experiment.

## Measurement Method

Latency is the per-operation value reported by nccl-tests. Energy primarily comes from differences in `nvmlDeviceGetTotalEnergyConsumption` across stdout-delimited benchmark regions, summed over participating GPUs. Each launch repeats the same collective long enough to create a multi-second region. Energy is divided by all operations inside the region, including warmup operations. Sampled NVML power, clocks, temperature, utilization, and performance state provide telemetry and an energy cross-check.

Each trial block starts fresh nccl-tests processes, and condition order is randomized within the block. Requested NCCL settings are compared with the algorithm, protocol, and channel range reported by an initial NCCL TUNING probe. The timed processes did not emit their own TUNING records.

See [docs/methodology.md](docs/methodology.md) for definitions and [docs/limitations.md](docs/limitations.md) before interpreting the results.

## Dataset Status

Five repeated trial blocks were planned. Three were completed before access to the experimental system was lost. The retained full-run dataset contains:

- 3 completed trial blocks;
- 3,528 successful final timed nccl-tests launches;
- 7,056 retained placement-specific measurement regions;
- 2-GPU and 4-GPU measurements;
- no 8-GPU measurements;
- no Full Trial 4 or Trial 5 measurements; and
- no closing TUNING probe.

`order_trial4.json` is a generated randomized plan, not evidence that Trial 4 ran. The current dataset is treated as pilot evidence.

## Main Observations

On this system, the tested configuration had a large effect in some cells. For 4-GPU out-of-place AllReduce at 512 KiB, the three-trial means were:

| requested configuration | configuration inferred from the initial probe | latency | participating-GPU board energy |
|---|---|---:|---:|
| `Tree/Simple` | `TREE/SIMPLE[0-7]` | 80.3 µs/op | 34.4 mJ/op |
| `auto/auto` | `RING/LL[0-7]` | 165.4 µs/op | 69.8 mJ/op |

This particular case differs by about 2.1× in latency and 2.0× in energy. It is an observation for this machine and topology, not a universal NCCL performance claim.

Across 126 analyzed collective × GPU-set × message-size cells, the mean latency-optimal and raw-energy-optimal effective configurations differed in 3 cells. The raw-energy differences were about 2.3–3.3%; idle-subtracted differences were about 3.7–4.5%. These effects are comparable with measurement and run-to-run uncertainty in the pilot data, so they identify candidate tradeoff regions for confirmation rather than proving a latency-energy tradeoff.

Generated descriptive results are in [data/processed/full/results_summary.md](data/processed/full/results_summary.md), and the figures are in [figures/full](figures/full).

## Limitations

- The effective configuration for timed launches is inferred from the initial probe; timed processes did not record TUNING output, and no closing probe was obtained.
- Energy regions include warmup operations, while reported nccl-tests latency covers timed operations.
- Counter boundaries are triggered by parsed stdout, not direct GPU execution markers.
- P8 idle subtraction is not pure communication-only dynamic energy.
- A short attempt preceded 406 accepted retries, which may have changed temperature or clocks before the accepted launch.
- The 4-GPU measurements combine a different GPU count with a different communication topology and show substantial shared-host/topology-related variability.

## Repository Layout

- `src/ncclenergy/` — orchestration, parsing, energy accounting, analysis, and plotting
- `configs/` — planned full and smoke-test configurations
- `data/processed/` — compact tables used for the checked-in summaries and figures
- `data/raw/` — retained run-level config, ordering, events, and sanitized system metadata
- `tests/` — parser, integration, and statistics checks using representative sanitized logs
- `docs/` — methodology, limitations, and smoke-test notes
- `figures/` — figures generated from the processed tables

The per-launch stdout, stderr, NCCL logs, power traces, and `launches.jsonl` are not included in the public tree because they contain private host/network identifiers and create tens of thousands of small files. Their omission is documented in [data/raw/README.md](data/raw/README.md). The processed tables and representative fixtures remain available, but a public checkout alone cannot reconstruct the processed tables from raw launch records.

## Reproducing the Analysis

Python 3.11 was used for the recorded runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run the tests and regenerate summaries and figures from the retained processed tables:

```bash
python tests/run_tests.py
python -m ncclenergy report --processed data/processed/full
python -m ncclenergy plot --processed data/processed/full --out figures/full
```

If a complete private raw run directory is available locally, process it with:

```bash
python -m ncclenergy analyze --run /path/to/raw/full --out /path/to/processed/full
```

That command requires the omitted `launches.jsonl` and per-launch directories. It does not launch GPU experiments.

New GPU experiments are separate from analysis reproduction. They require Linux, compatible NVIDIA hardware, NVML, CUDA, NCCL, and a built nccl-tests checkout; see `scripts/build_nccl_tests.sh` and `scripts/run_full.sh`. Do not run them merely to reproduce the checked-in analysis.

## Citation / Status

This is an independent research project and pilot characterization study. It is not a peer-reviewed publication and should not be cited as a completed five-trial confirmatory study.

The code is released under the [MIT License](LICENSE).
