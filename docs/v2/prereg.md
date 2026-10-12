# Pre-registration — confirmatory campaign (methodology v2)

Frozen 2026-10-09T01:19:19Z at git HEAD `d9c1297ff1c356198baeaaa9b8545b4b0a89cf0a` (dirty=True), **before any confirmatory block**.
Machine-readable version: [`prereg.json`](prereg.json); frozen campaign: `configs/v2/confirmatory.yaml`; per-cell power table: [`prereg_rq1_cells.csv`](prereg_rq1_cells.csv).

## Hashes

- calibration_entries_sha256: `ea235d079dcb8d461d7ad8bcdb4da2cf6e64647a6dd75e97ecfae43d91e1346c`
- patch_sha256: `293527e3ab71ece12ff42088b84f47763ba48e62f08999edbfa4ff04bad2e0b8`
- plugin_sha256: `0b96c10d120f461c876958dc98ad686aa97316f2c51d08248b19ea2703b2817b`
- confirmatory_schedule_sha256: `5e1f09a1ff2f66a1728d2b374de9d0aec6e5631cc8421eea26f1d5813dd4c312`
- confirmatory_yaml_sha256: `451b910fe2a8ef001c6bfea33939af2c060e4754cf7a660a9986efa701f52442`
- exploratory processed files: 8 files hashed in `prereg.json`

## Research questions

- **RQ1** Does the Energy-SLO policy reduce raw participating-GPU board energy per collective relative to NCCL automatic while meeting the latency SLO?
- **RQ2** Do latency-optimal and raw-energy-optimal configurations differ after block-level uncertainty?
- **RQ3** Does the best microbenchmark policy change energy per step of the synthetic PyTorch workload?
- **RQ4** Sensitivity to placement (3 fingerprints), temperature, clocks; NCCL version not tested.

## Endpoints and tests

- primary: ln(raw J per collective), block-paired difference policy - auto
- latency: ln(t_gpu) block-paired difference
- meaningful_effect: >= 5% lower raw energy
- latency_slo: policy latency <= 1.05 x auto latency (95 % upper bound)
- secondary: ['idle-adjusted J', 'EDP', 'bytes/J', 'counter vs integration']
- RQ1 test: one-sided paired t on block log-differences, Holm across RQ1 cells (alpha 0.025); confirmed if Holm p < 0.025 and latency-ratio 95 % upper bound <= 1.05; meaningful if also ratio <= 0.95
- RQ2 test: two one-sided paired t (energy of lat-opt > en-opt; latency of en-opt > lat-opt), Holm per family; DIFFER if both < 0.05; else TOST +-3 % on energy; else INCONCLUSIVE
- robustness: Wilcoxon signed-rank, Hodges-Lehmann, block bootstrap, idle-adjusted, order/temperature regression
- QC: docs/v2/protocol.md sections 9 and 12; QC_SHORT / MEASUREMENT_INVALID / UNVERIFIED / EXCLUDED_DEVIATION are excluded and counted; no outcome-dependent re-runs
- missing data: complete pairs per cell; missing counts reported per cell; outliers: none removed
- randomization: per block: random.Random('confirmatory-<seed>-<block>') shuffle of all units

## Matrix

- 39 RQ1 cells (exploratory Energy-SLO pick executes a configuration different from NCCL automatic) and 2 RQ2 cells (exploratory latency- and energy-optimal executed configurations differ); 41 cells, 82 units per block (auto + policy, or lat-opt + en-opt).
- Iteration schedule: R = max(10, ceil(3.0 s / min exploratory t_gpu)), w = max(5, ceil(0.2 s / t)) (`data/v2/raw/setup/calibration_frozen_confirmatory.json`), identical in every block.
- Cells where every exploratory policy coincided with NCCL automatic are not in the confirmatory matrix; for them the recommender falls back to automatic by construction.

## Block count

- Rule: smallest N <= 12 with predicted 95 % CI half-width <= ln(1.025) in >= 80 % of RQ1 cells, using max(exploratory-v2 SD, pilot SD); minimum 5.
- Result: the cap binds — **N = 12**, with 72% of RQ1 cells meeting the precision target. Listed before the campaign, as pre-registered: the cells below are under-powered for a ±2.5 % interval (predicted 95 % half-width at N = 12 from max(v2 SD, pilot SD)); large effects can still be resolved.

| cell | policy | exploratory E/auto | SD v2 | SD pilot | predicted half-width |
|---|---|---|---|---|---|
| P2_PCI_SHM all_reduce 268435456 | Tree/Simple | 0.970 | 0.0651 |  | ±4.2 % |
| P4_MIX all_gather 262144 | Ring/Simple | 0.767 | 0.0673 | 0.0443 | ±4.4 % |
| P4_MIX all_gather 524288 | Ring/Simple | 0.618 | 0.2797 | 0.0259 | ±19.4 % |
| P4_MIX all_reduce 65536 | Tree/LL | 0.923 | 0.0172 | 0.0663 | ±4.3 % |
| P4_MIX all_reduce 131072 | Tree/LL | 0.626 | 0.2860 | 0.0133 | ±19.9 % |
| P4_MIX all_reduce 262144 | Tree/Simple | 0.415 | 0.0107 | 0.0751 | ±4.9 % |
| P4_MIX all_reduce 1048576 | Tree/Simple | 0.769 | 0.0692 | 0.0273 | ±4.5 % |
| P4_MIX all_reduce 2097152 | Tree/Simple | 0.863 | 0.0062 | 0.1337 | ±8.9 % |
| P4_MIX all_reduce 8388608 | Tree/Simple | 0.715 | 0.0103 | 0.1317 | ±8.7 % |
| P4_MIX all_reduce 67108864 | Tree/Simple | 0.596 | 0.1862 | 0.1356 | ±12.6 % |
| P4_MIX reduce_scatter 524288 | Ring/Simple | 0.599 | 0.4451 | 0.0273 | ±32.7 % |

## RQ1 cells

| placement · collective · size | policy (executes) | auto executes | exploratory E/auto | exploratory T/auto |
|---|---|---|---|---|
| P2_NVL · all_gather · 524288 | Ring/LL (RING/LL/4) | RING/SIMPLE/4 | 0.804 | 0.771 |
| P2_NVL · all_gather · 1048576 | Ring/LL128 (RING/LL128/4) | RING/SIMPLE/4 | 0.919 | 0.881 |
| P2_NVL · all_reduce · 524288 | Ring/LL (RING/LL/4) | RING/SIMPLE/4 | 0.853 | 0.815 |
| P2_NVL · all_reduce · 8388608 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.958 | 0.964 |
| P2_NVL · all_reduce · 16777216 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.950 | 0.945 |
| P2_NVL · all_reduce · 33554432 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.950 | 0.925 |
| P2_NVL · all_reduce · 67108864 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.952 | 0.924 |
| P2_NVL · all_reduce · 134217728 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.975 | 0.950 |
| P2_NVL · all_reduce · 268435456 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.986 | 0.960 |
| P2_NVL · all_reduce · 536870912 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.993 | 0.971 |
| P2_NVL · reduce_scatter · 524288 | Ring/LL (RING/LL/4) | RING/SIMPLE/4 | 0.778 | 0.744 |
| P2_NVL · reduce_scatter · 1048576 | Ring/LL (RING/LL/4) | RING/SIMPLE/4 | 0.974 | 0.936 |
| P2_PCI_SHM · all_gather · 262144 | Ring/Simple (RING/SIMPLE/4) | RING/LL/4 | 0.929 | 0.932 |
| P2_PCI_SHM · all_reduce · 131072 | Tree/Simple (TREE/SIMPLE/4) | RING/LL/4 | 0.846 | 0.843 |
| P2_PCI_SHM · all_reduce · 262144 | Tree/Simple (TREE/SIMPLE/4) | RING/LL/4 | 0.681 | 0.680 |
| P2_PCI_SHM · all_reduce · 524288 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.948 | 0.948 |
| P2_PCI_SHM · all_reduce · 1048576 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.941 | 0.943 |
| P2_PCI_SHM · all_reduce · 2097152 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.985 | 0.990 |
| P2_PCI_SHM · all_reduce · 4194304 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.985 | 0.981 |
| P2_PCI_SHM · all_reduce · 16777216 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.990 | 0.998 |
| P2_PCI_SHM · all_reduce · 268435456 | Tree/Simple (TREE/SIMPLE/4) | RING/SIMPLE/4 | 0.970 | 0.970 |
| P2_PCI_SHM · reduce_scatter · 262144 | Ring/Simple (RING/SIMPLE/4) | RING/LL/4 | 0.879 | 0.881 |
| P4_MIX · all_gather · 262144 | Ring/Simple (RING/SIMPLE/8) | RING/LL/8 | 0.767 | 0.747 |
| P4_MIX · all_gather · 524288 | Ring/Simple (RING/SIMPLE/8) | RING/LL/8 | 0.618 | 0.618 |
| P4_MIX · all_reduce · 1024 | Tree/LL (TREE/LL/1) | RING/LL/1 | 0.984 | 0.982 |
| P4_MIX · all_reduce · 65536 | Tree/LL (TREE/LL/8) | RING/LL/4 | 0.923 | 0.877 |
| P4_MIX · all_reduce · 131072 | Tree/LL (TREE/LL/8) | RING/LL/8 | 0.626 | 0.609 |
| P4_MIX · all_reduce · 262144 | Tree/Simple (TREE/SIMPLE/8) | RING/LL/8 | 0.415 | 0.419 |
| P4_MIX · all_reduce · 524288 | Tree/Simple (TREE/SIMPLE/8) | RING/LL/8 | 0.312 | 0.316 |
| P4_MIX · all_reduce · 1048576 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.769 | 0.746 |
| P4_MIX · all_reduce · 2097152 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.863 | 0.835 |
| P4_MIX · all_reduce · 4194304 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.730 | 0.710 |
| P4_MIX · all_reduce · 8388608 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.715 | 0.694 |
| P4_MIX · all_reduce · 16777216 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.657 | 0.644 |
| P4_MIX · all_reduce · 33554432 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.640 | 0.633 |
| P4_MIX · all_reduce · 67108864 | Tree/Simple (TREE/SIMPLE/8) | RING/SIMPLE/8 | 0.596 | 0.576 |
| P4_MIX · reduce_scatter · 131072 | Ring/Simple (RING/SIMPLE/4) | RING/LL/8 | 0.940 | 0.973 |
| P4_MIX · reduce_scatter · 262144 | Ring/Simple (RING/SIMPLE/8) | RING/LL/8 | 0.720 | 0.723 |
| P4_MIX · reduce_scatter · 524288 | Ring/Simple (RING/SIMPLE/8) | RING/LL/8 | 0.599 | 0.610 |

## RQ2 cells

| cell | latency-optimal | energy-optimal |
|---|---|---|
| P2_NVL · all_gather · 2097152 | Ring/LL128 (RING/LL128/4) | auto (RING/SIMPLE/4) |
| P2_NVL · all_reduce · 1073741824 | Tree/Simple (TREE/SIMPLE/4) | auto (RING/SIMPLE/4) |
