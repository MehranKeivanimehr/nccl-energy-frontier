# Results summary (generated)

Source: `data/processed/full`

Bootstrap intervals and optimum labels are descriptive, not confirmatory.

## Coverage

- final units: 4536 — status counts: {'ok': 3528, 'not_run_probe_unsupported': 1008}
- launch attempts: 4942 (406 units needed a retry)
- regions used in statistics: 7056 (excluded: 0, reasons {})
- trials: [1, 2, 3]; GPU sets: ['2gpu_nvlink', '4gpu']; wall clock of timed launches: 7.43 h

## NCCL's own selection (auto/auto), from the probe

- AllReduce · 2gpu_nvlink: 1K–8K: RING/LL[0-0]; 16K–16K: RING/LL[0-1]; 32K–256K: RING/LL[0-3]; 512K–1G: RING/SIMPLE[0-3]
- AllGather · 2gpu_nvlink: 1K–8K: RING/LL[0-0]; 16K–16K: RING/LL[0-1]; 32K–256K: RING/LL[0-3]; 512K–1G: RING/SIMPLE[0-3]
- ReduceScatter · 2gpu_nvlink: 1K–8K: RING/LL[0-0]; 16K–16K: RING/LL[0-1]; 32K–256K: RING/LL[0-3]; 512K–1G: RING/SIMPLE[0-3]
- AllReduce · 4gpu: 1K–16K: RING/LL[0-0]; 32K–32K: RING/LL[0-1]; 64K–64K: RING/LL[0-3]; 128K–512K: RING/LL[0-7]; 1M–64M: RING/SIMPLE[0-7]; 128M–1G: TREE/SIMPLE[0-7]
- AllGather · 4gpu: 1K–16K: RING/LL[0-0]; 32K–32K: RING/LL[0-1]; 64K–64K: RING/LL[0-3]; 128K–512K: RING/LL[0-7]; 1M–1G: RING/SIMPLE[0-7]
- ReduceScatter · 4gpu: 1K–16K: RING/LL[0-0]; 32K–32K: RING/LL[0-1]; 64K–64K: RING/LL[0-3]; 128K–512K: RING/LL[0-7]; 1M–1G: RING/SIMPLE[0-7]

## Latency-optimal vs energy-optimal (out-of-place)

| collective · GPUs | cells | disagree (raw E) | interval excludes 0 (raw E) | max penalty (raw E) | disagree (idle-sub. E) | interval excludes 0 (idle-sub. E) | max penalty (idle-sub. E) | disagree (EDP) |
|---|---|---|---|---|---|---|---|---|
| AllReduce · 2gpu_nvlink | 21 | 2 | 1 | +3.3 % | 2 | 1 | +4.5 % | 2 |
| AllGather · 2gpu_nvlink | 21 | 1 | 1 | +2.3 % | 1 | 1 | +3.7 % | 0 |
| ReduceScatter · 2gpu_nvlink | 21 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| AllReduce · 4gpu | 21 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| AllGather · 4gpu | 21 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| ReduceScatter · 4gpu | 21 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| **total** | 126 | 3 | 2 | | 3 | 2 | | 2 |

## Candidate disagreements whose descriptive interval excludes 0

| objective | cell | latency-optimal (ran) | P(best) | energy-optimal (ran) | P(best) | energy penalty of latency-optimal [95% CI] | latency penalty of energy-optimal [95% CI] |
|---|---|---|---|---|---|---|---|
| raw | AllReduce · 2gpu_nvlink · 1MB | auto/LL128 (RING/LL128[0-3]) | 0.99 | auto/Simple (RING/SIMPLE[0-3]) | 0.67 | +3.3 % [+2.1 %, +4.7 %] | +0.9 % [+0.6 %, +1.1 %] |
| raw | AllGather · 2gpu_nvlink · 2MB | Ring/LL128 (RING/LL128[0-3]) | 0.78 | auto/Simple (RING/SIMPLE[0-3]) | 0.32 | +2.3 % [+0.9 %, +3.8 %] | +2.9 % [+2.5 %, +3.3 %] |
| idle-sub. | AllReduce · 2gpu_nvlink · 1MB | auto/LL128 (RING/LL128[0-3]) | 0.99 | auto/Simple (RING/SIMPLE[0-3]) | 0.63 | +4.5 % [+3.0 %, +6.0 %] | +0.9 % [+0.6 %, +1.1 %] |
| idle-sub. | AllGather · 2gpu_nvlink · 2MB | Ring/LL128 (RING/LL128[0-3]) | 0.78 | auto/Simple (RING/SIMPLE[0-3]) | 0.38 | +3.7 % [+1.7 %, +5.6 %] | +2.9 % [+2.4 %, +3.3 %] |

## Latency vs energy across executed configurations within a cell

- cells with ≥2 distinct executed configurations: 126 (≥3: 126)
- Spearman(latency, raw energy): median 1.000, min 0.500
- Spearman(latency, idle-subtracted energy): median 1.000, min 0.500
- max/min latency across executed configurations in a cell: median 2.42×
- max/min average power across executed configurations in a cell: median 1.052×, max 1.199×

## NCCL default (auto/auto) vs the best measured configuration

- cells: 126
- energy above the energy-optimal configuration: median +1.2 %, max +102.6 % at ['all_reduce', '4gpu', '524288']; within 1 %: 58 cells
- latency above the latency-optimal configuration: median +0.4 %, max +106.0 % at ['all_reduce', '4gpu', '524288']; within 1 %: 77 cells

- out-of-place vs in-place: same raw-energy disagreement call in 126 of 126 cells
