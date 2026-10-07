# Results summary (generated)

Source: `data/processed/smoke`

Bootstrap intervals and optimum labels are descriptive, not confirmatory.

## Coverage

- final units: 216 — status counts: {'ok': 168, 'not_run_probe_unsupported': 48}
- launch attempts: 228 (12 units needed a retry)
- regions used in statistics: 336 (excluded: 0, reasons {})
- trials: [1, 2]; GPU sets: ['2gpu_nvlink', '4gpu']; wall clock of timed launches: 0.39 h

## NCCL's own selection (auto/auto), from the probe

- AllReduce · 2gpu_nvlink: 4K–4K: RING/LL[0-0]; 1M–64M: RING/SIMPLE[0-3]
- AllGather · 2gpu_nvlink: 4K–4K: RING/LL[0-0]; 1M–64M: RING/SIMPLE[0-3]
- ReduceScatter · 2gpu_nvlink: 4K–4K: RING/LL[0-0]; 1M–64M: RING/SIMPLE[0-3]
- AllReduce · 4gpu: 4K–4K: RING/LL[0-0]; 1M–64M: RING/SIMPLE[0-7]
- AllGather · 4gpu: 4K–4K: RING/LL[0-0]; 1M–64M: RING/SIMPLE[0-7]
- ReduceScatter · 4gpu: 4K–4K: RING/LL[0-0]; 1M–64M: RING/SIMPLE[0-7]

## Latency-optimal vs energy-optimal (out-of-place)

| collective · GPUs | cells | disagree (raw E) | interval excludes 0 (raw E) | max penalty (raw E) | disagree (idle-sub. E) | interval excludes 0 (idle-sub. E) | max penalty (idle-sub. E) | disagree (EDP) |
|---|---|---|---|---|---|---|---|---|
| AllReduce · 2gpu_nvlink | 3 | 1 | 1 | +4.7 % | 1 | 1 | +6.5 % | 1 |
| AllGather · 2gpu_nvlink | 3 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| ReduceScatter · 2gpu_nvlink | 3 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| AllReduce · 4gpu | 3 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| AllGather · 4gpu | 3 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| ReduceScatter · 4gpu | 3 | 0 | 0 | +0.0 % | 0 | 0 | +0.0 % | 0 |
| **total** | 18 | 1 | 1 | | 1 | 1 | | 1 |

## Candidate disagreements whose descriptive interval excludes 0

| objective | cell | latency-optimal (ran) | P(best) | energy-optimal (ran) | P(best) | energy penalty of latency-optimal [95% CI] | latency penalty of energy-optimal [95% CI] |
|---|---|---|---|---|---|---|---|
| raw | AllReduce · 2gpu_nvlink · 1MB | Ring/LL128 (RING/LL128[0-3]) | 1.00 | auto/auto (RING/SIMPLE[0-3]) | 0.69 | +4.7 % [+4.0 %, +5.5 %] | +1.2 % [+1.0 %, +1.4 %] |
| idle-sub. | AllReduce · 2gpu_nvlink · 1MB | Ring/LL128 (RING/LL128[0-3]) | 1.00 | auto/auto (RING/SIMPLE[0-3]) | 0.68 | +6.5 % [+5.5 %, +7.4 %] | +1.2 % [+1.0 %, +1.4 %] |

## Latency vs energy across executed configurations within a cell

- cells with ≥2 distinct executed configurations: 18 (≥3: 6)
- Spearman(latency, raw energy): median 0.900, min 0.800
- Spearman(latency, idle-subtracted energy): median 0.900, min 0.800
- max/min latency across executed configurations in a cell: median 1.50×
- max/min average power across executed configurations in a cell: median 1.023×, max 1.097×

## NCCL default (auto/auto) vs the best measured configuration

- cells: 18
- energy above the energy-optimal configuration: median +0.8 %, max +49.4 % at ['all_reduce', '4gpu', '67108864']; within 1 %: 11 cells
- latency above the latency-optimal configuration: median +0.1 %, max +53.4 % at ['all_reduce', '4gpu', '67108864']; within 1 %: 13 cells

- out-of-place vs in-place: same raw-energy disagreement call in 18 of 18 cells
