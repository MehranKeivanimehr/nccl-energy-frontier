# Smoke-test report

Run directory: `data/raw/smoke`

Overall: **PASS**

- **PASS** accounting: 216 planned units, 216 final records, 216 unique; status counts {'ok': 168, 'not_run_probe_unsupported': 48}
- **PASS** Tree for AllGather/ReduceScatter marked unsupported: 24 rows; unsupported conditions: ['all_gather/2gpu_nvlink/Tree/LL128', 'all_gather/2gpu_nvlink/Tree/auto', 'all_gather/4gpu/Tree/LL128', 'all_gather/4gpu/Tree/auto', 'reduce_scatter/2gpu_nvlink/Tree/LL128', 'reduce_scatter/2gpu_nvlink/Tree/auto', 'reduce_scatter/4gpu/Tree/LL128', 'reduce_scatter/4gpu/Tree/auto']
- **PASS** no other probe failures: probe status counts {'ok': 84, 'unsupported': 24}
- **PASS** forced algo/proto honoured by NCCL: 66/66 forced (cond,size) rows match
- **INFO** auto selection recorded: all_gather/2gpu_nvlink/4096B -> RING/LL[0-0]; all_gather/2gpu_nvlink/1048576B -> RING/SIMPLE[0-3]; all_gather/2gpu_nvlink/67108864B -> RING/SIMPLE[0-3]; all_gather/4gpu/4096B -> RING/LL[0-0]; all_gather/4gpu/1048576B -> RING/SIMPLE[0-7]; all_gather/4gpu/67108864B -> RING/SIMPLE[0-7]; all_reduce/2gpu_nvlink/4096B -> RING/LL[0-0]; all_reduce/2gpu_nvlink/1048576B -> RING/SIMPLE[0-3]; all_reduce/2gpu_nvlink/67108864B -> RING/SIMPLE[0-3]; all_reduce/4gpu/4096B -> RING/LL[0-0]; all_reduce/4gpu/1048576B -> RING/SIMPLE[0-7]; all_reduce/4gpu/67108864B -> RING/SIMPLE[0-7]; reduce_scatter/2gpu_nvlink/4096B -> RING/LL[0-0]; reduce_scatter/2gpu_nvlink/1048576B -> RING/SIMPLE[0-3]; reduce_scatter/2gpu_nvlink/67108864B -> RING/SIMPLE[0-3]; reduce_scatter/4gpu/4096B -> RING/LL[0-0]; reduce_scatter/4gpu/1048576B -> RING/SIMPLE[0-7]; reduce_scatter/4gpu/67108864B -> RING/SIMPLE[0-7]
- **PASS** data check (#wrong == 0) for every supported (cond,size): 84 rows checked
- **PASS** selection identical in start and end probe: 108/108 stable
- **PASS** stdout re-parse identical to recorded rows: 168 launches re-parsed, 0 mismatches
- **PASS** all used regions have timed portion >= 2 s: timed_s min 2.078 s, region_s median 2.518 s, max 2.745 s
- **PASS** region wall-clock per op vs nccl-tests time (oop): ratio median 1.0079, min 1.0030, max 1.0150
- **PASS** region wall-clock per op vs nccl-tests time (ip): ratio median 1.0000, min 0.9982, max 1.0047
- **PASS** energy counter present and positive in every region: 336 regions
- **PASS** idle-subtracted energy positive in every region: dynamic fraction of raw energy: median 0.769, min 0.730
- **INFO** integrated PowerUsage vs energy counter (raw window): ratio median 0.9880, IQR [0.9379, 1.0059], min 0.8570, max 1.0251
- **PASS** integrated PowerUsage (window shifted +0.5 s) vs energy counter: ratio median 0.9933, IQR [0.9867, 0.9992], min 0.9716, max 1.0166
- **PASS** energy/op increases with message size (every config): 28/28 configs monotone
- **INFO** out-of-place vs in-place energy/op: relative difference median +0.0006, IQR [-0.0139, +0.0077]
- **INFO** trial-to-trial CV: energy/op CV median 0.0057 (max 0.0351); latency CV median 0.0022 (max 0.0254)
- **PASS** idle measurements reached P8: idle W by GPU {"4": {"median": 28.9566335124706, "min": 28.780206610603493, "max": 29.361147108013473, "count": 6}, "5": {"median": 20.95255359819307, "min": 20.839740685658228, "max": 21.10220458696937, "count": 6}, "6": {"median": 24.937703446888968, "min": 24.451585556872654, "max": 25.36735693657393, "count": 6}, "7": {"median": 27.61034343982905, "min": 27.031260740709754, "max": 27.69687668264528, "count": 6}}
- **PASS** sampler cadence: median dt 0.0500 s (requested 0.05), max dt 0.1474 s, NVML query 2.20 ms/GPU/sample, overruns 99
- **PASS** no foreign GPU processes during timed launches: 0 launches with foreign PIDs
- **INFO** QC flags: {}
- **INFO** temperatures / clocks / P-states: temp mean range [32.71818181818182, 40.23529411764706], max 41.0 C; graphics clock mean range [1906.8, 1938.75] MHz; P-states ['[2]']
