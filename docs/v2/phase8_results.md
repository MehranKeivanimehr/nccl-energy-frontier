# Phase 7–8 — confirmatory results and scientific analysis (methodology v2)

> **v2.2 audit corrections (2026-10-09).** Numbers are unchanged: the
> pre-registered analysis reproduces byte-for-byte. Reanalysis is in
> `data/v2/reanalysis/2026-10-09_v2.2-audit/`; details in
> [`v22_audit/stage2_report.md`](v22_audit/stage2_report.md). Corrections:
> 1. RQ2 AllGather 2 MiB: the energy penalty is counter-based and
>    estimator-sensitive, hence preliminary (§RQ2).
> 2. "≥ 5 % in 25 cells" is the pre-registered point-estimate rule. A
>    separate threshold test supports 21; "family-wise" was wrong.
> 3. Order: no statistically resolved association in the tested model;
>    block-, cell- and two-way-clustered intervals added.
> 4. Blocks are within-session replicates, not independent days or hosts.
> 5. The counter-reset mechanism is the interpretation supported by local
>    experiments, not an NVML guarantee.
> 6. RQ1 compares a selected pick with automatic; the global optimum is
>    not established.

## Status labels

* **CONFIRMATORY**: tests fixed in `prereg.json` (sha256 `d2053633…`)
  before block 1.
* **SECONDARY**: robustness and diagnostics on the same data, not
  pre-registered as tests.
* **EXPLORATORY**: Phase 5 data, used only to choose the confirmatory matrix.

Pilot (v1, driver 550) trials are not combined with any v2 number.

## Data

| item | value |
|---|---|
| raw data | `data/v2/raw/confirmatory/blocks/block_01 … block_12` (per-unit `unit.json`, `region.jsonl`, `telemetry.csv.gz`, `prof/`, `nccl.log`); SHA-256 manifest in `provenance/` |
| processed | `data/v2/processed/confirmatory/` (`units.csv`, `rq1.csv`, `rq2.csv`, `robustness.csv`, `order_temperature.json`, `summary.json`) |
| schedule | frozen (`calibration_frozen_confirmatory.json`, `configs/v2/confirmatory.yaml`, sha256 `451b910f…`) |
| blocks | 12 of 12 `COMPLETE`, 2026-10-09 01:20–03:28 UTC; 82 units per block in randomized order |
| GPUs | 4–7 only; ≤ 4 concurrent; one benchmark process at a time |
| units | 984 planned, 984 run, 984 `OK`, 984 `VERIFIED`; 0 retries; 0 region flags; 0 QC exclusions |
| region length | 2.79–6.12 s (median 3.01 s) |
| health | no HW-slowdown, HW-thermal or power-brake counter increase in any block (`blocks/*/summary.json`) |
| temperature / clock | max 47 °C; per-unit mean graphics clock 1906–1946 MHz |
| unit of replication | the **block** (n = 12 per cell); every statistic below is over blocks. The 12 blocks ran back to back in one ~2 h session on one host: within-session replicates, not independent days or hosts |

The 984/984 `VERIFIED` units each have a profiler record on every rank. That
record shows the requested ALGO/PROTO executed, with the exact
collective count `w + R + 1`.

## RQ1 — Energy-SLO policy vs NCCL automatic (CONFIRMATORY)

* **Test**: for each of 39 pre-registered cells, a one-sided paired t-test
  on the block log-ratio of policy energy to automatic energy. Holm
  correction across the 39 cells.
* **"Confirmed"**: Holm p < 0.025, and the latency-ratio 95 % upper bound
  is ≤ 1.05 (the SLO).
* **"Meaningful"**: confirmed, with point estimate ≤ 0.95.
* **Policy**: chosen from the exploratory data and frozen before block 1.
* **Outcome**:
  * **36 of 39 savings confirmed**;
  * **25 of 39 meaningful**;
  * 3 not confirmed, one of them a significant reversal.

Per-block ratios for every cell are in `rq1.csv`
(`per_block_E_ratio`, `per_block_T_ratio`) and
`figures/v2/confirmatory/rq1_per_block.png`.

| placement | cells | confirmed | meaningful | median E ratio | smallest E ratio |
|---|---|---|---|---|---|
| P2_NVL (GPUs 4,5, NVLink) | 12 | 12 | 5 | 0.956 | 0.780 |
| P2_PCI_SHM (GPUs 5,6, PCIe, `NCCL_P2P_DISABLE=1`) | 10 | 8 | 4 | 0.958 | 0.675 |
| P4_MIX (GPUs 4–7, 2 NVLink pairs over PCIe) | 17 | 16 | 16 | 0.674 | 0.312 |

Largest confirmed effects (energy ratio policy/auto, 95 % t-CI over 12
blocks):

| placement | collective | size | auto executed | policy executed | E ratio | T ratio |
|---|---|---|---|---|---|---|
| P4_MIX | AllReduce | 512 KiB | RING/LL/8 | TREE/SIMPLE/8 | 0.312 [0.308, 0.317] | 0.316 |
| P4_MIX | AllReduce | 256 KiB | RING/LL/8 | TREE/SIMPLE/8 | 0.408 [0.401, 0.415] | 0.414 |
| P4_MIX | ReduceScatter | 512 KiB | RING/LL/8 | RING/SIMPLE/8 | 0.529 [0.502, 0.557] | 0.538 |
| P4_MIX | AllGather | 512 KiB | RING/LL/8 | RING/SIMPLE/8 | 0.555 [0.535, 0.576] | 0.555 |
| P4_MIX | AllReduce | 8–64 MiB | RING/SIMPLE/8 | TREE/SIMPLE/8 | 0.647–0.674 | 0.637–0.663 |
| P2_PCI_SHM | AllReduce | 256 KiB | RING/LL/4 | TREE/SIMPLE/4 | 0.675 [0.665, 0.686] | 0.675 |
| P2_NVL | ReduceScatter | 512 KiB | RING/SIMPLE/4 | RING/LL/4 | 0.780 [0.771, 0.789] | 0.744 |
| P2_NVL | AllGather | 512 KiB | RING/SIMPLE/4 | RING/LL/4 | 0.810 [0.802, 0.817] | 0.772 |

Not confirmed:

| cell | E ratio [95 % CI] | note |
|---|---|---|
| P2_PCI_SHM AllReduce 4 MiB, Tree/Simple | 0.994 [0.987, 1.000] | no effect at this resolution |
| P4_MIX AllReduce 1 KiB, Tree/LL | 0.994 [0.987, 1.001] | no energy effect; 1.6 % faster |
| **P2_PCI_SHM AllReduce 256 MiB, Tree/Simple** | **1.014 [1.006, 1.021]** | **reversal**: the exploratory pick used 1.4 % more energy and was 1.7 % slower in the confirmatory blocks |

**Latency tracks energy.** In every confirmed cell the policy is also
faster (T ratio < 1). The energy saving comes from a shorter run at
nearly unchanged board power. On this hardware, at these sizes, the
"energy-optimal" choice is the latency-optimal choice in almost every
cell. That was already visible in the exploratory screen: across executed
configurations, latency differs by 2.18× (median max/min) while average
power differs by only 1.03×.

## RQ2 — does the latency-optimal configuration cost energy? (CONFIRMATORY)

These are the two cells where the exploratory latency-optimal and
energy-optimal configurations differed. Both are on P2_NVL.

| cell | latency-optimal vs energy-optimal | latency gain | energy penalty | verdict |
|---|---|---|---|---|
| AllGather 2 MiB | Ring/LL128 vs auto (Ring/Simple) | 2.87 % [2.69, 3.05] faster | **+1.23 % [0.28, 2.18]** (counter) | `DIFFER` by the pre-registered counter-based test; **estimator-sensitive** (see below) |
| AllReduce 1 GiB | Tree/Simple vs auto | 1.75 % [1.60, 1.90] faster | +0.15 % [−0.57, 0.88] | `EQUIVALENT_ENERGY_±3%` (TOST p = 1.7e-6) |

Twelve blocks each. The penalty CIs are 95 % paired t-CIs on the log ratio,
and the p-values are Holm-adjusted (`rq2.csv`).

**What this supports (corrected in the v2.2 audit).**
* **Latency (supported)**: LL128 has 2.8 % lower latency than auto at AllGather 2 MiB; auto is 2.9 % slower than LL128.
* **Energy penalty (preliminary)**: about 1 %, counter-based only.

| estimator (same 12 blocks) | LL128 / auto energy [95 % CI] | Holm p (2 cells) |
|---|---|---|
| corrected counter (pre-registered) | 1.0123 [1.0028, 1.0218] | 0.015 |
| integrated `POWER_INSTANT` | 1.0048 [0.9905, 1.0194] | 0.47 |
| lag-adjusted `PowerUsage` | 1.0092 [0.9992, 1.0192] | 0.069 |
| idle-adjusted counter | 1.0214 [1.0102, 1.0328] | 0.0015 |

The point estimates all lie above 1, but two of the four CIs include 1, so
this is **not an estimator-robust physical tradeoff**. The four estimators
share the board's sensor, so neither their agreement nor their disagreement
is an independent calibration. (`rq2_estimators.csv`,
`rq2_estimator_inputs.csv`)

AllReduce 1 GiB is energy-equivalent within ±3 % under all four estimators
(TOST p ≤ 0.024; `ar1g_tost.json`). The pilot (driver 550, 3 trials) had
pointed at the same two cells. That is consistent, but the pilot was not
used as evidence.

## Energy-measurement agreement (SECONDARY)

All 984 confirmatory regions are commit-aligned counter deltas (patch v2.1)
minus an idle tail. They are compared with the integral of `POWER_INSTANT`
sampled at 50 ms over the same region, which is non-destructive on driver
580.

| statistic (n = 984 regions) | ∫instant / counter | ∫instant − counter (J per region) |
|---|---|---|
| median | 0.980 | −23.1 |
| 5th–95th percentile | 0.941–0.991 | −56.9 to −5.6 |
| min–max | 0.923–0.996 | −77.1 to −2.9 |

* **Per placement**, the median ratio is:
  * P2_NVL 0.968;
  * P2_PCI_SHM 0.988;
  * P4_MIX 0.979.
* **The gap does not grow with region length.** Medians are −21 J for
  2.95–3.05 s regions and −28 J for > 3.2 s.
* **The gap does grow with power.** Medians are −7 J in the lowest power
  quartile (195–222 W summed over GPUs) and −32 J in the highest
  (439–460 W).

Phase 4 also observed a per-region difference of about 10 J per GPU.
Sensor/sampling lag was an initial explanation, not a demonstrated cause.
The later gate 3b matched-window analysis leaves the physical gap
unexplained (`data/v2.2/processed/v2.2-measval-01/posthoc_temperature.json`,
`window_gap`). Lack of an observed trend does not identify the mechanism.

The `PowerUsage` integral (1 s average, lag-corrected) gives a ratio of
0.978 (median; 5th–95th percentile 0.955–0.987).

**The checked RQ1 point ratios change little under this estimator.**
This is not a blanket robustness result for RQ2 or sensor accuracy.
Recomputing every RQ1 ratio from integrated
`POWER_INSTANT` energy changes it by at most 0.0075 (largest change:
P2_NVL AllReduce 128 MiB, 0.973 → 0.980). Computed from `units.csv`.

* **Idle tail subtracted by patch v2.1**: median 13.1 J per region, 1.3 %
  of region energy (range 0.3–2.4 %).
* **Idle-adjusted energy (SECONDARY)**:
  * energy above the measured P8 idle baseline is 75–85 % of raw energy;
    this is distinct from post-loop context power used for tail correction;
  * no idle-adjusted value is negative;
  * every RQ1 idle-adjusted ratio is within 0.014 of the raw ratio
    (largest: P4_MIX AllReduce 64 KiB, 0.921 raw vs 0.934 idle-adjusted);
  * the unadjusted 95 % CI of the idle-adjusted ratio lies below 1 in all
    36 confirmed cells and in one unconfirmed cell (P2_PCI_SHM AllReduce
    4 MiB, Holm p = 0.078 on raw energy).

## Robustness of RQ1 (SECONDARY; `robustness.csv`)

* **Nonparametric**: Hodges–Lehmann estimates are within 0.006 of the
  t-based ratios.
  * Exact Wilcoxon p = 0.00049 (the minimum for n = 12) in 32 of 39
    cells.
  * The three unconfirmed cells have p = 0.064 (PCI 4 MiB), 0.129 (P4
    1 KiB) and 0.0049 (PCI 256 MiB, two-sided, against the policy).
* **Block bootstrap**: 10,000 resamples of blocks. Percentile CI bounds
  are within 0.006 of the t-CI bounds in every cell.
* **Order and drift**:
  * Model: pooled regression of within-cell-centred paired log-energy
    differences (468 pairs) on the paired difference in execution
    position and in mean GPU temperature.
  * Position slope: 0.0037 [−0.0072, 0.0145] per full-block difference
    (iid, as published). No statistically resolved order association in
    the tested model.
  * v2.2 reanalysis (`order_temperature.json`):
    * block-clustered [−0.0042, 0.0116];
    * cell-clustered [−0.0069, 0.0143];
    * two-way [−0.0061, 0.0134];
    * exact wild cluster bootstrap by block, p = 0.35.
    * Position and temperature are partly confounded within a block, and 12
      clusters is few.
  * Temperature slope: **+0.37 % energy per °C [0.19, 0.55]** (iid),
    consistent with leakage power.
    * Clustered intervals: block [0.22, 0.52], two-way [0.18, 0.56].
    * Wild cluster bootstrap p = 0.003.
    * This is an association: temperature was not randomized.
  * Paired temperature differences have mean −0.006 °C, median absolute
    1.1 °C and 95th-percentile absolute 5.6 °C. So temperature adds up to
    about 2 % modeled variation to individual pairs. The near-zero average
    difference and randomized order do not prove absence of confounding or
    systematic bias; temperature was not independently manipulated.
* **Block-level temperature**: block means 39.8–40.7 °C; there is no
  trend over the 2 h campaign.

## Topology and thermal analysis (SECONDARY / descriptive)

* **P4_MIX is the main finding.** It has two NVLink pairs joined over
  PCIe across a single CPU socket (NUMA 1).
  * NCCL automatic picks Ring for AllReduce from 64 KiB to 64 MiB. Tree
    (LL at 64–128 KiB, Simple from 256 KiB) uses 8–69 % less energy
    (ratios 0.921 down to 0.312).
  * At 256–512 KiB, automatic stays on the LL protocol. The Simple-protocol
    picks (Tree/Simple for AllReduce, Ring/Simple for AllGather and
    ReduceScatter) are 1.4–3.2× faster.
  * 16 of 17 RQ1 cells here have meaningful savings.
  * From 1 MiB up, the profiler records show both arms use 8 channels (for
    example, AllReduce: RING/SIMPLE/8 vs TREE/SIMPLE/8). So the large-message
    effect is the algorithm, not the channel count. Below 256 KiB the
    channel counts can differ: AllReduce 64 KiB is RING/LL/4 vs TREE/LL/8.
  * **Hypothesis, not tested here**: NCCL's analytic model in 2.27.5
    misprices Tree on this mixed NVLink/PCIe topology. A test would
    compare NCCL's `TUNING` table predictions with the measured times. The
    available initial probe contains TUNING tables; timed processes used
    INIT,ENV logs, not per-process TUNING cost tables. Profiler records verify
    executed algorithm/protocol separately. Probe-based predictions cannot
    be claimed as exact timed-process cost-model evidence.
* **On the NVLink pair (P2_NVL)**, the gains are smaller:
  * 3.5–5 % from Tree at 8–64 MiB, and 1.3–2.7 % at 128–512 MiB;
  * 15–22 % from switching to LL at 512 KiB, where automatic already
    uses Simple;
  * 8 % from LL128 for AllGather at 1 MiB.
* **On the PCIe pair without P2P (P2_PCI_SHM)**, gains are concentrated
  at 128–256 KiB:
  * Tree/Simple AllReduce saves 15–33 %;
  * Ring/Simple AllGather and ReduceScatter save 6–12 %;
  * gains shrink to ≤ 2 % from 2 MiB up and reverse at 256 MiB.
* **Thermal**:
  * the highest temperature in any region was 47 °C;
  * no HW thermal or power-brake events occurred;
  * no unit was classified `THROTTLED` (SW power cap at ≥ 90 % of the
    power limit).

Recorded mean-clock variation was 1906–1946 MHz. No clock intervention was
performed, so these observations do not isolate or rule out clock effects.

## Claim-evidence table

| # | claim | status | evidence | blocks |
|---|---|---|---|---|
| C1 | NCCL automatic is not energy-optimal in 36 of 39 pre-registered cells (the selected pick uses less board energy); ≥ 5 % by the pre-registered point-estimate rule in 25, supported by a separate threshold test (Holm) in 21 | CONFIRMATORY (25 = pre-registered rule; 21 = v2.2 separate analysis) | `data/v2/processed/confirmatory/rq1.csv` ← `data/v2/raw/confirmatory/blocks/*/units/*/{unit.json,region.jsonl,prof/}` | 12 |
| C2 | Largest saving: P4_MIX AllReduce 512 KiB, Tree/Simple uses 0.312× the energy of automatic | CONFIRMATORY | `rq1.csv` row `P4_MIX,all_reduce,524288` | 12 |
| C3 | Energy savings come with latency savings in every confirmed cell | CONFIRMATORY data, descriptive comparison | `rq1.csv` `T_ratio` | 12 |
| C4 | P2_NVL AllGather 2 MiB: LL128 has 2.8 % lower latency than auto (supported); its ≈ 1 % energy penalty is counter-based and estimator-sensitive (**preliminary**) | CONFIRMATORY (latency) / PRELIMINARY (energy) | `rq2.csv`, `rq2_estimators.csv` | 12 |
| C5 | P2_NVL AllReduce 1 GiB: tested Tree/Simple is faster than auto and energy-equivalent within ±3 % for this pair | CONFIRMATORY | `rq2.csv` | 12 |
| C6 | P2_PCI_SHM AllReduce 256 MiB exploratory pick reverses (1.4 % more energy) | CONFIRMATORY | `rq1.csv` | 12 |
| C7 | The reported RQ1 sensitivity analyses are broadly consistent; the AllGather RQ2 energy penalty is not robust across estimators | SECONDARY / UNRESOLVED (RQ2) | `robustness.csv`, `rq2_estimators.csv` | 12 |
| C8 | No statistically resolved execution-order association in the tested model; +0.37 %/°C temperature association (not randomized; paired differences average ≈ 0 °C) | SECONDARY | `order_temperature.json` (+ v2.2 clustered fits) | 12 |
| C9 | ∫POWER_INSTANT is 2.0 % (median) below the commit-aligned counter; the gap scales with power, not duration | SECONDARY | `units.csv` (`inst_over_counter`) | 12 |

## Not supported / out of scope

* **Other hardware and software.** No claim is made about other GPUs,
  drivers, NCCL versions or multi-node systems. Every result is for RTX
  A6000, driver 580.178.04, NCCL 2.27.5, single node, 2–4 GPUs, FP32 sum.
* **Mechanism.** No claim is made about why NCCL's model misprices Tree on
  P4_MIX; that is a hypothesis.
* **Absolute energy.** No claim is made about absolute energy accuracy
  beyond the board sensor. The counter is the board's own measurement; it
  was not checked against an external meter.
* **Under-powered cells.** `prereg.md` listed 11 cells as under-powered
  for a ±2.5 % interval before the campaign:
  * 10 of them were confirmed anyway, because their effects are large;
  * the 11th is the P2_PCI_SHM 256 MiB reversal.
  The other two unconfirmed cells (PCI 4 MiB, P4 1 KiB) were not on that
  list. Their effects (≈ 0.6 %) are below the 5 % threshold the design
  targeted.
