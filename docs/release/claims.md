# Claim status — v2.2 pilot release (2026-10-09)

The historical categories and numerical claims below remain separate from
the completed [A6000 Stage 2 claims](../../studies/a6000-stage2-v2.5/docs/claims.md)
and [held-out results](../../studies/a6000-stage2-v2.5/docs/results.md).
The new selected comparison does not promote historical C7 to a global
optimum claim, establish physical accuracy, or repair unrecorded training
correctness.

## Scope of every claim

* One host: 8× RTX A6000, NCCL 2.27.5.
* Driver 580.178.04 for v2 and v2.2; driver 550 for the pilot.
* 2–4 GPUs, single node, FP32; sum for the reduction collectives.
* **NVML-reported GPU-board energy of the participating GPUs.** This is not
  communication-only energy and not whole-system energy.
* No external power reference.
* Blocks of each campaign ran back to back in one session. They are
  **within-session replicates**, not independent days or hosts.

## Status labels

| label | meaning |
|---|---|
| **supported** | the pre-specified or confirmatory analysis supports it, and it is robust to the checked alternatives |
| **supported (descriptive)** | measured, no inferential claim |
| **preliminary** | suggestive, method-sensitive or under-powered |
| **inconclusive** / **unresolved** | resolved neither way |
| **not established** | tested; the evidence does not show it |
| **not assessed** | not tested in this project |
| **unsupported** | not shown, or contradicted |
| **exploratory (post hoc)** | chosen after seeing the data; generates hypotheses, supports no claim |
| **superseded** | historical result replaced by a later method |

## Where the numbers come from

* Every number below is recomputed from the retained outputs by
  `python -m ncclenergy.v2.release_tables`.
* `results/release-corrections_2026-10-11/key_numbers.csv` lists the
  number, its source file and the documents that must quote it.
* `doc_check.json` checks the required rounded numerical strings, not whether
  a statement is scientifically justified. Historical presentations remain
  separately frozen under `results/v2.2-pilot-release_2026-10-09/`.

## 1. Historical pilot (method v1; superseded)

| # | claim | status | basis |
|---|---|---|---|
| P1 | The latency-optimal and energy-optimal executed configurations coincide in 123 of 126 cells | **superseded** (driver 550; uncorrected boundary counter reads; 3 of 5 trials) | `data/processed/full/results_summary.json` |
| P2 | The 3 disagreeing cells have raw energy penalties of 2.3–3.3 %, the size of the 2-GPU noise floor | **superseded**; never evidence of a tradeoff | same |

## 2. v2 exploratory and confirmatory (method v2.1; Stage 2 corrections applied)

| # | claim | status | basis |
|---|---|---|---|
| C1 | The frozen Energy-SLO pick uses less board energy than NCCL automatic in 36 of 39 pre-registered cells | **supported** | `data/v2/processed/confirmatory/rq1.csv` (Holm, one-sided) |
| C2 | The saving is ≥ 5 %: 25 cells by the pre-registered point-estimate rule; 21 cells by the separately identified threshold test | **supported (21 cells)**; **preliminary (the other 4)** | `rq1.csv`; `data/v2/reanalysis/2026-10-09_v2.2-audit/rq1_threshold.csv` |
| C3 | Largest effect: P4_MIX AllReduce 512 KiB, Tree/Simple vs automatic's Ring/LL, energy ratio 0.312 [0.308, 0.317] | **supported** | `rq1.csv` |
| C4 | In every confirmed cell the pick is also faster; energy differences track time differences | **supported (descriptive)** | `rq1.csv` time ratios |
| C5 | The confirmatory matrix compares a selected pick with automatic only; the best configuration overall was **not** established | scope statement | — |
| C6 | P2_NVL AllGather 2 MiB: LL128 has 2.8 % lower latency than auto; auto is 2.9 % slower than LL128 | **supported** (latency; denominator corrected) | `data/v2/processed/confirmatory/rq2.csv` |
| C7 | P2_NVL AllGather 2 MiB: LL128 costs ≈ 1 % more board energy. Counter 1.012 [1.003, 1.022]; instant power 1.005 [0.991, 1.019]; lag-adjusted PowerUsage 1.009 [0.999, 1.019] | **preliminary, unresolved**: estimator-sensitive; the v2.2 validation (§3) did not resolve it | `rq2_estimators.csv`; gate 3b report |
| C8 | P2_NVL AllReduce 1 GiB: Tree is energy-equivalent within ±3 % | **supported** under all four estimators (TOST); one shared sensor | `ar1g_tost.json` |
| C9 | P2_PCI_SHM AllReduce 256 MiB: the pick reverses (+1.4 % energy) | **supported** | `rq1.csv` |
| C10 | No statistically resolved execution-order association in the tested model | **inconclusive about order effects** | `order_temperature.json` (block-clustered, wild bootstrap) |
| C11 | Paired energy is associated with paired temperature difference, 0.37 %/°C | **supported (association)**; temperature is not randomized | `order_temperature.json` |
| C12 | Counter and sampled-power estimates differ, by a median of 2.0 % in v2 (instant power below counter) | **supported (descriptive)**; not an independent calibration. Gate 3b post hoc (§4, H2) adds a roughly constant per-region component on P2_NVL AllGather 2 MiB; its source is unexplained | `units.csv`; gate 3b §5.2 |
| C13 | The fixed example tuner reproduces the measured executions: 36/36 rules and 189/189 sizes | **supported** | `data/v2/raw/phase10/verify_tuner_r3/summary.json` |
| C14 | Synthetic DDP, batch 128, P4_MIX: recorded policy/auto energy per step is 0.791 [0.784, 0.799], time per step 0.706 | **supported (descriptive observation only)**, 5 within-session blocks. Numerical equivalence is unverified: numerics not recorded; identical data on every rank; v2.1 meter had no QC flags (200/200 readings plausible post hoc); unguarded example tuner | `data/v2/raw/phase10/workload/runs.jsonl`, referenced `result.json`, reanalysis `phase10_v22/phase10_summary.json` |
| C15 | DDP batch 2048: 0.885 [0.880, 0.889] at the 300 W power limit | **supported (descriptive, synthetic, power-capped)**; training equivalence unverified | same raw workload records |
| C16 | FSDP energy ratio is 1.001 [0.973, 1.030]; the tested comparison meets the original ±3 % energy-equivalence criterion | **supported for reported synthetic board energy**, not numerical training equivalence or exact equality | same |
| C17 | The `no_sync` difference measures communication energy | **unsupported**: different power regimes and overlap | — |
| C18 | Benefits for real-model training, BF16, time to quality or whole-system energy | **not assessed** | — |
| C19 | The recommender/tuner is safe to deploy generally | **unsupported**: validated only inside the controlled harness on this host. dtype/op are declared, not verified. The guarded plugin was never loaded by real NCCL | `native/nccl_tuner_guarded` |

## 3. v2.2 measurement validation (pre-specified, gate 3b; methods `v2.2-fail-closed`, `v2.2-aligned-meter-logged`)

| # | claim | status | basis |
|---|---|---|---|
| M1 | The v2.2 pipeline runs on this hardware: owned-process control, watchdog, fail-closed QC, logged boundaries, all-rank evidence | **supported (smoke test, 26/26 checks; not for inference)** | `data/v2.2/processed/v2.2-smoke-02/smoke_check.json` |
| M2 | The controller makes no cumulative-counter read inside a measured region | **supported** for the gate 3a units on real NVML, including the window up to the end of the post read (`counts.json`). Code paths are covered by CPU tests | gate 3a audit; `tests/v2/test_v2_no_counter_reads.py` |
| M3 | Gate 3b ran as pre-specified: 70 of 72 processes accepted, 2 excluded and not rerun | **supported** (counts match the raw records) | `counts.json` |
| M4 | Pre-specified decisions: D1 INCONCLUSIVE, D2 INCONCLUSIVE (LL128 term −0.12 % [−0.48, +0.25]), D3 SYSTEMATIC_OFFSET (20 s B ÷ A −2.44 % [−4.80, −0.01]), D4 NOT_SUFFICIENT (40 s half-width 3.98 % vs ≤ 0.4 %) | **result as defined** | `data/v2.2/processed/v2.2-measval-01/decisions.json` |
| M5 | Identical execution of auto and forced Ring/Simple, for the parameters NCCL's profiler records | **supported** (6/6 blocks) | same |
| M6 | Measurement agreement between identically executed configurations within ±0.5 % | **not established** (D3) | same |
| M7 | Absolute accuracy of the board-energy measurement | **not assessed** (no external reference) | — |
| M8 | The method resolves a ≈ 1 % energy difference for AllGather 2 MiB on this host | **not established** (D4); hence C7 stays preliminary | same |

## 4. Post-hoc exploratory (gate 3b data; changes no claim above)

| # | observation | status | basis |
|---|---|---|---|
| H1 | ln(J/op) is associated with GPU temperature: within-block slope 0.43 [0.38, 0.48] %/°C. Under an arm × length + temperature model, the residual SD at ≥ 10 s falls from 1.62 % to 0.25 %. It does not establish temperature as the sole cause, or accuracy | **exploratory (post hoc)** | `posthoc_temperature.json` |
| H2 | Over matched windows, steady-state power of counter vs sampled agrees within about 0.06 % per GPU. Whole-region energy differs by +13.5 [12.2, 14.8] J (instant) and +13.1 [12.9, 13.3] J (lag) per GPU per region. The pre-start-window explanation is not supported. **Unexplained** | **exploratory (post hoc)** | `posthoc_temperature.json` (`window_gap`) |
| H3 | Temperature-adjusted C ÷ A at 40 s is about +1.2 % | **exploratory (post hoc)**; conditional intervals omit slope uncertainty and overlapping LOBO-fit dependence; **does not change C7** | same |

## Scientific basis and evidence boundaries

The [scientific resolution and literature review](scientific_resolution.md)
separates each paper's findings from this project's measurements. Literature
supports mechanisms and validation methods, not the numerical percentages
above. Raw campaign inputs and the public evidence omissions are indexed
there and in [OMITTED.md](../../OMITTED.md). A document-number check alone
does not establish measurement validity.
