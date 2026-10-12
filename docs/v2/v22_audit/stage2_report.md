# v2.2 audit — Stage 2 report (CPU-only repairs, tests, reanalysis)

> **v2.2 pilot release (2026-10-09).** The current claim status, limitations and reproduction instructions are in
> [`../../release/claims.md`](../../release/claims.md), [`../../release/limitations.md`](../../release/limitations.md) and
> [`../../release/reproduction.md`](../../release/reproduction.md). This document is kept as the record of its stage.


> **Addendum (after review).** [`stage3_plan_rev1.md`](stage3_plan_rev1.md)
> supersedes §4.1 and §10.
> * **§4.1**: the 0.10 % bounds only read-timestamp and commit-phase
>   effects. The decomposition is in
>   `data/v2/reanalysis/2026-10-09_v2.2-audit-addendum1/`; the tail model
>   (up to about 1.4 %) and sensor accuracy (not characterized) are
>   separate.
> * **§10**: replaced by the smoke gate 3a and measurement-validation gate
>   3b, with pre-specified decision rules.

## Scope and status

| item | value |
|---|---|
| branch | `ext/v2.2-audit`, created from `63c5332` |
| baseline check | HEAD `63c5332`, clean tree, no stash (2026-10-09) |
| GPU work | **none**: no benchmark, no CUDA training, no GPU allocation assumed |
| other processes | tests ran only CPU child processes they created themselves; none left running |
| push / publish | none |
| historical data | raw data, frozen schedule, pre-registration and frozen processed outputs/figures unchanged |
| frozen-output check | 36 frozen files under `data/v2/processed` and `figures/v2`, SHA-256 identical before and after the reanalysis (`summary.json` `frozen_outputs_unchanged: true`) |

The pilot's 27,491 files still match `provenance/pilot_v1_sha256_2026-10-08.txt`.
That manifest's own hash is also unchanged.

## 1. Audit findings: what was done

| # | finding | verified? | resolution | evidence |
|---|---|---|---|---|
| 1 | Interrupting the controller orphans GPU processes | yes: CPU reproduction with SIGTERM and Ctrl-C | P1, `procctl.py`: owned process tree tracked by (pid, start time, uid); controller is a child subreaper; SIGTERM, then SIGKILL after a grace period, then reap, on every exit path | `test_v2_procctl.py` (9 tests) |
| 2 | Health/contention checks only after execution | yes (`runner.py` `run_unit`) | P2, `watchdog.py`: about 1 Hz checks for foreign process, temperature, HW health bits, SW thermal, telemetry stale/errors/truncation and low disk; a trip cancels the owned tree and stops the campaign; no retry | `test_v2_watchdog.py` (5) |
| 3 | AllGather 2 MiB tradeoff depends on the estimator | yes, numbers reproduced exactly | wording corrected; four-estimator table | `rq2_estimators.csv`, `rq2_estimator_inputs.csv` |
| 4 | Workload energy log lacks raw counter values and timestamps | yes | P7, `aligned_meter.py`: logs C0/C1/Cpost, NVML status, read brackets, probe edges, phase, predicted commits, period used and estimated, QC flags; energy recomputable with `aligned_energy()` | `test_v2_meters.py` |
| 5 | Recommender ignores CUDA; weak hardware/dtype guards; tuner does not validate environment | yes | P3: exact scope match (GPU model, driver, CUDA driver, NCCL version, dtype, op, in-place, ranks/nodes, fingerprint, placement env), else fallback. Scope derived only from raw evidence (`profile_scope.py`). New guarded plugin `native/nccl_tuner_guarded`. dtype/op are **declared**, not verified | `test_v2_recommender.py` (8), `test_v2_tuner_guard.py` (19 fallback cases) |
| 6 | Interpolation copied endpoint ratios and CIs | yes | P4: off by default; the experimental opt-in has a separate status and carries no ratio or CI; tuner export is exact-size only; range rules need two explicit opt-ins | `test_v2_recommender.py`, `test_v2_tuner_guard.py` |
| 7 | Sampled fallback still read the counter (`idle_baseline`) | yes | P5: sampled idle baseline; `SampledMeter`; idle adjustment uses sampled idle power | end-to-end test with a mock counter that fails if called: 0 calls |
| 8 | "25 cells, ≥ 5 %, family-wise" overstated | yes: 25 by point estimate, 21 by threshold test | pre-registered definition kept (25); the 21 is reported as a separately identified analysis | `rq1_threshold.csv`, `figures/rq1_threshold.png` |
| 9 | Pooled regression ignores clustering | yes; conclusions unchanged | block-, cell- and two-way-clustered fits plus an exact wild cluster bootstrap | `order_temperature.json` |
| 10 | no_sync ≠ communication energy; numerics unchecked | yes | wording removed or corrected; workload records loss trajectory, finite checks, parameter norms and all-rank consistency; comparison logic in `numerics.py` | `test_v2_numerics.py` (includes a 2-process CPU run of the real workload script) |

Additional issues found and fixed in this stage:
- **Fixed 100 ms period** assumed by both meters. The Python meter now estimates the period from probe edges and flags a mismatch. The C patch is unchanged; its records already contain edges, so the period is checked offline (section E).
- **Python meter had no QC flags.** It now has `region.py`-equivalent flags.
- **"Independent blocks" wording** corrected to within-session blocks.
- **Hard-coded GPU allocation:** the `phase10` default `ALLOC=[4,5,6,7]`, the `runner preflight --alloc` default and the GPU list in the script's cool-down loop are removed. Allocation must be explicit.
- **Missing dependencies:** `scipy` was missing from the requirements; `torch` was marked optional although Phase 10 needs it.
- **Analysis depended on NVML:** `phase10 analyze` initialized NVML. It no longer does.

## 2. Changed files

Changed and new files, grouped:
* **Safety**:
  * `src/ncclenergy/v2/procctl.py` (new), `watchdog.py` (new);
  * `runner.py`: owned runs, watchdog, `SAFETY_STOP`, `INTERRUPTED` status, explicit `--alloc`, sampled idle;
  * `phase10.py`: explicit allocation, owned torchrun, all-rank evidence, lazy NVML, analysis output directories;
  * `scripts/run_confirmatory_v2.sh`: signal forwarding, no `timeout`, allocation and log directory from config/env.
* **QC**: `qc.py`, fail-closed (`QC_METHOD = "v2.2-fail-closed"`), and `telemetry_problems()`.
* **Measurement / workload**:
  * `aligned_meter.py`: v2.2 logged meter and `SampledMeter`;
  * `workload.py`: numerics, seed, meter choice, CPU/gloo test mode;
  * `numerics.py` (new);
  * `analysis.py`: idle power from sampled baselines in sampled mode.
* **Recommender / deployment**:
  * `recommender.py`: scope guards, interpolation off, exact-size export with `#@scope` header;
  * `profile_scope.py` (new);
  * `phase89.py`: profile scope from raw evidence;
  * `native/nccl_tuner_guarded/` (new plugin; the historical `native/nccl_tuner_example` is unchanged).
* **Reanalysis**: `reanalysis.py` (new) and output `data/v2/reanalysis/2026-10-09_v2.2-audit/`.
* **Dependencies**: `requirements-analysis.txt`, `requirements-gpu.txt` (new); `requirements.txt` (pilot list kept, with a header pointer).
* **Tests**:
  * `tests/v2/test_v2_{procctl,watchdog,meters,numerics,tuner_guard,deps_and_script}.py` (new);
  * `test_v2_core.py` (fail-closed semantics added), `test_v2_recommender.py` (v2.2 semantics);
  * helpers: `tests/v2/fake_nvml/`, `tests/v2/helpers/`, `tests/v2/native/`.
* **Docs**: this report; wording corrections, each with an audit box, in `README.md`, `docs/v2/phase8_results.md`, `phase9_recommender.md`, `phase10_results.md`, `final_report.md`, `phase4_validation.md`, `replication_1080ti_titan.md`; plus `telemetry.py` comments.

## 3. Tests

`python tests/run_tests.py`: **78 / 78 pass** (baseline 45). All are CPU-only.

| file | tests | covers |
|---|---|---|
| `test_v2_procctl.py` | 9 | the run ends by normal exit, timeout, SIGTERM-resistant tree, watchdog trip, controller exception, failing cleanup step, controller SIGTERM and Ctrl-C, or a second signal during cleanup; orphaned and new-session grandchildren are cleaned; unrelated and recycled pids are never signalled |
| `test_v2_watchdog.py` | 5 | each trip condition; SW power cap alone does not trip; a check that raises trips (fail closed); foreign process cancels an owned run |
| `test_v2_meters.py` | 7 | v2.2 accounting identical to v2.1 on 200 random inputs; QC flags; raw-value logging and offline recomputation; unsupported counter reported; sampled meter and **sampled mode end to end through `runner.idle_baseline` + `runner.run_unit` with a counter mock that fails if called (0 calls)**; negative control showing the mock detects counter use |
| `test_v2_numerics.py` | 3 | comparison logic (tolerance, semantics, non-finite, rank-inconsistent); **real workload script on CPU (gloo, 2 ranks)**: finite checks, rank consistency, same-seed runs agree, different-seed runs flagged not comparable |
| `test_v2_tuner_guard.py` | 2 (19 + 4 cases) | guarded plugin loaded the way NCCL loads it, with a mock NVML that aborts if the counter is touched: exact rules apply only with a full match; every missing, unknown or incompatible fact disables all rules |
| `test_v2_deps_and_script.py` | 3 | analysis modules import and run with `torch` and `pynvml` blocked; requirement files split and pinned; campaign script forwards SIGTERM/SIGINT, the owned tree is cleaned, no further block starts |
| `test_v2_recommender.py` | 8 | scope guards incl. missing legacy metadata (never filled in), op, interpolation off, experimental opt-in separate, exact-size export |
| `test_v2_core.py` | 27 | existing tests plus fail-closed QC and telemetry-problem detection |

**Untested paths** (they need GPUs or are outside unprivileged control):
* **Real GPU and NCCL**:
  * the nccl-tests v2.1 patch in sampled mode on a board without the counter;
  * `AlignedMeter` against real NVML;
  * the watchdog fed by the real NVML sampler;
  * the guarded plugin inside a real NCCL communicator (tested only with the harness);
  * FSDP and CUDA paths of the workload;
  * the campaign script with the real runner.
* **SIGKILL of the controller itself**: it cannot be handled. Orphans would be re-parented to init or an outer subreaper. This is mitigated by bounded per-process timeouts and by never running the controller under an external `kill -9`.
* **Kernels without `PR_SET_CHILD_SUBREAPER`**: the code falls back to tracking only; recorded in `res["subreaper"]`.
* **Corner case:** a pre-existing child of the controller that spawns new processes during a run would have those adopted as owned. The runner never has such children.

## 4. Numerical-change ledger

The reanalysis output is in `data/v2/reanalysis/2026-10-09_v2.2-audit/`. Its machine-readable summary is `summary.json`; `ledger.csv` lists only the rows that differ.

| item | frozen value | v2.2 value | kind | why |
|---|---|---|---|---|
| RQ1 / RQ2 tables, `units.csv` | `data/v2/processed/confirmatory/` | `reproduced/` | **identical (byte-for-byte)** | the pre-registered analysis reproduces exactly |
| RQ1 savings confirmed | 36 / 39 | 36 / 39 | unchanged | — |
| RQ1 "meaningful" (pre-registered: confirmed and point estimate ≤ 0.95) | 25 | 25 | unchanged definition | — |
| RQ1 ratio ≤ 0.95 supported by a corrected threshold test | not reported | **21** (Holm, one-sided, α 0.025; same with the Bonferroni bound) | new, separately identified analysis | 4 cells meet the point-estimate rule only: P2_NVL AR 32 MiB, P2_PCI_SHM AG 256 KiB, P4_MIX AR 64 KiB, P4_MIX RS 128 KiB |
| RQ2 AllGather 2 MiB | "DIFFER", counter: +1.23 % [0.28, 2.18] | counter 1.0123 [1.0028, 1.0218]; integrated instant 1.0048 [0.9905, 1.0194]; lag PowerUsage 1.0092 [0.9992, 1.0192] (Holm p 0.069); idle-adjusted 1.0214 [1.0102, 1.0328] | sensitivity; wording | the verdict depends on the estimator, so "DIFFER" is counter-based and **preliminary** |
| RQ2 AllReduce 1 GiB equivalence (±3 %) | counter TOST p 1.7e-6 | counter 1.7e-6; instant 0.023; lag PowerUsage 2.9e-6; idle-adjusted 2.1e-5 (inputs and 90 % CIs in `ar1g_tost.json`) | sensitivity | equivalence holds under all four estimators; they share one sensor, so agreement is not independent calibration |
| order slope (per block of position) | 0.0037 [−0.0072, 0.0145] (iid) | block-clustered [−0.0042, 0.0116]; cell-clustered [−0.0069, 0.0143]; two-way [−0.0061, 0.0134]; wild cluster bootstrap p = 0.353 | reanalysis | "no statistically resolved order association in the tested model" |
| temperature slope (per °C) | +0.0037 [0.0019, 0.0055] (iid) | block [0.0022, 0.0052]; cell [0.0019, 0.0055]; two-way [0.0018, 0.0056]; wild bootstrap p = 0.0029 | reanalysis | same point estimate; association resolved under every variant |
| boundary-read timing uncertainty | not reported | see §4.1 | new | derived from recorded read brackets |
| maximum sample interval | "≈ 50.5 ms" (my Stage 1 note) | measurement windows: max 50.5 ms; complete traces: max 162.6 ms (236 of 2,784 GPU traces > 150 ms, around launch/teardown) | clarified | the integrals use only measurement-window samples |
| recommender LOBO, robustness | `recommender_lobo*`, `robustness.csv` | identical | unchanged | all LOBO queries are at measured sizes with matching scope |
| tuner files | 4–5 range rules per file, no scope header | one rule per measured size (7–17 per file) + `#@scope` header | code change | ranges covered unmeasured sizes; frozen files kept |
| Phase 10 contrasts, arms, overhead | `phase10_summary.json` | identical | unchanged | — |
| Phase 10 all-rank execution | rank 0 only | 50 / 50 runs consistent across all ranks | new check | — |
| Phase 10 numerical comparison | none | `NOT_RECORDED` for all historical runs | new field | v2.1 workload did not record losses |
| accepted units under v2.2 fail-closed QC | exploratory 1,542; confirmatory 984 | 1,542; 984 | identical | §5 |

### 4.1 Boundary-read timing uncertainty

Source: `read_latency.json`, 2,784 GPU-regions, 984 units.

Read-call durations, from the recorded before/after timestamps:

| read | median | p95 | max |
|---|---|---|---|
| start read | 1.37 ms | 3.93 ms | 7.54 ms |
| end read | 1.55 ms | 1.63 ms | 1.94 ms |
| commit-phase bracket | 2.35 ms | — | 2.55 ms |

* **Correction magnitude**: the idle-tail correction removes a median 1.34 % of region energy (max 2.41 %).
* **Uncertainty of the correction**: the reset instant can fall anywhere inside the end read, so the post window has ±half a read of slack. That gives the correction a median relative uncertainty of 0.85 % (max 1.07 %) of the correction itself.
* **Uncertainty of total region energy**: take worst-case half-widths of the start read, the end read and the phase bracket, summed over a unit's GPUs. That is median 0.42 J (max 1.31 J) per unit, or **0.057 % (max 0.10 %)** of region energy.

These figures rest on the interpretation that a counter read resets the open accumulation window. That interpretation is supported by the local Phase 4 dose and linearity experiments; NVML does not guarantee it. They cover boundary timing only. They do not cover the sensor's own error, the counter's internal averaging, or calibration; the board sensor is the only reference available.

## 5. Historical acceptance under the v2.2 QC

| campaign | v2.1 accepted (OK + VERIFIED) | v2.2 accepted | units only in v2.1 | units only in v2.2 | v2.2 status counts |
|---|---|---|---|---|---|
| exploratory (2 blocks) | 1,542 | 1,542 | 0 | 0 | OK 1,542; QC_SHORT 11; EXCLUDED_DEVIATION 1 |
| confirmatory (12 blocks) | 984 | 984 | 0 | 0 | OK 984 |

The counts don't change because none of the historical attempts that reached a QC decision had any of the four new fail-closed conditions:
* a missing data check;
* sampler errors, truncation or sample gaps inside the region;
* missing temperature or clock-event telemetry;
* unverified execution evidence.

That fits the run records: 0 sampler errors, 0 window gaps, and every run had `-c 1` data checks. The 11 `QC_SHORT` units and the one excluded deviation are unchanged. Sampler overruns (562 / 984 confirmatory units) are not failures: no measurement-window gap exceeded 50.5 ms.

## 6. Figures

* New:
  * `data/v2/reanalysis/2026-10-09_v2.2-audit/figures/rq1_threshold.png`: point estimates and Bonferroni bounds against 0.95; the 21 threshold-supported cells vs the 4 point-only cells.
  * `.../figures/rq2_estimators.png`: RQ2 under the four estimators.
* Regenerated with the v2.2 code into the same directory, numerically unchanged:
  * `phase89_v22/rq1_per_block.png`
  * `phase89_v22/counter_vs_integration.png`
  * `phase10_v22/workload_blocks.png`
* Frozen figures under `figures/v2/` are untouched.

## 7. Corrected claim table

**Status labels**:
* **supported**: confirmatory, robust to the checked alternatives;
* **preliminary**: suggestive, method-sensitive or under-powered;
* **inconclusive**: resolved neither way;
* **unsupported**: not shown.

All claims are for this host only (RTX A6000, driver 580.178.04, NCCL 2.27.5), 2–4 GPUs, FP32, **board energy** of the participating GPUs, and 12 blocks run back to back in **one ~2 h session**: within-session replicates, not independent days or hosts.

| # | claim | status | basis |
|---|---|---|---|
| C1 | The frozen Energy-SLO pick uses less board energy than NCCL automatic in 36 of 39 pre-registered cells | **supported** | `rq1.csv`; Holm one-sided |
| C2 | The saving is ≥ 5 %: 25 cells by the pre-registered point-estimate rule; 21 cells supported by a corrected threshold test | **supported (21) / preliminary (4)** | `rq1_threshold.csv` |
| C3 | Largest effect: P4_MIX AllReduce 512 KiB, Tree/Simple vs automatic Ring/LL, ratio 0.312 [0.308, 0.317] (68.8 % lower board energy) | **supported** | `rq1.csv` |
| C4 | In every confirmed cell the pick is also faster; energy differences track time differences | **supported (descriptive)** | `rq1.csv` T ratios |
| C5 | These are comparisons of a selected pick against automatic. The best configuration overall was **not** established | — (scope statement) | confirmatory matrix = automatic + one pick |
| C6 | P2_NVL AllGather 2 MiB: LL128 has 2.8 % lower latency; auto is 2.9 % slower | **supported** (latency; denominator wording corrected 2026-10-11) | `rq2.csv` |
| C7 | P2_NVL AllGather 2 MiB: LL128 costs ≈ 1 % more energy | **preliminary**: counter-based; the instant-power and lag-PowerUsage CIs include 1 | `rq2_estimators.csv` |
| C8 | P2_NVL AllReduce 1 GiB: Tree is energy-equivalent within ±3 % | **supported** under all four estimators (one shared sensor) | `ar1g_tost.json` |
| C9 | P2_PCI_SHM AllReduce 256 MiB pick reverses (+1.4 %) | **supported** | `rq1.csv` |
| C10 | No statistically resolved execution-order association in the tested model | **inconclusive about order effects** | `order_temperature.json` |
| C11 | Paired energy rises about 0.37 % per °C of paired temperature difference | **supported (association)**; temperature is not randomized | `order_temperature.json` |
| C12 | Counter and sampled-power estimates differ by median 2.0 % (instant power below counter); the gap scales with power | **supported (descriptive)**; not independent calibration | `units.csv` |
| C13 | The fixed tuner reproduces the measured executions (36 / 36 rules, 189 / 189 sizes) | **supported** | `verify_tuner_r3` |
| C14 | Synthetic DDP batch 128 on P4_MIX: tuner policy gives 0.791 [0.784, 0.799] **board energy of GPUs 4–7 per step**, 0.706 step time | **supported for this synthetic workload, 5 within-session blocks**; numerics were not recorded | `phase10_summary.json` |
| C15 | DDP batch 2048: 0.885 [0.880, 0.889] at the 300 W power limit | **supported (synthetic, power-capped)** | same |
| C16 | FSDP negative control unchanged: 1.001 [0.973, 1.030] | **supported** | same |
| C17 | The `no_sync` difference measures communication energy | **unsupported**: it compares runs in different power regimes and with overlap; it is not communication-only energy | — |
| C18 | Real-model training, BF16, time-to-quality or whole-system energy benefits | **unsupported (not tested)** | — |
| C19 | The recommender/tuner is safe to deploy generally | **unsupported**: validated only inside the controlled harness on this host; dtype/op are declared, not verified | `native/nccl_tuner_guarded` |

## 8. Method identifiers (never pooled)

| id | what | where |
|---|---|---|
| `v1` | pilot: boundary counter reads, driver 550 | `data/raw/`, `data/processed/` |
| `v2.1` | patch v2.1 commit-aligned region, v2.1 QC, v2.1 aligned meter (workload) | `data/v2/raw/*` (all historical v2 data) |
| `v2.2-fail-closed` QC | same accounting; stricter acceptance (§5 shows identical historical acceptance) | `qc.py` |
| `v2.2-aligned-meter-logged` | identical energy formula to v2.1 (tested); adds raw values, brackets, status codes, period estimate, flags; adds possible rejections (`PERIOD_MISMATCH`, `END_READ_BEFORE_COMMIT`, …) | `aligned_meter.py` |
| `v2.2-sampled-meter` | different estimator (integrated sampled power) | `aligned_meter.py` |

**Accounting check**: v2.2 does not change energy accounting for the counter path.
* The formula is identical to v2.1 (test on 200 random inputs).
* Historical acceptance is identical (§5).
* The pre-registered tables reproduce byte-for-byte (§4).

New runs still get the identifier `v2.2`, because acceptance, logging and process control differ. They are stored under a new campaign directory and never pooled with v2.1 observations without an explicit, justified model.

## 9. Remaining limitations

* **Single host and session**:
  * one host and one driver;
  * one session for the 12 confirmatory blocks and one for the 5 workload blocks;
  * no between-day or between-host variance.
* **No external reference**: board sensor only; no external power meter. Counter/sampled agreement shares the sensor.
* **RQ2 AllGather 2 MiB is estimator-sensitive (~1 % effect).** Boundary timing (≤ 0.1 %) is not the issue. Sensor-level disagreement between estimators (about 1 %) is the same size as the effect.
* **Native tuner scope**: dtype and reduction op cannot be verified by NCCL's v4 tuner interface. Deployment is limited to a controlled harness that declares the scope and verifies the rest at startup.
* **Process cleanup**: it cannot survive a `kill -9` of the controller.
* **Replication blockers (1080 Ti / Titan)**:
  * PyTorch 2.9.1+cu128 is compiled for sm_70–sm_120 only (checked from the wheel's compile flags). The Phase 10 workload therefore **cannot run on Pascal (sm_61)**: GTX 1080 Ti, Titan X (Pascal), Titan Xp.
  * The NCCL 2.27.5 wheel does contain sm_61 code (`cuobjdump`), so nccl-tests microbenchmarks rebuilt for sm_61 remain possible.
  * The exact Titan models on that cluster are unknown. A Titan V (sm_70) or Titan RTX (sm_75) would be supported by the wheel.
  * Counter availability on those boards must be established by `capabilities.py` on idle GPUs.

## 10. Minimal Stage 3 GPU plan (proposal only — not started)

| item | proposal |
|---|---|
| campaign / method id | `v2.2-measval-01`; raw data under `data/v2.2/raw/v2.2-measval-01/`, never pooled with `data/v2/raw` |
| allocation | **two GPUs forming one NVLink (NV4) pair, named and confirmed by you**. On this host the NV4 pairs are (0,1), (2,3), (4,5), (6,7). For comparability with the historical P2_NVL profile (GPUs 4,5, NUMA 1, fingerprint `e65ef2bdbf86ae93`), (4,5) or (6,7). Nothing runs until you name the pair and confirm it is allocated to you. Preflight then re-verifies with read-only checks: topology NV4, NUMA node, no processes, ≤ 16 MiB used, temperature < 45 °C |
| configurations | P2_NVL AllGather 2 MiB, FP32, out-of-place: **A** automatic (historically RING/SIMPLE/4); **B** forced `NCCL_ALGO=Ring NCCL_PROTO=Simple` (identical-execution control vs A); **C** forced Ring/LL128 (historically RING/LL128/4). No channel or other settings changed |
| region lengths and operation counts | target 2.5, 10, 20 and 40 s. R = ceil(T / t), with t = 52.3 µs (A, B) and 50.8 µs (C) from the confirmatory data: A/B R ≈ 47,800 / 191,200 / 382,400 / 764,800; C R ≈ 49,200 / 196,900 / 393,700 / 787,400 (exact values fixed in the frozen schedule before collection). Warm-up 0.2 s |
| design | 6 rounds. Each round = 3 configurations × 4 lengths = 12 fresh processes in seeded random order, plus a 30 s idle baseline before and after (counter and sampled, giving a low-power regime). 72 processes in total |
| recorded | per GPU: C0/C1/Cpost, NVML status, read-call before/after timestamps, probe edges, phase, bracket, predicted commit, loop end, CUDA event time, R and w, profiler evidence on every rank; 50 ms sampler (PowerUsage, POWER_INSTANT, clocks, temperature, utilization, P-state, clock-event reasons, processes) |
| resources | GPU memory per process < 0.5 GiB; 1 benchmark process (2 GPU threads) under `nice 10` on the GPUs' NUMA node; disk < 0.5 GB; RAM < 2 GiB |
| wall-time budget | ≈ 5.3 min per round (217.5 s regions + ≈ 8 s overhead × 12) + 1.3 min idle ≈ 6.6 min; 6 rounds ≈ **40 min**; hard stop at 60 min |
| health thresholds and watchdog | stop on: any HW slowdown, HW thermal or power-brake bit; SW thermal; SW power cap with region power ≥ 90 % of limit; GPU temperature ≥ 80 °C (A6000 slowdown threshold reads 95 °C); foreign process on either allocated GPU; telemetry stale > 2 s or ≥ 3 sampler errors; disk < 20 GB free; per-process timeout 3 × planned + 60 s; campaign wall time > 60 min. The trip cancels the owned process tree (SIGTERM, then SIGKILL after 10 s), reaps it, records the reason and stops the campaign |
| acceptance / exclusion (fixed now) | v2.2 fail-closed QC. Execution evidence on every rank must equal the intended execution (A and B: RING/SIMPLE/4; C: RING/LL128/4); otherwise the process is excluded and reported, not rerun. Exact operation counts. Region length within ±10 % of target. No retries of any kind (`max_attempts: 1`). A safety stop ends the campaign |
| analysis (fixed now) | per length: paired per-round log ratios C/A and B/A (negative control, expected 1) under counter, integrated instant and lag-PowerUsage estimators; counter-vs-sampled disagreement modelled against 1/T, protocol and mean power (including idle baselines); correction magnitude (tail/E) reported separately from correction uncertainty (from brackets); update period estimated from the edges of every process (no 100 ms assumption); B/A spread as the noise floor for ~1 % differences |
| stopping | fixed 6 rounds; no significance-based stopping; an inconclusive result is reported as such |
| what this cannot do | it is a single session and single pair, and has no external meter, so it cannot separate sensor bias from physical energy |

I'm waiting for your review. No GPU work will start without your approval of this plan and your explicit GPU allocation.
