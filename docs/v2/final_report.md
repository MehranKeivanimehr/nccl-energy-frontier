# Methodology v2 — final report (2026-10-09)

> **v2.2 pilot release (2026-10-09).** The current claim status, limitations and reproduction instructions are in
> [`../release/claims.md`](../release/claims.md), [`../release/limitations.md`](../release/limitations.md) and
> [`../release/reproduction.md`](../release/reproduction.md). This document is kept as the record of its stage.


> **v2.2 audit corrections (2026-10-09, branch `ext/v2.2-audit`).** The
> independent audit's ten findings were verified and addressed. Numbers are
> unchanged and the pre-registered analysis reproduces byte-for-byte.
> Corrected statements are marked below:
> * RQ2 AllGather 2 MiB energy penalty is preliminary (estimator-sensitive).
> * ≥ 5 %: 25 by the pre-registered point-estimate rule, 21
>   threshold-supported.
> * Order: no statistically resolved association.
> * Blocks are within-session replicates.
> * The counter mechanism is a local interpretation.
> * Energies are board energy only.
>
> Corrected claim table, ledger, tests and the Stage 3 plan:
> [`v22_audit/stage2_report.md`](v22_audit/stage2_report.md).

**Scope**: Phases 0–11 of the methodology-v2 specification. Everything ran
on one host: 8× RTX A6000, driver 580.178.04, NCCL 2.27.5.

**GPU use**: only GPUs 4–7. At most 4 GPUs were in use at once, never more
than the 6 allowed, with one experiment process at a time. Every launch
checked that the GPUs were idle first.

**Data handling**:
* the pilot (v1, driver 550, 3 trials) is untouched and never pooled;
* nothing was pushed or published;
* commits are local only.

## Per-phase report

| phase | what | GPU work (IDs, UTC, max concurrency) | health events | failed / excluded attempts | validation | commit |
|---|---|---|---|---|---|---|
| 0 | inventory, pilot checksums (27,491 files) | none (read-only NVML queries, no counter reads) | — | — | pilot tables regenerate byte-identically | 45b60b8 |
| 1 | protocol (`protocol.md`) | none | — | — | — | 45b60b8 |
| 2 | region patch, profiler plugin, runner, tests | none | — | — | CPU unit tests | c9f20de |
| 3 | unprivileged preflight (`phase3_preflight.md`) | read-only NVML on GPUs 4–7 | none (SW-cap and SW-thermal counters tick at idle, a driver artefact) | — | Xid event registration works; kernel log `UNAVAILABLE_DUE_TO_PERMISSIONS` | cc504be |
| 4 | validation (`phase4_validation.md`) | GPUs 4–7, 10-08 21:37–22:19 (validation, probe, calibration, smoke block), ≤ 4 | none | v2.0 end-read bias → patch v2.1; one run with a relative-path bug (kept, README); 3 P2P probes on GPUs 5,6 hung at 120 s (stopped; placement → `P2_PCI_SHM`) | linearity intercept +1.6/+3.0 J; selection 30/30 verified; plugin overhead < 1 % | 6cb51eb, de5423c, 360ed6b |
| 5 | exploratory screen, 2 blocks (`phase5_exploratory.md`) | GPUs 4–7, 10-08 22:22 → 10-09 01:18, ≤ 4 | none | 1 unit `EXCLUDED_DEVIATION` (my capability script read counters during block 1; `deviations.jsonl`); 11 `QC_SHORT` | 1,554/1,554 `VERIFIED` | e7c9cf6 |
| 6 | pre-registration (`prereg.md`, `prereg.json` sha `d2053633…`) | none | — | `freeze_prereg` path error after writing the schedule; partial file renamed `…PARTIAL_FAILED_RUN…`; re-run gives identical schedule hash | hashes recorded | d9c1297, 240bc3e |
| 7 | confirmatory, 12 blocks | GPUs 4–7, 10-09 01:20–03:28, ≤ 4 | none in 12 blocks | none | 984/984 `OK` + `VERIFIED`, 0 retries | 53f7390 |
| 8 | analysis (`phase8_results.md`) | none | — | — | robustness, bootstrap, idle-adjusted, order/temperature | 4c69425 |
| 9 | recommender (`phase9_recommender.md`) | none | — | tuner ranges first merged across unmeasured gaps → adjacency rule + test | LOBO 492 rows; exploratory→confirmatory | 53f7390, 4c69425 |
| 10 | tuner + workload (`phase10_plan.md`, `phase10_results.md`) | GPUs 4–7 (and 4,5 or 5,6 subsets), 10-09 03:29–04:08, ≤ 4 | none (no HW events; SW power cap active, as expected, in the compute-heavy workload arms at the 300 W board limit) | tuner verify attempt 1: stock example plugin (nChannels = 1, cost-table indexing bug); attempt 2: relative config path; both kept with READMEs; 1 diagnostic segfault (stock plugin) | verify r3: 36/36 rules, 189/189 executions match measured; overhead not measurable; workload 5 blocks | bf866a7, 4bb5704, 4c69425 |
| 11 | replication package (`replication_1080ti_titan.md`) | none | — | — | — | 5f3e717 |

Tests: `python tests/run_tests.py` passes **45/45** (CPU only).

## 1. Completed and missing matrices

| matrix | status |
|---|---|
| Phase 4 validation (basic, idle counter, dose, plugin overhead, linearity, selection) | complete (rounds 1 and 2) |
| allowed-set probe, 3 placements × 3 collectives | complete for `P2_NVL`, `P4_MIX`, `P2_PCI_SHM`; `P2_PCI` with P2P **not measurable** (hang) |
| smoke block (36 units) | complete |
| exploratory (3 placements × 3 collectives × 21 sizes × allowed configurations; 777 executed units per block) | 2 blocks complete (exploratory config allowed up to 3) |
| confirmatory (39 RQ1 + 2 RQ2 cells, 82 units per block) | 12 / 12 blocks, 984 / 984 units |
| tuner verification (3 placements × 3 collectives × 1 KiB–1 GiB) | complete (r3) |
| tuner overhead (AllReduce 4 KiB, 64 KiB; 3 ABAB reps) | complete |
| workload A/B (5 blocks × 10 units, plus 2 smoke rounds not analysed) | complete: 50/50 runs, rc 0 |
| **missing / not run** | 6-GPU and 8-GPU placements; other dtypes/ops; in-place variant (v2 measures out-of-place only); multi-node; other NCCL versions; `P2_PCI` with P2P enabled; replication on other hardware (package prepared, not run) |

## 2. Raw-data manifest

`provenance/v2_raw_sha256_2026-10-09.txt` holds the SHA-256 of every file
under:
* `data/v2/raw`, `data/v2/processed`, `figures/v2`;
* `configs/v2`, `patches`;
* the two native plugin sources;
* `docs/v2/prereg.json`.

The manifest's own hash is in `….txt.sha256`.

Per-process directories are kept on disk but not in git. These are
`units/`, probe and calibration runs, and Phase 10 runs. They contain NCCL
logs with the hostname and a private IP; this is the pilot's policy,
`.gitignore`. Block-level metadata, setup tables, Phase 10 run tables and
all processed tables are in git.

**Privacy note.** 59 `nccl.log` files and some nccl-tests stdout files
under `data/v2/raw/validation*/` were committed in Phase 4 (local commits
6cb51eb, de5423c, 360ed6b). They contain the hostname and the private IP
address of the server's NIC. Nothing has been pushed. I did not rewrite
history (the spec forbids it). Before any push you need to decide how to
handle them: rewrite with your approval, or publish a sanitised export.

## 3. Measurement validity

* **Energy**: commit-aligned NVML counter (patch v2.1), one region per fresh
  process, start read after warm-up. The end read is taken 8 ms after the
  first commit following the loop end, and the idle tail is subtracted
  using the next commit window.
  * Validated by:
    * linearity in R (intercept +1.6/+3.0 J per 2-GPU region; J per op
      flat in R);
    * warm-up exclusion (≤ 0.7 %);
    * dose test (interpretation supported locally, not an NVML guarantee): counter reads during a region destroy energy on driver
      580 (−27 %), `POWER_INSTANT` reads do not.
  * No counter or `POWER_INSTANT` read touched a GPU with a running
    process, except the one recorded deviation (Phase 5).
* **Configuration**: every confirmatory unit is `VERIFIED` by the NCCL
  profiler plugin on every rank: executed ALGO/PROTO/nChannels and the
  exact count `w + R + 1`.
* **QC (confirmatory)**: 0 `MEASUREMENT_INVALID`, 0 `QC_SHORT`, 0
  `THROTTLED`, 0 retries. Region length 2.79–6.12 s.
* **Residual bias**: the remaining intercept implies ≤ +0.5 % at the chosen
  region length. It cancels in paired ratios to first order and is not
  corrected.

## 4. Counter vs integration (984 confirmatory regions)

| | ∫POWER_INSTANT / counter | ∫PowerUsage (lag 0.75 s) / counter | ∫instant − counter |
|---|---|---|---|
| median | 0.980 | 0.978 | −23.1 J |
| 5th–95th percentile | 0.941–0.991 | 0.955–0.987 | −56.9 to −5.6 J |
| min–max | 0.923–0.996 | 0.951–0.991 | −77.1 to −2.9 J |

* The gap scales with power, not with region length. It matches Phase 4's
  start-of-region offset of about 10 J per GPU.
* RQ1 ratios computed from integrated instant power differ from the
  counter-based ratios by ≤ 0.0075.

## 5. Topology and thermal

* **P4_MIX** (two NVLink pairs over PCIe): automatic is far from optimal.
  * AllReduce 64 KiB–64 MiB with Tree is 8–69 % less energy.
  * At 256–512 KiB the Simple-protocol picks are 1.4–3.2× faster.
* **P2_NVL**: smaller gains, 1.3–22 %.
* **P2_PCI_SHM**: gains at 128–256 KiB, and a reversal at 256 MiB.
* **Thermal**: max 47 °C, no HW thermal or power-brake events; mean
  graphics clock 1906–1946 MHz.
* **Temperature effect**: +0.37 %/°C [0.19, 0.55] on paired energy. Paired
  temperature differences average −0.006 °C, so randomization balances it.
* **Order**: no statistically resolved order association in the tested model (0.0037 [−0.0072, 0.0145] per block, iid). Block-clustered [−0.0042, 0.0116]; wild cluster bootstrap p = 0.35 (v2.2).

Details are in `phase8_results.md`.

## 6. Trial-level (block-level) statistics

Every RQ1 and RQ2 cell has 12 block-paired values, listed in `rq1.csv` and
`rq2.csv` (`per_block_*` columns) and plotted in
`figures/v2/confirmatory/rq1_per_block.png`. Blocks are the units of
replication; there is no pooling across blocks before the paired test. All
12 ran back to back in one ~2 h session on one host: within-session
replicates, not independent days or hosts.

## 7. Supported and unsupported conclusions

**Supported** (confirmatory, 12 blocks; this host only):

* **RQ1**: the frozen Energy-SLO picks save board energy vs NCCL automatic
  in 36/39 cells. The saving is ≥ 5 % in 25 by the pre-registered
  point-estimate rule, and in 21 by a separate threshold test (v2.2).
  Each cell compares the selected pick with automatic only; the global
  optimum is not established.
  * Largest: P4_MIX AllReduce 512 KiB, ratio 0.312 [0.308, 0.317].
  * In all confirmed cells the pick is also faster.
* **RQ2**: in P2_NVL AllGather 2 MiB, LL128 has 2.8 % lower latency than auto (supported; auto is 2.9 % slower).
  * Its energy penalty, +1.2 % [0.3, 2.2] with the counter, is
    **preliminary**: integrated instant power gives +0.5 % [−1.0, 1.9],
    and lag-adjusted PowerUsage gives Holm p 0.069 (v2.2).
  * P2_NVL AllReduce 1 GiB is energy-equivalent within ±3 % under all four
    estimators.
* **Energy tracks latency**: board power changes little across
  configurations (1.03× median spread), so energy follows latency.
* **One reversal**: P2_PCI_SHM AllReduce 256 MiB Tree/Simple uses 1.4 %
  more energy than automatic.
* **Tuner**: the fixed NCCL example tuner reproduces the measured
  executions exactly (36/36 rules, 189/189 executions).
* **End to end (RQ3)**: deploying the policy via the fixed tuner reduces
  GPU energy per training step of a synthetic FP32 MLP on P4_MIX
  (5 blocks):
  * DDP batch 128: 0.791 [0.784, 0.799], with step time 0.706;
  * DDP batch 2048, power-capped: 0.885 [0.880, 0.889];
  * FSDP (64 MiB collectives, no rule) is unchanged at 1.001
    [0.973, 1.030], as predicted.

**Not supported / not claimed**:

* any other GPU, driver, NCCL version, dtype, op, GPU count ≥ 6, or
  multi-node setting;
* the mechanism behind NCCL's P4_MIX misselection (hypothesis only);
* absolute energy accuracy beyond the board sensor;
* recommender behaviour outside the profiled fingerprints, which returns
  `INSUFFICIENT_EVIDENCE` by design;
* workload transfer beyond this synthetic FP32 MLP on P4_MIX: real
  models, mixed precision and other placements were not tested. The
  `no_sync` subtraction does not isolate exposed communication because
  power regime and compute/communication overlap differ; the synthetic
  result is not an established upper bound on training benefits.

## 8. Recommender validation

* **Exploratory (2 blocks) → confirmatory (12 blocks)**:
  * 36/39 picks confirmed;
  * predicted energy ratio median error +0.0003, median absolute error
    0.005, maximum 0.081;
  * log-correlation 0.987;
  * only 26/39 confirmatory CIs contain the exploratory point, so the
    point predictions are overconfident.
* **LOBO (confirmatory, 492 rows)**:
  * regret median 0, p90 0, max 3.4 %, against automatic's median regret
    of 11.5 %;
  * 1 SLO violation;
  * 9/436 recommended picks used more energy than automatic, mostly in
    cells with ≤ 2 % true effect.
* **Scope**: LOBO covers only the 41 pre-selected cells.

Details are in `phase9_recommender.md`.

## 9. End-to-end results (Phase 10)

Details are in `phase10_results.md`; the plan is in `phase10_plan.md`
(amendment A1 made before block 1).

| item | result |
|---|---|
| tuner deployment | verify r3: 36/36 rules executed, 189/189 sizes match the measured executions incl. nChannels; two failed attempts kept |
| tuner overhead | 4 KiB: 0.999 [0.984, 1.015]; 64 KiB: 0.986 [0.906, 1.073] (same executions, 3 reps): not detectable |
| DDP b128 `energy_slo/auto` (primary) | energy 0.791 [0.784, 0.799], time 0.706 [0.696, 0.716]; 64 MiB AllReduce RING/SIMPLE/8 → TREE/SIMPLE/8. The original criterion passed; this is a descriptive synthetic result, not verified equivalent training or real-model transfer |
| FSDP `energy_slo/auto` (primary, negative control) | 1.001 [0.973, 1.030]; identical executions (RING/SIMPLE/8). **Passes** |
| DDP `lat_opt/energy_slo` | 0.995 [0.983, 1.007] (same executions), TOST ±3 % p = 0.002 |
| DDP b2048 `energy_slo/auto` (secondary) | 0.885 [0.880, 0.889], time 0.823; measured at the 300 W power limit (SW power cap 96–98 % of samples) |
| plan's rough magnitude estimate | b128: predicted 0.719, measured 0.791. b2048: predicted 0.916, measured 0.885. The linear estimate fails in both directions (overlap, power regime) |

## 10. Code changes that affect numerical results

| change | when | affects |
|---|---|---|
| patch v2.0 → v2.1 (commit-aligned end read, tail subtraction) | Phase 4, before any exploratory or confirmatory block | all v2 energies |
| NVML v2 memory API for the allocation check | Phase 3 | which runs may start (not values) |
| SW-power-cap QC rule (≥ 90 % of limit) and HW-only health events | Phases 3–4, before the exploratory blocks | QC classification only (no confirmatory unit affected) |
| evidence matching on bytes, re-verified from stored profiler files | Phase 4 smoke | `VERIFIED` status (AllGather) |
| effective-configuration collapse (identical executions merged) | Phase 5 analysis, before pre-registration | exploratory policy choice |
| confirmatory R schedule from exploratory sustained latency, T* = 3.0 s | Phase 6 (frozen) | confirmatory region lengths |
| recommender adjacency rule (no merging across unmeasured sizes) | Phase 9 | tuner files and recommender answers only |
| tuner conf total-byte ranges; example-plugin fixes (channels, cost table) | Phase 10 | tuner behaviour (workload A/B), not RQ1/RQ2 |
| Phase 10 plan amendment A1 (step counts, batch-2048 family) | before workload block 1 | workload region lengths and arms |

No change was made to measurement or analysis code after the first
confirmatory block. The runner, region, QC, evidence, telemetry and
preflight modules are identical in commits 240bc3e and eb0be07, the two
commits under which the confirmatory blocks ran (`block_summary.json`
`git.head`). Neither had uncommitted tracked changes (`diff_sha256` null).

## 11. Server health before and after

| | before v2 (Phase 0/3, 2026-10-08) | after v2 (this report) |
|---|---|---|
| GPU processes | none on any GPU | none on any GPU (04:10 UTC); no experiment process left |
| GPUs 4–7 HW thermal / HW power brake counters | 0 | 0 |
| idle power GPUs 4–7 (P8, 30 s counter baseline) | 20.6–28.7 W (confirmatory block baselines, 01:21–03:28) | 20.8–28.3 W, P8, 28–31 °C |
| remapped rows | 0 | 0 |
| disk free (`/md127`) | 63 GB | 63 GB |

No experiment process was left running. No other user's process ever
appeared on GPUs 4–7 during a run.

The SW-power-cap and SW-thermal cumulative counters on GPUs 4–7 rose
between Phase 3 and now. Values in seconds since boot, Phase 3 → now:

| GPU | SW power cap | SW thermal |
|---|---|---|
| 4 | ≈ 31,000 → 38,760 | 7 → 5,375 |
| 5 | ≈ 1,450–1,800 → 5,262 | 0 → 0 |
| 6 | ≈ 31,000 → 40,117 | 150 → 5,948 |
| 7 | ≈ 1,450–1,800 → 7,412 | 0 → 0 |

Phase 4 showed both counters ticking at pure idle on these boards. The
power-capped workload arms add real cap time on top. Per-sample reasons
during regions never showed SW thermal.

## 12. Permission limitations and unknowns

* **Kernel log** (Xid, PCIe, NVLink messages): `UNAVAILABLE_DUE_TO_PERMISSIONS`
  (`kernel.dmesg_restrict=1`). NVML Xid event registration was used
  instead and polled after every unit.
* **DCGM**: not installed. No scheduler.
* **Process owners**: other users' process owners are shown only when
  `/proc` permits.
* **Not done, by policy**: changes to clocks, power limits, persistence
  mode, MIG/ECC or driver state. All results are at the default
  application clocks and the 300 W limit.
* **Unknown**: CPU, DRAM and PCIe-switch energy (board energy only); the
  cause of the SW-thermal / SW-power-cap counter growth at idle on GPUs 4
  and 6.

## 13. Replication

* **This host**: re-run with the frozen inputs. Build the patched
  nccl-tests and the two plugins first, as in
  `docs/v2/replication_1080ti_titan.md` §3, with
  `-gencode=arch=compute_86,code=sm_86` for the A6000. Also build the
  tuner plugin with `make -C native/nccl_tuner_example NCCL_SRC=<NCCL
  2.27.5 source>`.
  * Confirmatory blocks (sequential, GPUs 4–7, stops at the first
    incomplete block):

    ```bash
    bash scripts/run_confirmatory_v2.sh
    ```

  * Pre-registered analysis (RQ1/RQ2):

    ```bash
    PYTHONPATH=src python -m ncclenergy.v2.analyze_cli confirmatory --campaign data/v2/raw/confirmatory --prereg docs/v2/prereg.json --out data/v2/processed/confirmatory
    ```

  * Robustness, LOBO and tuner files:

    ```bash
    PYTHONPATH=src python -m ncclenergy.v2.phase89 --campaign data/v2/raw/confirmatory --prereg docs/v2/prereg.json --processed data/v2/processed/confirmatory --figures figures/v2/confirmatory
    ```

  * Tuner check (GPUs 4–7; `--tag` must be a new directory name):

    ```bash
    PYTHONPATH=src python -m ncclenergy.v2.phase10 verify-tuner --policy-dir data/v2/processed/confirmatory --tag verify_tuner_rerun
    ```

  * Phase 10 analysis of the stored runs (no GPU):

    ```bash
    PYTHONPATH=src python -m ncclenergy.v2.phase10 analyze --policy-dir data/v2/processed/confirmatory
    ```

* **Other hardware**: `docs/v2/replication_1080ti_titan.md`. It prepares,
  but does not run, a replication on 1080 Ti / Titan GPUs, with separate
  storage and no pooling.

## 14. Recommended next experiment

1. **No GPU needed for a probe-based diagnostic**: compare the recorded
   initial TUNING probe with measured P4_MIX times. Timed processes used
   INIT,ENV logging and do not contain every process's cost table. Actual
   executed algorithm/protocol is verified separately by the profiler.
   A probe/measured discrepancy suggests a hypothesis; it does not prove
   which internal model caused a particular timed choice.
2. **GPU, pre-registered**: real-model transfer on P4_MIX, for example a
   GPT-2-class model with DDP, BF16 gradients and default 25 MiB buckets.
   * First, a short confirmatory microbenchmark of BF16 AllReduce at the
     bucket sizes. Only FP32 was profiled, and the recommender correctly
     refuses other dtypes.
   * Then an auto vs Energy-SLO A/B with ≥ 8 blocks, reporting
     tokens/J and step time.
   * This is a separate test of practical value. The synthetic MLP result
     is not a demonstrated upper bound. Choose replication from a
     pre-specified precision/power design, not an arbitrary block count.
3. **Version sensitivity**: repeat the 17 P4_MIX RQ1 cells with a newer
   NCCL, user-space only, on the same GPUs. That tests whether the
   largest effect survives a library update before anyone deploys the
   tuner file.
