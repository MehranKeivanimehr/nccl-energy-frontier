# Final review report — NCCL Energy Frontier v2.2 pilot release (2026-10-09)

> **Historical producer report; corrected candidate notice (2026-10-11).**
> Commit/test statements below describe the supplied producer snapshot, not
> new work or independently verified current test passes. Subsequent review
> identified remaining presentation/forward-execution issues; see
> [corrections_2026-10-11.md](corrections_2026-10-11.md). No Git history is
> present in this sanitized candidate; do not infer a current local branch.

**What this is**:
* The extension is finished as a bounded pilot characterization.
* Nothing in this report required, or used, a new GPU experiment.
* All work is on local branch `ext/v2.2-audit`. Nothing is pushed,
  merged, tagged or published.

## 1. Outcome

* **Main message**:
  * Tested settings can substantially outperform NCCL's automatic choice
    on this measured deployment.
  * Transfer to a synthetic training workload shows promising
    improvements, with stated measurement and correctness limitations.
  * The small latency-versus-energy optimum difference (C7) was **not
    resolved** by the v2.2 measurement-validation campaign.
  * This is a pilot study, not a general deployment claim or a completed
    proof of a latency–energy tradeoff.
* **Gate 3b** is complete as pre-specified: 6 blocks, 72 processes, 70
  accepted, 2 excluded and not rerun. The decisions were D1 INCONCLUSIVE,
  D2 INCONCLUSIVE, D3 SYSTEMATIC_OFFSET and D4 NOT_SUFFICIENT. C7 stays
  preliminary and unresolved.
* **Reproduction**: every published analysis reproduces from the retained
  evidence, both in the repository and in the sanitized export (§3). No
  frozen output changed.

## 2. Files and code changed in this release step

These commits follow the gate 3b result (`042625d`).

| commit | content |
|---|---|
| `f3ca565` | gate 3b report corrected (association language; D3 competing explanations; property distinctions; one number corrected); post-hoc matched-window diagnostic; analysis-amendment record; frozen copies of the collection-time analysis and campaign script; AM-1 and AM-2 checks |
| `3b499b5` | `procctl`: fail-closed cleanup; thread-safe ownership set |
| `4975780` | fail-closed QC, process-list, meter, Phase 10 meter-QC, workload numerics (`--data per_rank`) and guarded-tuner rule parsing |
| `8500116` | synthetic GPU UUID in a test (it was a real UUID of this host) |
| `261f147` | README rewrite (pilot section moved verbatim to `docs/pilot_v1/`); `docs/release/*`; `release_repro`, `release_tables`, the open-audit hook, the export builder; v2.2 manifest; release tables; two D2 roundings and the start-phase evidence in the gate 3b report |
| `f39c3ac` | `release_tables` added as a reproduction step |
| `415ef08` | tests create the git-ignored `logs/` directory (found by testing the export) |
| (this commit) | export builder writes a `.gitignore` that tracks the exported evidence; this report |

**Totals**:
* 24 code and test files changed: +1,628 / −71 lines.
* New modules:
  * `release_repro.py`, `release_tables.py`, `amendment_check.py`;
  * `scripts/build_release_export.py`, `scripts/open_audit/sitecustomize.py`.
* Code behaviour changed only for future runs. No collected data and no
  published output changed (§3).

**Post-collection fixes of the gate 3b analysis**: see
[`../v2/v22_audit/analysis_amendments.md`](../v2/v22_audit/analysis_amendments.md).

* **AM-1** (`0be9c1d`, `df.flags` → `df["flags"]`):
  * The frozen pre-collection analysis (`configs/v2.2/frozen_code/measval_f4735b1.py`,
    git blob `fb378505`) and the current analysis give identical
    decisions, numbers and processes table on the same raw data.
  * The frozen `descriptive()` fails with `KeyError: 0`.
  * Check: `amendment_check`.
* **AM-2** (`4438d07`, campaign-script exit status):
  * The frozen script (blob `20c7e532`) and the fixed script start the
    runner identically; a test with a recording stub shows this.
  * The raw data were committed unmodified (`e963e12`) before either fix.
    No raw file changed afterwards.

## 3. Test and reproduction results

| check | result |
|---|---|
| CPU test suite (repository) | **108/108 passed** (`python tests/run_tests.py`); fake NVML, mocks, gloo, C harness |
| regression tests for this release's fixes | each fails on the previous code and passes now (procctl 4, QC/telemetry/meters/watchdog 5, runner end-to-end 1, numerics 1, tuner parsing 1, Phase 10 meter-QC 1, AM-2 1) |
| release reproduction (repository, audited) | **14/14 steps PASS**; 145 frozen files hashed before and after, 0 changed; 0 reads outside the repository |
| release tables | 65 key numbers recomputed and found verbatim in README (17/17), claims (29/29) and the gate 3b report (45/45). Counts consistent: gate 3b 72/70/2 with 0 retries; v2 confirmatory 984/984; gate 3a counter reads inside the strict region window 0; Phase 10 meter plausibility 200/200 |
| historical provenance | pilot manifest 27,491 files and v2 manifest 24,541 files re-verified, **0 failures** |
| sanitized export: test suite | **108/108 passed** in a fresh extraction of the archive. The first export build exposed 5 failures (tests needed the git-ignored `logs/` directory), fixed in `415ef08` |
| sanitized export: reproduction (audited) | **14/14 steps PASS** in a fresh extraction of the archive; 145 frozen files unchanged; **0 reads outside the export**. One extra status, `EQUAL_EXCEPT_REPO_PATH`, for a reference file whose recorded path was sanitized to `<REPO>` |

**Per-file reproduction statuses**:
* All `IDENTICAL`, except the following, which are named and explained:
  * 3 gzip files `NUMERIC_EQUAL` (header timestamp only; maximum difference
    0.0);
  * the pilot `results_summary.md`, `IDENTICAL_EXCEPT_SOURCE_PATH`;
  * `placement_meta.json`, `EQUAL_EXCEPT_REPO_PATH`;
  * `figures/full/method_example_trace.png`, `KNOWN_UNREPRODUCED`: an
    illustrative figure made once for an unrecorded launch; no number
    depends on it.
* `phase89` and `phase10` are gated against the Stage 2 outputs of the
  same (v2.2) code. Their designed differences from the frozen v2.1
  outputs are listed, not hidden (ledger L21, L22).

**Untested**: GPU-dependent paths are listed in
[`limitations.md`](limitations.md) §5. The CPU tests do not validate real
CUDA, NVML or NCCL behaviour.

## 4. Claim-status table (summary; full table with sources in [`claims.md`](claims.md))

| # | claim | status |
|---|---|---|
| P1–P2 | pilot: latency-optimal = energy-optimal in 123 of 126 cells; 3 small disagreements | superseded |
| C1 | the pick uses less board energy than automatic in 36 of 39 cells | supported |
| C2 | ≥ 5 % saving: 25 cells (point estimate) / 21 cells (threshold test) | supported (21) / preliminary (4) |
| C3 | P4_MIX AllReduce 512 KiB ratio 0.312 [0.308, 0.317] | supported |
| C6 | AllGather 2 MiB LL128 2.8 % lower latency than auto | supported (denominator corrected) |
| C7 | AllGather 2 MiB LL128 ≈ 1 % more energy | **preliminary, unresolved** |
| C8 | AllReduce 1 GiB Tree energy-equivalent within ±3 % | supported (one sensor) |
| C10, C11 | order association / temperature association 0.37 %/°C | inconclusive / supported (association) |
| C13 | tuner reproduces measured executions 36/36, 189/189 | supported |
| C14–C16 | synthetic DDP 0.791 [0.784, 0.799] (step time 0.706); b2048 0.885 [0.880, 0.889]; FSDP 1.001 [0.973, 1.030] | descriptive board-energy observations; training equivalence unverified |
| C17–C19 | `no_sync` = communication energy; real-model, BF16, whole-system benefits; general tuner deployment | unsupported / not assessed / unsupported |
| M1–M8 | v2.2: pipeline works; no in-region counter reads; gate 3b as pre-specified; identical execution established; agreement within ±0.5 % not established; absolute accuracy not assessed; ≈ 1 % not resolvable | as labeled in `claims.md` |
| H1–H3 | post hoc: temperature association 0.43 [0.38, 0.48] %/°C; unexplained per-region estimator gap about 13 J per GPU; temperature-adjusted C ÷ A about +1.2 % | exploratory; changes no claim |

## 5. Data inventory and checksums

The full table is in [`data_inventory.md`](data_inventory.md).

**Manifests**:

| data | files | manifest (SHA-256 of the manifest) |
|---|---|---|
| pilot | 27,491 | `provenance/pilot_v1_sha256_2026-10-08.txt` (`fba0598e…`) |
| v2 | 24,541 | `provenance/v2_raw_sha256_2026-10-09.txt` (`aeff3e7c…`) |
| v2.2 | 823 | `provenance/v22_sha256_2026-10-09.txt` (`83486f67…`) |

**Local data**:
* 52,881 files and 193 MB in the data, figures and results trees.
* All preserved unmodified.

**Export**:
* The original export manifest lists 13,302 files, of which 3,178 are sanitized derivatives. Every
  derivative is listed in `DERIVATIVES.csv` with the rules applied and
  its original and export SHA-256.
* 12 files omitted for privacy.
* 39,747 on-disk raw files not exported, because no published analysis
  reads them. They are listed per directory in `OMITTED.md`.

## 6. Numerical-change ledger

[`change_ledger.md`](change_ledger.md) has 25 rows (CSV alongside). The
corrections to earlier reports:

| row | change | why |
|---|---|---|
| L11 | gate 3b D2 instant intercept: +0.03 → +0.02 % | hand rounding |
| L12 | D2 LL128 lower bound: −0.49 → −0.48 % | hand rounding; the decision is unchanged |
| L13 | per-pair estimator agreement: "within 0.1 %" → 0.05–1.16 pp by length and estimator | overstated |
| L14–L16 | wording: counter, temperature, D3 | — |
| L17–L18 | new matched-window numbers for the estimator gap | — |

Stage 2 entries: L01–L09. Gate 3a reinterpretation: L10.

## 7. Remaining untested paths and limitations

See [`limitations.md`](limitations.md). The most consequential:

| limitation | detail |
|---|---|
| scope | one host and one session per campaign (within-session replicates); board energy only; no external reference |
| C7 | unresolved |
| synthetic workload | replicated data, numerics not recorded, unguarded example tuner |
| never run on real hardware | the guarded tuner inside NCCL; the v2.2 workload and meters on CUDA; the SIGKILL path with a GPU process; this release's code fixes on GPU paths |

## 8. Private-information handling

**Local repository**:
* The original scientific data are unchanged and kept locally. That
  includes files containing the host name, private IPs, the user's paths,
  GPU UUIDs and serial numbers.
* Commits in this release used a gated private-information scan of the
  staged files, which aborts on any unreviewed hit. Two earlier commits
  (smoke-02, and `4975780` in this step) ran despite scan hits. In
  `4975780` the hit was a pre-existing real GPU UUID in a test, already in
  history since `cc504be`. It was replaced in `8500116`.
* **History is not clean.** Local history (38 commits, from the first
  commit `f39af08` onward) contains, in committed files:
  * the host name, private IPv4 addresses (NCCL logs), the user's
    absolute paths and GPU UUIDs and serial numbers (pilot metadata);
  * the author's email address in every commit.

  History was not rewritten.
* **Do not push this branch or any of its commits to a public
  remote.** Do not create patches from them.

**Sanitized export** (`../release_exports/`, outside the repository):
* No `.git`, no logs, no credentials.
* Rules:
  * repository and home paths → `<REPO>` / `<HOME>`;
  * user and host names → `<user>` / `<host>`;
  * private IPv4 → `<ip>`;
  * GPU UUIDs → `GPU-<uuid>`;
  * serial numbers → `<serial>`.
* The private tokens are taken from the environment at build time, so the
  builder script contains none of them.
* Scientific values are unchanged; the reproduction inside the export
  confirms it.
* Omitted:
  * `nvidia-smi -q` dumps (serials and identifiers);
  * process lists that include another user's process;
  * unused per-process raw files.
* The post-build scan of every exported file, including inside `.gz`,
  found **0 hits**.
* Remaining personal information, by design:
  * the author's name in `LICENSE` (MIT copyright);
  * the GPU model, driver and NCCL versions, and PCI bus IDs, which are
    needed scientifically;
  * provenance manifests listing original files with their original
    SHA-256 (paths sanitized).

## 9. Integrating the sanitized extension into the existing public repository

Current corrected-candidate instructions are in
[integration_review.md](integration_review.md). The commands below describe
the historical producer export, not the revised candidate package. Do not reuse
the fixed temporary directory or private branch for current integration.

The public repository's location is not recorded in this local
repository (no git remote). The steps assume a clone of it at
`$PUBLIC` whose layout follows this project's.

Use only the sanitized export, never the local history.

1. **Verify the archive**:

   ```bash
   cd ../release_exports && sha256sum -c nccl-energy-frontier_v2.2-pilot-release_2026-10-09.tar.gz.sha256
   ```

2. **Extract** into an empty directory and verify the file checksums:

   ```bash
   mkdir -p /tmp/nef_release && tar -xzf nccl-energy-frontier_v2.2-pilot-release_2026-10-09.tar.gz -C /tmp/nef_release
   ```

   ```bash
   cd /tmp/nef_release/nccl-energy-frontier_v2.2-pilot-release_2026-10-09 && sha256sum -c --quiet SHA256SUMS && echo OK
   ```

3. **Re-run the checks in the extracted tree**: the test suite and the
   audited reproduction (expected: all PASS, `outside_reads.tsv` empty).
   Then remove the generated `logs/`, `native/*/build/` and
   `__pycache__/`:

   ```bash
   python tests/run_tests.py && PYTHONPATH=src python -m ncclenergy.v2.release_repro --audit --out logs/release_repro/check
   ```

4. **In the public clone, on a new branch**:

   ```bash
   cd $PUBLIC && git switch -c v2.2-pilot-release
   ```

5. **Copy the tree over the public files.** Do not use `--delete`; review
   deletions by hand. The export carries the old README's pilot section as
   `docs/pilot_v1/README_pilot_v1.md`.

   ```bash
   rsync -a --exclude logs/ --exclude '__pycache__/' --exclude 'native/*/build/' /tmp/nef_release/nccl-energy-frontier_v2.2-pilot-release_2026-10-09/ $PUBLIC/
   ```

6. **Review, scan and commit**:
   * Review `git status` and `git diff --stat`.
   * Re-run your own private-information scan on the staged files.
   * Commit **one** new commit authored as you choose, with the release
     description below.
   * The export's `.gitignore` already drops the original rules that would
     ignore exported evidence; they are listed in its header. Check that
     `git status --ignored` shows no exported file as ignored.
7. **Push and publish** only after your own review. These steps were not
   done here.

## 10. Proposed release description

> **NCCL Energy Frontier — v2.2 pilot release.**
>
> * A pilot characterization of NCCL collective latency and GPU-board
>   energy on one 8× RTX A6000 server (NCCL 2.27.5).
> * Pre-registered, configuration-verified pick-vs-automatic comparisons
>   (12 within-session blocks) show that tested settings can substantially
>   outperform NCCL's automatic choice on this deployment: 36 of 39 cells
>   lower board energy; up to 0.312× on 4-GPU AllReduce at 512 KiB.
> * A synthetic DDP workload with a tuner-deployed policy used 0.79× the
>   board energy per step, with stated limitations.
> * A pre-specified measurement-validation campaign did not resolve the
>   small (≈ 1 %) latency-vs-energy optimum difference on AllGather 2 MiB.
>   That finding stays preliminary.
> * Includes:
>   * raw evidence for every published analysis (sanitized);
>   * a one-command CPU reproduction with an input audit;
>   * a claim-status table, a change ledger and limitations.
> * Not a general deployment claim. GPU-board energy only. No external
>   power reference.

## 11. Optional future experiments (not required for this release)

1. **Thermally controlled measurement validation**: a temperature band or
   temperature-matched interleaving, with a pre-specified temperature
   covariate.
2. **Per-region counter-vs-sampled gap**: varied start phase, pre- and
   post-roll windows, idle bracketing.
3. **External power reference**: needs hardware or administrator action.
4. **Other days, hosts, GPU models and drivers**: re-run the Phase 4
   POWER_INSTANT dose test first on any other driver.
5. **Real-model transfer**: BF16 profiling, then a DDP real model with
   per-rank data and recorded vectors; a separate precision/power plan determines
   the repetition count, not an arbitrary minimum.
6. **A newer NCCL version** on the P4_MIX RQ1 cells.
7. **The guarded tuner inside real NCCL**, within the controlled harness.
