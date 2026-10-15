# Change ledger: smoke-1 CPU corrections

Written before any edit at 2026-10-13T02:42Z. The "Actual changes" section is filled in after the edits.

## Provenance

* Reviewed baseline: `2b73580e848986905c45bc8ee10706383dc5cd72` (GitHub `extension-pilot-reviewed`). HEAD is unchanged; nothing is committed or pushed.
* Starting state is the cg1 implementation, uncommitted on top of `2b73580`. The 34 worktree files are byte-identical to `delivery/MANIFEST.sha256` (`repo_worktree_changes/`); verified 2026-10-13, 0 mismatches.
* Pre-edit snapshot is in `baseline_before_edits/`:
  * `worktree_snapshot.tar`, sha256 `88cff109ee26b3c6f9ce08201626b37242c1eb08a52ee705f4728a9e60683c51`;
  * `worktree_sha256.txt`, one line per file;
  * `HEAD.txt`.
* Authorization (2026-10-13, user message):
  * the CPU corrections A, B and C;
  * a three-launch software smoke, only after the user explicitly confirms the allocated GPU pair and the execution window. That confirmation has **not** been given at the time of writing.
  * It is not approval of calibration, held-out collection, publication or scientific conclusions.
* Untouched: historical raw data, frozen outputs, historical protocols, pre-registration and published claims. The cg1 packet and the proposed manifests stay as delivered; new documents are written next to them.

## Planned changes and what they affect

| id | change | affects |
|---|---|---|
| A | S5 is reconciled with the implemented L3 dose rule. Eligible completed reads are kept apart from straddling and shutdown reads, which are logged separately and never delivered. An in-flight read at stop is not a failure. Active pollers must satisfy work_end ≤ stop_request ≤ join_end. An inactive poller (E0) is not judged on stop/join times. The 95 % threshold is unchanged. The smoke criterion text is updated | **Future acceptance** only: S5 and dose QC. It does not touch historical data |
| B | Guard on the primary period estimate. If the least-squares P_hat lies outside the hard feasible period interval, `aligned_hat` is unavailable with an explicit reason. The estimate is not clipped. Raw-counter and nominal-period results are retained | **Availability of the future aligned-energy estimator** (E_aligned(P_hat)) |
| C | Baseline raw evidence. Per GPU, save the e0/e1 counter values, return statuses, read-call start/end times, GPU identity, clock domain and units. First-read evidence is kept if the baseline later fails. The existing reads are reused. The formula and safety gates are unchanged | **Provenance only**. It must add **no** NVML calls (checked by the counter-call counts in the CPU tests) |
| F | Section 3 finalization requirements, implemented only where the code did not already satisfy them: (1) smoke execution deadline = min(reservation end, smoke start + 30 min); (2) refuse a non-empty `LD_PRELOAD` and record the effective loader/NCCL environment | Future smoke execution and provenance. They are listed separately because they come from section 3, not from items A–C |

## Actual changes

Recorded 2026-10-13. The unified diff against the pre-edit snapshot is `smoke1_changes.diff` (sha256 `a45d705e…622b25`).

| file | before (sha256) | after (sha256) |
|---|---|---|
| `src/ncclenergy/v2/stage2_calibration.py` | `46be8db763d5f606d4aa14191391cdfb4949293058be8b6583b83888de78ffdf` | `b88ad3c110800293718a642c2351c1c04389edd43bab0bd0efe91db0a7803124` |
| `src/ncclenergy/v2/stage2_energy.py` | `9365d9379d7e2a4d62c576e1608ac975b541c829205fd4c297c0b57e5dc972d6` | `d3317beab273fd475cc97ff4e0f97b7cb6d47208386571e5ce8798a568c61c00` |
| `src/ncclenergy/v2/stage2_phase.py` | `74688deed8994f8ad9f8b00b8d71143670a267e613938d64ad4ba4e62c2d8ab6` | `b399bf6cbe9de7ac92132983f176535cbbca23ecd150f30b84084bf961d837ed` |
| `src/ncclenergy/v2/stage2_runner.py` | `3d60404afcf8a521cc83d46fc5e5c635b0f4fc85f867d1debf4372a2e4c503e2` | `9f6d2536ac6ed1a3d4f726232dcc984df3036541f65cbddf0182f79d707d142a` |
| `tests/v2/helpers/stage2_session.py` | `36fca45ab9a082dfd61966fb9622434e76482000b23067958190e096930abe30` | `0dd94b40c4655c10757147a7f821fab804f1c79e5f9a2c7279c274c54111c481` |
| `tests/v2/test_stage2_cg1_regressions.py` | `b0cdf84f2cf3eae323be4a99048ec16bb90c460f24096575127088251b28a91e` | `b0ba9af407a5aa3d1214577185bd7c0958c505ad27d7bd6a797b1f523f695e7f` |
| `tests/v2/test_stage2_spec.py` | `2787db4d9e3ece909249610b187644621f0aa19519c8b13c483758ecf54c6b0b` | `21f4f8279af7b9cc4a5334a76181a8b5b3169b2ffa9327fd15a38f5fbc470894` |
| `tests/v2/test_stage2_smoke1_regressions.py` | (new) | `8df2113f56b6c2f491b4a95daa7c9a02e081eb9aafd0e0534cf71483bf75985d` |

**A (S5), in `stage2_energy.dose_summary` / `s2_metrics` and `stage2_runner`:**
* `dose_summary` now:
  * requires `joined_ns` (`DOSE_TIMES_MISSING` otherwise);
  * adds `DOSE_JOINED_BEFORE_STOP` and an `order` record (work_end ≤ stop ≤ join);
  * lists straddling and shutdown reads separately (`straddling_detail`, `shutdown_detail`) and adds `eligible_completed`;
  * labels `inflight_at_stop` as logged, not a failure.
* The delivery rule, its 10 ms / 20 ms constants and the 95 % threshold are unchanged.
* `s2_metrics`:
  * marks active pollers;
  * gives inactive pollers (E0) a summary with no stop/join requirement;
  * flags poller times without a dose request (`DOSE_POLLER_TIMES_WITHOUT_DOSE_REQUEST`).
* `stage2_runner.smoke_s5` is new. `smoke_problems` adds S5 and stores `rec["smoke_checks"]` per launch.
* The S5 text is in the new `protocol/STAGE2_SMOKE1_PROTOCOL.md` (sha256 `514f1279c43e98885dc6d2ba8bce907f484ae4a691ff43faa886edb15bac9401` at writing; Freeze-S binds the final hash). `STAGE2_CG1_PACKET.md` is unchanged.

**B (period guard), in `stage2_phase` / `stage2_calibration`:**
* `estimate_period_phase` adds `primary_period`: COMPATIBLE or INCOMPATIBLE_WITH_FEASIBLE_MODEL against the hard LP interval, with a 1 ns tolerance. `period_ns` is still reported unclipped.
* `aligned_hat_period()` is new.
* `unit_energies` sets `aligned_hat = None` plus `aligned_hat_unavailable_reason` and a unit-level reason. `raw_J` and `aligned_nom` are always computed.
* `end_alignment_timing` records any nominal-period denominator substitution (`denominator_nominal_period_substitutions`) instead of falling back silently.

**C (baseline evidence), in `stage2_runner.session_baseline`:**
* The new `_read_evidence` and `counter_read_evidence` store the e0/e1 [index, value mJ, status, call start ns, call end ns] immediately after each existing read, plus GPU index/UUID/PCI, clock info (`perf_counter` = CLOCK_MONOTONIC) and units.
* This evidence is written to `baseline.json` by the existing `finally` block, so a later failure keeps it.
* No new NVML call: the counter-call counts in the tests are unchanged (8 per calibration session; 2 when the baseline fails after the first read).

**F (section 3), in `stage2_runner`:**
* `SMOKE_CEILING_S = 1800` and `execution_budget()`: the smoke deadline is min(reservation end, start + 30 min), recorded in `summary["deadline"]`.
* A non-empty `LD_PRELOAD` is refused in `check_config` and in `stage2_env`.
* Recorded per launch: `loader_env` (`LD_PRELOAD`, `LD_LIBRARY_PATH`). Recorded per session: `loader_evidence()` (`ldd` of the binary under the Stage 2 library path).

**Test changes:**
* New `test_stage2_smoke1_regressions.py` (12 tests).
* `stage2_session.py` gains the scenarios `smoke` (the frozen 3-launch schedule from `smoke01_DRAFT.yaml`) and `baseline_fail_after_first_read`, plus evidence outputs.
* Fixture fix: `test_stage2_spec._region_and_s2` E0 poller times are now −1, as the native v2.5 records them.
* `test_stage2_cg1_regressions::test_D2…`: its synthetic records now carry the inactive-poller dose summary that real records always have. The original pause and post-probe assertions are unchanged.

**Not changed:**
* native patch, binary, plugin, build record and libnccl (hashes identical to cg1);
* configs and drafts;
* historical data, frozen outputs, protocols, pre-registration and claims;
* `IMPLEMENTATION_ID` / `METHOD_ID`. Freeze-S binds the code hashes; the measurement method is unchanged.

## Test results

* Before the edits, the affected files had 72/72 passing (`affected_tests_before.log`).
* The first run after the edits had 83/84 passing (`affected_tests_after.log`). The one failure was `test_D2…`: its synthetic record had no dose summary, and the new S5 check reported a misleading reason. Fixed as listed above.
* Final full suite with `tests/run_tests.py`: **206/206 passed**, exit 0. That is the 194 tests of cg1 plus 12 new ones (`full_suite_smoke1.txt`, sha256 `d86a8549…f84a41`). The git worktree now has 31 status entries: the 30 before plus the new test file.

## Correction G (written before editing, 2026-10-13)

**Approval.** The user's reply, 2026-10-13: "Approve fix G, choose an idle pair and window again." G is the proposal in `SMOKE1_STOP_REPORT.md` §4.

**Change.** In `stage2_readiness._row_problems`, the idle-gate row rule used by:
* the baseline P8 entry/validity;
* R1 readiness;
* the readiness-derivation replay.

On rows whose P-state is P8, the clock-event bits 0x1 (GPU idle) and 0x4 (SW power cap) are tolerated. Any other bit still fails, and the raw bitmask is kept. Each baseline and wait also reports the share of rows carrying each bit.

**Unchanged:**
* rows that are not P8 (they fail as NOT_P8; their clock-event check stays exact);
* in-region QC (masks 0x1 only);
* held-out pre-region check R2;
* all other gates.

**Affects:** future acceptance by the idle gates only: S3 baselines, held-out readiness and readiness derivation. No historical data is touched.

**Pre-edit hashes:**
* `src/ncclenergy/v2/stage2_readiness.py`: `stage2_readiness_pre_G` (see the line below)
* recorded: d6a3d5b98cb4a7c78a173d584f9711a93a1432333bc3b9c2531bd995cd5105a2 src/ncclenergy/v2/stage2_readiness.py (pre-G)

**G, actual changes:**
* `stage2_readiness`:
  * `P8_TOLERATED_CLOCK_EVENTS = 0x1 | 0x4`;
  * `_row_problems` tolerates those bits only when `pstate == 8`; the reason is renamed `CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING`;
  * new `clock_event_bit_shares()`.
  * Diff: `G_changes_readiness.diff`.
* `stage2_runner`: records `clock_event_bit_shares` in each baseline (`baseline.json`), fixed wait and held-out live wait. Recording only; no decision changed.
* Tests:
  * 3 new G tests in `test_stage2_smoke1_regressions.py`;
  * `stage2_session.py` gains the scenarios `idle_ce_gpuidle` (0x1), `idle_ce_swpc` (0x4) and `idle_ce_thermal` (0x40), plus `baseline_bit_shares` and `unit_qc_reasons` outputs.
* Protocol: §3 of `STAGE2_SMOKE1_PROTOCOL.md` documents G.
* Post-G hashes: `G_after_hashes.txt`.
* The smoke-1 regression file has 15/15 passing. Full-suite result: `full_suite_smoke1_G.txt`.
