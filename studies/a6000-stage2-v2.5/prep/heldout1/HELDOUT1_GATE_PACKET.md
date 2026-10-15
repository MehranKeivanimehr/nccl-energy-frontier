# Held-out gate packet (heldout-1), 2026-10-13

## Status (results first)

* **Done, CPU only.** No GPU was used, no NVML call was made, no held-out data exist, nothing was committed, pushed or published, and the GitHub-cleanup prompt was not run.
* **Part 1:** `prep/calib1/CALIB1_REPORT_r1.md` (sha256 `5256c8fb…`). It contains the correction ledger and leaves the calibration outcomes unchanged. The original report is preserved (`bcecb960…`).
* **Part 2:** `stage2_heldout.py` is implemented as offline analysis.
  * New tests: **20/20 passed**.
  * Full CPU suite: **228/229 passed; 1 failed (`v2/test_stage2_identity_gate.py::test_cli_check_refuses_the_drafts`)**. The one failure is a pre-existing stale test, not caused by this change (§8).
* **Freeze chain intact:**
  * `verify_freeze(Freeze-C)` reports no problems.
  * Each of the three proposed Freeze-H manifests (one per session, parent Freeze-C) is refused **only** for the missing user approval. The shared roles are identical to Freeze-C.
  * No freeze record was written and no old freeze hash was touched.
* **What the held-out result can be at best:** a conditional comparison of NVML-reported board energy on this pair and host (pilot).
  * Physical adequacy is **NOT_ESTABLISHED by construction**: no systematic hard-bound rule is frozen, and the calibration Q1 primary is not bounded.

## 1. What I ask you for (only this)

**A. Approve this gate.** Approval means:
1. `stage2_heldout.py` (sha256 `4826f195a9c59dfe92660bc37c2bfae52f725751607ac4beb9c12b87daa89c7d`) with the rules in `CHANGE_LEDGER.md` (N1–N18, N11a) is the held-out analysis;
2. `protocol/STAGE2_HELDOUT1_PROTOCOL.md` (`759e2f62f4511f1b462b17faff3f454dcf6e3ce8e25f0617b7c6e955d769aced`) is the held-out protocol;
3. the calibration-derived readiness settings in §5 are used unchanged;
4. one Freeze-H record is created per session, after your authorization of that session, with parent Freeze-C and the shared roles unchanged;
5. you accept the items in §7 that are **not** implemented. Among them is the K2 historical D1–D4 replay.

**B. Fresh authorization, once per session (three windows).** For each session I need:
* **Pair:** NVML GPUs 4 and 5, the physical pair of smoke-1 and calibration (UUID aliases A6000-b10caa4a / A6000-f5f81ac3).
  * The readiness bands and T_launch were derived on this pair, so no other pair can be substituted under this gate.
  * Before any launch, a read-only scope read (no counter reads) must match the scope recorded at calibration.
* **Window:** an absolute start and a fixed absolute end, with time zone.
  * Proposed length: **120 min** per session (the protocol ceiling).
  * **Minimum 116 min.** That is enough to complete every schedule even if every readiness wait reaches its 240 s cap (§6).
* **Scope:** exactly the 24 launches of that session's frozen schedule, once. No retries, replacements or optional stopping.

An idle GPU is not an allocation, and I do not assume the earlier GPU 4/5 approvals still hold.

## 2. Change ledger

`prep/heldout1/CHANGE_LEDGER.md` (sha256 `14d1eb42431c9c18a99e71a2311fd62e6b5cd68998a06ac5fe30190a0e66160a`) was written before any code. It lists every numerical-result-affecting choice where the protocol was silent:
* N1–N15: the base rules;
* N16–N18: offline re-verification of acceptance, non-positive energies, and pair/scope guards. Recorded before the module was first executed;
* N11a: the R3 override applies only when R3 was evaluated and failed. A missing arm or session is missing data. Found by the first test run and recorded before the code change.

The ledger also lists every change made during implementation:
* the H-D2 key fix;
* session-mean heterogeneity kept in the output;
* recursive unit hashing;
* UUID aliasing in the scope comparison;
* `scope_expected` required.

The aliasing fix matters on real data. Configs carry raw UUIDs and session summaries carry the runner's aliases, so without it every real session would have been refused.

## 3. Tests (`code/tests/v2/test_stage2_heldout.py`, sha256 `0b652a16eab3c0d1768932cbfe6157fcab27be46154330c1d9e497f9d0fe55c5`)

| requirement | test(s) |
|---|---|
| known answers | `test_known_answers_HD3_…` checks session means, t, 95 %/90 % CIs, two-sided p, Holm, TOST and verdicts against the closed-form df = 2 t distribution and the textbook quantiles 4.3027/2.9200, not against scipy. `test_known_answers_HD1_HD2_HD4` covers 9-member Holm, exact OLS recovery of the protocol term, and H-D4 precision. `test_end_to_end_…` reproduces the same answers through session directories |
| three-session requirements | `test_three_sessions_are_required_…`: 2 sessions → INSUFFICIENT_SESSIONS; a member losing one session → INSUFFICIENT_SESSIONS for that member only; the family size stays 3 |
| missing primary and partial data | `test_missing_primary_stays_missing_…`: unavailable primary (the raw-counter sensitivity keeps both blocks, nothing substituted), a NOT_RUN block, a NOT_ACCEPTED unit, and a session stopped by the readiness gate |
| exact operation accounting | `test_exact_operation_accounting_…`: R must be an int equal to the frozen count in config, record and region; J = E ÷ R exactly; duplicates and unscheduled launches refused. `test_end_to_end_…` checks all 72 units |
| invalid period, counter, coverage | `test_invalid_period_counter_and_coverage_…` runs the **unchanged** `unit_energies` on test copies of a fake-NVML runner session. Each fault makes only the affected estimator unavailable, with its reason |
| clock domains | `test_clock_domains_…`: shifting the CLOCK_REALTIME-epoch field timestamps by 1 h leaves the sampled energy identical; shifting the CLOCK_MONOTONIC query brackets changes it. `test_R1_and_R2_…`: per-GPU bands (a GPU 5-band clock on GPU 4 fails) |
| thermal imbalance, R2 refusal | `test_R3_is_per_gpu_…`: GPU-mean balance but per-GPU violation; 0.5 °C boundary; non-finite T; override only on H-D3/H-D4. `test_R1_and_R2_…`: out-of-band T or clock, clock event 0x1 at P2, stale power, and a fixed wait (not READY) are all excluded as ACCEPTANCE_INCONSISTENT. `test_readiness_rules_must_be_…`: swapped or changed bands refused |
| estimator separation | `test_estimator_separation_…`: sensitivities computed separately, never promoted; adequacy refuses non-primary input; H-D2 is NOT_APPLICABLE for the sampled estimator |
| TOST/Holm labels | label and `family_size` assertions in the known-answer tests; `test_labels_and_scope_statement`; `test_adequacy_is_not_established_…` (intersection-union: all members met yet NOT_ESTABLISHED; one member failing → NOT_MET) |
| provenance binding | `test_provenance_refusals_and_bindings`: calibration and smoke dirs refused; scope mismatch, pair substitution, GPU order, rules and counts checked; Freeze-H config-hash binding; input hashes. `test_real_calibration_only_for_interface_…`: the real calibration dir is refused as held-out; calibration Q3 and Q1 regression; real scope aliasing. `test_module_is_analysis_only_…`: no pynvml import; the runner never imports the module; the Freeze-C code role is byte-identical |

## 4. Exact 72-launch schedule (unchanged)

The schedule files are byte-identical to the cg1 implementation record:

| file | sha256 |
|---|---|
| `heldout_s1_DRAFT.yaml` | `2200fad1ff3ed25d9e3c0ed53e8eb3144b4f1bcdc579c9d221365e60078b00d3` |
| `heldout_s2_DRAFT.yaml` | `28fbbb5a1d6d3cbb359ee9825a7ede227333b396181c461f8f4b4d1bfbeaddfb` |
| `heldout_s3_DRAFT.yaml` | `e4656b69fbaf2b704bd5c7ae4b767c823b2569dfb359032bbdaae6bcbe10ae3d` |

The full table (launch id, block, position, triple, arm, length, requested pause, R, w) is `gate/SCHEDULE_72.tsv`, sha256 `63474b856d5a2b8c86a250c8def82926c15844aa18781b9901954c5724329354`. Order within each block (arm and length in s):

| session, block | launches 1–12 |
|---|---|
| s1 b1 | C10 A40 B20 A10 C2.5 B40 B2.5 A20 C40 A2.5 B10 C20 |
| s1 b2 | C40 B10 A2.5 B2.5 A10 C20 B40 C10 A40 C2.5 A20 B20 |
| s2 b1 | C10 A10 B20 A20 C40 B40 B10 A2.5 C2.5 C20 B2.5 A40 |
| s2 b2 | B40 C20 A20 A2.5 B20 C10 C2.5 B2.5 A40 B10 A10 C40 |
| s3 b1 | B2.5 C2.5 A20 B10 C40 A40 C20 A10 B20 C10 B40 A2.5 |
| s3 b2 | B10 C40 A40 A2.5 C10 B2.5 A20 B40 C20 A10 B20 C2.5 |

Counts are the same in every session:

| arm | w | R at 2.5 / 10 / 20 / 40 s |
|---|---|---|
| A | 95604 | 47802 / 191207 / 382413 / 764825 |
| B | 95597 | 47799 / 191193 / 382385 / 764769 |
| C | 98289 | 49145 / 196577 / 393154 / 786308 |

## 5. Hashes, parent and derived readiness settings

| item | path | sha256 |
|---|---|---|
| held-out protocol (new) | `protocol/STAGE2_HELDOUT1_PROTOCOL.md` | `759e2f62f4511f1b462b17faff3f454dcf6e3ce8e25f0617b7c6e955d769aced` |
| analysis module (new) | `code/src/ncclenergy/v2/stage2_heldout.py` | `4826f195a9c59dfe92660bc37c2bfae52f725751607ac4beb9c12b87daa89c7d` |
| decision helpers (unchanged) | `stage2_decisions.py` | `ea82214b90f9d26d54d12544ee3763c9cd028b06e5a47cace75197cd4e692b72` |
| other analysis-role helpers (unchanged, all in the Freeze-C code role) | `stage2_calibration.py` / `stage2_energy.py` / `stage2_phase.py` / `stage2_readiness.py` | `b88ad3c1…` / `d3317bea…` / `b399bf6c…` / `2491ebe6…` |
| **parent Freeze-C** | `protocol/FREEZE_C_CALIB1.json` | `45b127f3f29a65f64552f00525e13b26670226dac6ccd765cd78ae6aa390278c` |
| Freeze-C's parent | `protocol/FREEZE_S_SMOKE1.json` | `22cfae14ba04fa1f9468bb7e19ce01b88201bdd184f1c7ba7c8fe36c458816ff` |
| calibration analysis (Q1, Q3 input) | `prep/calib1/run/calibration_analysis.json` | `152dd0a5a8cac1e8f587941eef1ac490c8095de073b2e0422580045fc59d89f4` |
| shared roles (unchanged) | patch v2.5 / binary / build record / plugin / 22 code files | `492523e7…` / `38d7a327…` / `43c5c4fa…` / `0b96c10d…` / as in Freeze-C |
| external libnccl (unchanged) | `…/nvidia/nccl/lib/libnccl.so.2` | `1d9c1032…` |
| gate builder | `prep/heldout1/build_gate.py` | `c624a12c696e9e2b88d72138b66dcb34d448f23664780dfa128514041158b77c` |
| Freeze-H proposals (dry run) | `gate/freeze_h_dryrun/FREEZE_H_s{1,2,3}_PROPOSED.json` | `1227340b99442d01994057e165fc0f9e1f7193cc84c03ac4fa08498836fe5633` / `f4f552a496ac2d9cb15d25807fec8009ba4ffdcd4e95d241f61c90115dd83a40` / `240868ceba9640ac8b9e48dfdba015e6d50e9f318410e1d522f955b745efa2cc` |
| dry-run verification | `gate/verify.json` | `8d68715c17c7b89a0166c19756053ce4964cccd5ac942d4879a9e096cdfd1c8b` |

**Derived readiness settings** come from calibration Q3, status OK. They are not re-derived and are not changed:
* **T_launch = 26 °C.** The derivation raised it once from 25 °C, where the q90 time to the full gate was censored. At 26 °C the q90 was 102 s over 21 traces.
* **R1:** both GPUs in P8, no foreign process, T ≤ 26 °C and stable within 1 °C over 30 s, idle bits 0x1/0x4 only, wait cap 240 s. If the cap expires, nothing is launched and the session stops.
* **R2 bands:**

| GPU | T_region | clock | P-state |
|---|---|---|---|
| 4 | 28.0–31.2 °C | 1950–1980 MHz | P2 |
| 5 | 29.0–32.0 °C | 1920–1950 MHz | P2 |

  Clock events must be exactly 0, and the POWER_INSTANT row ≤ 1 s old.
* **R3:** ≤ 0.5 °C per GPU, per session/length/pair.

The proposed configs `gate/proposed_configs/heldout_s{1,2,3}_PROPOSED.yaml` are the DRAFT plus these rules plus the calibration-time `scope_expected`. Their allocation, window and freeze record are null, so the runner refuses them.

## 6. Time and deadline budget per window

These are planning values from the frozen `stage2_budget` module (`gate/budget.json`). The per-launch components are gate-3b medians, so these are **not upper bounds**. The only enforced bound is each window's fixed end.

A unit, readiness wait included, starts only if the time remaining is at least the unit-start gate:

| unit length | gate |
|---|---|
| 2.5 s | 519.2 s |
| 10 s | 541.7 s |
| 20 s | 571.7 s |
| 40 s | 631.7 s |

The gate covers the 240 s wait cap, the wall limit, 15 s cleanup, the 100 s end baseline and the 75 s tail.

| | session 1 | session 2 | session 3 |
|---|---|---|---|
| last launch | s1-b2-p12 (B, 20 s) | s2-b2-p12 (C, 40 s) | s3-b2-p12 (C, 2.5 s) |
| expected (frozen design, 120 s waits) | 62.9 min | 62.9 min | 62.9 min |
| calibration-informed (q90 102 s waits) | 55.7 min | 55.7 min | 55.7 min |
| floor (30 s waits) | 26.9 min | 26.9 min | 26.9 min |
| cap-based estimate (240 s waits) | 112.7 min | 112.7 min | 112.7 min |
| window needed to complete, waits at q90 | 61.3 min | 62.0 min | 60.7 min |
| **window needed to complete, every wait at cap** | **115.1 min** | **115.8 min** | **114.6 min** |
| proposed window (fixed end, hard stop) | 120 min | 120 min | 120 min |

* The deadline is the authorized window end, held on CLOCK_MONOTONIC from session start.
* With 120 min, the margin over the every-wait-at-cap case is 4.9 / 4.2 / 5.4 min (sessions 1 / 2 / 3). That margin absorbs overruns of the median-based per-launch overheads. A stop rule ends the session regardless.
* If time runs out, the session ends partial, its data are kept, and the affected contrasts report INSUFFICIENT_SESSIONS. No replacement session is run without an approved amendment.
* Total: at most 3 × 120 min. Expected GPU occupancy is about 3 × 56–63 min.

## 7. How the module is bound, and what is not implemented

**Binding.**
* `stage2_heldout.py` appears only in the Freeze-H `analysis` role, which `MANDATORY_ANALYSIS["Freeze-H"]` already requires. That role also lists the unchanged helpers `stage2_decisions`, `stage2_calibration`, `stage2_energy`, `stage2_phase` and `stage2_readiness`.
* It is **not** in the `code` role. The runner never imports it (tested), so `loaded_code_files()` of a held-out run cannot contain it.
* The shared roles (patch, binary, build record, plugin, and the 22 code files) are copied from Freeze-C. `verify_freeze` checks that their hashes equal the parent's.
* Each Freeze-H record will be a **new** file. Freeze-S and Freeze-C are not rewritten, and their hashes are re-verified through the parent chain.
* Each proposal binds 50 files: the roles, the stage-2 tests including the new one, the three DRAFT schedules, the calibration analysis, the ledger, the builder and the schedule table.

**Not implemented (accepted by approving this gate).**
* **K2 historical D1–D4 replay** (v2 K2).
  * The frozen `measval_f4735b1.py` reads the historical campaign layout (`blocks/block_*/`, `requested` config names, frozen target entries).
  * Stage 2 sessions use another layout, so an adapter would be new data-mapping code. It is not written. Raw data are kept, so it could be added later only by an approved amendment, ideally before the sessions run.
* **K3 secondary covariate analysis:** the resampling count and seed are not frozen. Reported as NOT_COMPUTED.
* **E_samp,hold,inst sensitivity:** no frozen implementation exists.
* **Delay-shifted sampled estimator:** calibration Q4 = NO_SECONDARY_ESTIMATOR.
* **Systematic budget:** no hard-bound rule is frozen. v3 allows start-phase and thermal rules at Freeze-H, but none is proposed, and end alignment is ALIGNMENT_NOT_ESTABLISHED from calibration. Every component is UNKNOWN, so every adequacy scope is NOT_ESTABLISHED. The conditional statistical criteria are still reported.
* **M4d** stays calibration-only (ALIGNMENT_NOT_ESTABLISHED); nothing in held-out turns it into a PASS.

## 8. Known issue left unchanged

`test_stage2_identity_gate.py::test_cli_check_refuses_the_drafts` fails.
* The test expects every `configs/stage2/*.yaml` to be refused as an unfinalized draft.
* The executed final configs `smoke01.yaml` and `calib01.yaml` are refused only with "now is outside the confirmed access window" (`tests/stale_test_evidence.txt`). That is the correct refusal for used configs.
* The test file is bound by Freeze-C. Editing it would break `verify_freeze` for Freeze-C and every Freeze-H, so it is left as is.
* Final held-out configs will add to the same failure. A fix needs an approved amendment.

## 9. After approval (not started)

For each session separately, after its authorization:
1. Record the authorization verbatim.
2. Do a read-only scope read: no counter reads, no GPU state change.
3. Write `configs/stage2/heldout_s<n>.yaml` (DRAFT plus the rules, scope, allocation, window and freeze record) and `protocol/FREEZE_H_S<n>.json`, both new files.
4. Run `verify_freeze`.
5. Run the 24 launches once.
6. Deliver an evidence ZIP and stop.

The analysis (`analyse_heldout`, with all three Freeze-H records) runs once, after the third session.
