# Change ledger: held-out analysis module (heldout-1), CPU only

Written before any code edit, 2026-10-13T22:14Z.
* Pre-edit worktree hashes: `baseline_before_edits/worktree_sha256.txt` (387 files).
* `src/ncclenergy/v2/stage2_heldout.py` does not exist yet.

**Authorization** (user, 2026-10-13: "Finite next step after independent calib-1 review"):
* bounded CPU work: a documents-only `CALIB1_REPORT_r1.md`; the missing `stage2_heldout.py` as offline analysis only; CPU tests; one held-out gate packet.
* **Not authorized:** GPU collection, retries, publication, commits, pushes, or running the GitHub-cleanup prompt.

## What changes, and what does not

* **New files:**
  * `src/ncclenergy/v2/stage2_heldout.py` (offline analysis);
  * `tests/v2/test_stage2_heldout.py`;
  * proposed held-out session configs and a Freeze-H dry run (not freeze records).
* **Unchanged:**
  * every acquisition, native, NVML, cleanup, watchdog, acceptance and runner path;
  * every existing analysis helper: `stage2_decisions`, `stage2_calibration`, `stage2_energy`, `stage2_phase`, `stage2_readiness`;
  * all frozen records, calibration data and outputs.
* **Binding:**
  * The new module is used only offline, and only in the Freeze-H `analysis` role.
  * The shared `code` role must stay identical to Freeze-C: same 22 files, same hashes.
  * The runner never imports the module.

## Numerical-result-affecting choices (listed before applying)

They apply to prospective held-out data only, never to calibration or history. Where the frozen protocol was silent, the choice is marked **(choice)**.

| # | item | rule implemented | source |
|---|---|---|---|
| N1 | unit quantities | J = E / R with the exact frozen integer R (`stage2_energy.joules_per_op`). E is the per-participating-GPU sum from `stage2_calibration.unit_energies`, unchanged: primary E_aligned(P̂); sensitivities E_counter_raw, E_aligned(P_nom), E_samp,trap,inst. Only accepted units. An unavailable primary stays unavailable with its reason, **never substituted** | v2 K3/I; cg1 L2, L11, L12; smoke-1 B |
| N2 | estimator separation | every family is computed separately per estimator. Decisions and adequacy use the **primary only**; sensitivities are labelled and never promoted | v2 K3 |
| N3 | H-D1 | 9 contrasts ln(J_L/J_40), L ∈ {2.5, 10, 20} × arms A/B/C, per block (both units of that arm accepted). Holm over the 9; margin ln 1.003 | cg1 L14; `stage2_decisions.HD1_LENGTHS` |
| N4 | H-D2 | per block, the coefficient of 1{C} in OLS of ln(E_samp,trap,inst / E) on [1, 1/T, 1{C}] over the block's accepted units with both estimators; margin ln 1.003; family of 1. **(choice)** T = the unit's measured region duration `T_region_s` (C0 to work end). **(choice)** A block needs ≥ 4 such units, including ≥ 1 C and ≥ 1 non-C, and a full-rank design (rank 3); otherwise that block's H-D2 contrast is missing. For the sampled sensitivity family, H-D2 is NOT_APPLICABLE | v2 K3 table |
| N5 | H-D3 | ln(J_B/J_A) at L ∈ {10, 20, 40}, per block, **control-valid blocks only**. Holm over the 3 lengths; margin ln 1.005 | v2 K3, J |
| N6 | control validity | **(choice, strict reading of "if A's execution differs from B's in any block")**. A block is control-invalid if, at any length, A and B are both accepted and their verified execution keys (algorithm/protocol/channels) differ. Invalid blocks are excluded from H-D3. If **any** block is control-invalid, every H-D3 verdict gets the override INVALID_CONTROL_OR_MEASUREMENT and `control_valid = False` for adequacy | v2 J, K3 verdict table |
| N7 | H-D4 | ln(J_C/J_A) at 40 s per block. Reported: summary, unadjusted evidence at margin ln 1.004 (labelled), and precision ok iff 95 % half-width ≤ ln 1.004 | v2 K3; `stage2_decisions.precision` |
| N8 | aggregation | session mean = mean of that session's available block contrasts (1 or 2). Primary t on the 3 session means (df 2), **S = 3 required**, otherwise INSUFFICIENT_SESSIONS. Conditional t over all block contrasts labelled "conditional on these sessions". Implemented with the unchanged `session_summary`, `family`, `evidence`, `holm`, `verdict` | v2 K3; `stage2_decisions` |
| N9 | R1 | a unit counts only if the runner recorded readiness READY. A fixed-wait expiry is never readiness | v2 F; cg1 L5 |
| N10 | R2 | recomputed offline with `pre_region_check` and the frozen bands. A unit accepted by the runner but failing the recomputation is excluded and flagged `ACCEPTANCE_INCONSISTENT_R2` | v2 F; cg1 L7 |
| N11 | R3 | `thermal_balance` per (session, length, pair B−A / C−A), per GPU, both arms required, accepted units only, ≤ 0.5 °C. A violation at L gives the override THERMAL_COMPARABILITY_NOT_ESTABLISHED on H-D3(L) (pair B−A) and on H-D4 (L = 40, pair C−A). **(choice)** H-D1 and H-D2 are within-arm or estimator contrasts and get no R3 override; R3 is still reported. `r3_ok[L]` for adequacy = both pairs OK in all sessions | v2 F; cg1 L8 |
| N12 | adequacy | `stage2_decisions.adequacy` unchanged, for all scopes:
  * Q1 comes from the frozen calibration analysis (primary INSUFFICIENT_ROUNDS, so not bounded);
  * every systematic component UNKNOWN (no hard-bound rule was frozen; calibration end alignment ALIGNMENT_NOT_ESTABLISHED).

  The physical result is therefore **NOT_ESTABLISHED by construction**. Conditional statistical criteria are reported separately | v3 K3; cg1 L15, L17 |
| N13 | not implemented | the secondary covariate analysis (end-of-R1 temperature, slope refit in a block resampling) is **not implemented**: the protocol gives no resample count or seed. It is reported as NOT_COMPUTED with that reason. Q4: no delay-shifted estimator (calibration: NO_SECONDARY_ESTIMATOR) | v2 K3; calib-1 |
| N14 | inputs and provenance | the inputs are exactly three held-out session directories:
  * mode STAGE2_HELDOUT, session numbers 1–3, launch ids and units equal to the frozen schedule, implementation/method ids;
  * calibration or smoke directories are **refused**;
  * the readiness rules in the analysis call must equal the calibration Q3 output (T_launch 26 °C, bands);
  * every input file used is hashed into the output;
  * with a Freeze-H record, each session's `config_sha256` must equal that record's config hash | user prompt |
| N15 | missing data | no imputation, optional stopping or replacement:
  * missing or unaccepted unit → any contrast needing it is missing for that block;
  * unavailable estimator → missing for that estimator only;
  * block with no valid contrast → the session uses the other block;
  * session with none → INSUFFICIENT_SESSIONS;
  * incomplete session → data kept, same rules | v2 N (partial sessions) |

### Addendum: rules added while drafting the module, recorded before any execution of it

| # | item | rule implemented | source |
|---|---|---|---|
| N16 | offline re-verification of acceptance | for every runner-accepted unit, re-check R1 READY, R2 (N10), the frozen execution checks (evidence VERIFIED; for arms B/C, the forced algorithm/protocol from the runner's `ARM_EXPECTED`; a single execution with an execution key; op count R + w + 1), region present and a v2.5 work end on every GPU. Any failure → unit EXCLUDED with `ACCEPTANCE_INCONSISTENT` and the specific reasons (the runner rejects all of these, so a failure means the record is inconsistent; it is reported, never repaired). **R** must equal the frozen count in the config, in the record and in the region record; otherwise the input is **refused** (exact operation accounting) | v2 J; cg1 L2; runner `acceptance` |
| N17 | non-positive or non-finite energy | an estimator value ≤ 0 or non-finite is unavailable for that estimator only (reason recorded); no other estimator is affected | N1, N15 |
| N18 | pair and scope guards | each session's `scope_observed` must match its config's `scope_expected` (the runner's own `scope_problems`, imported unchanged). All analysed sessions must have identical observed scope (no pair substitution). Every record's GPU list must equal the analysed pair. `counts` must be identical across session configs | user prompt; v2 scope rules |

| N11a | R3 override vs missing data (found by the first CPU test run, recorded before the code change) | the R3 override applies at L when R3 was **evaluated and failed** for some session: a per-GPU difference > 0.5 °C, or a non-finite/missing pre-region temperature. When a session has no unit of one arm at L (`ARM_MISSING`), or is absent entirely, it contributes no contrast at L. That is missing data, giving INSUFFICIENT_SESSIONS when fewer than 3 sessions remain, not a thermal verdict. `r3_ok[L]` for adequacy stays strict: missing counts as not OK. *First draft:* any missing (session, L, pair) key overrode the verdict, so a run missing session 3 reported THERMAL_COMPARABILITY_NOT_ESTABLISHED instead of INSUFFICIENT_SESSIONS | v2 K3 verdict table (R3 "violated"); N11 |

Corrections to the table above:
* N4: the "≥ 4 units including C and non-C" rule is from v2 K3 (not a choice). The definition T = `T_region_s` and the rank-3 requirement remain choices.
* N1: v2 K3 also lists E_samp,hold,inst as a sensitivity. No frozen implementation of it exists (`unit_energies` computes the trapezoid only), so it is **not computed** and is reported as NOT_COMPUTED. Adding it would be a new numerical path.

## Actual changes (2026-10-13; CPU only, no GPU, no NVML call)

**New files**:

| file | sha256 | role |
|---|---|---|
| `code/src/ncclenergy/v2/stage2_heldout.py` (458 lines) | `4826f195a9c59dfe92660bc37c2bfae52f725751607ac4beb9c12b87daa89c7d` | offline held-out analysis; Freeze-H `analysis` role only |
| `code/tests/v2/test_stage2_heldout.py` (20 tests) | `0b652a16eab3c0d1768932cbfe6157fcab27be46154330c1d9e497f9d0fe55c5` | CPU tests |
| `protocol/STAGE2_HELDOUT1_PROTOCOL.md` | `759e2f62f4511f1b462b17faff3f454dcf6e3ce8e25f0617b7c6e955d769aced` | proposed held-out protocol (binding only through approved Freeze-H records) |
| `prep/heldout1/build_gate.py` | `c624a12c696e9e2b88d72138b66dcb34d448f23664780dfa128514041158b77c` | builds the proposed configs, schedule, budget and Freeze-H dry run |
| `prep/calib1/CALIB1_REPORT_r1.md` | `5256c8fb88e77f12fa16b3e34c79585a7b7a11d51dd9837c365d2d01de9909f5` | documents-only corrected calibration report (the original is unchanged) |

**Unchanged** (verified):
* all 387 files hashed in `baseline_before_edits/worktree_sha256.txt`: 0 changed, 0 missing;
* `verify_freeze(FREEZE_C_CALIB1.json)` returns no problems, so every Freeze-C-bound file (the 22 code files, patch, binary, plugin, build record, config, protocol and tests) is byte-identical;
* no file in `protocol/` other than the new held-out protocol was written.
* The DRAFT held-out configs are byte-identical to the cg1 implementation record.

**Changes made during implementation**, before any held-out data exists:
* H-D2 looked up energies by the wrong key names. Found while writing the tests and fixed before the first test run.
* Session-mean heterogeneity (K3) was dropped from the output; it is now kept.
* Unit files are now hashed recursively.
* R3 override semantics (N11a above).
* N18 scope comparison (found while assembling the gate, before any held-out data). Configs carry raw GPU UUIDs; session summaries store them as the runner's stable aliases (`common.redact`). The expected scope is now aliased with the same `redact` before `scope_problems`. Without this, every real session would have been refused (a refusal only, never a numerical change). Regression-checked on the real calibration (`calib01.yaml` vs its session summary: no mismatch).
* `analyse_heldout` refuses a session config without `scope_expected`. Before this, the scope check was skipped silently; the runner already refuses such configs.

**Tests**:
* `test_stage2_heldout.py`: **20/20 passed**.
* Full suite (`tests/run_tests.py`): **228/229 passed**. Log: `tests/full_suite_heldout1.txt`, sha256 `bbdea9bfe4f03edc3c2f64dea85bbe8489b6ed751e6f591f8674e1063599cb39`.
  The final run on the final module is byte-identical to the run before the scope fix (`tests/full_suite_heldout1_before_scope_fix.txt`, same sha256). The log has no timestamps.
* **The one failure is pre-existing and not caused by this change:** `test_stage2_identity_gate.py::test_cli_check_refuses_the_drafts`.
  * It expects every `configs/stage2/*.yaml` to be refused as an unfinalized draft.
  * The executed `smoke01.yaml` and `calib01.yaml` are final configs. The runner refuses them only with "now is outside the confirmed access window" (`tests/stale_test_evidence.txt`), which is the correct behaviour for used configs.
  * The test file is bound by Freeze-C, so editing it would break `verify_freeze` for Freeze-C and every Freeze-H. It is left unchanged and reported.
