# Change ledger r1: production provenance and disclosure corrections (CPU only)

Written before any edit, 2026-10-13. Pre-edit hashes: `pre_edit_sha256.txt`.

**Authorization** (user, "One bounded correction, then the existing held-out endpoint"):
* CPU corrections to the new offline held-out analysis, its unfrozen tests and the proposed documents and Freeze-H artifacts.
* **Not authorized:** an allocation, an experiment window, a commit or a push.

**Accepted prospective scope reductions:**
* the K2 historical replay, the secondary covariate analysis and the sampled-hold sensitivity are not computed in this stage;
* no new estimator and no delay-shifted estimator.

## What may change, and what may not

* **May change:**
  * `code/src/ncclenergy/v2/stage2_heldout.py`;
  * `code/tests/v2/test_stage2_heldout.py` (unfrozen);
  * `protocol/STAGE2_HELDOUT1_PROTOCOL.md` (proposed, never frozen);
  * new files under `prep/heldout1_r1/`, including a new builder and new proposed Freeze-H artifacts.
* **Must not change:**
  * raw data, calibration and smoke outputs, acquisition modules and helpers, native files;
  * executed configs, the DRAFT schedules, Freeze-S/C bytes, the stale frozen identity test;
  * the delivered heldout-1 packet and ZIP, and `prep/heldout1/` (kept as delivered).

## Corrections

None of P1–P7 changes a computed number. Each one only accepts or refuses inputs, or adds labels. P3a changes only a status label.

| # | correction | rule |
|---|---|---|
| P1 | production boundary | New strict entry point `analyse_heldout(session_dirs, freeze_records, base=REPO)`. Configs and calibration inputs are read **only** from the files bound by the Freeze-H records. The previous dictionary-driven entry point becomes `analyse_sessions_unverified`: a labelled low-level/synthetic test interface whose output says `provenance_status: UNVERIFIED (low-level test interface)`. `load_session` and `analyse_units` stay as labelled low-level interfaces. Production output says `VERIFIED_OFFLINE` and lists exactly which identities could not be verified offline |
| P2 | identity | The permitted identity is the runner's frozen `IMPLEMENTATION_ID` / `METHOD_ID`. It must match in the config, in the session summary (with its `mode`) and in every executed unit record that carries a `unit` (with its `mode`, `campaign` and `gpus` = config `allocation`, in order). For an **accepted** record any missing field refuses the analysis; for other executed records any present but inconsistent value refuses it. Log-only entries (NOT_RUN, INTERRUPTED) carry none of these fields and are kept as non-accepted |
| P3 | Freeze-H coverage and validity | Every supplied session directory must have exactly one Freeze-H record. Each record must have: `schema` and `kind` Freeze-H; a non-empty `approved_by_user`; `session` equal to its key; the roles config, protocol, analysis, `calibration_analysis`, `schedule` and the shared roles, all listed in `files`; every listed file present with its hash (exceptions in P7). The analysis role must contain the six analysis modules, each byte-identical to the module actually running. The parent must equal the frozen Freeze-C anchor (`45b127f3…`, file bytes checked); it must be an approved Freeze-C whose shared roles equal the H record's (record-level comparison) and whose own parent (Freeze-S) file matches. All H records must agree on protocol, analysis, parent and calibration analysis, and their config paths must be distinct. A JSON containing only a matching hash token is refused |
| P3a | H without data | A session with an approved H but no supplied directory is reported as **NOT_STARTED**, with all 24 launches NOT_RUN. It is never hidden. A directory without its H is refused |
| P4 | config | Loaded from the H-bound bytes; its sha256 must equal the H entry and the session summary's `config_sha256`. A nonexistent file is refused. It must equal the H-bound DRAFT schedule except for the finalization keys (allocation, placements, allocation_confirmation, scope_expected, freeze_record, readiness_rules), and the DRAFT bytes must equal the frozen cg1 anchors. So counts (R, w), schedule, preparation and identity cannot change. `readiness_rules` must equal calibration Q3. `allocation` and `scope_expected` must equal the Freeze-C calibration config's (same physical pair) and be identical across sessions. An optional caller-supplied config dictionary must be semantically equal to the loaded one, or the analysis is refused |
| P5 | calibration analysis | Its actual bytes must equal the H-bound hash **and** the frozen anchor (`152dd0a5…`). Its recorded `analysis_module_sha256` must equal the Freeze-C analysis-role hashes. A changed Q1 with an unchanged Q3 is refused |
| P6 | R and w | Record `R` and `w` must equal the loaded config's counts, and region R must equal R. Values are checked, never changed |
| P7 | portable versus live | Production analysis makes no NVML call, needs no GPU and does not call the runner's live `verify_freeze` (which needs the private absolute NCCL path). These checks are portable artifact checks. Not verifiable offline, and listed in the output rather than silently skipped: the external libnccl (verified only if its recorded path exists with the recorded hash); the compiled binary and plugin (build outputs; verified if present, otherwise listed as NOT_VERIFIED_OFFLINE (absent)). Every other bound file is required |

| # | disclosure correction |
|---|---|
| D1 | With the actual frozen calibration Q1 (primary INSUFFICIENT_ROUNDS, not bounded), **every** aggregate adequacy scope has `conditional_statistical` = NOT_MET and physical adequacy NOT_ESTABLISHED, whatever the held-out results. This goes into the output as `aggregate_adequacy_ceiling` and into the documents. The individual H-D1–H-D4 results can still be informative |
| D2 | 114.6–115.8 min and the 116-min minimum are median-overhead **planning estimates**, never guaranteed completion. Windows are fixed and absolute, at most 120 min. A deadline or readiness failure may end a session partial; there are no retries and no replacement sessions |
| D3 | The offline R1 check verifies the runner's **recorded READY status** only. It is not a replay of the readiness trace. R2 is recomputed from the recorded pre-region state |
| D4 | Full suite reported with its exit status (1), the stale frozen test and its baseline cause. Added: an unfrozen regression test showing that DRAFT configs are refused as drafts and the finalized, executed configs are refused only for the expired window |
| D5 | Machine-readable NOT_COMPUTED in the output: K2 replay, secondary covariate analysis, sampled-hold sensitivity, delay-shifted estimator. The hold reason is corrected: the frozen helper `stage2_energy.sampled_energy_hold_J` exists; only its pipeline wiring (in `unit_energies`) is missing, and it is not added in this stage |
| D6 | Interpretation limits in the output and documents. H-D1 length dependence does not by itself identify a sensor-only mechanism: thermal and clock changes over the region can also change board energy. H-D4 is one selected C/A comparison (AllGather 2 MiB, 40 s, this pair), not evidence that global latency and energy optima differ |
| D7 | The protocol document is revised in place (proposed only, never frozen). A new gate packet r1 and new proposed Freeze-H artifacts are built under `prep/heldout1_r1/` with the corrected hashes. Historical freezes and the delivered heldout-1 packet are untouched |

## Actual changes (2026-10-13; CPU only, no GPU, no NVML call in any test of this module)

| file | sha256 | change |
|---|---|---|
| `code/src/ncclenergy/v2/stage2_heldout.py` (767 lines) | `ae8c9c6b504f45ed3039fa04abb5a53abb51dc3736daef9d5fec33c4cc29b4ee` | P1–P7, D1, D3, D5, D6. Contrast and decision code is unchanged (all earlier known-answer tests pass unchanged) |
| `code/tests/v2/test_stage2_heldout.py` (26 tests; unfrozen) | `4de5f5376a991e7682fdfda2563a24255eef6cf1370496db12da0902633dc9ee` | 6 new tests (below); the earlier ones moved to the labelled low-level interface |
| `protocol/STAGE2_HELDOUT1_PROTOCOL.md` (proposed, never frozen) | `67dad4292d6b2f9f5d6470934858b020718adb9f5481e25e65745962f16e9b64` | D1–D6 text |
| `prep/heldout1_r1/build_gate_r1.py` (new) | `5ff158d8b3b07e48493442a24a034e8f2dc3bb8a71b19e3c5f0893b0b138d3bf` | r1 Freeze-H schema plus the portable check; writes only `prep/heldout1_r1/gate/` |
| `prep/heldout1_r1/check_preserved.py` (new) | `d0b73129b4db4d0cf4640fd84390b64fd6d7ba334d3006462c74e3e8e1fafd26` | read-only preservation check (`preserved_check.txt`) |

**New tests**, covering the cases the independent review reproduced:
* a real-format happy path, through approved **test-only** Freeze-H records in a temp dir that bind the real frozen files;
* the Q1 ceiling: with every held-out criterion met, the only unmet item is Q1;
* NOT_STARTED, and data without its Freeze-H;
* wrong config, summary and record identity;
* an accepted record lacking `mode` or `implementation_id`;
* a wrong campaign, GPU order, changed w, and an accepted record without its unit;
* changed counts, schedule, rules, pair or scope in the config;
* a summary config hash that differs from the H;
* a config edited after freezing, and a nonexistent config;
* an altered config dictionary versus the bound bytes;
* an omitted H, an unapproved H, wrong kind, wrong session, a token-only JSON and an unreadable record;
* a Freeze-S parent instead of Freeze-C;
* the analysis role missing the module, or carrying a different module hash;
* a changed shared role;
* a changed Q1 with Q3 unchanged;
* an unfrozen regression test: the runner `check` command, run in a fresh process, refuses DRAFTs as drafts and the executed `smoke01.yaml`/`calib01.yaml` only as "now is outside the confirmed access window".

**Results:**
* `test_stage2_heldout.py`: **26/26 passed**.
* Full suite: **234/235 passed, exit status 1** (`tests/full_suite_r1.txt`, sha256 `5751dbaa15f43f05ce7a5715ded590d3dc8d308f483dee51f65047b2b679859e`). The single failure is the frozen stale test `test_stage2_identity_gate.py::test_cli_check_refuses_the_drafts`. Its baseline cause is that it expects every config to be a draft, but `smoke01.yaml` and `calib01.yaml` were finalized and executed. It is Freeze-C-bound and left unchanged.
* An in-process config check from a process that has imported the analysis module is refused by the live freeze check ("loaded code not bound by the freeze"). This confirms that a runner process must never load the analysis, which is why the regression test runs `check` in a fresh process.

**Preservation** (`preserved_check.txt`; all OK):
* the 387 pre-heldout-1 baseline files;
* the 16 r1 pre-edit preserved files: Freeze-S/C, executed and DRAFT configs, the calibration analysis, the heldout-1 packet, ledger and builder, and the calibration/smoke/heldout-1 ZIPs;
* the live `verify_freeze` of Freeze-S and Freeze-C;
* the 305 calibration and 45 smoke raw session files against their manifests;
* `prep/heldout1/` against the delivered heldout-1 manifest.

**Incident (found and repaired during this step).**
* What happened: a guarded edit of the copied builder failed one string match, but the next shell line ran the still-unmodified copy anyway. That copy wrote to `prep/heldout1/gate/`. Four delivered files were regenerated with different bytes (the three dry-run proposals and `verify.json`); the other five regenerated byte-identically.
* Repair: the four were restored byte-for-byte from the delivered ZIP and checked against its manifest.
* Scope: nothing in `protocol/`, no raw data and no freeze record was touched. The builder writes no such files.
* Prevention: the r1 builder's output path is hard-coded to `prep/heldout1_r1/gate/`. Before each run, the command checks that path in the builder source and refuses to run otherwise. It then compares the hash of `prep/heldout1/` before and after the run. This is a run-command guard, not code inside the builder.
