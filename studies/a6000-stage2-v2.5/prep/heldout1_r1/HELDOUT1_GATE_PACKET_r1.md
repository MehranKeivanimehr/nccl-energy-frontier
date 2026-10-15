# Held-out gate packet r1: production provenance and disclosure corrections (2026-10-13)

**Status.**
* CPU only: no GPU, no NVML call, no held-out data, no commit, push or publication; the GitHub-cleanup prompt was not run.
* The production-analysis provenance gap found by the independent review is closed. **26/26** held-out tests pass.
* Full suite: **234/235, exit status 1.** The one failure is the frozen stale test described below.
* All preserved artifacts recheck OK.
* The delivered heldout-1 packet and ZIP are kept as delivered; `prep/heldout1/` matches its manifest.

## Endpoint, stated plainly

* With the actual frozen calibration Q1 (primary INSUFFICIENT_ROUNDS, not bounded), **all five aggregate adequacy scopes stay `conditional_statistical` = NOT_MET, and physical adequacy stays NOT_ESTABLISHED, even if every held-out criterion succeeds.** A test shows this directly: with every held-out criterion met, the only unmet item is Q1.
* The individual H-D1–H-D4 verdicts can still be informative, as conditional comparisons of NVML-reported board energy on the calibration pair.
* Accepted prospective scope reductions, which are **not** a completion of every original protocol output, are each reported as NOT_COMPUTED in the machine-readable output:
  * K2 replay;
  * secondary covariate analysis;
  * the sampled-hold sensitivity. Its helper `sampled_energy_hold_J` exists; only its wiring is missing;
  * the delay-shifted estimator, which calibration Q4 refused;
  * systematic bounds.
* **H-D1:** a length dependence is not a sensor-only mechanism; thermal and clock changes can also change board energy.
* **H-D4:** one selected C/A comparison, not evidence that global latency and energy optima differ.
* **Offline R1:** checks the runner's **recorded** READY status, not a replay of the readiness trace.

## Correction ledger (`CHANGE_LEDGER_r1.md`, written before editing)

| # | correction |
|---|---|
| P1 | Strict production entry point `analyse_heldout(session_dirs, freeze_records)`. The old dictionary-driven entry point is now `analyse_sessions_unverified`, whose output is labelled UNVERIFIED |
| P2 | The permitted implementation/method identity must hold in the config, session summary and executed records, together with mode, campaign and GPU order. Missing on an accepted record → refused |
| P3 | Exactly one approved Freeze-H per supplied session. Checked: schema, kind, approval, session number, every bound file, the analysis role equal to the running code, parent = frozen Freeze-C (`45b127f3…`) with equal shared roles and an intact Freeze-S, and agreement across the H records. A token-only JSON is refused |
| P3a | A frozen session without data is reported NOT_STARTED (24 NOT_RUN); data without its H is refused |
| P4 | The config is read from its H-bound bytes, which must equal the run's `config_sha256` and the frozen DRAFT except the finalization keys. Its rules must equal calibration Q3; its pair and scope must equal the calibration pair's. A supplied dictionary must equal the loaded config |
| P5 | The calibration analysis bytes must equal the frozen anchor (`152dd0a5…`) and be produced by the Freeze-C code. A changed Q1 with the same Q3 is refused |
| P6 | Record R, w and region R must equal the config counts (checked, never changed) |
| P7 | Portable offline checks only: no NVML, no GPU, no live `verify_freeze`. The binary/plugin (if absent) and the external libnccl (if its path is absent on this host) are **listed** as NOT_VERIFIED_OFFLINE, never skipped silently |
| D1–D7 | The disclosures above, the planning-estimate wording for windows, the honest suite report, the unfrozen regression test, the protocol text revision, and new proposed Freeze-H artifacts |

**Incident in this step.**
* An accidental run of an unedited copy of the old builder regenerated four files in `prep/heldout1/gate/`.
* They were restored byte-for-byte from the delivered ZIP and verified against its manifest. Nothing else was touched.
* Details are in the ledger.

## Tests

* **New (6):**
  * production happy path in the real record format, through approved test-only H records in a temp dir that bind the real frozen files;
  * identity: wrong config, summary or record identity; accepted record missing `mode`/`implementation_id`; campaign; GPU order; changed w; accepted record without its unit;
  * config: counts, schedule, rules, pair, scope; summary hash ≠ H; edited after freeze; nonexistent; supplied dictionary ≠ bytes;
  * Freeze-H: omitted; unapproved; wrong kind or session; token-only; unreadable; Freeze-S as parent; analysis role missing the module or with another hash; changed shared role;
  * changed Q1 with Q3 unchanged;
  * unfrozen regression: DRAFTs are refused as drafts, and the executed `smoke01.yaml`/`calib01.yaml` only as an expired window.
* **Earlier (20):** unchanged in substance. They now use the labelled low-level interface, and all known answers are the same.
* **Results:**
  * `test_stage2_heldout.py`: 26/26.
  * Full suite: 234/235, exit 1 (`tests/full_suite_r1.txt`).
* **Stale frozen test:** `test_stage2_identity_gate.py::test_cli_check_refuses_the_drafts` fails.
  * Baseline cause: it expects every config to be a draft, but `smoke01.yaml` and `calib01.yaml` were finalized and executed.
  * It is bound by Freeze-C and not edited.

## Hashes (sha256)

| artifact | sha256 |
|---|---|
| `code/src/ncclenergy/v2/stage2_heldout.py` (analysis) | `ae8c9c6b504f45ed3039fa04abb5a53abb51dc3736daef9d5fec33c4cc29b4ee` |
| `code/tests/v2/test_stage2_heldout.py` | `4de5f5376a991e7682fdfda2563a24255eef6cf1370496db12da0902633dc9ee` |
| `protocol/STAGE2_HELDOUT1_PROTOCOL.md` (proposed) | `67dad4292d6b2f9f5d6470934858b020718adb9f5481e25e65745962f16e9b64` |
| `prep/heldout1_r1/CHANGE_LEDGER_r1.md` | `65c3a8577a43ed8a2677100a7cde4002104a72b38d89db4fc93fcd4346d5c7cd` |
| `prep/heldout1/CHANGE_LEDGER.md` (N1–N18, N11a; unchanged) | `14d1eb42431c9c18a99e…` (as delivered) |
| proposed Freeze-H s1 / s2 / s3 (`gate/freeze_h_dryrun/`, 51 bound files each, no approval) | `c2e0a5f4a6cbf18a…` / `6e20ca5e9a9eebf0…` / `d971c830567809703…` |
| `gate/verify.json` | `74b13e59a67e3b3b55465f1c9ccbe6a5c7c03a418b0ad7ee5d1e775a9786dd15` |
| parent Freeze-C / its parent Freeze-S (unchanged) | `45b127f3f29a65f6…` / `22cfae14ba04fa1f…` |
| calibration analysis (unchanged) | `152dd0a5a8cac1e8f587941eef1ac490c8095de073b2e0422580045fc59d89f4` |
| schedules s1 / s2 / s3 DRAFT (unchanged, = cg1) | `2200fad1…` / `28fbbb5a…` / `e4656b69…` |
| `gate/SCHEDULE_72.tsv` / `gate/budget.json` (identical to heldout-1) | `63474b85…` / `ee49c813…` |
| full suite log | `5751dbaa15f43f05ce7a5715ded590d3dc8d308f483dee51f65047b2b679859e` |

**Dry-run verification** (`gate/verify.json`), per proposal:
* the runner's live `verify_freeze` → only "freeze record lacks the user's approval entry";
* the portable `freeze_h_problems` → only `FREEZE_H_NOT_APPROVED`;
* NOT_VERIFIED_OFFLINE on this host: none;
* config items still pending: `allocation` and `placements` (they need your authorization).

**Preservation** (`preserved_check.txt`), all OK:
* the 387 baseline files;
* 16 preserved files: Freeze-S/C, configs, calibration analysis, heldout-1 documents, the three evidence ZIPs;
* live `verify_freeze` of Freeze-S and Freeze-C;
* the 305 calibration and 45 smoke raw session files;
* `prep/heldout1/` against the delivered manifest.

## Windows

* The window needed to complete each schedule if every readiness wait hits its 240 s cap (115.1 / 115.8 / 114.6 min for sessions 1 / 2 / 3), and the 116-min minimum, are **median-overhead planning estimates, never guaranteed completion.**
* Each window must be fixed and absolute, at most 120 min.
* A deadline or readiness failure may end a session partial. Its data are kept; there are no retries and no replacement sessions.

## What is needed next (session 1 only)

1. The **actual allocation** of the same physical NVML GPUs **4 and 5** to you for this session (an idle GPU is not an allocation).
2. An **absolute start and end with time zone**, at most 120 min apart.

With that, I create `configs/stage2/heldout_s1.yaml` and `protocol/FREEZE_H_S1.json` as new files, verify their bindings, run the read-only scope checks, and execute the 24 launches of session 1 once under the unchanged gates. Sessions 2 and 3 each need their own fresh authorization.
