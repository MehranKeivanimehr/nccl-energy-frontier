# Calibration gate sheet (calib-1)

Date: 2026-10-13. CPU only. No GPU was used, nothing is committed or pushed, and held-out collection is not started.

## 1. What changed since smoke-1

**Documents only. No code or measurement change.** All 27 bound acquisition, control, acceptance and analysis files still match `FREEZE_S_SMOKE1.json` (`22cfae14…`). Unchanged: the original smoke ZIP, raw files, Freeze-S and historical outputs.

| document | sha256 | change |
|---|---|---|
| `audit/SCIENTIFIC_VALIDITY_AUDIT_2026-10-13.md` (r1) | `83622da0…` | now supplied. Corrections: Q1 "guaranteed to fail" withdrawn; mechanism wording (read-dependent reporting artifact; reset mechanism unverified, not an NVML contract); M4c purpose. r0 preserved as `…r0_as_delivered.md` (`bf74569c…`) |
| `prep/smoke1/SMOKE1_EVIDENCE_PACKET_r1.md` | `cdd6dc83…` | corrections: 0x24 at t = 38.000093434 s, P8, 26 °C; scoped "no event" and "host load" statements; SMOKE-01 warm-up ranks (4.966 / 4.964 s); "never 0 at P8" scoped to the samples (smoke: 0/3,963 P8 rows; 75 zero rows were P0/P3 wind-down). Original `f159195b…` unchanged |
| `protocol/STAGE2_CALIB1_PROTOCOL.md` | `9da7a802…` | new. Restates the existing rules: matrix, Q1 as pre-specified, clock-event policy, M4c, endpoint, deadline. Bound at Freeze-C |

**Do any acquisition, safety or acceptance dependencies change?** **No.**
* Freeze-C will reuse Freeze-S as parent with identical shared roles. A dry run of `verify_freeze` passed.
* Only the calibration config (allocation, window, scope, freeze record) and the protocol are new at Freeze-C, and neither is a shared role.
* No provenance amendment and no new smoke are needed.

## 2. Remaining blocker

1. **Your authorization of the pair and a fresh window (§9).** This is the only blocker.
   * After it: complete `calib01.yaml` from the draft; build Freeze-C (parent Freeze-S, your approval quoted); run `stage2_runner check` (must give no refusals); run the read-only identity, scope and health preflight; then calibration.

**Not blockers, but disclosed risks:**
* A transient 0x20 (like the smoke-1 0x24) has these effects, and no retry or replacement follows:
  * at the start baseline: the **session stops before any launch**;
  * during a unit's process lifetime or in a QC window: the **session stops** (§5);
  * at the end baseline: recorded as FAILED;
  * in a fixed wait: recorded only.
* High host load from other users: effect not assessed.

## 3. Matrix (existing, 23 launches, `calib01_DRAFT.yaml` `57976c02…`)

All launches: AllGather 2 MiB, arm B (Ring/Simple), w = 95,597, R(2.5/10/20/40 s) = 47,799 / 191,193 / 382,385 / 764,769, M4d on, frozen M4c pauses. In execution order:

| launches | content |
|---|---|
| 1 | S0a E0 2.5 s (the gate) |
| 2 | S0b D100 2.5 s |
| 3–6 | C1 round 1: E0 at 20, 2.5, 40, 10 s |
| 7–9 | C2 round 1: E0, D100, D500 at 10 s |
| 10–13 | C1 round 2: E0 at 20, 2.5, 10, 40 s |
| 14–16 | C2 round 2: D100, D500, E0 at 10 s |
| 17–20 | C1 round 3: E0 at 2.5, 20, 40, 10 s |
| 21–23 | C2 round 3: D500, E0, D100 at 10 s |

No retries, replacement units, optional stopping or added launches.

## 4. Budget and deadline

* **Expected 58.9 min**; cap-based planning estimate 60.7 min (not a bound).
* **Confirmed window: 2026-10-13 05:08:30Z–06:23:30Z.** Its end is the absolute enforced deadline: no separate automatic cap, and no extension if the start is late.
* A 40 s unit starts only if ≥ 513.0 s remain. The unit-start gate may therefore stop the run before all 23 launches; that is recorded, not retried.

## 5. Clock-event policy (existing code, no change)

* **Recording:** everywhere.
* **Fixed 120 s wait:** a cooldown only. It records masks and does not stop on them; it stops only on a foreign or unknown process or ≥ 80 °C.
* **Idle gates** (baselines, R1, Q3 replay): P8 rows tolerate only 0x1 and 0x4. 0x20 fails.
* **Watchdog over the unit's whole process:** 0x8, 0x20, 0x40 or 0x80 → cancel and session stop.
* **Measured-region QC** **[C0, work end]**: 0x4, 0x8, 0x20, 0x40 or 0x80 → THROTTLED and session stop. The preceding 250 ms is used only for required-channel coverage and the recorded pre-state; held-out R2 gates it.
* **R2 (held-out):** exactly 0.

Full table: protocol §4.

## 6. Bound hashes

Full list: `bound_hashes.tsv`; all re-verified against the files today.

* **Acquisition:**
  * v2.5 patch `492523e7a79cc856…`;
  * binary `all_gather_perf` `38d7a32748721819…`;
  * build record `43c5c4fa297a3264…`;
  * profiler plugin `0b96c10d120f461c…`;
  * libnccl 2.27.5 `1d9c10322efacde6…`.
* **Control and acceptance code:**
  * `stage2_runner` `6a9793188debab82`;
  * `stage2_energy` `d3317beab273fd47`;
  * `stage2_readiness` `2491ebe6e70e61df`;
  * `runner` `98c47112a249de80`;
  * `telemetry` `63d85db5e5c36e9e`;
  * `watchdog` `1371a415d8e29eb8`;
  * `qc` `08299484208f7930`;
  * `procctl` `b55179ad4865dd06`;
  * `preflight` `f1a672aa02676130`;
  * `region` `def13f52bf4889b4`;
  * `evidence` `2fe3b99ea6c7d287`;
  * `stage2_schedule` `c4764e6a19f0cc7a`;
  * `stage2_budget` `ef1a7926cf5f6380`;
  * plus `common`, `attempts`, `selection`, `topology` and the package `__init__` files.
* **Analysis:**
  * `stage2_calibration` `b88ad3c110800293`;
  * `stage2_decisions` `ea82214b90f9d26d`;
  * `stage2_phase` `b399bf6cbe9de7ac`.

## 7. Calibration outputs and how they are reported

* **Q1** (D100/E0, D500/E0; counter, sampled and DiD; ±ln 1.003; 3 rounds; Holm on the difference/minimum-effect tests; individual TOST unadjusted; joint both-dose BOUNDED rule for adequacy): any class, including DETECTED and INCONCLUSIVE, is reported as a legitimate finding. Q1 tests only the additional in-region reads.
* **Q2:** descriptive.
* **Q3:** T_launch and R2 bands, or NOT_DERIVABLE / NOT_ACHIEVABLE.
* **Q4:** secondary estimator on or off.
* **M4c:** RETAINED or AMENDMENT_PROPOSED.
* **End alignment:** CONDITIONAL or NOT_ESTABLISHED; never a bound.
* **Thermal slope:** association only.
* **t_op:** within 3 % or an amendment proposal.
* **Adequacy:** NOT_ESTABLISHED (systematic part).

Smoke-1 data are excluded: `analyse()` refuses smoke directories.

## 8. Checks that decide whether held-out is allowed (existing rules)

Held-out needs **all** of the following:
1. calibration executed under Freeze-C, with S0a accepted;
2. **your review and decision**;
3. **Q3 status OK**. Otherwise there is no T_launch or R2 band, and Freeze-H needs an amendment;
4. **M4c RETAINED** (AMENDMENT_PROPOSED needs an amendment);
5. **t_op within 3 %** (otherwise an amendment proposal);
6. **Freeze-H**, which needs:
   * the held-out analysis module `stage2_heldout.py`. It does not exist yet; it is new analysis code needing CPU validation, bound in Freeze-H without changing the shared acquisition roles;
   * parent Freeze-C;
   * readiness rules from Q3;
   * your approval;
7. a fresh allocation confirmation for each held-out session window.

Q1, Q2, Q4, end alignment and the thermal slope are findings that are reported. They are not held-out gates. Nothing starts the 72 held-out launches automatically.

## 9. Authorization (received 2026-10-13T05:08:30Z; verbatim in `allocation/user_authorization.json`)

Please state, in your own words:
1. that you authorize **calibration (the 23 launches above)** on the **same physical pair**:
   * GPU 4 = UUID A6000-b10caa4a, PCI 00000000:81:00.0;
   * GPU 5 = UUID A6000-f5f81ac3, PCI 00000000:A1:00.0;
2. a **fresh execution window**: date, start, end and timezone, at least 75 min long.

Idle observations and the expired smoke delegation are not used as this authorization.
