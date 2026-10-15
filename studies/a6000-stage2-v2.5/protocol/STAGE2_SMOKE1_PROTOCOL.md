# Stage 2 software smoke (smoke-1): protocol

**Status.** Finalized text for Freeze-S, written 2026-10-13. It is bound into Freeze-S only after the user confirms the allocated pair and the window.

**What this run is.**
* A bounded GPU **software** validation: exactly three launches.
* Its data never enter calibration, held-out or historical analysis, and are never pooled.
* It is not approval of calibration (23 launches), held-out collection (72 launches), publication or any scientific conclusion.

**What this protocol replaces.** It supersedes the smoke proposal in `STAGE2_CG1_PACKET.md` §8 in two places:
* criterion S5;
* the execution deadline.

The cg1 packet itself is unchanged.

## 1. Identity and implementation

| item | value |
|---|---|
| implementation / method | `stage2-cg1-2026-10-12` / `v2.5-stage2-region`. The smoke-1 CPU corrections change acceptance and provenance code, not the measurement method. Freeze-S binds the exact code hashes |
| native | v2.5 patch `492523e7…876f2f`, binary `all_gather_perf` `38d7a327…ab2cf7`, profiler plugin `0b96c10d…2817b`, libnccl 2.27.5 `1d9c1032…fb7168`. Unchanged since cg1; no rebuild |
| baseline commit | `2b73580e848986905c45bc8ee10706383dc5cd72`, plus the uncommitted Stage 2 worktree. Not committed or pushed |

## 2. Preconditions

1. **Allocation and window.** The user explicitly confirms one A6000 NVLink pair and the current execution window, recorded verbatim as `allocation_confirmation` (`confirmed_by: user`). An idle pair is not an allocation.
2. **Scope.** `scope_expected` is read from that pair before freezing:
   * GPU model, driver, NVLink link, UUIDs in rank order, PCI IDs, NUMA node, power limit, max graphics clock;
   * topology and the rank → NVML-index → UUID/PCI mapping.

   The runner compares it again before any counter read or launch.
3. **Environment.**
   * `LD_PRELOAD` must be unset or empty in the controller; the config check refuses otherwise, and so does every launch.
   * The effective loader and NCCL environment is recorded per launch.
   * `ldd` of the binary is recorded once.
4. **Freeze-S.** The record hash-binds:
   * the completed config;
   * this protocol;
   * the patch, binary, build record and plugin;
   * every loaded code module;
   * the external libnccl.

   It carries the user's authorization and the allocation confirmation, quoted and not paraphrased.
5. **Limits.**
   * No administrator privileges.
   * No driver, clock, power-limit or persistence changes.
   * No action on other users' processes.
   * No automatic retries and no pair substitution.

## 3. Matrix

All launches: AllGather, 2 MiB, arm B (forced Ring/Simple), frozen counts w = 95,597 and R(2.5 s) = 47,799, M4d enabled. Each launch is one fresh process.

| launch | dose | M4c requested pause |
|---|---|---|
| SMOKE-01 | E0 (no poller) | 95,504,453 ns |
| SMOKE-02 | D100 | 47,298,398 ns |
| SMOKE-03 | D500 | 73,968,092 ns |

Preparation, waits, baselines and safety thresholds are unchanged:
* warm-up target 5.0 s;
* 120 s fixed wait before each launch;
* gated session start and end baselines (P8 entry within 60 s, 10 s settle, 30 s measure);
* temperature stop 80 °C;
* watchdog, owned-process cleanup, fail-closed QC and scope checks.

**Idle-gate clock events (correction G, approved by the user 2026-10-13).**
* **Where it applies:** the baseline P8 entry/validity, readiness R1 and the readiness-derivation replay. These accept, on rows in P8 only, the clock-event bits 0x1 (GPU idle) and 0x4 (SW power cap). Any other bit fails, and rows that are not P8 keep the exact rule.
* **Why:** at P8 idle, the A6000s on driver 580 never report 0. Gate 3b recorded 0 of 14,904 idle rows at 0.
* **Recording:** the raw bitmask is kept, and each baseline and wait reports the share of rows carrying each bit.
* **Unchanged:** in-region QC (masks only 0x1, so 0x4 inside [C0 − 0.25 s, work end] still fails) and the held-out pre-region check R2.

## 4. Execution deadline

The deadline is the earlier of:
* the reservation end;
* the smoke start + 30 min, where the smoke start is the moment the smoke command starts after the confirmation.

`stage2_runner.execution_budget` computes it, and the session summary records it under `deadline`. The unit-start and baseline budget gates are unchanged.

## 5. Stop rules

Stop on any of:
* a safety event;
* a foreign or unknown process;
* a scope mismatch;
* a power or clock policy change;
* an Xid event;
* a failed software gate.

A smoke launch that is not accepted stops the session. Failed attempts are never repaired, discarded or replaced, and all attempted and partial output is kept.

## 6. Smoke criteria

The automatic verdict is PASS only if S1–S5 hold. **S6, review, is always required**: an automatic PASS does not replace inspection of all three launches.

* **S1** Scope and identity match `scope_expected`.
* **S2** All required telemetry channels are valid: power, temperature, graphics clock, P-state, clock events and POWER_INSTANT. TOTAL_POWER samples are optional and only recorded.
* **S3** The gated start and end baselines are OK.
* **S4**, for every launch:
  * exit 0 and QC OK;
  * evidence VERIFIED: RING/SIMPLE executed, operation count R + w + 1;
  * a valid Stage 2 record and an exact counter-call inventory;
  * warm-up 5.0 s ± 10 %;
  * region length 2.5 s ± 10 %;
  * |pause deviation| ≤ 2 ms;
  * post-probe ≥ 3 brackets within 0.41 s, all statuses 0.
* **S5** (corrected): dose rule and poller order.
  * **Active poller (D100, D500), per GPU:**
    * delivered ÷ eligible ≥ **95 %** (threshold unchanged) under the L3 rule:
      * a request is eligible iff req + 10 ms ≤ work end;
      * it is delivered iff the read has status 0, starts within 20 ms of the request and ends ≤ work end;
    * work end ≤ stop request ≤ join end, all three present;
    * no dose read starts before the end of C0.
  * **Not failures:**
    * Reads that straddle the work end, or start after it before the stop request ("shutdown" reads), are logged separately with their times and never counted as delivered.
    * A read in flight at the stop request is physically possible. It is logged (`inflight_at_stop`), not required to be absent.
  * **Inactive poller (E0):** no dose reads and no poller times. Missing stop/join times are expected and are not a failure.

  Per-launch detail is kept in `unit.json` → `smoke_checks`.

The M4d end-alignment status is reported as a finding, not as a criterion.

## 7. What is returned, and what is not claimed

**Returned:**
* per-launch execution, QC and smoke-acceptance outcomes;
* the raw native and profiler records, telemetry and baseline raw reads (e0/e1 values, statuses, call times, identity, clock, units);
* requested vs verified configuration and operation counts;
* work-end/stop/join order, dose eligibility and delivery, straddling/shutdown reads, pause and post-probe checks;
* safety and cleanup evidence;
* a readiness recommendation for calibration.

**Not claimed:**
* physical sub-percent energy accuracy;
* resolved latency/energy optima;
* real-training savings.

Unknown systematic components stay unknown. Further runs need separate approval.
