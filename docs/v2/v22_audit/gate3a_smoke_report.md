# Gate 3a — real-GPU smoke test: result

**Label**: SMOKE. These data are not used for any scientific inference.

**Allocation**: GPUs 4,5 (one NVLink pair), approved by the user on
2026-10-09.

**Plan**: [`stage3_plan_rev1.md`](stage3_plan_rev1.md) §6.

## Attempts

| campaign | result | evidence |
|---|---|---|
| `v2.2-smoke-01` | **stopped at S2** by a v2.2 software defect: the watchdog flagged **our own** exiting benchmark as a foreign process, because NVML's 1 s process list lagged process exit. S3 and S4 not run; no retry | `data/v2.2/raw/v2.2-smoke-01/README_STOPPED.txt`; fixed and regression-tested in commit `9935e72` |
| `v2.2-smoke-02` (approved rerun) | **passed: 26 / 26 checks** | `data/v2.2/processed/v2.2-smoke-02/smoke_check.json` |

## v2.2-smoke-02 (2026-10-09, 19:12–19:16 UTC)

| step | what | outcome |
|---|---|---|
| S1 preflight (read-only) | GPUs 4,5 idle (P8, 4.3 MiB, 26 °C, no process); NV4, NUMA 1, 4 NVLinks each; Xid registration OK; counter available; idle update period 99.98 / 100.00 ms (p10–p90 99.0–100.6 ms) | PASS |
| S2 three 2.5 s units through `runner block` | block `COMPLETE`, 3 / 3 OK and `VERIFIED`, one attempt each, no watchdog trip, every process reaped; idle baselines before and after | PASS |
| S3 self-initiated bounded shutdown | 20 s unit with a 5 s per-unit wall limit: our watchdog tripped `WALL_TIME_EXCEEDED`; the benchmark ended on SIGTERM within the grace period (no SIGKILL); 0 survivors; block `STOPPED`, unit `SAFETY_STOP` | PASS |
| S4 controller SIGTERM | SIGTERM sent to **our own** runner about 5 s into a 20 s unit: the runner terminated its benchmark (SIGTERM, no SIGKILL), recorded the cleanup (`owned_process.json`) and exited within 1 s; block `INTERRUPTED` | PASS |
| after all steps | no process on GPUs 4,5; P8, 5 MiB, 28 °C | PASS |

**Details**:
* **Region records**: every record holds start, end and post counter
  values with status 0, the read-call brackets, 4 probe edges per GPU,
  phase, bracket, predicted commit, loop end and CUDA event time. No QC
  flags.
* **Execution evidence**: both ranks recorded each run.

  | configuration | algorithm / protocol | channels | warps |
  |---|---|---|---|
  | auto | RING / SIMPLE | 4 | 17 |
  | forced Ring/Simple | RING / SIMPLE | 4 | 17 |
  | Ring/LL128 | RING / LL128 | 4 | 20 |

  The identical-execution control (auto vs forced Ring/Simple) holds.
* **Per-operation time**: 52.2, 51.9 and 50.5 µs (historical medians
  52.3, 52.3, 50.9 µs).
* **Controller NVML audit** (S2, real NVML):
  * 33,342 calls in total;
  * 8 cumulative-counter reads, all in the idle baselines (2 GPUs × start
    and end × before and after);
  * **0 inside any measured region**, where all 6,176 calls came from the
    sampler thread.
* **Health**: no health events; maximum temperature 33 °C.

**Descriptive only (one run per configuration; no inference)**:
* The integrated `POWER_INSTANT` / counter ratio was about 0.98 (auto),
  0.96 (forced Ring/Simple) and 0.94 (LL128).
* Whether that disagreement really depends on protocol or region length
  is exactly what gate 3b's D2 is designed to measure, over blocks. One
  smoke run per configuration cannot say.
* **Later result (gate 3b)**: the D2 protocol term was −0.12 %
  [−0.48, +0.25] (INCONCLUSIVE). Ratios of single 2.5 s runs are dominated
  by a roughly constant per-region gap between the estimators, so they are
  not evidence of a protocol dependence. See
  [`gate3b_measval_report.md`](gate3b_measval_report.md) §2 D2 and §5.2.

## What this does and does not establish

**It establishes that the v2.2 pipeline works on this hardware**:
* the runner, owned-process control, watchdog, fail-closed QC, NVML
  logging and all-rank execution evidence;
* bounded cleanup, both when our own watchdog cancels a run and when our
  own runner is signalled.

**It does not establish** measurement accuracy or any energy comparison.

**Gate 3b** (measurement validation, `v2.2-measval-01`) still needs a
separate approval.
