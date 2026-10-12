Gate 3a smoke test v2.2-smoke-01 (GPUs 4,5; approved by the user 2026-10-09). STOPPED at step S2; S3 and S4 not run.

S1 (read-only preflight, s1_preflight/): passed. GPUs 4,5 idle, NV4, NUMA 1, 4 NVLinks each; builds identical to the
historical v2 campaign (NCCL lib, patch v2.1, profiler plugin, all_gather_perf); counter available, idle update period
99.92 / 100.03 ms (p10-p90 99.26-100.64 ms).

S2 (blocks/block_01, 2026-10-09 19:03:06-19:04:05 UTC): units Ring/Simple and Ring/LL128 OK; unit auto ended
SAFETY_STOP "WATCHDOG:FOREIGN_PROCESS:GPU5:[218029]" and the block stopped (no retry, as pre-specified).
Diagnosis: pid 218029 was OUR OWN benchmark process (owned_process.json of that unit: pid 218029). The watchdog's
ownership set contained only live owned processes, while NVML's process list (sampled every 1 s) still listed the
benchmark for up to a sampling period after it exited. The same false trip was recorded after exit in the other two
units (owned_process.json watchdog_trip). A software defect of v2.2, not contention: no other process used GPUs 4,5.
Checks on the collected evidence (data/v2.2/processed/v2.2-smoke-01/smoke_check_partial_s2.json): 17/20 pass; the 3
failures are this stop. Region records, execution evidence on both ranks (auto = forced Ring/Simple =
RING/SIMPLE/4ch/17 warps; LL128 = RING/LL128/4ch/20 warps), telemetry, cleanup, and the controller NVML audit
(0 counter reads inside regions; 4 counter reads, all in the idle baseline) all passed.
After the stop: no process of ours; GPUs 4,5 P8, 4.3 MiB used; no health events.
Fix (commit after this one): ownership for foreign-process checks = every pid owned during the run unless recycled;
a watchdog trip recorded after exit is now itself a safety problem. Regression tests reproduce the race.
This campaign is not rerun; a repeat needs a new campaign id and a new approval.
