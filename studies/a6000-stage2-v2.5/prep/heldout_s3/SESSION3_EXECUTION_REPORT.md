# Held-out session 3: execution, QC and provenance report

**No energy, J/op or contrast is computed here.** The inferential analysis runs once, after the three planned sessions, with the strict production entry point and all three Freeze-H records. Nothing in the study is adjusted on interim outcomes.

## Execution

* Session summary: execution_status **COMPLETE**; stop_reason `None`; n_planned 24; n_attempts 24; n_accepted **24**.
* Units: 24 records; 24 accepted; 0 not accepted; 0 scheduled launches with no record (NOT_RUN): none.
* UTC 2026-10-14T02:28:30Z → 2026-10-14T03:10:35Z; deadline `2026-10-14T04:27:58+00:00` (reservation end); remaining at start 7167.930687852029 s, at end 4642.007570485992 s.
* Baselines: start `OK`, end `OK`.
* Postflight: {"owned_processes_left_on_gpus": [], "allocation_problems": []}; incomplete: null.
* R1 time to READY (s): n 24, min 30.00990919797914, max 105.10931259900099.
* Not accepted: {}.

## Provenance pre-check (no energies; same rules as the final analysis)

```
{
 "freeze_h": "no problems",
 "not_verified_offline": [],
 "config": "no problems",
 "run_config_sha256_equals_freeze_h": true,
 "summary_identity": [
  "STAGE2_HELDOUT",
  "stage2-cg1-2026-10-12",
  "v2.5-stage2-region"
 ],
 "record_identity_problems": "none",
 "scope_observed_matches": "yes"
}
```

## Per-launch table

`run/execution_qc.tsv`, one row per attempt record, in execution order.

## QC summary (from `run/execution_qc.tsv` and the raw records)

**Every launch:**
* QC OK; R1 READY (time to READY 30.0–105.1 s);
* R2 recomputed OK. GPU 4 pre-region 29.0–31.0 °C / 1950–1965 MHz; GPU 5 30.0–31.6 °C / 1920–1935 MHz; P2; clock events 0 on both;
* evidence VERIFIED;
* no watchdog trip, owned-process cleanup OK, nothing unsafe;
* measured region length 0.978–0.996 × target.

**Execution keys:**
* A and B: `RING/SIMPLE/4`, so the A/B control execution was identical in every launch;
* C: `RING/LL128/4`.

**Session:**
* Start and end P8 baselines OK (4 counter reads each, only in the gated baselines).
* Xid event registration OK on both GPUs.
* Postflight: no owned process left; no allocation problem.

**Host conditions (recorded, not adjusted):**
* 1-min load at unit start was 28.2–40.7 (mean 36.0).
* Another user's jobs were running on GPUs 0, 1 and 2 at the pre-launch check and after the session; none was on the allocated pair. None of the runner's checks for GPUs 4/5 tripped.

**Window:** 02:27:58Z → fixed end 04:27:58Z (120 min). The session ran 02:28:30Z → 03:10:35Z (42.1 min), and 4,642 s remained at the end.
