# Held-out session 1: execution, QC and provenance report

**No energy, J/op or contrast is computed here.** The inferential analysis runs once, after the three planned sessions, with the strict production entry point and all three Freeze-H records. Nothing in the study is adjusted on interim outcomes.

## Execution

* Session summary: execution_status **COMPLETE**; stop_reason `None`; n_planned 24; n_attempts 24; n_accepted **24**.
* Units: 24 records; 24 accepted; 0 not accepted; 0 scheduled launches with no record (NOT_RUN): none.
* UTC 2026-10-14T00:42:51Z → 2026-10-14T01:26:03Z; deadline `2026-10-14T02:41:33+00:00` (reservation end); remaining at start 7121.643949012971 s, at end 4529.50263981102 s.
* Baselines: start `OK`, end `OK`.
* Postflight: {"owned_processes_left_on_gpus": [], "allocation_problems": []}; incomplete: null.
* R1 time to READY (s): n 24, min 30.009621384961065, max 117.12420733098406.
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
* QC OK; R1 READY (time to READY 30.0–117.1 s);
* R2 recomputed OK. GPU 4 pre-region 30.0–31.0 °C / 1950–1965 MHz; GPU 5 30.0–31.6 °C / 1920–1935 MHz; P2; clock events 0 on both;
* evidence VERIFIED;
* no watchdog trip, owned-process cleanup OK, nothing unsafe;
* measured region length 0.982–0.999 × target.

**Execution keys:**
* A and B: `RING/SIMPLE/4`, so the A/B control execution was identical in every launch;
* C: `RING/LL128/4`.

**Session:**
* Start and end P8 baselines OK (4 counter reads each, only in the gated baselines).
* Xid event registration OK on both GPUs.
* Postflight: no owned process left; no allocation problem.

**Host conditions (recorded, not adjusted):**
* 1-min load at unit start was 28.6–39.6 (mean 34.4).
* About 3 min after the session started, another user's processes appeared on GPUs 0, 1 and 2, not on the allocated pair. The runner's ownership and foreign-process checks cover GPUs 4/5 only, and none tripped.

**Window:** 00:41:33Z → fixed end 02:41:33Z (120 min). The session ran 00:42:51Z → 01:26:03Z (43.2 min), and 4,529 s remained at the end.
