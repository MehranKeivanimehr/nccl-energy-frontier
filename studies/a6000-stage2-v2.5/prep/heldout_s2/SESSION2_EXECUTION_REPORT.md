# Held-out session 2: execution, QC and provenance report

**No energy, J/op or contrast is computed here.** The inferential analysis runs once, after the three planned sessions, with the strict production entry point and all three Freeze-H records. Nothing in the study is adjusted on interim outcomes.

## Execution

* Session summary: execution_status **COMPLETE**; stop_reason `None`; n_planned 24; n_attempts 24; n_accepted **24**.
* Units: 24 records; 24 accepted; 0 not accepted; 0 scheduled launches with no record (NOT_RUN): none.
* UTC 2026-10-14T01:39:12Z → 2026-10-14T02:20:25Z; deadline `2026-10-14T03:38:31+00:00` (reservation end); remaining at start 7158.412390790007 s, at end 4685.41381928901 s.
* Baselines: start `OK`, end `OK`.
* Postflight: {"owned_processes_left_on_gpus": [], "allocation_problems": []}; incomplete: null.
* R1 time to READY (s): n 24, min 30.01006674702512, max 105.11726493400056.
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
* R2 recomputed OK. GPU 4 pre-region 29.4–31.0 °C / 1950–1965 MHz; GPU 5 30.0–31.0 °C / 1920–1935 MHz; P2; clock events 0 on both;
* evidence VERIFIED;
* no watchdog trip, owned-process cleanup OK, nothing unsafe;
* measured region length 0.980–0.998 × target.

**Execution keys:**
* A and B: `RING/SIMPLE/4`, so the A/B control execution was identical in every launch;
* C: `RING/LL128/4`.

**Session:**
* Start and end P8 baselines OK (4 counter reads each, only in the gated baselines).
* Xid event registration OK on both GPUs.
* Postflight: no owned process left; no allocation problem.

**Host conditions (recorded, not adjusted):**
* 1-min load at unit start was 26.3–40.9 (mean 34.1).
* Another user's jobs were running on GPUs 0 and 2 at the pre-launch check, and on GPUs 0, 1 and 2 after the session. None was on the allocated pair. The runner's ownership and foreign-process checks cover GPUs 4/5 only, and none tripped.

**Window:** 01:38:31Z → fixed end 03:38:31Z (120 min). The session ran 01:39:12Z → 02:20:25Z (41.2 min), and 4,685 s remained at the end.
