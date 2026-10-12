FAILED ATTEMPT (kept unmodified as a record; not used for any rule-application claim).
Run: 2026-10-09 03:38-03:41 UTC, GPUs 4-7 (<= 4 concurrent), fixed example tuner (patches/nccl-2.27.5-example-tuner-fixes.patch),
confs regenerated with total bytes.
Result: 0/36 ruled sizes executed the rule; 153/153 unruled sizes executed exactly what NCCL automatic executed in the
measurement campaigns (ALGO/PROTO/nChannels), i.e. the nChannels=1 problem of attempt 1 is fixed.
Cause: experimenter error. --policy-dir was passed as a relative path and nccl-tests runs with cwd = its output
directory, so NCCL_TUNER_CONFIG_FILE did not resolve; the example plugin silently uses defaults when the file is
missing. Diagnostic runs with absolute paths (../tuner_diag/r2_*) applied the rules (TUNING log "Applied config").
Fix: phase10.main resolves --policy-dir and checks the conf files exist. Re-run: ../verify_tuner_r3.
