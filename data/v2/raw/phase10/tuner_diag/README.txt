Diagnostic run (stock example tuner, NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING, P4_MIX AllReduce 16 MiB). Segfaulted (rc -11)
inside the example plugin's cost-table write; see ../verify_tuner/README_FAILED_ATTEMPT.txt. Diagnostic only.
r2_* (2026-10-09 ~03:42-03:43 UTC, GPUs 4,5): fixed plugin, absolute conf path, NCCL_DEBUG_SUBSYS incl. TUNING.
Single size 512 KiB, sweep 128K-2M and sweep 1K-1G all show "Applied config" and the profiler records the rule's
ALGO/PROTO; this located the relative-path error of verify_tuner_r2.
