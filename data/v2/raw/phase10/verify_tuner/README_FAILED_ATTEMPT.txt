FAILED ATTEMPT (kept unmodified as a record; not used for any claim).
Run: 2026-10-09, GPUs 4-7, stock NCCL 2.27.5 ext-tuner/example plugin + tuner confs generated before the fixes.
Result: 0/28 ruled sizes executed the rule (verify.jsonl). Every collective ran nChannels=1 (profiler files).
Causes, found by reading NCCL 2.27.5 source (src/enqueue.cc getAlgoInfo, src/include/enqueue.h):
 1. Example plugin sets *nChannels = 1 on every call -> NCCL uses 1 channel for every collective.
 2. Example plugin indexes the cost table as float** although NCCL passes a contiguous
    float[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS]; writes go through garbage pointers (rules have no effect;
    the TUNING-log diagnostic run in ../tuner_diag segfaulted, rc -11).
 3. Our generated confs divided AllGather/ReduceScatter sizes by nranks; NCCL passes total bytes
    (elementSize * ncclFuncMaxSendRecvCount = nRanks*count).
Fixes: patches/nccl-2.27.5-example-tuner-fixes.patch (1, 2); recommender.tuner_conf (3).
Re-run: ../verify_tuner_r2.
