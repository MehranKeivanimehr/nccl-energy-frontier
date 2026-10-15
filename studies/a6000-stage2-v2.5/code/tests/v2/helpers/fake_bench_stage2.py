"""Stand-in for the v2.4 Stage 2 nccl-tests binary (CPU only). Parses the nccl-tests arguments the runner passes,
honours the Stage 2 environment exactly as the native code does (refusals included), sleeps the region for real and
writes region.jsonl (v2.1 schema), s2.jsonl (nccle-s2-v1), profiler files for 2 ranks, one stdout row and a log.

Test knobs: FAKE_S2_PREP_S (actual warm-up duration, default 5.0), FAKE_S2_EXTRA_READS (adds to the reported
counter-call count), FAKE_REGION_S (default from R at 2.5 s per 47,799 ops).
"""
import json
import os
import sys
import time

argv = sys.argv[1:]
R = int(argv[argv.index("-n") + 1])
w = int(argv[argv.index("-w") + 1])
now = time.monotonic_ns
P = 100_000_000
env = os.environ

mode = env.get("NCCLE_STAGE2_MODE")
s2vars = [k for k in ("NCCLE_S2_FILE", "NCCLE_START_PAUSE_NS", "NCCLE_DOSE_PERIOD_MS", "NCCLE_POST_PROBE") if k in env]


def refuse(msg):
    sys.stderr.write(f"NCCLE: {msg}\n")
    sys.exit(3)


if mode is None and s2vars:
    refuse("Stage 2 variables set without NCCLE_STAGE2_MODE")
if mode not in (None, "calibration", "heldout"):
    refuse("NCCLE_STAGE2_MODE must be calibration or heldout")
cal = mode == "calibration"
dose_ms = int(env["NCCLE_DOSE_PERIOD_MS"]) if "NCCLE_DOSE_PERIOD_MS" in env else 0
pp = env.get("NCCLE_POST_PROBE", "0") == "1"
if (dose_ms or pp) and not cal:
    refuse("dose / post-probe refused outside calibration")
if dose_ms not in (0, 100, 500):
    refuse("NCCLE_DOSE_PERIOD_MS must be 100 or 500")
pause_ns = int(env.get("NCCLE_START_PAUSE_NS", "-1"))
if mode and not 0 <= pause_ns < 100_000_000:
    refuse("NCCLE_START_PAUSE_NS must be an integer in [0, 100000000)")

region_s = float(env.get("FAKE_REGION_S", R * 2.5 / 47799))
prep_s = float(env.get("FAKE_S2_PREP_S", 5.0))
time.sleep(0.5)                                      # the controller's sampler sees >= 250 ms before C0
t0a = now()
t0b = t0a + 1_300_000
phase = t0a - 23_000_000                             # start phase 23 ms
# pre-C0 times are synthetic and in the past (only their order and spans are used)
t_pause_b = t0a - 50_000
t_pause_a = t_pause_b - pause_ns - 100_000
t_probe_end = t_pause_a - 50_000
t_probe_start = t_probe_end - 350_000_000
t_prep_end = t_probe_start - 100_000
t_prep_start = t_prep_end - int(prep_s * 1e9)
edges = [[phase - k * P - 1_100_000, phase - k * P + 1_100_000] for k in (4, 3, 2)]
dose_reads = []
base = t0b
t_end_target = t0a + int(region_s * 1e9)
skipped = 0
STOP_AFTER_WORK_END_NS = 20_000                       # the controller requests the stop right after the work end
if dose_ms:                                           # same behaviour as the v2.5 native poller (one rule, see parser)
    Pd = dose_ms * 1_000_000
    k = 1
    while True:
        req = base + k * Pd
        if req >= t_end_target + STOP_AFTER_WORK_END_NS:
            break
        d = req - now()
        if d > 0:
            time.sleep(d / 1e9)
        ta = now()
        time.sleep(0.0013)                            # a real ~1.3 ms read
        dose_reads.append([k, req, ta, now(), 1000 + k, 0])
        k_next = (now() - base) // Pd + 1
        if k_next > k + 1:
            skipped += k_next - k - 1
            k = k_next
        else:
            k += 1
d = t_end_target - now()
if d > 0:
    time.sleep(d / 1e9)
t_work_end = t_end_target                             # synchronized GPU-work end, recorded before the poller stop
stop_req = t_work_end + STOP_AFTER_WORK_END_NS
inflight = int(any(r[2] < stop_req < r[3] for r in dose_reads))
joined = max(now(), stop_req) + 30_000
while now() < joined + 10_000:
    pass
t_loop_end = now()                                    # v2.1 loop end, taken after the join
k = -(-(t_loop_end + 2_000_000 - phase) // P)
t_commit = phase + k * P


def sleep_until(t):
    d = t - now()
    if d > 0:
        time.sleep(d / 1e9)


sleep_until(t_commit + 8_000_000)
te_a = now()
te_b = te_a + 1_300_000
sleep_until(t_commit + P + 8_000_000)
tp_a = now()
tp_b = tp_a + 1_300_000
pp_reads = []
if pp:
    t, v, grid = tp_b + 100_000, 9000, t_commit + 2 * P
    n_edges = 0
    while t < tp_b + 4 * P and n_edges < 4:
        if t > grid:
            v += 7
            grid += P
            n_edges += 1
        pp_reads.append([t, t + 1_350_000, v, 0])
        t += 1_400_000
gpus_r, gpus_s = [], []
for rank, (pci, idx) in enumerate((("00000000:81:00.0", 4), ("00000000:A1:00.0", 5))):
    e0 = 1_000_000 + rank
    e1 = e0 + int(150.0 * (t_commit - (t0a + t0b) // 2) / 1e6)
    ep = e1 + int(80.0 * (t_commit + P - (te_a + te_b) // 2) / 1e6)
    gpus_r.append({"rank": rank, "cuda_dev": rank, "pci": pci, "nvml_index": idx, "handle_rc": 0,
                   "e_start_mJ": e0, "e_end_mJ": e1, "rc_start": 0, "rc_end": 0, "t_start_a_ns": t0a, "t_start_b_ns": t0b,
                   "t_end_a_ns": te_a, "t_end_b_ns": te_b, "event_ms": region_s * 1000.0, "event_rc": 0,
                   "n_edges": len(edges), "phase_ns": phase, "bracket_ns": 2_200_000, "t_loop_end_ns": t_loop_end,
                   "t_commit_ns": t_commit, "e_post_mJ": ep, "rc_post": 0, "t_post_a_ns": tp_a, "t_post_b_ns": tp_b,
                   "edges": edges})
    probe_reads = 254
    bad_part = env.get("FAKE_S2_BAD_PART")                 # test-only: only launches whose directory names this part
    extra = int(env.get("FAKE_S2_EXTRA_READS", 0)) + (1 if bad_part and bad_part in os.getcwd() else 0)
    n = 1 + probe_reads + 1 + len(dose_reads) + 1 + 1 + len(pp_reads) + extra
    gpus_s.append({"rank": rank, "pci": pci, "n_counter_reads": n, "e_prep0_mJ": e0 - 500_000, "rc_prep0": 0,
                   "t_prep0_a_ns": t_prep_start - 1_400_000, "t_prep0_b_ns": t_prep_start - 10_000,
                   "t_prep_start_ns": t_prep_start, "t_prep_end_ns": t_prep_end, "t_probe_start_ns": t_probe_start,
                   "t_probe_end_ns": t_probe_end, "probe_reads": probe_reads, "t_pause_a_ns": t_pause_a,
                   "t_pause_b_ns": t_pause_b, "t_work_end_ns": t_work_end,
                   "dose": {"base_ns": base if dose_ms else -1, "stop_req_ns": stop_req if dose_ms else -1,
                            "joined_ns": joined if dose_ms else -1, "n": len(dose_reads), "skipped": skipped,
                            "overflow": 0, "start_failed": 0, "inflight_at_stop": inflight, "reads": dose_reads},
                   "post_probe": {"start_ns": tp_b + 100_000 if pp else -1,
                                  "end_ns": (pp_reads[-1][1] if pp_reads else -1), "n": len(pp_reads),
                                  "edges": 4 if pp else 0, "overflow": 0, "reads": pp_reads}})
count = 524288
rec = {"schema": "nccle-region-v2", "commit_period_ns": P, "coll": "all_gather", "size_bytes": 2097152, "count": count,
       "type": "float", "op": 0, "in_place": 0, "warmup_iters": w, "iters": R, "agg_iters": 1, "graph_launches": 0,
       "R": R, "nranks": 2, "host_sec_per_coll": region_s / R, "pid": os.getpid(), "nvml_loaded": 1, "nvml_init_rc": 0,
       "gpus": gpus_r}
with open(env["NCCLE_REGION_FILE"], "a") as f:
    f.write(json.dumps(rec) + "\n")
if mode:
    with open(env["NCCLE_S2_FILE"], "a") as f:
        f.write(json.dumps({"schema": "nccle-s2-v1", "pid": os.getpid(), "mode": mode, "coll": "all_gather",
                            "size_bytes": 2097152, "R": R, "warmup_iters": w, "pause_req_ns": pause_ns,
                            "dose_period_ms": dose_ms, "post_probe": int(pp), "commit_period_ns": P,
                            "gpus": gpus_s}) + "\n")
algo = env.get("NCCL_ALGO", "Ring").upper()
proto = env.get("NCCL_PROTO", "Simple").upper()
prof = env.get("NCCLE_PROFILER_DIR", "prof")
os.makedirs(prof, exist_ok=True)
for rank in (0, 1):
    with open(os.path.join(prof, f"prof_1_r{rank}_abc.json"), "w") as f:
        json.dump({"schema": "nccle-profiler-v1", "plugin": "nccle-algo-v1", "pid": 1, "rank": rank, "nranks": 2,
                   "nNodes": 1, "commHash": "abc", "groups": R + w + 1, "overflow": 0,
                   "colls": [{"func": "AllGather", "algo": algo, "proto": proto, "nChannels": 2, "nWarps": 16,
                              "count": count, "datatype": "ncclFloat32", "n": R + w + 1}]}, f)
print("# fake stage2 bench")
print(f"     2097152        {count}     float    none      -1   {region_s / R * 1e6:.2f}    0.01    0.01       0")
with open(env.get("NCCL_DEBUG_FILE", "nccl.log"), "w") as f:
    f.write("host:1:1 [0] NCCL INFO NCCL version 2.27.5+cuda12.9\n")
