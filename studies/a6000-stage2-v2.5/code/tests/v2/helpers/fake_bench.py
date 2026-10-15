"""Stand-in for a patched nccl-tests process (CPU only). Sleeps for the region and writes the files the runner
reads: region.jsonl, profiler files for 2 ranks, one nccl-tests stdout row and an NCCL log.


    python fake_bench.py <R> <w> <region_s> [counter]
With "counter", the record carries a valid commit-aligned counter region (100 W per GPU during the region,
80 W idle-with-context afterwards, 100 ms commit period); without it the counter reads report NOT_SUPPORTED.
"""
import json
import os
import sys
import time

R, w, region_s = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
counter = len(sys.argv) > 4 and sys.argv[4] == "counter"
now = time.monotonic_ns
P = 100_000_000


def sleep_until(t):
    d = t - now()
    if d > 0:
        time.sleep(d / 1e9)


t0 = now()
phase = t0 - 37_000_000
edges = [[phase + P - 1_000_000, phase + P + 1_000_000], [phase + 2 * P - 1_000_000, phase + 2 * P + 1_000_000]]
sleep_until(phase + 2 * P + 5_000_000)
ts_a = now()
ts_b = ts_a + 1000
time.sleep(region_s)
t_loop_end = now()
k = -(-(t_loop_end + 2_000_000 - phase) // P)
t_commit = phase + k * P
sleep_until(t_commit + 8_000_000)
te_a = now()
te_b = te_a + 1000
sleep_until(t_commit + P + 8_000_000)
tp_a = now()
tp_b = tp_a + 1000
gpus = []
for rank, (pci, idx) in enumerate((("00000000:81:00.0", 4), ("00000000:A1:00.0", 5))):
    e0 = 1_000_000 + rank
    e1 = e0 + int(100.0 * (t_commit - (ts_a + ts_b) // 2) / 1e6)          # 100 W up to the commit (region + tail)
    ep = e1 + int(80.0 * (t_commit + P - (te_a + te_b) // 2) / 1e6)       # 80 W idle-with-context afterwards
    rc = 0 if counter else 3
    gpus.append({"rank": rank, "cuda_dev": rank, "pci": pci, "nvml_index": idx, "handle_rc": 0,
                 "e_start_mJ": e0 if counter else 0, "e_end_mJ": e1 if counter else 0, "rc_start": rc, "rc_end": rc,
                 "t_start_a_ns": ts_a, "t_start_b_ns": ts_b, "t_end_a_ns": te_a, "t_end_b_ns": te_b,
                 "event_ms": region_s * 1000.0, "event_rc": 0,
                 "n_edges": len(edges) if counter else 0, "phase_ns": phase if counter else -1,
                 "bracket_ns": 2_000_000 if counter else -1, "t_loop_end_ns": t_loop_end,
                 "t_commit_ns": t_commit if counter else -1, "e_post_mJ": ep if counter else 0, "rc_post": rc,
                 "t_post_a_ns": tp_a, "t_post_b_ns": tp_b, "edges": edges if counter else []})
rec = {"schema": "nccle-region-v2", "commit_period_ns": 100000000, "coll": "all_reduce", "size_bytes": 1048576,
       "count": 262144, "type": "float", "op": 0, "in_place": 0, "warmup_iters": w, "iters": R, "agg_iters": 1,
       "graph_launches": 0, "R": R, "nranks": 2, "host_sec_per_coll": region_s / R, "pid": 1, "nvml_loaded": 1,
       "nvml_init_rc": 0, "gpus": gpus}
open("region.jsonl", "w").write(json.dumps(rec) + "\n"
                                + ("{corrupt\n{}\n" if os.environ.get("FAKE_BENCH_CORRUPT_REGION") == "1" else ""))
if os.environ.get("FAKE_BENCH_ERRLINE"):         # test-only: an error line in the benchmark's stderr
    sys.stderr.write(os.environ["FAKE_BENCH_ERRLINE"] + "\n")
for rank in (0, 1):
    json.dump({"schema": "nccle-profiler-v1", "plugin": "nccle-algo-v1", "pid": 1, "rank": rank, "nranks": 2,
               "nNodes": 1, "commHash": "abc", "groups": R + w + 1, "overflow": 0,
               "colls": [{"func": "AllReduce", "algo": "RING", "proto": "SIMPLE", "nChannels": 4, "nWarps": 16,
                          "count": 262144, "datatype": "ncclFloat32", "n": R + w + 1}]},
              open(f"prof/prof_1_r{rank}_abc.json", "w"))
print("# fake bench")
print("     1048576        262144     float     sum      -1   250000.0    0.01    0.01       0")
open("nccl.log", "w").write("host:1:1 [0] NCCL INFO NCCL version 2.27.5+cuda12.9\n")
