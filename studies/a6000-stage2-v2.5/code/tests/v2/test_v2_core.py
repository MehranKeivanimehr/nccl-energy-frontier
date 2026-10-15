"""CPU-only tests for methodology-v2 accounting, QC and evidence logic (no GPU, no NVML)."""

import copy
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from ncclenergy.v2.attempts import AttemptLog, final_attempt, should_retry
from ncclenergy.v2.common import append_jsonl, read_jsonl
from ncclenergy.v2.evidence import verify
from ncclenergy.v2.qc import classify
from ncclenergy.v2.region import idle_adjusted, parse_region_text, region_metrics
from ncclenergy.v2.telemetry import gaps, integrate_trapezoid, window_summary
from ncclenergy.v2.topology import category, fingerprint, parse_topo_matrix

ROOT = Path(__file__).resolve().parents[2]


P_NS = 100_000_000   # counter commit period on driver 580 (measured in Phase 4)
# v2.2 QC is fail-closed: an OK classification needs complete telemetry and verified execution evidence.
OKA = {"telemetry_problems": [], "evidence_status": "VERIFIED"}


def _gpu(rank, pci, e0, e1, t0=1_000_000_000, dur_s=3.0, ev_ms=2990.0, rc=(0, 0, 0), tail_ms=0.0, p_ctx=60.0):
    """v2-schema GPU record. e1 is the aligned end value; it already contains the idle tail energy."""
    t_loop_end = t0 + 50_000 + int(dur_s * 1e9)
    t_commit = t_loop_end + int(tail_ms * 1e6)
    t1a, t1b = t_commit + 8_000_000, t_commit + 9_000_000
    e1 = e1 + int(round(p_ctx * tail_ms))                       # mJ of idle tail inside the aligned delta
    post_s = (t_commit + P_NS - (t1a + t1b) / 2) * 1e-9
    return {"rank": rank, "cuda_dev": rank, "pci": pci, "nvml_index": 4 + rank, "handle_rc": rc[0],
            "e_start_mJ": e0, "e_end_mJ": e1, "rc_start": rc[1], "rc_end": rc[2],
            "t_start_a_ns": t0, "t_start_b_ns": t0 + 100_000, "t_end_a_ns": t1a, "t_end_b_ns": t1b,
            "event_ms": ev_ms, "event_rc": 0, "n_edges": 3, "phase_ns": t_commit - 7 * P_NS, "bracket_ns": 3_000_000,
            "t_loop_end_ns": t_loop_end, "t_commit_ns": t_commit, "e_post_mJ": e1 + int(round(p_ctx * post_s * 1000)),
            "rc_post": 0, "t_post_a_ns": t_commit + P_NS + 8_000_000, "t_post_b_ns": t_commit + P_NS + 9_000_000,
            "edges": [[t_commit - 7 * P_NS - 1_500_000, t_commit - 7 * P_NS + 1_500_000],
                      [t_commit - 6 * P_NS - 2_000_000, t_commit - 6 * P_NS + 2_000_000]]}


def _region(gpus, iters=1000, warmup=200, agg=1, R=None):
    return {"schema": "nccle-region-v2", "commit_period_ns": P_NS, "coll": "all_reduce", "size_bytes": 1048576,
            "count": 262144, "type": "float", "op": 0, "in_place": 0, "warmup_iters": warmup, "iters": iters,
            "agg_iters": agg, "graph_launches": 0, "R": iters * agg if R is None else R, "nranks": len(gpus),
            "host_sec_per_coll": 3.0e-3, "pid": 1, "nvml_loaded": 1, "nvml_init_rc": 0, "gpus": gpus}


PCI = ["00000000:81:00.0", "00000000:A1:00.0", "00000000:C1:00.0", "00000000:E1:00.0"]


def test_unit_conversion_and_participating_sum():
    rec = _region([_gpu(0, PCI[0], 100_000, 400_000), _gpu(1, PCI[1], 5_000, 305_500)])
    m = region_metrics(rec, expected_pci=PCI[:2], expected_R=1000)
    assert m["flags"] == []
    assert abs(m["E_raw_J"] - (300.0 + 300.5)) < 1e-12           # mJ -> J, summed over participants
    assert abs(m["J_per_coll"] - 600.5 / 1000) < 1e-12
    assert abs(m["t_gpu_s"] - 2.990 / 1000) < 1e-12               # max event time / R
    assert abs(m["T_region_s"] - 3.0) < 1e-6
    assert abs(m["per_gpu"][0]["p_avg_W"] - 100.0) < 1e-6


def test_warmup_excluded_from_denominator():
    a = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)], iters=1000, warmup=10))
    b = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)], iters=1000, warmup=5000))
    assert a["J_per_coll"] == b["J_per_coll"] == 0.3              # denominator is R, never R + w


def test_exact_operation_denominator():
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)], iters=250, agg=4))
    assert m["R"] == 1000 and abs(m["J_per_coll"] - 0.3) < 1e-12
    bad = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)], iters=250, agg=4, R=250))
    assert "R_INCONSISTENT" in bad["flags"]
    notfrozen = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)]), expected_R=999)
    assert "R_NOT_FROZEN_VALUE" in notfrozen["flags"]
    assert classify({"exit_code": 0, "region": notfrozen, "wrong": 0}, T_min=2.0)["qc_status"] == "MEASUREMENT_INVALID"


def test_inactive_gpu_exclusion():
    rec = _region([_gpu(0, PCI[0], 0, 300_000), _gpu(1, PCI[1], 0, 300_000)])
    m = region_metrics(rec, expected_pci=[PCI[0], PCI[2]])        # GPU 6 expected, GPU 5 recorded
    assert "GPU_SET_MISMATCH" in m["flags"]
    ok = region_metrics(rec, expected_pci=PCI[:2])
    # idle adjustment uses only participating GPUs and per-GPU durations
    adj = idle_adjusted(ok, {PCI[0].lower(): 20.0, PCI[1].lower(): 30.0, PCI[2].lower(): 999.0})
    assert abs(adj - (600.0 - (20.0 + 30.0) * 3.0)) < 1e-6


def test_negative_idle_adjusted_not_clamped():
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 30_000)]))     # 10 W average
    assert idle_adjusted(m, {PCI[0].lower(): 25.0}) < 0


def test_counter_failures_and_reversal():
    cases = {
        "COUNTER_NOT_SUPPORTED": _gpu(0, PCI[0], 0, 0, rc=(0, 3, 3)),
        "UNAVAILABLE_DUE_TO_PERMISSIONS": _gpu(0, PCI[0], 0, 0, rc=(4, 0, 0)),
        "COUNTER_NOT_UPDATING": dict(_gpu(0, PCI[0], 500, 500), e_end_mJ=500),
        "COUNTER_REVERSAL_OR_WRAP": _gpu(0, PCI[0], 900_000, 100),
        "COUNTER_IMPLAUSIBLE": _gpu(0, PCI[0], 0, 3_000_000),       # 1000 W
    }
    for flag, g in cases.items():
        m = region_metrics(_region([g]))
        assert flag in m["flags"], (flag, m["flags"])
        if flag != "COUNTER_IMPLAUSIBLE":
            assert m["E_raw_J"] is None and m["J_per_coll"] is None  # never repaired or clamped
        assert classify({"exit_code": 0, "region": m, "wrong": 0}, 2.0)["qc_status"] == "MEASUREMENT_INVALID"
    g = _gpu(0, PCI[0], 0, 300_000)
    g["t_end_a_ns"], g["t_end_b_ns"] = g["t_start_a_ns"] - 10, g["t_start_a_ns"] - 5
    assert "TIME_REVERSAL" in region_metrics(_region([g]))["flags"]
    g = _gpu(0, PCI[0], 0, 300_000)
    g["t_loop_end_ns"] = g["t_start_a_ns"] - 1
    assert "TIME_REVERSAL" in region_metrics(_region([g]))["flags"]


def test_partial_logs():
    good = json.dumps(_region([_gpu(0, PCI[0], 0, 300_000)]))
    recs, flags = parse_region_text(good + "\n" + good[:40])
    assert len(recs) == 1 and flags == ["PARTIAL_REGION_LOG"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.jsonl"
        append_jsonl(p, {"x": 1})
        with open(p, "a") as f:
            f.write('{"x": 2')                                      # crashed writer
        out, partial = read_jsonl(p)
        assert out == [{"x": 1}] and partial


def test_append_only():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.jsonl"
        append_jsonl(p, {"unit_id": "u", "attempt": 0, "qc_status": "OK"})
        before = p.read_bytes()
        append_jsonl(p, {"unit_id": "u", "attempt": 1, "qc_status": "OK"})
        assert p.read_bytes().startswith(before)
        log = AttemptLog(p)
        try:
            log.append({"unit_id": "u"})
            raise AssertionError("missing fields accepted")
        except ValueError:
            pass


def test_irregular_and_missing_samples():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.02, 0.09, 200))                   # irregular sampling
    p = 50 + 10 * t                                                # linear power -> trapezoid exact
    t0, t1 = 1.0, 6.0
    exact = 50 * (t1 - t0) + 5 * (t1 ** 2 - t0 ** 2)
    assert abs(integrate_trapezoid(t, p, t0, t1) - exact) < 1e-9
    assert np.isnan(integrate_trapezoid(t, p, -1.0, 2.0))          # not covered -> NaN, not extrapolated
    tg = np.concatenate([np.arange(0, 1, 0.05), np.arange(2, 3, 0.05)])
    g = gaps(tg, 0.05)
    assert g["n_gaps"] == 1 and g["max_dt"] > 0.9
    arr = np.array([[ti, 4, 100.0, 40, 1900, 8000, 90, 10, 2, 0, 300000, 1000, 2.0, 100.0] for ti in tg])
    s = window_summary(arr, 4, 0.1, 2.5, 0.05)
    assert s["gaps"]["n_gaps"] == 1


def test_unsupported_and_unverified_configurations():
    assert verify([], "AllReduce", 262144, 2, "Tree", "LL", 1000)["status"] == "UNVERIFIED"
    f = lambda rank, algo, proto, n: {"rank": rank, "overflow": 0, "colls": [
        {"func": "AllReduce", "algo": algo, "proto": proto, "nChannels": 4, "nWarps": 16, "count": 262144,
         "datatype": "ncclFloat32", "n": n, "seq_first": 0, "seq_last": n - 1}]}
    ok = verify([f(0, "TREE", "LL", 1201), f(1, "TREE", "LL", 1201)], "AllReduce", 262144, 2, "Tree", "LL", 1000, 1201)
    assert ok["status"] == "VERIFIED" and ok["actual_key"] == "TREE/LL/4" and ok["n_matches_expected"]
    wrong = verify([f(0, "RING", "LL", 1201), f(1, "RING", "LL", 1201)], "AllReduce", 262144, 2, "Tree", "LL", 1000)
    assert wrong["status"] == "UNVERIFIED" and "FORCED_ALGO_NOT_USED" in wrong["reasons"]
    mixed = verify([f(0, "RING", "LL", 1201), f(1, "RING", "SIMPLE", 1201)], "AllReduce", 262144, 2, "default",
                   "default", 1000)
    assert "MIXED_CONFIG" in mixed["reasons"]
    missing = verify([f(0, "RING", "LL", 1201)], "AllReduce", 262144, 2, "default", "default", 1000)
    assert "RANK_FILES_MISSING" in missing["reasons"]
    short = verify([f(0, "RING", "LL", 10), f(1, "RING", "LL", 10)], "AllReduce", 262144, 2, "default", "default", 1000)
    assert "COUNT_BELOW_R" in short["reasons"]


def test_failed_attempts_and_retry_rule():
    infra = {"unit_id": "u", "attempt": 0, "qc_status": "INFRA_FAIL", "retryable": True}
    assert classify({"exit_code": 134}, 2.0)["qc_status"] == "INFRA_FAIL"
    assert classify({"timed_out": True}, 2.0)["retryable"]
    assert classify({"exit_code": 0, "region": None}, 2.0)["qc_status"] == "INFRA_FAIL"
    assert should_retry([infra]) and not should_retry([infra, dict(infra, attempt=1)])
    qcfail = {"unit_id": "u", "attempt": 0, "qc_status": "QC_SHORT", "retryable": False}
    assert not should_retry([qcfail])                              # QC failures are never re-run


def test_retry_selection_is_outcome_blind():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 2)
        atts = []
        for k in range(n):
            st = rng.choice(["INFRA_FAIL", "OK", "QC_SHORT"]) if k == n - 1 else "INFRA_FAIL"
            atts.append({"unit_id": "u", "attempt": k, "qc_status": st, "retryable": st == "INFRA_FAIL",
                         "J_per_coll": rng.uniform(0.1, 10)})
        chosen = final_attempt(atts)["attempt"]
        for _ in range(5):                                          # permute measured values only
            shuffled = copy.deepcopy(atts)
            vals = [a["J_per_coll"] for a in shuffled]
            rng.shuffle(vals)
            for a, v in zip(shuffled, vals):
                a["J_per_coll"] = v
            assert final_attempt(shuffled)["attempt"] == chosen


def test_qc_short_and_throttle():
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 150_000, dur_s=1.5)]))
    assert classify({**OKA, "exit_code": 0, "region": m, "wrong": 0}, 2.0)["qc_status"] == "QC_SHORT"
    m2 = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)]))
    th = classify({**OKA, "exit_code": 0, "region": m2, "wrong": 0, "clock_events_or": 0x40}, 2.0)
    assert th["qc_status"] == "THROTTLED" and th["stop_block"]
    assert classify({**OKA, "exit_code": 0, "region": m2, "wrong": 3}, 2.0)["qc_status"] == "CORRECTNESS_FAIL"
    assert classify({**OKA, "exit_code": 0, "region": m2, "wrong": 0, "foreign_pids": [123]}, 2.0)["stop_block"]
    assert classify({**OKA, "exit_code": 0, "region": m2, "wrong": 0, "clock_events_or": 0x1}, 2.0)["qc_status"] == "OK"


def test_topology_fingerprint_from_real_output():
    text = (ROOT / "data/raw/full/metadata_start/nvidia-smi-topo.txt").read_text()
    topo = parse_topo_matrix(text)
    assert topo["n"] == 8 and topo["links"][(4, 5)] == "NV4" and topo["links"][(5, 6)] == "NODE"
    assert topo["numa"][4] == 1 and topo["numa"][0] == 0
    assert category([4, 5], topo) == "NVLINK_ALL"
    assert category([5, 6], topo) == "PCIE_ONLY"
    assert category([4, 5, 6, 7], topo) == "MIXED_NVLINK_PCIE"
    assert category([2, 3, 4, 5, 6, 7], topo) == "MIXED_NVLINK_PCIE+CROSS_SOCKET"
    pci = {g: f"pci{g}" for g in range(8)}
    a, b = fingerprint([4, 5], topo, pci), fingerprint([6, 7], topo, pci)
    assert a["fingerprint"] == b["fingerprint"]                    # same link pattern
    assert a["concrete_fingerprint"] != b["concrete_fingerprint"]  # different physical GPUs
    assert fingerprint([4, 5], topo, pci)["fingerprint"] != fingerprint([5, 6], topo, pci)["fingerprint"]


def test_sampler_clock_matches_patch_clock():
    info = time.get_clock_info("perf_counter")
    assert "CLOCK_MONOTONIC" in info.implementation and "RAW" not in info.implementation


def test_pci_normalisation():
    from ncclenergy.v2.common import norm_pci
    assert norm_pci("00000000:81:00.0") == norm_pci("0000:81:00.0") == "0000:81:00.0"
    assert norm_pci("00000000:A1:00.0") == "0000:a1:00.0"
    rec = _region([_gpu(0, "0000:81:00.0", 0, 300_000)])
    assert region_metrics(rec, expected_pci=["00000000:81:00.0"])["flags"] == []


def test_tuning_table_allowed_sets():
    from ncclenergy.v2.selection import enabled, parse_tuning_table
    t2 = parse_tuning_table((ROOT / "tests/v2/fixtures/tuning_table_2gpu_nvlink_default.log").read_text())
    t4 = parse_tuning_table((ROOT / "tests/fixtures/nccl_probe_allreduce_4gpu_default.log").read_text())
    assert enabled(t2, "AllReduce", "Ring", "LL128") and enabled(t2, "AllReduce", "Tree", "LL128")
    assert enabled(t4, "AllReduce", "Ring", "LL128") is False and enabled(t4, "AllReduce", "Tree", "LL128") is False
    assert enabled(t4, "AllReduce", "Tree", "Simple") and enabled(t4, "AllReduce", "Ring", "LL")
    for t in (t2, t4):
        assert enabled(t, "AllGather", "Tree", "Simple") is False      # Tree not implemented for AllGather
        assert enabled(t, "ReduceScatter", "Ring", "Simple")
    assert enabled(t4, "AllReduce", "PAT", "Simple") is None or enabled(t4, "AllReduce", "PAT", "Simple") is False


def test_v2_stdout_row_with_skipped_in_place():
    from ncclenergy.v2.runner import parse_stdout_rows
    txt = ("#       size         count      type   redop    root     time   algbw   busbw  #wrong     time\n"
           "     1048576        262144     float     sum      -1    49.56   21.16   21.16       0" + " " * 32 + "\n")
    rows = parse_stdout_rows(txt)
    assert rows == [{"size_bytes": 1048576, "count": 262144, "time_us": 49.56, "wrong": 0}]
    assert parse_stdout_rows(txt.replace("       0", "     N/A"))[0]["wrong"] is None


def test_sw_power_cap_flag_below_limit_is_not_throttling():
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)]))
    low = classify({**OKA, "exit_code": 0, "region": m, "wrong": 0, "clock_events_or": 0x4, "power_frac_max": 0.35}, 2.0)
    assert low["qc_status"] == "OK" and "SW_POWER_CAP_FLAG_BELOW_LIMIT" in low["reasons"]
    high = classify({**OKA, "exit_code": 0, "region": m, "wrong": 0, "clock_events_or": 0x4, "power_frac_max": 0.95}, 2.0)
    assert high["qc_status"] == "THROTTLED"
    both = classify({**OKA, "exit_code": 0, "region": m, "wrong": 0, "clock_events_or": 0x4 | 0x20, "power_frac_max": 0.3}, 2.0)
    assert both["qc_status"] == "THROTTLED" and both["reasons"] == ["SW_THERMAL"]


def test_uuid_redaction():
    from ncclenergy.v2.common import gpu_alias, redact
    u = "GPU-00000000-1111-2222-3333-444444444444"     # synthetic (was a real UUID of this host)
    out = redact(f'{{"CUDA_VISIBLE_DEVICES": "{u},{u}"}}')
    assert u not in out and out.count(gpu_alias(u)) == 2



def test_commit_aligned_tail_correction():
    # 300 J of collectives in 3 s; 40 ms of 60 W idle tail inside the aligned delta (2.4 J)
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000, tail_ms=40.0, p_ctx=60.0)]))
    assert m["flags"] == [], m["flags"]
    pg = m["per_gpu"][0]
    assert abs(pg["aligned_delta_J"] - 302.4) < 1e-9
    assert abs(pg["p_ctx_W"] - 60.0) < 0.05 and abs(pg["tail_J"] - 2.4) < 0.01
    assert abs(m["E_raw_J"] - 300.0) < 0.01                     # tail removed
    assert abs(m["T_region_s"] - 3.0) < 1e-3                     # region = start read -> loop end


def test_commit_phase_failures_are_hard():
    g = dict(_gpu(0, PCI[0], 0, 300_000), n_edges=0, t_commit_ns=-1)
    m = region_metrics(_region([g]))
    assert "COUNTER_PHASE_UNKNOWN" in m["flags"]
    assert classify({"exit_code": 0, "region": m, "wrong": 0}, 2.0)["qc_status"] == "MEASUREMENT_INVALID"
    g = _gpu(0, PCI[0], 0, 300_000)
    g["edges"].append([g["phase_ns"] + 3 * P_NS + 40_000_000, g["phase_ns"] + 3 * P_NS + 42_000_000])
    assert "COUNTER_PHASE_INCONSISTENT" in region_metrics(_region([g]))["flags"]
    g = _gpu(0, PCI[0], 0, 300_000)
    g["t_end_a_ns"] = g["t_commit_ns"] - 1_000_000
    assert "COUNTER_END_READ_BEFORE_COMMIT" in region_metrics(_region([g]))["flags"]
    g = dict(_gpu(0, PCI[0], 0, 300_000), bracket_ns=30_000_000)
    assert "COUNTER_PHASE_IMPRECISE" in region_metrics(_region([g]))["flags"]
    g = _gpu(0, PCI[0], 0, 300_000)
    g["e_post_mJ"] = g["e_end_mJ"] + 100_000                      # ~1100 W idle tail: implausible
    assert "COUNTER_TAIL_IMPLAUSIBLE" in region_metrics(_region([g]))["flags"]


def test_v1_schema_still_parses_but_is_marked():
    g = _gpu(0, PCI[0], 0, 300_000)
    for k in ("n_edges", "phase_ns", "bracket_ns", "t_loop_end_ns", "t_commit_ns", "e_post_mJ", "rc_post",
              "t_post_a_ns", "t_post_b_ns", "edges"):
        g.pop(k)
    rec = dict(_region([g]), schema="nccle-region-v1")
    rec.pop("commit_period_ns")
    m = region_metrics(rec)
    assert "SCHEMA_V1_UNALIGNED" in m["flags"] and m["E_raw_J"] is not None



def test_evidence_matches_on_bytes_for_retyped_allgather():
    # NCCL runs AllGather as ncclInt8 with count x 4 bytes; nccl-tests reports float elements
    f = lambda r: {"rank": r, "overflow": 0, "colls": [{"func": "AllGather", "algo": "RING", "proto": "SIMPLE",
        "nChannels": 8, "nWarps": 17, "count": 16777216, "datatype": "ncclInt8", "n": 533, "seq_first": 0, "seq_last": 532}]}
    ev = verify([f(0), f(1)], "AllGather", 4194304, 2, "default", "default", min_expected=10, exact_expected=533)
    assert ev["status"] == "VERIFIED" and ev["n_matches_expected"] and ev["actual_key"] == "RING/SIMPLE/8"
    other = verify([f(0), f(1)], "AllGather", 1048576, 2, "default", "default", min_expected=10)
    assert other["status"] == "UNVERIFIED"            # a different size must not match


def test_sampled_power_fallback_is_explicit():
    g = _gpu(0, PCI[0], 0, 0, rc=(0, 3, 3))                       # counter not supported (e.g. Pascal)
    m = region_metrics(_region([g]))
    assert classify({**OKA, "exit_code": 0, "region": m, "wrong": 0}, 2.0)["qc_status"] == "MEASUREMENT_INVALID"
    ok = classify({**OKA, "exit_code": 0, "region": m, "wrong": 0, "sampled_energy_J": 300.0}, 2.0, energy_source="sampled")
    assert ok["qc_status"] == "OK"
    none = classify({**OKA, "exit_code": 0, "region": m, "wrong": 0}, 2.0, energy_source="sampled")
    assert none["qc_status"] == "MEASUREMENT_INVALID"


def test_v22_qc_fails_closed_on_missing_check_telemetry_or_evidence():
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)]))
    base = {"exit_code": 0, "region": m, "wrong": 0}
    assert classify({**OKA, **base}, 2.0)["qc_status"] == "OK"
    r = classify({**OKA, **base, "wrong": None}, 2.0)
    assert r["qc_status"] == "MEASUREMENT_INVALID" and r["reasons"] == ["NO_DATA_CHECK"]
    r = classify({**OKA, **base, "wrong": None, "data_check_required": False}, 2.0)
    assert r["qc_status"] == "OK" and "NO_DATA_CHECK_NOT_REQUIRED" in r["reasons"]       # only when declared
    r = classify({**base, "evidence_status": "VERIFIED"}, 2.0)                          # telemetry not reported
    assert r["qc_status"] == "MEASUREMENT_INVALID" and r["reasons"] == ["TELEMETRY_UNKNOWN"]
    r = classify({**base, "evidence_status": "VERIFIED", "telemetry_problems": ["SAMPLE_GAP_GPU4"]}, 2.0)
    assert r["qc_status"] == "MEASUREMENT_INVALID" and "SAMPLE_GAP_GPU4" in r["reasons"]
    for ev in (None, "UNVERIFIED"):
        r = classify({**base, "telemetry_problems": [], "evidence_status": ev}, 2.0)
        assert r["qc_status"] == "EXECUTION_UNVERIFIED"


def test_v22_telemetry_problem_detection():
    from ncclenergy.v2.qc import telemetry_problems
    good = {"n_samples": 60, "gaps": {"n_gaps": 0}, "temp_max_c": 40.0, "clock_events_or": 0, "energy_int_inst_J": 10.0}
    assert telemetry_problems({"errors": 0, "truncated": False}, {4: good, 5: good}, 2, "counter") == []
    assert "SAMPLER_ERRORS" in telemetry_problems({"errors": 2, "truncated": False}, {4: good, 5: good}, 2, "counter")
    assert "TELEMETRY_GPU_MISSING" in telemetry_problems({"errors": 0}, {4: good}, 2, "counter")
    gap = dict(good, gaps={"n_gaps": 1})
    assert "SAMPLE_GAP_GPU5" in telemetry_problems({"errors": 0}, {4: good, 5: gap}, 2, "counter")
    notemp = dict(good, temp_max_c=None)
    assert "NO_TEMP_OR_CLOCK_EVENTS_GPU4" in telemetry_problems({"errors": 0}, {4: notemp}, 1, "counter")
    nosamp = dict(good, energy_int_inst_J=float("nan"), energy_int_lag_J=None)
    assert "NO_SAMPLED_ENERGY_GPU4" in telemetry_problems({"errors": 0}, {4: nosamp}, 1, "sampled")
    assert telemetry_problems(None, None, 2, "counter") == ["NO_TELEMETRY"]


def test_v22_release_audit_fail_closed_paths():
    """Missing exit status, region-parse flags, unaligned schema, unknown power fraction with SW power cap, NaN
    lag integral: none may yield OK (v2.2 release code audit)."""
    from ncclenergy.v2.qc import telemetry_problems
    m = region_metrics(_region([_gpu(0, PCI[0], 0, 300_000)]))
    base = {**OKA, "exit_code": 0, "region": m, "wrong": 0}
    assert classify(base, 2.0)["qc_status"] == "OK"
    nox = {k: v for k, v in base.items() if k != "exit_code"}
    assert classify(nox, 2.0)["reasons"] == ["EXIT_UNKNOWN"]
    r = classify({**base, "region_parse_flags": ["CORRUPT_REGION_LINE"]}, 2.0)
    assert r["qc_status"] == "MEASUREMENT_INVALID" and r["reasons"] == ["REGION_PARSE:CORRUPT_REGION_LINE"]
    r = classify({**base, "region": dict(m, flags=["SCHEMA_V1_UNALIGNED"])}, 2.0)
    assert r["qc_status"] == "MEASUREMENT_INVALID"
    cap = {**base, "clock_events_or": 0x4}
    assert classify({**cap, "power_frac_max": 0.5}, 2.0)["qc_status"] == "OK"          # documented idle artefact
    assert classify({**cap, "power_frac_max": None}, 2.0)["qc_status"] == "THROTTLED"  # unknown: kept
    good = {"n_samples": 60, "gaps": {"n_gaps": 0}, "temp_max_c": 40.0, "clock_events_or": 0}
    nan_lag = dict(good, energy_int_inst_J=float("nan"), energy_int_lag_J=float("nan"))
    assert "NO_SAMPLED_ENERGY_GPU4" in telemetry_problems({"errors": 0}, {4: nan_lag}, 1, "sampled")


def test_window_gaps_include_the_window_edges():
    """A sampling gap that spans the start of the window is a gap (was missed: only in-window samples counted)."""
    import numpy as np
    from ncclenergy.v2.telemetry import window_summary
    t = np.concatenate([np.arange(0.0, 1.0, 0.05), np.arange(1.6, 3.0, 0.05)])     # 0.6 s gap before 1.6
    arr = np.zeros((len(t), 14))
    arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 13] = t, 4, 100.0, 100.0
    inside_only = window_summary(arr, 4, 1.5, 2.5, 0.05)          # window starts inside the gap
    assert inside_only["gaps"]["n_gaps"] == 1
    assert window_summary(arr, 4, 1.7, 2.5, 0.05)["gaps"]["n_gaps"] == 0
