"""Parser tests on real nccl-tests / NCCL output captured on the benchmark machine
(network-interface lines removed from the NCCL logs)."""

from pathlib import Path

import numpy as np

from ncclenergy.analyze import _forced_ok, label
from ncclenergy.energy import integrate_power_j
from ncclenergy.nccl_parse import (assign_events, parse_nccl_log, parse_stdout, region_offsets,
                                   selection_by_size, unsupported_reason)
from ncclenergy.stats import pareto_front, prob_best

FIX = Path(__file__).parent / "fixtures"


def test_stdout_rows_and_header():
    out = parse_stdout((FIX / "all_gather_2gpu_stdout.txt").read_text(encoding="utf-8"))
    assert out["version"]["nccl_tests"] == "2.21.1"
    assert out["version"]["nccl_library"] == 22705
    assert out["params"]["nthread"] == 2 and out["params"]["iters"] == 3
    assert [d["pci"] for d in out["devices"]] == ["0000:81:00", "0000:a1:00"]
    assert len(out["rows"]) == 2
    r = out["rows"][0]
    assert r["size_bytes"] == 1024 and r["count"] == 128 and r["redop"] == "none"
    assert r["oop_time_us"] == 32.02 and r["ip_time_us"] == 16.76
    assert r["oop_wrong"] is None  # -c 0 prints N/A
    assert out["oob"] == {"count": 0, "status": "OK"}


def test_probe_rows_have_wrong_counts():
    out = parse_stdout((FIX / "stdout_probe_allreduce_4gpu_default.txt").read_text(encoding="utf-8"))
    assert out["params"]["validation"] == 1
    assert all(r["oop_wrong"] == 0 and r["ip_wrong"] == 0 for r in out["rows"])


def test_region_offsets_point_at_token_ends():
    text = (FIX / "all_gather_2gpu_stdout.txt").read_text(encoding="utf-8")
    offs = region_offsets(text)
    assert len(offs) == 2
    data = text.encode()
    o = offs[0]
    assert data[:o["pre_end"]].decode().endswith("-1")          # root column
    assert data[:o["oop_end"]].decode().endswith("N/A")         # out-of-place #wrong
    assert data[:o["ip_end"]].decode().endswith("N/A")
    assert o["pre_end"] < o["oop_end"] < o["ip_end"]
    # chunk mapping: a chunk ending exactly at a token end delivers that token
    ev = assign_events(offs, [o["pre_end"], o["oop_end"], o["ip_end"] + 1, len(data)])
    assert (ev[0]["pre_chunk"], ev[0]["oop_chunk"], ev[0]["ip_chunk"]) == (0, 1, 2)


def test_nccl_log_tuning_and_transports():
    log = parse_nccl_log((FIX / "nccl_probe_allreduce_4gpu_default.log").read_text(encoding="utf-8"))
    assert log["nccl_version"] == "2.27.5+cuda12.9"
    assert log["env"]["NCCL_RUNTIME_CONNECT"] == "0."
    assert [r["nvml_dev"] for r in log["ranks"]] == [4, 5, 6, 7]
    kinds = {t[2] for t in log["transports"]}
    assert "P2P/direct pointer" in kinds and "SHM/direct/direct" in kinds
    sel = selection_by_size(log["tuning"], "AllReduce")
    assert sel[4096] == [("RING", "LL", 0, 0)]
    assert log["enabled_matrix"] == {}  # NCCL prints the matrix only when NCCL_ALGO/PROTO are set


def test_unsupported_detection():
    log = parse_nccl_log((FIX / "nccl_unsupported_allgather_tree.log").read_text(encoding="utf-8"))
    out = parse_stdout((FIX / "stdout_unsupported_allgather_tree.txt").read_text(encoding="utf-8"))
    reason = unsupported_reason(log, out, "")
    assert reason and "no algorithm/protocol available for function AllGather" in reason
    assert log["env"]["NCCL_ALGO"] == "Tree"
    # the matrix reflects the requested mask, not what the collective implements
    assert log["enabled_matrix"]["AllGather"]["algo"]["Tree"] == 1
    assert log["enabled_matrix"]["AllGather"]["algo"]["Ring"] == 0
    assert out["rows"] == [] and out["failures"]


def test_integration_trapezoid():
    t = np.linspace(0, 10, 101)
    p = np.full_like(t, 100.0)
    assert abs(integrate_power_j(t, p, 2.05, 7.05) - 500.0) < 1e-9
    p2 = 10 * t  # linear ramp: integral of 10 t from 1 to 3 = 40
    assert abs(integrate_power_j(t, p2, 1.0, 3.0) - 40.0) < 1e-9
    assert np.isnan(integrate_power_j(t, p, -1, 3))  # not covered


def test_labels_and_forced_check():
    assert label("default", "LL128") == "auto/LL128"
    assert _forced_ok("Ring", "default", [{"algo": "RING", "proto": "LL"}])
    assert not _forced_ok("Tree", "LL", [{"algo": "RING", "proto": "LL"}])


def test_pareto_and_pbest():
    pts = [("a", 1, 5), ("b", 2, 2), ("c", 3, 3), ("d", 0.5, 9)]
    assert pareto_front(pts) == {"a", "b", "d"}
    rng = np.random.default_rng(0)
    pb = prob_best({"x": np.array([1.0, 1.1, 0.9]), "y": np.array([5.0, 5.1, 4.9])}, 1000, rng)
    assert pb["x"] == 1.0
