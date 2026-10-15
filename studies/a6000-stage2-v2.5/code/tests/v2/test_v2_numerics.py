"""Numerical-comparison logic and the workload's finite/consistency checks. The workload test runs the real
training script on CPU only (torch gloo backend, 2 processes, tiny model, no energy meter, no CUDA, no NVML)."""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ncclenergy.v2.numerics import compare_numerics, finite_summary, global_param_norm

ROOT = Path(__file__).resolve().parents[2]


def _res(traj, pnorm=10.0, seed=0, finite=True, ranks=True, **sem):
    s = {"mode": "ddp", "world": 2, "seed": seed, "batch": 8, "global_batch": 16, "steps": len(traj), "warmup": 2,
         "hidden": 32, "layers": 2, "dtype": "float32", "data": "identical"}
    s.update(sem)
    return {"semantics": s, "numerics": {"loss_trajectory": traj, "loss_all_finite": finite,
                                         "grads_all_finite": finite, "ranks_consistent": ranks, "param_norm": pnorm}}


def test_compare_numerics_tolerance_and_failure_modes():
    a = _res([1.0, 0.9, 0.8])
    assert compare_numerics(a, _res([1.0, 0.9, 0.8]))["within_tolerance"] is True
    near = compare_numerics(a, _res([1.0, 0.9, 0.8 * (1 + 5e-4)], pnorm=10.0 * (1 + 5e-5)))
    assert near["within_tolerance"] is True and 4e-4 < near["max_rel_loss_diff"] < 6e-4
    far = compare_numerics(a, _res([1.0, 0.9, 0.85]))
    assert far["comparable"] and far["within_tolerance"] is False
    pfar = compare_numerics(a, _res([1.0, 0.9, 0.8], pnorm=10.1))
    assert pfar["within_tolerance"] is False
    # not comparable -> never "within tolerance"
    for b, why in ((_res([1.0, 0.9, 0.8], seed=1), "SEMANTICS_DIFFER:seed"),
                   (_res([1.0, 0.9]), "SEMANTICS_DIFFER:steps"),
                   (_res([1.0, 0.9, float("nan")], finite=False), "NONFINITE:b"),
                   (_res([1.0, 0.9, 0.8], ranks=False), "RANKS_INCONSISTENT:b")):
        r = compare_numerics(a, b)
        assert not r["comparable"] and r["within_tolerance"] is None and why in r["reasons"], (why, r)
    missing = _res([1.0, 0.9, 0.8])
    del missing["semantics"]["dtype"]
    assert "SEMANTICS_MISSING:dtype" in compare_numerics(a, missing)["reasons"]


def test_finite_summary_and_param_norm():
    f = finite_summary([1.0, float("inf"), 0.5])
    assert not f["all_finite"] and f["nonfinite_steps"] == [1]
    assert finite_summary([1.0, 0.5])["all_finite"]
    assert global_param_norm("ddp", [4.0, 4.0]) == 2.0              # replicated: rank 0 holds all parameters
    assert global_param_norm("fsdp", [4.0, 5.0]) == 3.0             # sharded: sum over ranks
    assert global_param_norm("ddp", [float("nan")]) is None


def _cpu_run(tmp, seed, name, mode="ddp", data=None):
    out = Path(tmp) / f"{name}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NCCL_", "CUDA_"))}
    env.update(PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           "-m", "ncclenergy.v2.workload", "--mode", mode, "--device", "cpu", "--backend", "gloo", "--meter", "none",
           "--hidden", "32", "--layers", "2", "--batch", "8", "--warmup", "2", "--steps", "6", "--seed", str(seed),
           "--policy", name, "--out", str(out)] + (["--data", data] if data else [])
    p = subprocess.run(cmd, env=env, cwd=tmp, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(out.read_text())


def test_cpu_workload_records_numerics_and_comparison_works():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = _cpu_run(tmp, 0, "a"), _cpu_run(tmp, 0, "b"), _cpu_run(tmp, 1, "c")
        for r in (a, b, c):
            n = r["numerics"]
            assert n["loss_all_finite"] and n["grads_all_finite"] and n["ranks_consistent"]
            assert len(n["loss_trajectory"]) == 6 and all(math.isfinite(x) for x in n["loss_trajectory"])
            assert r["semantics"]["global_batch"] == 16 and r["meter"] is None and r["energy_J"] is None
            assert [p["rank"] for p in n["per_rank"]] == [0, 1]
        same = compare_numerics(a, b)
        assert same["comparable"] and same["within_tolerance"] and same["max_rel_loss_diff"] <= 1e-6
        assert compare_numerics(a, c)["reasons"] == ["SEMANTICS_DIFFER:seed"]


def test_per_rank_data_detects_a_missing_gradient_allreduce():
    """v2.2 release audit: with identical data on every rank (historical Phase 10), DDP without the gradient
    AllReduce (ddp_nosync) still looks consistent; with per-rank data the replicas diverge and the check fails."""
    with tempfile.TemporaryDirectory() as tmp:
        ok = _cpu_run(tmp, 0, "ok", "ddp")["numerics"]
        miss = _cpu_run(tmp, 0, "miss", "ddp_nosync")["numerics"]
        blind = _cpu_run(tmp, 0, "blind", "ddp_nosync", data="replicated")["numerics"]
    assert ok["consistency_check"] == "DDP_REPLICAS_EQUAL_PER_RANK_DATA" and ok["ranks_consistent"] is True
    assert miss["ranks_consistent"] is False and miss["ranks_consistent_expected"] is False
    assert blind["consistency_check"] == "LOSS_TRAJECTORIES_EQUAL_REPLICATED_DATA" and blind["ranks_consistent"] is True
