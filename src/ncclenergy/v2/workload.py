"""Small synthetic distributed PyTorch workload (Phase 10). Launch with torchrun, one process per GPU.

    GPU collection is disabled pending a validated guarded harness and numerical protocol.

Synthetic data only. Measures step time (CUDA events), throughput and raw
participating-GPU board energy per step in the same run.

Semantics (recorded in "semantics"): torch.manual_seed(seed) before model
construction, so every rank holds the same initial weights; the global batch is
batch x world. FP32, TF32 off. Data (v2.2 release, --data): "per_rank" (default)
draws a different synthetic batch on every rank from seed and rank, so a missing
or local-only gradient reduction makes DDP replicas diverge and is detected by the
replicated-parameter check; "replicated" (the historical Phase 10 runs) gives every
rank the same batch, under which a skipped AllReduce is NOT detectable.

v2.2 additions (recorded in "numerics" and "meter"): per-step loss trajectory on
every rank (kept on the device during the measured region and copied after it),
finite-loss and finite-gradient checks, local parameter sum of squares per rank,
an all-rank consistency check, and the full meter record (raw counter values,
read brackets, status codes, alignment and QC flags; or the sampled meter).

--device cpu --backend gloo --meter none runs the same code path on CPU only
(used by the CPU regression tests; no CUDA, no NVML).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(h, h, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.lin(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ddp", "ddp_nosync", "fsdp"], required=True)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bucket-mb", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nvml-gpus", default="", help="physical NVML indices of the participating GPUs, rank order")
    ap.add_argument("--policy", default="unlabelled")
    ap.add_argument("--meter", choices=["aligned", "sampled", "none"], default="aligned")
    ap.add_argument("--power-lag-s", type=float, default=0.0, help="PowerUsage lag for --meter sampled")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--backend", default="nccl")
    ap.add_argument("--data", choices=["per_rank", "replicated"], default="per_rank")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cuda":
        raise SystemExit("CUDA workload collection is disabled in this pilot release; validate the guarded "
                         "harness and preregister numerical comparisons first. CPU regression tests remain available.")
    if a.meter != "none" and not a.nvml_gpus:
        raise SystemExit("--nvml-gpus is required with an energy meter")

    dist.init_process_group(a.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    cuda = a.device == "cuda"
    if cuda:
        torch.cuda.set_device(lr)
        torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", lr) if cuda else torch.device("cpu")
    torch.manual_seed(a.seed)
    model = nn.Sequential(*[Block(a.hidden) for _ in range(a.layers)]).to(dev)
    if a.mode.startswith("ddp"):
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[lr] if cuda else None, bucket_cap_mb=a.bucket_mb)
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({Block}), device_id=lr if cuda else None)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    if a.data == "per_rank":
        gen = torch.Generator().manual_seed(a.seed * 100_003 + rank + 1)
        x = torch.randn(a.batch, a.hidden, generator=gen).to(dev)
    else:
        x = torch.randn(a.batch, a.hidden, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        if a.mode == "ddp_nosync":
            with model.no_sync():          # compute-only reference: no gradient AllReduce
                loss = model(x).pow(2).mean()
                loss.backward()
        else:
            loss = model(x).pow(2).mean()
            loss.backward()
        opt.step()
        return loss.detach()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    for _ in range(a.warmup):
        step()
    sync()
    dist.barrier()
    meter, probe = None, None
    if rank == 0 and a.meter != "none":
        from .aligned_meter import AlignedMeter, SampledMeter
        gpus = [int(g) for g in a.nvml_gpus.split(",")]
        meter = AlignedMeter(gpus) if a.meter == "aligned" else SampledMeter(gpus, lag_s=a.power_lag_s)
        probe = meter.probe()
    sync()
    dist.barrier()
    if meter is not None:
        meter.start()
    dist.barrier()
    losses = []
    if cuda:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    if cuda:
        s.record()
    for _ in range(a.steps):
        losses.append(step())          # stays on the device: no host sync inside the measured region
    if cuda:
        e.record()
    sync()
    dist.barrier()
    host_s = time.perf_counter() - t0
    res = meter.stop() if meter is not None else None
    region_s = (s.elapsed_time(e) / 1000.0) if cuda else host_s

    # ---- numerical checks, outside the measured region
    traj = torch.stack(losses).double().cpu().tolist()
    grads_finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    sq = float(sum(float(p.detach().double().pow(2).sum()) for p in model.parameters()))
    psum = float(sum(float(p.detach().double().sum()) for p in model.parameters()))
    mine = {"rank": rank, "region_s": region_s, "loss_trajectory": traj, "param_sum_local": psum,
            "loss_all_finite": all(math.isfinite(v) for v in traj), "grads_all_finite": grads_finite and n_grads > 0,
            "n_grads": n_grads, "param_sq_sum_local": sq}
    allr = [None] * world
    dist.all_gather_object(allr, mine)
    if rank == 0:
        from .numerics import finite_summary, global_param_norm
        step_s = max(r["region_s"] for r in allr) / a.steps
        trajs = [r["loss_trajectory"] for r in allr]
        close = lambda u, v, tol: abs(u - v) <= tol * max(1.0, abs(v))
        if a.data == "replicated":
            # same batch and synchronized (DDP) or gathered (FSDP) weights: loss trajectories agree across ranks.
            # Cannot detect a skipped reduction (every rank's local gradient equals the average).
            check = "LOSS_TRAJECTORIES_EQUAL_REPLICATED_DATA"
            ranks_consistent = all(len(t) == len(trajs[0]) and all(close(u, v, 1e-6) for u, v in zip(t, trajs[0]))
                                   for t in trajs)
        elif a.mode.startswith("ddp"):
            # per-rank data: replicas stay identical only if gradients were averaged across ranks
            check = "DDP_REPLICAS_EQUAL_PER_RANK_DATA"
            ranks_consistent = all(close(r["param_sq_sum_local"], allr[0]["param_sq_sum_local"], 1e-9)
                                   and close(r["param_sum_local"], allr[0]["param_sum_local"], 1e-9) for r in allr)
        else:
            check, ranks_consistent = "NOT_APPLICABLE_SHARDED", None       # FSDP shards: no replica to compare
        semantics = {"mode": a.mode, "world": world, "seed": a.seed, "batch": a.batch, "global_batch": a.batch * world,
                     "steps": a.steps, "warmup": a.warmup, "hidden": a.hidden, "layers": a.layers, "dtype": "float32",
                     "tf32": False, "bucket_mb": a.bucket_mb, "device": a.device, "backend": a.backend,
                     "data": ("different synthetic batch per rank (generator seeded by seed and rank)"
                              if a.data == "per_rank" else
                              "identical synthetic batch on every rank (torch.manual_seed(seed) before model and data)")}
        numerics = {"loss_trajectory": trajs[0], "loss_summary": finite_summary(trajs[0]),
                    "loss_all_finite": all(r["loss_all_finite"] for r in allr),
                    "grads_all_finite": all(r["grads_all_finite"] for r in allr),
                    "ranks_consistent": ranks_consistent, "consistency_check": check,
                    "ranks_consistent_expected": a.mode != "ddp_nosync",
                    "param_sq_sum_per_rank": [r["param_sq_sum_local"] for r in allr],
                    "param_norm": global_param_norm(a.mode, [r["param_sq_sum_local"] for r in allr]),
                    "per_rank": [{k: r[k] for k in ("rank", "region_s", "loss_all_finite", "grads_all_finite", "n_grads")}
                                 for r in allr]}
        energy = res["energy_J"] if res else None
        out = {"mode": a.mode, "policy": a.policy, "world": world, "hidden": a.hidden, "layers": a.layers,
               "batch": a.batch, "steps": a.steps, "warmup": a.warmup, "bucket_mb": a.bucket_mb,
               "semantics": semantics, "numerics": numerics,
               "step_time_s": step_s, "host_step_time_s": host_s / a.steps,
               "throughput_samples_s": a.batch * world / step_s, "energy_J": energy,
               "energy_per_step_J": (energy / a.steps) if energy is not None else None,
               "meter_method": getattr(meter, "method", None), "meter": res, "probe": probe,
               "torch": torch.__version__,
               "nccl": ".".join(map(str, torch.cuda.nccl.version())) if cuda else None,
               "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "NCCLE_", "CUDA_VISIBLE"))}}
        from .common import redact
        with open(a.out, "w") as f:
            f.write(redact(json.dumps(out, indent=1, default=str)))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
