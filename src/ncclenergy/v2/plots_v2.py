"""Methodology-v2 figures (block-level; executed configurations).

Colour = executed protocol (Simple/LL/LL128: first three reference-palette slots, safe
for all-pairs comparison); marker = executed algorithm (Ring circle, Tree triangle);
a black ring marks the configuration NCCL automatic executed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
PROTO = {"SIMPLE": "#2a78d6", "LL": "#eb6834", "LL128": "#1baf7a"}
ALGO = {"RING": "o", "TREE": "^"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COLL = {"all_reduce": "AllReduce", "all_gather": "AllGather", "reduce_scatter": "ReduceScatter"}
PLACE = ["P2_NVL", "P2_PCI_SHM", "P4_MIX"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK,
                     "legend.frameon": False})


def fmt_bytes(b):
    for u, f in (("G", 2 ** 30), ("M", 2 ** 20), ("K", 2 ** 10)):
        if b >= f:
            return f"{b / f:g}{u}"
    return str(int(b))


def _eff_parts(actual: str):
    a, p, _ = (actual.split("|")[0] + "//").split("/")[:3]
    return a, p


def _legend(fig, y=1.0):
    h = [Line2D([], [], color=c, marker="s", ls="", ms=7, label=f"protocol {k.title() if k == 'SIMPLE' else k}")
         for k, c in PROTO.items()]
    h += [Line2D([], [], color=INK2, marker=m, ls="", ms=6, label=f"algorithm {k.title()}") for k, m in ALGO.items()]
    h += [Line2D([], [], color=INK, marker="o", mfc="none", ls="", ms=11, label="NCCL automatic executed this")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, y), fontsize=8)


def pareto(v: pd.DataFrame, out: Path, sizes=(4096, 262144, 16 << 20, 1 << 30), title_tag=""):
    """Block-mean latency vs energy per executed config; bars = t-based 95 % CI over blocks."""
    for coll in [c for c in COLL if c in set(v.coll)]:
        places = [p for p in PLACE if p in set(v.placement)]
        szs = [s for s in sizes if s in set(v.size_bytes)]
        fig, axes = plt.subplots(len(places), len(szs), figsize=(3.2 * len(szs), 2.7 * len(places) + 0.9), squeeze=False)
        for i, pl in enumerate(places):
            for j, s in enumerate(szs):
                ax = axes[i][j]
                c = v[(v.placement == pl) & (v.coll == coll) & (v.size_bytes == s)]
                if c.empty:
                    ax.set_visible(False)
                    continue
                auto_eff = c[c.config == "auto"].actual.iloc[0] if (c.config == "auto").any() else None
                pts = []
                for eff, g in c.groupby("actual"):
                    bm = g.groupby("block")[["logT", "logE"]].mean()
                    n = len(bm)
                    from scipy import stats
                    q = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
                    mt, me = bm.logT.mean(), bm.logE.mean()
                    st = bm.logT.std(ddof=1) / math.sqrt(n) if n > 1 else 0
                    se = bm.logE.std(ddof=1) / math.sqrt(n) if n > 1 else 0
                    a, p = _eff_parts(eff)
                    x, y = math.exp(mt) * 1e6, math.exp(me) * 1e3
                    xe = [[x - math.exp(mt - q * st) * 1e6], [math.exp(mt + q * st) * 1e6 - x]] if n > 1 else None
                    ye = [[y - math.exp(me - q * se) * 1e3], [math.exp(me + q * se) * 1e3 - y]] if n > 1 else None
                    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=ALGO.get(a, "s"), color=PROTO.get(p, INK2), ms=6,
                                mec=SURFACE, mew=0.8, elinewidth=0.7, zorder=3)
                    if eff == auto_eff:
                        ax.plot(x, y, "o", mfc="none", mec=INK, ms=12, mew=1.0, zorder=4)
                    pts.append((x, y))
                front = sorted([p for p in pts if not any((q[0] <= p[0] and q[1] <= p[1]) and q != p for q in pts)])
                if len(front) > 1:
                    ax.plot(*zip(*front), color=INK, lw=0.8, zorder=2)
                ax.set_title(f"{pl} · {fmt_bytes(s)}B", loc="left")
                if i == len(places) - 1:
                    ax.set_xlabel("latency per collective (µs)")
                if j == 0:
                    ax.set_ylabel("raw energy per collective (mJ)")
        _legend(fig)
        fig.suptitle(f"Latency–energy per executed configuration · {COLL[coll]}{title_tag}", x=0.01, ha="left",
                     y=1.04, fontsize=11, color=INK)
        fig.text(0.01, -0.01, "Points: geometric mean over within-session blocks; bars: descriptive 95 % t-intervals; "
                 "black line: Pareto front of the means.", fontsize=7.5, color=INK2, ha="left", va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"pareto_{coll}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def regret_heatmap(pol: pd.DataFrame, out: Path, col="energy_slo_E_vs_auto", title="", fname="default_regret.png"):
    rows = [(p, c) for p in PLACE for c in COLL if ((pol.placement == p) & (pol.coll == c)).any()]
    sizes = sorted(pol.size_bytes.unique())
    img = np.full((len(rows), len(sizes)), np.nan)
    same = []
    for i, (p, c) in enumerate(rows):
        for j, s in enumerate(sizes):
            r = pol[(pol.placement == p) & (pol.coll == c) & (pol.size_bytes == s)]
            if len(r):
                rr = r.iloc[0]
                if rr.get("energy_slo_actual") == rr.get("auto_actual"):
                    same.append((j, i))                     # policy executes what NCCL automatic executes
                else:
                    img[i, j] = 1 / rr[col] - 1             # auto energy relative to policy energy
    bounds = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 10]
    fig, ax = plt.subplots(figsize=(0.42 * len(sizes) + 3.2, 0.42 * len(rows) + 1.8), layout="constrained")
    ax.imshow(np.zeros_like(img), cmap=ListedColormap([NEUTRAL]), aspect="auto")
    im = ax.imshow(np.ma.masked_invalid(np.clip(img, 0, None)), cmap=ListedColormap(SEQ),
                   norm=BoundaryNorm(bounds, len(SEQ)), aspect="auto")
    for j, i in same:
        ax.text(j, i, "=", ha="center", va="center", fontsize=7, color=INK2)
    for i, j in zip(*np.where(np.isfinite(img) & (img < 0))):
        ax.text(j, i, "−", ha="center", va="center", fontsize=9, color="#b22222")
    ax.set_xticks(range(len(sizes)), [fmt_bytes(s) for s in sizes], rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(rows)), [f"{COLL[c]} · {p}" for p, c in rows], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(sizes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    fig.text(0.01, -0.02, "'=' grey: the Energy-SLO pick executes the same algorithm/protocol/channels as NCCL automatic. "
             "Blue: auto energy relative to pick; red '−': policy worse than auto (colour clipped at zero).",
             fontsize=7.5, color=INK2, ha="left", va="top")
    cb = fig.colorbar(im, ax=ax, ticks=bounds[:-1], fraction=0.03)
    cb.ax.set_yticklabels([f"{b:.0%}" for b in bounds[:-1]], fontsize=7.5)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("message size (bytes)")
    fig.savefig(out / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def counter_vs_integration(units: pd.DataFrame, out: Path):
    ok = units[(units.qc_status == "OK") & units.inst_over_counter.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), layout="constrained")
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, pl in enumerate([p for p in PLACE if p in set(ok.placement)]):
        d = ok[ok.placement == pl]
        axes[0].hist(100 * (d.inst_over_counter - 1), bins=60, histtype="step", lw=1.4, color=cols[k], label=pl)
        axes[1].scatter(d.T_region_s, 100 * (d.inst_over_counter - 1), s=5, color=cols[k], alpha=0.5, lw=0, label=pl)
    axes[0].set_xlabel("∫POWER_INSTANT / counter − 1 (%)")
    axes[0].set_ylabel("regions")
    axes[1].set_xlabel("region duration (s)")
    axes[1].set_ylabel("∫POWER_INSTANT / counter − 1 (%)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Counter vs instantaneous-power integration, all QC-passed regions", loc="left")
    axes[1].set_title("…vs region length; physical cause not established", loc="left")
    fig.savefig(out / "counter_vs_integration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def rq1_blocks(rq1: pd.DataFrame, out: Path):
    if rq1.empty:
        return
    r = rq1.sort_values(["placement", "coll", "size_bytes"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(r) + 1.6), sharey=True, layout="constrained")
    for ax, key, lo, hi, ttl in ((axes[0], "per_block_E_ratio", "E_ratio_ci_lo", "E_ratio_ci_hi", "raw energy / auto"),
                                 (axes[1], "per_block_T_ratio", None, "T_ratio_ci_hi", "latency / auto")):
        for i, row in r.iterrows():
            vals = json.loads(row[key])
            ax.scatter(vals, [i] * len(vals), s=10, color=INK2, alpha=0.6, lw=0, zorder=2)
            m = float(np.exp(np.mean(np.log(vals))))
            ax.plot(m, i, "o", color="#2a78d6", ms=6, zorder=3)
            if lo:
                ax.hlines(i, row[lo], row[hi], color="#2a78d6", lw=1.4, zorder=2)
        ax.axvline(1.0, color=INK, lw=0.8)
        if key == "per_block_T_ratio":
            ax.axvline(1.05, color="#e34948", lw=0.8)
        ax.set_title(ttl, loc="left")
    axes[0].axvline(0.95, color=INK2, lw=0.6)
    axes[0].set_yticks(range(len(r)), [f"{COLL[c]} · {p} · {fmt_bytes(s)}B → {pol}" for p, c, s, pol in
                                       zip(r.placement, r.coll, r.size_bytes, r.policy)], fontsize=7.5)
    axes[0].invert_yaxis()
    fig.suptitle("RQ1: pick vs auto, within-session blocks (grey), geometric mean and 95 % CI (blue)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(out / "rq1_per_block.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def workload_blocks(runs: pd.DataFrame, out: Path):
    """Phase 10: energy per step and step time per arm, every block shown (lines join a block's arms)."""
    fams = [("ddp", "DDP, batch 128", ["auto", "lat_opt", "energy_slo"]),
            ("ddpb2048", "DDP, batch 2048", ["auto", "energy_slo"]),
            ("fsdp", "FSDP, batch 128 (negative control)", ["auto", "lat_opt", "energy_slo"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for j, (fam, title, pols) in enumerate(fams):
        for i, (col, lab) in enumerate((("energy_per_step_J", "energy per step (J, GPUs 4-7)"),
                                        ("step_time_s", "step time (ms)"))):
            ax = axes[i, j]
            scale = 1000 if col == "step_time_s" else 1
            for b, g in runs[runs.arm.str.startswith(fam + "_") & ~runs.arm.str.contains("nosync")].groupby("block"):
                y = [g[g.arm == f"{fam}_{p}"][col].mean() * scale for p in pols]
                ax.plot(range(len(pols)), y, "-o", color="0.5", alpha=.6, ms=4)
            ax.set_xticks(range(len(pols)))
            ax.set_xticklabels(pols)
            ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.grid(alpha=.3)
    fig.suptitle("Historical synthetic Phase 10 on P4_MIX; training equivalence unverified; one line per block",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "workload_blocks.png", dpi=150)
    plt.close(fig)
