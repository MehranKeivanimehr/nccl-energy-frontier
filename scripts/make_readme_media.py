"""README animations from retained processed tables: python scripts/make_readme_media.py [--out docs/media]

Presentation only. Reads data/v2/processed/confirmatory/rq1.csv and data/processed/full/summary.csv; collects
nothing, changes no statistic and writes only into --out. The animations replay published numbers:
  rq1_cells.gif               v2 confirmatory: each pre-registered pick vs NCCL automatic, cell by cell
  pilot_frontier_sweep.gif    historical pilot v1 (superseded): latency vs board energy relative to auto, by size
Scope labels are drawn into every frame so a frame cannot be quoted without them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RQ1 = REPO / "data/v2/processed/confirmatory/rq1.csv"
PILOT = REPO / "data/processed/full/summary.csv"
PLACEMENT_COLOR = {"P2_NVL": "#2a6fdb", "P2_PCI_SHM": "#e8663d", "P4_MIX": "#18a77a"}
PLACEMENT_LABEL = {"P2_NVL": "2 GPUs, NVLink", "P2_PCI_SHM": "2 GPUs, PCIe/SHM", "P4_MIX": "4 GPUs, mixed"}
COLL = {"all_reduce": "AllReduce", "all_gather": "AllGather", "reduce_scatter": "ReduceScatter"}
PROTO_COLOR = {"LL": "#e8663d", "LL128": "#18a77a", "Simple": "#2a6fdb", "auto": "#444444"}
ALGO_MARKER = {"Ring": "s", "Tree": "^", "auto": "o"}
INK, MUTED, GRID = "#222222", "#666666", "#e4e4e4"


def _size(b: int) -> str:
    for unit, f in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= f:
            return f"{b / f:g} {unit}"
    return f"{b} B"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def rq1_cells(out: Path, hold: int = 18) -> None:
    d = pd.read_csv(RQ1)
    d["order"] = d.placement.map({k: i for i, k in enumerate(PLACEMENT_COLOR)})
    d = d.sort_values(["order", "coll", "size_bytes"]).reset_index(drop=True)
    labels = [f"{COLL[r.coll]} · {_size(int(r.size_bytes))} → {r.policy}" for r in d.itertuples()]
    n = len(d)
    fig, ax = plt.subplots(figsize=(9.2, 10.8), dpi=80)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.15)
    _style(ax)
    ax.set_xlim(0.25, 1.12)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(range(n), labels, fontsize=8.5)
    ax.axvline(1.0, color=INK, lw=1.2)
    ax.axvline(0.95, color=MUTED, lw=1, ls=":")
    ax.text(1.0, -1.2, "auto", ha="center", fontsize=8.5, color=INK)
    ax.text(0.95, -1.2, "−5 %", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xlabel("pick ÷ NCCL automatic (1.0 = same as automatic; left = less)", fontsize=10)
    fig.suptitle("v2 confirmatory: pre-registered pick vs NCCL automatic, cell by cell", fontsize=12.5, x=0.02,
                 ha="left", color=INK)
    fig.text(0.02, 0.935, "● board energy ratio with 95 % CI over 12 within-session blocks    "
             "◇ latency ratio    ✕ saving not confirmed", fontsize=9, color=MUTED)
    fig.text(0.02, 0.015, "One host (8× RTX A6000), one session. Each cell compares one selected pick with automatic "
             "only;\nthe best configuration overall was not established. Source: data/v2/processed/confirmatory/rq1.csv",
             fontsize=8.5, color=MUTED)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=PLACEMENT_LABEL[k])
                        for k, c in PLACEMENT_COLOR.items()], loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.06))
    counter = fig.text(0.02, 0.9, "", fontsize=10.5, color=INK, weight="bold")
    for t in ax.get_yticklabels():
        t.set_alpha(0.25)

    def draw(i):
        if i < n:
            r = d.iloc[i]
            c = PLACEMENT_COLOR[r.placement]
            ax.hlines(i, r.E_ratio_ci_lo, r.E_ratio_ci_hi, color=c, lw=2.2)
            ax.plot(r.T_ratio, i, marker="D", ms=5, mfc="none", mec=c, ls="")
            ax.plot(r.E_ratio, i, marker="o" if r.saving_confirmed else "X", ms=7, color=c, ls="")
            ax.get_yticklabels()[i].set_alpha(1.0)
            shown = d.iloc[: i + 1]
            counter.set_text(f"lower board energy than automatic (confirmed): "
                             f"{int(shown.saving_confirmed.sum())} of {i + 1} cells shown")
        return []

    FuncAnimation(fig, draw, frames=n + hold, interval=260, blit=False).save(out / "rq1_cells.gif",
                                                                         writer=PillowWriter(fps=4))
    plt.close(fig)


def pilot_frontier_sweep(out: Path, hold: int = 6) -> None:
    s = pd.read_csv(PILOT)
    s = s[s.placement == "oop"]
    panels = [("all_reduce", "4gpu"), ("all_gather", "2gpu_nvlink")]
    sizes = sorted(s.size_bytes.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=80)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.2, wspace=0.22)
    fig.text(0.01, 0.95, "Historical pilot v1 (SUPERSEDED): where each NCCL setting lands relative to automatic",
             fontsize=12.5, color=INK)
    fig.text(0.01, 0.025, "Driver 550, uncorrected boundary counter reads, 3 trial means per point, out-of-place. "
             "Descriptive only; superseded by\nthe v2 method. ○ = lowest-latency setting, □ = lowest-energy setting "
             "at that size. Source: data/processed/full/summary.csv", fontsize=8.5, color=MUTED)
    title = fig.text(0.01, 0.89, "", fontsize=11, color=INK, weight="bold")
    data = {}
    for coll, gs in panels:
        p = s[(s.coll == coll) & (s.gpu_set == gs)]
        auto = p[p.config == "auto/auto"].set_index("size_bytes")
        p = p.assign(t_rel=p.t_op_us / p.size_bytes.map(auto.t_op_us), e_rel=p.e_op_j / p.size_bytes.map(auto.e_op_j))
        data[(coll, gs)] = p
    handles = [Line2D([], [], marker="o", ls="", color=c, label=f"proto {k}") for k, c in PROTO_COLOR.items()]
    handles += [Line2D([], [], marker=m, ls="", color=MUTED, mfc="none", label=f"algo {k}") for k, m in ALGO_MARKER.items()]
    fig.legend(handles=handles, loc="upper left", ncol=7, fontsize=8.5, frameon=False, bbox_to_anchor=(0.005, 0.87))
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16]

    def draw(i):
        b = sizes[min(i, len(sizes) - 1)]
        title.set_text(f"message size {_size(int(b))}")
        for ax, key in zip(axes, panels):
            ax.clear()
            _style(ax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            p = data[key]
            lim = (min(p.t_rel.min(), p.e_rel.min()) * 0.9, max(p.t_rel.max(), p.e_rel.max()) * 1.1)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator(ticks))
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_locator(NullLocator())
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.plot(lim, lim, color=GRID, lw=1)
            ax.axhline(1, color=MUTED, lw=0.8, ls=":")
            ax.axvline(1, color=MUTED, lw=0.8, ls=":")
            f = p[p.size_bytes == b]
            for r in f.itertuples():
                algo, proto = r.config.split("/")
                ax.plot(r.t_rel, r.e_rel, marker=ALGO_MARKER[algo], ms=8, ls="", color=PROTO_COLOR[proto],
                        mfc=PROTO_COLOR[proto] if algo == "auto" else "none", mew=1.8)
            lo_t, lo_e = f.loc[f.t_op_us.idxmin()], f.loc[f.e_op_j.idxmin()]
            ax.plot(lo_t.t_rel, lo_t.e_rel, marker="o", ms=17, mfc="none", mec=INK, mew=1.2, ls="")
            ax.plot(lo_e.t_rel, lo_e.e_rel, marker="s", ms=19, mfc="none", mec=INK, mew=1.0, ls="")
            coll, gs = key
            ax.set_title(f"{COLL[coll]} · {gs.replace('_', ' ')}", fontsize=10.5, loc="left", color=INK)
            ax.set_xlabel("latency ÷ automatic", fontsize=9.5)
            ax.set_ylabel("board energy per op ÷ automatic", fontsize=9.5)
        return []

    FuncAnimation(fig, draw, frames=len(sizes) + hold, interval=700, blit=False).save(
        out / "pilot_frontier_sweep.gif", writer=PillowWriter(fps=1.5))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs/media")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rq1_cells(a.out)
    pilot_frontier_sweep(a.out)


if __name__ == "__main__":
    main()
