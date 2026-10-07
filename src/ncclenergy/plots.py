"""Figures from processed tables (data/processed/<run>/*.csv).

Encoding (fixed across all figures):
  colour  = requested NCCL protocol (auto = neutral ink, Simple/LL/LL128 = the first
            three slots of the reference categorical palette, which are
            colour-vision-deficiency safe even when all pairs are compared)
  style   = requested algorithm (auto solid/circle, Ring dashed/square, Tree dotted/triangle)
Heatmaps colour the *effective* protocol NCCL used (from the probe's TUNING log)
and print the effective algorithm (R = Ring, T = Tree) in the cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL_CELL = "#f0efec"
PROTO_COLOR = {"auto": "#52514e", "Simple": "#2a78d6", "LL": "#eb6834", "LL128": "#1baf7a"}
EFF_PROTO_COLOR = {"SIMPLE": PROTO_COLOR["Simple"], "LL": PROTO_COLOR["LL"], "LL128": PROTO_COLOR["LL128"]}
ALGO_STYLE = {"auto": ("-", "o"), "Ring": ((0, (5, 2)), "s"), "Tree": ((0, (1, 1.6)), "^")}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COLL_TITLE = {"all_reduce": "AllReduce", "all_gather": "AllGather", "reduce_scatter": "ReduceScatter"}
COLLS = ["all_reduce", "all_gather", "reduce_scatter"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
})


def fmt_bytes(b: float) -> str:
    for unit, f in (("G", 2 ** 30), ("M", 2 ** 20), ("K", 2 ** 10)):
        if b >= f:
            v = b / f
            return f"{v:g}{unit}"
    return f"{int(b)}"


def _split_config(c: str) -> tuple[str, str]:
    a, p = c.split("/")
    return a, p


def _style_legend(fig, protos, algos, y=1.0):
    h = [Line2D([], [], color=PROTO_COLOR[p], lw=2, label=f"proto {p}") for p in protos]
    h += [Line2D([], [], color=INK2, lw=1.4, ls=ALGO_STYLE[a][0], marker=ALGO_STYLE[a][1], ms=4,
                 label=f"algo {a}") for a in algos]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, y), fontsize=8,
               handlelength=2.6, columnspacing=1.2)


def _grid(summ: pd.DataFrame):
    sets = [s for s in ("2gpu_nvlink", "4gpu", "8gpu") if s in set(summ.gpu_set)] + \
        sorted(set(summ.gpu_set) - {"2gpu_nvlink", "4gpu", "8gpu"})
    colls = [c for c in COLLS if c in set(summ.coll)]
    return sets, colls


def _plain_log_labels(ax):
    """Plain-number tick labels on a log y axis; minor labels only when it spans < ~1 decade."""
    from matplotlib.ticker import FuncFormatter
    lo, hi = ax.get_ylim() if ax.has_data() else (1, 10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    small = hi / max(lo, 1e-300) < 12
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if small else ""))


def size_lines(summ: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path, logy=True,
               placement="oop", note: str | None = None):
    s = summ[(summ.placement == placement) & (summ.n_trials >= 1)]
    sets, colls = _grid(s)
    fig, axes = plt.subplots(len(sets), len(colls), figsize=(3.6 * len(colls), 2.9 * len(sets) + 0.9),
                             squeeze=False, sharex=True)
    protos, algos = set(), set()
    for i, gs in enumerate(sets):
        for j, coll in enumerate(colls):
            ax = axes[i][j]
            d = s[(s.gpu_set == gs) & (s.coll == coll)]
            for cfg, g in sorted(d.groupby("config")):
                a, p = _split_config(cfg)
                protos.add(p)
                algos.add(a)
                g = g.sort_values("size_bytes")
                ls, mk = ALGO_STYLE[a]
                ax.plot(g.size_bytes, g[metric], color=PROTO_COLOR[p], ls=ls, marker=mk, ms=3.2, lw=1.3,
                        alpha=0.95, zorder=3 if a == "auto" and p == "auto" else 2)
                if f"{metric}_lo" in g:
                    ax.fill_between(g.size_bytes, g[f"{metric}_lo"], g[f"{metric}_hi"], color=PROTO_COLOR[p],
                                    alpha=0.10, lw=0)
            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
                _plain_log_labels(ax)
            ax.set_title(f"{COLL_TITLE[coll]} · {gs}", loc="left")
            if i == len(sets) - 1:
                ax.set_xlabel("message size (bytes)")
            if j == 0:
                ax.set_ylabel(ylabel)
            ticks = [2 ** k for k in range(10, 31, 4) if d.size_bytes.min() <= 2 ** k <= d.size_bytes.max()] \
                if len(d) else []
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([fmt_bytes(t) for t in ticks])
    _style_legend(fig, [p for p in ("auto", "Simple", "LL", "LL128") if p in protos],
                  [a for a in ("auto", "Ring", "Tree") if a in algos], y=0.995)
    fig.suptitle(title, x=0.01, ha="left", y=1.04, fontsize=11, color=INK)
    if note:
        fig.text(0.01, -0.01, note, fontsize=7.5, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pareto_fig(summ: pd.DataFrame, par: pd.DataFrame, gs: str, energy_metric: str, path: Path,
               placement="oop", n_cols=4):
    s = summ[(summ.placement == placement) & (summ.gpu_set == gs)]
    if s.empty:
        return
    sizes_all = sorted(s.size_bytes.unique())
    idx = np.unique(np.linspace(0, len(sizes_all) - 1, n_cols).round().astype(int))
    sizes = [sizes_all[i] for i in idx]
    colls = [c for c in COLLS if c in set(s.coll)]
    fig, axes = plt.subplots(len(colls), len(sizes), figsize=(3.1 * len(sizes), 2.8 * len(colls) + 0.8),
                             squeeze=False)
    p_sub = par[(par.placement == placement) & (par.gpu_set == gs) & (par.energy_metric == energy_metric)]
    protos, algos = set(), set()
    for i, coll in enumerate(colls):
        for j, size in enumerate(sizes):
            ax = axes[i][j]
            d = s[(s.coll == coll) & (s.size_bytes == size)]
            for r in d.itertuples():
                a, p = _split_config(r.config)
                protos.add(p)
                algos.add(a)
                y, ylo, yhi = getattr(r, energy_metric), getattr(r, f"{energy_metric}_lo"), getattr(r, f"{energy_metric}_hi")
                ax.errorbar(r.t_op_us, y, xerr=[[r.t_op_us - r.t_op_us_lo], [r.t_op_us_hi - r.t_op_us]],
                            yerr=[[y - ylo], [yhi - y]], fmt=ALGO_STYLE[a][1], ms=5.5, color=PROTO_COLOR[p],
                            mec=SURFACE, mew=1.0, elinewidth=0.7, capsize=0, zorder=3)
            f = p_sub[(p_sub.coll == coll) & (p_sub.size_bytes == size) & p_sub.on_front].sort_values("t_op_us")
            if len(f):
                ax.plot(f.t_op_us, f.energy_j, color=INK, lw=0.9, zorder=2)
                ax.scatter(f.t_op_us, f.energy_j, s=110, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=4)
            ax.set_title(f"{COLL_TITLE[coll]} · {fmt_bytes(size)}B", loc="left")
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
            if i == len(colls) - 1:
                ax.set_xlabel("latency per op (µs)")
            if j == 0:
                ax.set_ylabel("energy per op (J)" + (" idle-subtracted" if "dyn" in energy_metric else ""))
    _style_legend(fig, [p for p in ("auto", "Simple", "LL", "LL128") if p in protos],
                  [a for a in ("auto", "Ring", "Tree") if a in algos], y=0.995)
    fig.suptitle(f"5 · Latency–energy Pareto fronts · {gs} · "
                 f"{'idle-subtracted' if 'dyn' in energy_metric else 'raw'} GPU energy",
                 x=0.01, ha="left", y=1.04, fontsize=11, color=INK)
    fig.text(0.01, -0.01, "Points: trial-block means per requested configuration (out-of-place); bars: descriptive "
             "95% bootstrap "
             "intervals; black rings joined by a line: Pareto front of the means (both axes minimized); a single ring "
             "means one configuration is fastest and least energy at once.",
             fontsize=7.5, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _cells(opt: pd.DataFrame, placement: str):
    o = opt[opt.placement == placement]
    sets = [s for s in ("2gpu_nvlink", "4gpu", "8gpu") if s in set(o.gpu_set)]
    rows = [(c, s) for s in sets for c in COLLS if ((o.coll == c) & (o.gpu_set == s)).any()]
    sizes = sorted(o.size_bytes.unique())
    return o, rows, sizes


def _parse_eff(eff) -> tuple[str | None, str | None]:
    if not isinstance(eff, str) or "|" in eff:
        return None, None
    algo, rest = eff.split("/", 1)
    return algo, rest.split("[")[0]


def optimum_heatmap(opt: pd.DataFrame, which: str, title: str, path: Path, placement="oop"):
    o, rows, sizes = _cells(opt, placement)
    if not rows:
        return
    protos = ["SIMPLE", "LL", "LL128"]
    cmap = ListedColormap([NEUTRAL_CELL] + [EFF_PROTO_COLOR[p] for p in protos])
    grid = np.zeros((len(rows), len(sizes)))
    fig, ax = plt.subplots(figsize=(0.42 * len(sizes) + 2.6, 0.42 * len(rows) + 1.6))
    texts = []
    for i, (c, s) in enumerate(rows):
        for j, size in enumerate(sizes):
            r = o[(o.coll == c) & (o.gpu_set == s) & (o.size_bytes == size)]
            if r.empty or pd.isna(r.iloc[0].get(f"{which}_opt")):
                continue
            algo, proto = _parse_eff(r.iloc[0][f"{which}_opt_selected"])
            if proto in protos:
                grid[i, j] = protos.index(proto) + 1
            pb = r.iloc[0][f"{which}_opt_pbest"]
            texts.append((j, i, (algo or "?")[0], pb))
    ax.imshow(grid, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N), aspect="auto")
    for j, i, t, pb in texts:
        ax.text(j, i, t, ha="center", va="center", fontsize=7.5, color="white" if pb >= 0.5 else INK,
                fontweight="bold" if pb >= 0.8 else "normal")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([fmt_bytes(x) for x in sizes], rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{COLL_TITLE[c]} · {s}" for c, s in rows], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(sizes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("message size (bytes)")
    handles = [Patch(color=EFF_PROTO_COLOR[p], label=f"protocol {p.title() if p == 'SIMPLE' else p}") for p in protos]
    handles.append(Patch(color=NEUTRAL_CELL, label="no data"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.text(0.01, -0.02, "Cell colour = protocol NCCL actually ran for the winning configuration; letter = algorithm "
             "(R Ring, T Tree). Letter style = P(best) over bootstrap resamples: bold white ≥80%, regular white 50–80%, "
             "dark <50%.",
             fontsize=7.5, color=INK2, ha="left", va="top")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def disagreement_heatmap(opt: pd.DataFrame, path: Path, placement="oop"):
    o, rows, sizes = _cells(opt, placement)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.36 * len(sizes) + 1.8) + 1.0, 0.42 * len(rows) + 2.1),
                             sharey=True, layout="constrained")
    bounds = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 10]
    cmap = ListedColormap(SEQ_BLUE)
    norm = BoundaryNorm(bounds, cmap.N)
    for ax, o_key, ttl in ((axes[0], "en", "raw GPU energy"), (axes[1], "dyn", "idle-subtracted GPU energy")):
        img = np.full((len(rows), len(sizes)), np.nan)
        nodata = np.zeros((len(rows), len(sizes)))
        marks = []
        for i, (c, s) in enumerate(rows):
            for j, size in enumerate(sizes):
                r = o[(o.coll == c) & (o.gpu_set == s) & (o.size_bytes == size)]
                if r.empty or pd.isna(r.iloc[0].get("lat_opt")):
                    nodata[i, j] = 1
                    marks.append((j, i, "–"))
                    continue
                r = r.iloc[0]
                if r[f"disagree_{o_key}_effective"]:
                    img[i, j] = max(r[f"{o_key}_penalty_of_lat_opt"], 0)
                    marks.append((j, i, "•" if r[f"disagree_{o_key}_resolved"] else ""))
                else:
                    marks.append((j, i, "="))
        ax.imshow(nodata, cmap=ListedColormap([NEUTRAL_CELL, SURFACE]), vmin=0, vmax=1, aspect="auto")
        im = ax.imshow(np.ma.masked_invalid(img), cmap=cmap, norm=norm, aspect="auto")
        for j, i, t in marks:
            v = img[i, j]
            ax.text(j, i, t, ha="center", va="center", fontsize=8 if t == "•" else 7,
                    color="white" if (np.isfinite(v) and v >= 0.05) else INK2)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([fmt_bytes(x) for x in sizes], rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{COLL_TITLE[c]} · {s}" for c, s in rows], fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(sizes)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.set_xlabel("message size (bytes)")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, ticks=bounds[:-1])
    cb.ax.set_yticklabels([f"{b:.0%}" for b in bounds[:-1]], fontsize=7.5)
    cb.outline.set_visible(False)
    fig.suptitle("8 · Where latency- and energy-optimal NCCL configurations disagree, and the energy cost of "
                 "choosing for latency", x=0.01, ha="left", fontsize=10, color=INK)
    fig.text(0.01, -0.02, "'=' gray: both optima run the same algorithm/protocol/channels (agree). Blue: they differ; "
             "shade = mean energy of the latency-optimal configuration relative to the energy-optimal one.\n"
             "• = descriptive 95% bootstrap interval of that penalty excludes 0.  '–' white: fewer than two "
             "configurations with "
             "enough valid trials.", fontsize=7.5, color=INK2, ha="left", va="top")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def example_trace(run_dir: Path, launch_dir: str, path: Path):
    """Power trace of one timed launch with the measured regions overlaid."""
    d = Path(run_dir) / launch_dir
    pw = pd.read_csv(d / "power.csv.gz")
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    t0 = pw.t_mono.min()
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    gpu_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference palette slots 1-4
    for k, (g, grp) in enumerate(pw.groupby("gpu")):
        ax.plot(grp.t_mono - t0, grp.power_w, lw=1.4, color=gpu_colors[k % 4],
                label=f"GPU {g}: nvmlDeviceGetPowerUsage (50 ms samples)")
    for reg in res["regions"]:
        if not reg.get("valid_boundaries"):
            continue
        for k, (g, v) in enumerate(reg["per_gpu"].items()):
            pc = v["energy_counter_j"] / reg["duration_s"]
            ax.hlines(pc, reg["t0"] - t0, reg["t1"] - t0, color=INK, lw=1.4)
        ax.axvspan(reg["t0"] - t0, reg["t1"] - t0, color=GRID, alpha=0.45, lw=0)
        ax.text((reg["t0"] + reg["t1"]) / 2 - t0, 1.01, f"{reg['placement']} region",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.plot([], [], color=INK, lw=1.4, label="per-GPU energy-counter average over each region")
    ax.set_xlabel("time since sampler start (s)")
    ax.set_ylabel("power (W)")
    c = res["cond"]
    ax.set_title(f"Example launch: {COLL_TITLE[c['coll']]} {c['gpu_set']} {c['algo']}/{c['proto']} "
                 f"{fmt_bytes(res['size_bytes'])}B, n={res['iters']} w={res['warmup']}", loc="left", fontsize=9,
                 pad=16)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.3), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def relative_to_auto(summ: pd.DataFrame) -> pd.DataFrame:
    keys = ["coll", "gpu_set", "size_bytes", "placement"]
    base = summ[summ.config == "auto/auto"].set_index(keys)[["t_op_us", "e_op_j"]]
    r = summ.join(base, on=keys, rsuffix="_auto")
    return r.assign(t_rel=r.t_op_us / r.t_op_us_auto, e_rel=r.e_op_j / r.e_op_j_auto)


def plot_all(processed: Path, out: Path):
    processed, out = Path(processed), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(processed / "summary.csv")
    opt = pd.read_csv(processed / "optima.csv")
    par = pd.read_csv(processed / "pareto.csv")
    note = "Lines: mean of trial blocks (out-of-place); bands: descriptive 95% bootstrap interval."
    size_lines(summ, "t_op_us", "latency per op (µs)", "1 · Latency per collective vs message size",
               out / "fig1_latency_vs_size.png", note=note)
    size_lines(summ, "e_op_j", "energy per op (J)", "2a · GPU energy per collective vs message size (raw)",
               out / "fig2a_energy_per_op_vs_size.png", note=note + " Energy summed over participating GPUs.")
    size_lines(summ, "e_op_dyn_j", "idle-subtracted energy per op (J)",
               "2b · Idle-subtracted GPU energy per collective vs message size",
               out / "fig2b_energy_dyn_per_op_vs_size.png", note=note)
    size_lines(summ, "gb_per_j", "GB per joule", "3a · Energy efficiency (message GB / raw J) vs message size",
               out / "fig3a_gb_per_j_vs_size.png", note=note)
    size_lines(summ, "gb_per_j_dyn", "GB per joule (idle-subtracted)",
               "3b · Energy efficiency (message GB / idle-subtracted J) vs message size",
               out / "fig3b_gb_per_j_dyn_vs_size.png", note=note)
    size_lines(summ, "edp_js", "energy × delay (J·s)", "4 · Energy-delay product per collective vs message size",
               out / "fig4_edp_vs_size.png", note=note)
    # Supplementary power and ratios relative to NCCL's default.
    pw = summ.assign(p_gpu=summ.p_avg_w / summ.n_gpus, p_gpu_lo=summ.p_avg_w_lo / summ.n_gpus,
                     p_gpu_hi=summ.p_avg_w_hi / summ.n_gpus)
    size_lines(pw, "p_gpu", "average power per GPU (W)", "S1 · Average board power per GPU during the collective",
               out / "figS1_power_per_gpu_vs_size.png", logy=False, note=note)
    rel = relative_to_auto(summ)
    size_lines(rel, "t_rel", "latency ÷ auto/auto", "S2 · Latency relative to NCCL's default (auto/auto)",
               out / "figS2_latency_rel_auto.png", logy=True,
               note="Ratio of trial means (out-of-place); 1 = same as NCCL's default choice.")
    size_lines(rel, "e_rel", "energy/op ÷ auto/auto", "S3 · Raw energy per op relative to NCCL's default (auto/auto)",
               out / "figS3_energy_rel_auto.png", logy=True,
               note="Ratio of trial means (out-of-place); 1 = same as NCCL's default choice.")
    for gs in sorted(summ.gpu_set.unique()):
        pareto_fig(summ, par, gs, "e_op_j", out / f"fig5a_pareto_{gs}_raw.png")
        pareto_fig(summ, par, gs, "e_op_dyn_j", out / f"fig5b_pareto_{gs}_dyn.png")
    optimum_heatmap(opt, "lat", "6 · Latency-optimal NCCL configuration", out / "fig6_heatmap_latency_optimal.png")
    optimum_heatmap(opt, "en", "7a · Energy-optimal NCCL configuration (raw GPU energy)",
                    out / "fig7a_heatmap_energy_optimal.png")
    optimum_heatmap(opt, "dyn", "7b · Energy-optimal NCCL configuration (idle-subtracted GPU energy)",
                    out / "fig7b_heatmap_dyn_energy_optimal.png")
    disagreement_heatmap(opt, out / "fig8_heatmap_disagreement.png")
    print(f"figures written to {out}")
