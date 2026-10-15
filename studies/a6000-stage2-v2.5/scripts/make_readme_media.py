"""Render retained Stage 2 estimates; no acquisition or inferential analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
INPUTS = {
    "analysis": "prep/heldout_final/heldout_analysis.json",
    "execution": "prep/heldout_final/execution.tsv",
    "calibration": "prep/calib1/run/calibration_analysis.json",
    "smoke": "code/data/stage2/v2.5-s2-smoke/smoke/session_summary.json",
}
ESTIMATORS = ("primary", "raw_counter", "nominal_period", "sampled_instant")
ESTIMATOR_NAMES = ("Aligned counter (primary)", "Raw counter", "Nominal-period counter", "Sampled power")
COMPARISONS = (("HD3", "10"), ("HD3", "20"), ("HD3", "40"), ("HD4", "C/A@40"))
COMPARISON_NAMES = ("B / A  |  10 s", "B / A  |  20 s", "B / A  |  40 s", "C / A  |  40 s")
ROW_NOTES = (
    "Mean-ratio compatibility only",
    "Thermal comparability failed",
    "Insufficient precision",
    "Positive effect; precision missed",
)
INK, MUTED, GRID = "#182b42", "#52647a", "#dbe4ed"
COLORS = ("#007e87", "#315ea8", "#a46a10", "#bf4964")
ROW_COLORS = ("#007e87", "#a46a10", "#a46a10", "#6f53ae")


def percent(log_ratio: float) -> float:
    """Convert a retained log ratio to percent, without refitting it."""
    value = float(log_ratio)
    if not math.isfinite(value):
        raise ValueError("A plotted log ratio must be finite.")
    return 100 * math.expm1(value)


def extract(analysis: dict, estimator: str, family: str, member: str) -> dict:
    result = analysis["estimators"][estimator]
    summary = result["summaries"][family][member]
    decision = result[family][member]
    if summary["n"] != 3 or summary["df"] != 2 or summary["status"] != "OK":
        raise ValueError("Presentation requires the retained three-session summary.")
    return {
        "estimator": estimator, "family": family, "member": member,
        "mean_pct": percent(summary["mean"]),
        "ci95_pct": [percent(v) for v in summary["ci95"]],
        "session_pct": [percent(summary["session_means"][str(s)]) for s in (1, 2, 3)],
        "n_sessions": summary["n"], "df": summary["df"],
        "verdict": decision["verdict"],
    }


def build_payload(root: Path = STUDY) -> dict:
    analysis = json.loads((root / INPUTS["analysis"]).read_text(encoding="utf-8"))
    calibration = json.loads((root / INPUTS["calibration"]).read_text(encoding="utf-8"))
    smoke = json.loads((root / INPUTS["smoke"]).read_text(encoding="utf-8"))
    with (root / INPUTS["execution"]).open(newline="", encoding="utf-8") as handle:
        execution = list(csv.DictReader(handle, delimiter="\t"))
    units = analysis["units"]
    if len(units) != 72 or len(execution) != 72 or any(u["status"] != "USED" for u in units):
        raise ValueError("Unexpected held-out unit inventory; review before rendering.")
    missing = sorted(u["launch_id"] for u in units if u["J"].get("primary") is None)
    if missing != ["s2-b2-p08", "s3-b2-p11"]:
        raise ValueError("Primary missingness differs from the retained study.")
    expected_execution = {"A": "RING/SIMPLE/4", "B": "RING/SIMPLE/4", "C": "RING/LL128/4"}
    for row in execution:
        if row["exec_key"] != expected_execution[row["arm"]] or row["evidence"] != "VERIFIED":
            raise ValueError("Arm execution was not verified as represented.")
    primary = analysis["estimators"]["primary"]
    contrasts = [extract(analysis, "primary", family, member) for family, member in COMPARISONS]
    duration = {
        est: {arm: [extract(analysis, est, "HD1", f"{arm}|{length}")
                    for length in ("2.5", "10", "20")] for arm in ("A", "B", "C")}
        for est in ESTIMATORS
    }
    half_width = primary["HD4_precision"]["half_width"] * 100
    if primary["HD4_precision"]["ok"] or half_width <= 100 * math.log(1.004):
        raise ValueError("The retained precision failure must not disappear.")
    return {
        "scope": "Presentation only; retained estimates, not new analysis or physical calibration.",
        "source_sha256": {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                          for rel in INPUTS.values()},
        "counts": {"smoke": smoke["n_accepted"], "calibration": calibration["result"]["n_units"],
                   "heldout": len(units), "primary_available": len(units) - len(missing), "sessions": 3},
        "missing_primary": missing, "execution": expected_execution,
        "selected": contrasts, "duration": duration,
        "precision_half_width_log_pct": half_width,
        "precision_target_log_pct": 100 * math.log(1.004),
        "overall_adequacy": analysis["physical_adequacy"],
        "animation_rule": "Only emphasis changes; every displayed estimate/CI uses all three sessions."
    }


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "a6000-stage2-presentation", "axes.unicode_minus": False,
    })
    return plt


def frame_image(fig):
    from PIL import Image
    fig.canvas.draw()
    return Image.frombytes("RGBA", fig.canvas.get_width_height(), bytes(fig.canvas.buffer_rgba())).convert("RGB")


def save_animation(frames: list, output: Path) -> None:
    # Discrete emphasis, never synthetic intermediate observations or evolving CIs.
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=[2200] * (len(frames) - 1) + [4500], loop=0,
                   disposal=2, optimize=False)


def overview(data: dict, output: Path, plt) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(12, 4.8), dpi=120, facecolor="#102338")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 12), ylim=(0, 4.8)); ax.axis("off")
    ax.text(.48, 4.33, "NCCL ENERGY FRONTIER", color="#79d5cd", size=12, weight="bold")
    ax.text(.48, 3.79, "A6000 measurement characterization", color="white", size=23, weight="bold")
    ax.text(.48, 3.38, "AllGather 2 MiB  |  2 RTX A6000 GPUs  |  one NVLink pair", color="#d7e2ed", size=12)
    counts = data["counts"]
    phases = ((.48, "SMOKE", counts["smoke"], "Pipeline check"),
              (4.35, "CALIBRATION", counts["calibration"], "Read-dose / readiness"),
              (8.22, "HELD-OUT", counts["heldout"], "3 same-evening sessions"))
    for x, title, count, note in phases:
        ax.add_patch(FancyBboxPatch((x, 1.68), 3.28, 1.34, boxstyle="round,pad=0.02,rounding_size=0.12",
                                   facecolor="#1b3651", edgecolor="none"))
        ax.text(x + .2, 2.7, title, color="#9bc3da", size=10, weight="bold")
        ax.text(x + .2, 2.15, str(count), color="white", size=31, weight="bold")
        ax.text(x + 1.05, 2.16, "launches", color="#d7e2ed", size=12)
        ax.text(x + .2, 1.85, note, color="#d7e2ed", size=10)
    for x in (3.88, 7.75):
        ax.add_patch(FancyArrowPatch((x, 2.32), (x + .31, 2.32), arrowstyle="-|>",
                                    mutation_scale=14, color="#79d5cd", linewidth=1.4))
    ax.text(.48, 1.22, "A  Automatic", color="white", size=12, weight="bold")
    ax.text(4.35, 1.22, "B  Ring / Simple", color="white", size=12, weight="bold")
    ax.text(8.22, 1.22, "C  Ring / LL128", color="white", size=12, weight="bold")
    ax.text(.48, .88, "Observed A = B: Ring / Simple, 4 channels; C: Ring / LL128, 4 channels.", color="#d7e2ed", size=11)
    ax.text(.48, .43, f"{counts['heldout']} accepted  /  {counts['primary_available']} primary estimates  |  NVML board-energy estimates only",
            color="#79d5cd", size=11, weight="bold")
    ax.text(.48, .13, "Pilot study; no external energy calibration; global latency / energy optima unresolved.",
            color="#f4cb86", size=10)
    fig.savefig(output / "study_overview.png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def comparisons(data: dict, output: Path, plt) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    frames = []
    for focus in (None, 0, 1, 2, 3, None):
        fig = plt.figure(figsize=(12, 6.7), dpi=110)
        ax = fig.add_axes([.28, .23, .44, .53])
        fig.text(.04, .94, "SAME EXECUTION OR A DIFFERENT PROTOCOL?", size=11, color=COLORS[0], weight="bold")
        fig.text(.04, .885, "Held-out energy comparisons", size=24, weight="bold")
        fig.text(.04, .835, "AllGather 2 MiB  |  RTX A6000 NVLink pair  |  three same-evening sessions", size=11, color=MUTED)
        ax.set(xlim=(-1.5, 2.65), ylim=(-.55, 3.65), yticks=[],
               xlabel="NVML-reported energy/op change (%)")
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.spines["left"].set_visible(False)
        for i, (record, label, note, color) in enumerate(zip(data["selected"], COMPARISON_NAMES, ROW_NOTES, ROW_COLORS)):
            y = 3 - i
            if i < 3:
                lo, hi = (100 * (1 / 1.005 - 1), .5)
                ax.add_patch(Rectangle((lo, y - .39), hi - lo, .8, facecolor="#edf3f8", edgecolor="none", zorder=0))
            active = focus is None or focus == i
            alpha = 1 if active else .32
            m = record["mean_pct"]; lo, hi = record["ci95_pct"]
            ax.errorbar(m, y - .10, xerr=[[m - lo], [hi - m]], fmt="D", color=color,
                        markersize=7, capsize=4, elinewidth=2.2, alpha=alpha, zorder=5)
            for j, (value, marker) in enumerate(zip(record["session_pct"], ("o", "s", "^"))):
                ax.plot(value, y + .20, marker=marker, markersize=5.5, linestyle="none",
                        color=color, alpha=alpha, markerfacecolor="white", markeredgewidth=1.3)
            fy = .23 + ((y + .55) / 4.2) * .53
            fig.text(.04, fy + .010, label, size=13, weight="bold", color=color, alpha=alpha)
            fig.text(.04, fy - .025, "Identical execution" if i < 3 else "LL128 vs automatic", size=10, color=MUTED, alpha=alpha)
            fig.text(.76, fy + .010, f"{m:+.2f}%  [{lo:+.2f}, {hi:+.2f}]", size=12, weight="bold", color=color, alpha=alpha)
            fig.text(.76, fy - .025, note, size=9.5, color=MUTED, alpha=alpha)
        handles = [Line2D([], [], marker="D", color=INK, label="All-session mean + 95% CI", markersize=6),
                   *[Line2D([], [], marker=m, color=MUTED, markerfacecolor="white", linestyle="none", label=f"Session {i + 1}")
                     for i, m in enumerate(("o", "s", "^"))]]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.03, .12), ncol=4,
                   frameon=False, fontsize=10, handlelength=1.3, columnspacing=1.5)
        fig.text(.04, .093, "Shading: exact [1/1.005, 1.005] control margin. All CIs use n=3 session means (df=2).", size=10, color=MUTED)
        fig.text(.04, .052, "Precision target missed; all aggregate adequacy scopes unmet; no physical-accuracy or global-optimum claim.", size=10, color="#8c5811")
        fig.text(.04, .02, "Source: prep/heldout_final/heldout_analysis.json  |  Animation changes emphasis only, not estimates or sample size.", size=9, color=MUTED)
        frames.append(frame_image(fig))
        if focus is None:
            fig.savefig(output / "heldout_comparisons.png", dpi=110)
        plt.close(fig)
    save_animation(frames, output / "heldout_comparisons.gif")


def duration_sensitivity(data: dict, output: Path, plt) -> None:
    from matplotlib.lines import Line2D
    limits = [v for est in ESTIMATORS for arm in ("A", "B", "C")
              for row in data["duration"][est][arm] for v in row["ci95_pct"]]
    ymin, ymax = min(limits), max(limits)
    pad = (ymax - ymin) * .13
    frames = []
    for focus in (None, 0, 1, 2, 3, None):
        fig, axes = plt.subplots(1, 3, figsize=(12, 6.2), dpi=110, sharey=True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.36, top=.72, wspace=.12)
        fig.text(.04, .94, "WHY THE ENERGY ESTIMATOR MATTERS", size=11, weight="bold", color=COLORS[0])
        fig.text(.04, .875, "Short regions do not tell the same story", size=23, weight="bold")
        focus_name = "All four estimators" if focus is None else ESTIMATOR_NAMES[focus]
        fig.text(.04, .815, f"{focus_name}  |  2.5, 10 and 20 s versus the same arm at 40 s", size=12, color=MUTED)
        for ax, arm, arm_name in zip(axes, ("A", "B", "C"), ("A  Automatic", "B  Ring / Simple", "C  Ring / LL128")):
            ax.set_title(arm_name, loc="left", size=13, weight="bold", pad=10)
            ax.axhline(0, color=INK, linewidth=1)
            ax.set(xticks=[0, 1, 2], xticklabels=["2.5", "10", "20"], xlim=(-.42, 2.42), ylim=(ymin - pad, ymax + pad),
                   xlabel="Region target (s)")
            ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, linewidth=.7)
            ax.spines["left"].set_visible(False)
            for ei, (est, color, marker) in enumerate(zip(ESTIMATORS, COLORS, ("D", "o", "s", "^"))):
                alpha = 1 if focus is None or focus == ei else .18
                for li, record in enumerate(data["duration"][est][arm]):
                    m = record["mean_pct"]; lo, hi = record["ci95_pct"]
                    ax.errorbar(li + (ei - 1.5) * .13, m, xerr=None, yerr=[[m - lo], [hi - m]],
                                fmt=marker, color=color, markersize=5, capsize=3, alpha=alpha, elinewidth=1.4)
        axes[0].set_ylabel("Energy/op change versus 40 s (%)", labelpad=10)
        handles = [Line2D([], [], marker=m, color=c, linestyle="none", label=n)
                   for c, m, n in zip(COLORS, ("D", "o", "s", "^"), ESTIMATOR_NAMES)]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.035, .16), ncol=2,
                   frameon=False, fontsize=10, handletextpad=.5, columnspacing=1.5)
        fig.text(.04, .11, "Dots + descriptive 95% CIs: three session means. No interpolation between measured durations.", size=10, color=MUTED)
        fig.text(.04, .065, "Sensitivity only: shared NVML infrastructure, not independent calibration. Mechanism unresolved.", size=10, color="#8c5811")
        fig.text(.04, .025, "AllGather 2 MiB; same A6000 pair / evening. Source: prep/heldout_final/heldout_analysis.json.", size=9, color=MUTED)
        frames.append(frame_image(fig))
        if focus is None:
            fig.savefig(output / "estimator_duration.png", dpi=110)
        plt.close(fig)
    save_animation(frames, output / "estimator_duration.gif")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY.parents[1] / "reproduced/stage2-readme-media")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh output directory; existing media will not be overwritten.")
    payload = build_payload()
    args.output.mkdir(parents=True, exist_ok=True)
    plt = setup_plotting()
    overview(payload, args.output, plt)
    comparisons(payload, args.output, plt)
    duration_sensitivity(payload, args.output, plt)
    (args.output / "media_values.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Rendered 2 data-driven GIFs, their static alternatives, a study overview and a source-value ledger.")


if __name__ == "__main__":
    main()
