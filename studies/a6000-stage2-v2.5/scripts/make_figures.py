"""Plot retained held-out contrasts; this performs no new inferential analysis."""
from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=root.parents[1] / "reproduced/stage2-figures")
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Choose a fresh figure directory; existing figures will not be overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    with (root / "prep/heldout_final/contrasts.tsv").open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    chosen = [next(r for r in rows if r["estimator"] == "primary" and r["family"] == fam and r["member"] == member)
              for fam, member in (("HD3", "10"), ("HD3", "20"), ("HD3", "40"), ("HD4", "C/A@40"))]
    labels = ["B/A, 10 s: mean-ratio compatibility", "B/A, 20 s: thermal criterion not met",
              "B/A, 40 s: insufficient precision", "C/A, 40 s: precision target missed"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, r in enumerate(chosen):
        m, lo, hi = (float(r[k]) for k in ("mean_pct", "ci95_lo_pct", "ci95_hi_pct"))
        ax.errorbar(m, i, xerr=[[m-lo], [hi-m]], fmt="o", color="#147e7a" if i in (0,3) else "#8d6e35", capsize=3)
    ax.axvline(0, color="#687a82", linewidth=1)
    ax.set(yticks=range(4), yticklabels=labels, xlabel="NVML-reported board energy per operation: relative change (%)")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
    fig.suptitle("AllGather 2 MiB, RTX A6000 NVLink pair: selected comparisons")
    fig.text(.99,.02,"Three same-evening sessions; descriptive 95% CIs; no physical accuracy or global-optimum claim.",ha="right",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94)); fig.savefig(args.output / "selected_comparisons.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.2))
    for est, color in (("primary", "#147e7a"), ("raw_counter", "#284a6f"), ("nominal_period", "#b78a29"), ("sampled_instant", "#bd4b50")):
        vals = [next(float(r["mean_pct"]) for r in rows if r["estimator"] == est and r["family"] == "HD1"
                     and r["member"] == f"{a}|2.5") for a in ("A","B","C")]
        ax.plot([0,1,2], vals, "o-", label=est, color=color)
    ax.axhline(0,color="#687a82",lw=1); ax.set(xticks=[0,1,2],xticklabels=["A: automatic", "B: Ring/Simple", "C: Ring/LL128"],
        ylabel="2.5 s versus 40 s: mean energy/op change (%)")
    ax.grid(alpha=.15); ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Estimator-dependent duration effects (sensitivity, not a new verdict)")
    fig.text(.99,.02,"Point estimates only; same observations/sensor infrastructure. Mechanism unresolved.",ha="right",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.93)); fig.savefig(args.output / "duration_sensitivity.png"); plt.close(fig)
    print("Two figures rendered from retained contrast-table values.")


if __name__ == "__main__":
    main()
