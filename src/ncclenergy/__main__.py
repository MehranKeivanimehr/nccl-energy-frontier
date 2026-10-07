"""Command-line interface for experiment capture and offline analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

PHASES = ["probe", "calibrate", "trials", "probe_end"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ncclenergy")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("metadata")
    m.add_argument("--config", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)

    r = sub.add_parser("run")
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--run-name", default=None)
    r.add_argument("--phases", default=",".join(PHASES))

    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--n-boot", type=int, default=10000)
    a.add_argument("--seed", type=int, default=12345)

    p = sub.add_parser("plot")
    p.add_argument("--processed", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    rp = sub.add_parser("report")
    rp.add_argument("--processed", required=True, type=Path)

    s = sub.add_parser("smoke-check")
    s.add_argument("--run", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "metadata":
        from .metadata import collect
        from .orchestrate import REPO, load_config
        cfg, _ = load_config(args.config)
        collect(args.out, REPO, cfg)
    elif args.cmd == "run":
        from .orchestrate import Orchestrator
        phases = [x for x in args.phases.split(",") if x]
        bad = set(phases) - set(PHASES)
        if bad:
            raise SystemExit(f"unknown phases {bad}; valid: {PHASES}")
        Orchestrator(args.config, args.run_name).run(phases)
    elif args.cmd == "analyze":
        from .analyze import analyze_run
        analyze_run(args.run, args.out, n_boot=args.n_boot, seed=args.seed)
    elif args.cmd == "plot":
        from .plots import plot_all
        plot_all(args.processed, args.out)
    elif args.cmd == "report":
        from .report import write_report
        write_report(args.processed)
    elif args.cmd == "smoke-check":
        from .smoke import smoke_check
        smoke_check(args.run, args.out)


if __name__ == "__main__":
    main()
