"""The single held-out inferential analysis (CPU only; reads files, no NVML, no GPU).

    python run_final_analysis.py <stage2_root>

One call of the strict production entry point stage2_heldout.analyse_heldout over the three authentic session
directories and their approved Freeze-H records, with the default production trust anchors. Writes
prep/heldout_final/: heldout_analysis.json (the complete machine-readable output), units.tsv, exclusions.tsv,
execution.tsv, contrasts.tsv, thermal_control.tsv, adequacy.tsv, evidence_manifests.tsv and FINAL_HELDOUT_REPORT.md.
Percentages are 100 * expm1(log contrast) (ordinary %), with the geometric ratio exp(x); log values are also given.
Older-method (pilot, v2, gate 3b, calibration) data are not read except the frozen calibration analysis that the
protocol uses for Q1/Q3.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
OUT_NAME = "prep/heldout_final"
FAMILIES = (("HD1", "length dependence ln(J_L/J_40) within arm"), ("HD2", "protocol term of ln(E_samp/E)"),
            ("HD3", "B vs A ln(J_B/J_A) (same execution, forced vs default)"), ("HD4", "C vs A at 40 s ln(J_C/J_A)"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(x):
    return None if x is None else 100.0 * math.expm1(x)


def f(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def tsv(path, cols, rows):
    path.write_text("\t".join(cols) + "\n" + "".join("\t".join(f(r.get(c), 6) if isinstance(r.get(c), float)
                                                               else ("" if r.get(c) is None else str(r.get(c)))
                                                               for c in cols) + "\n" for r in rows))


def contrast_rows(out):
    rows = []
    for est, res in out["estimators"].items():
        for fam, _ in FAMILIES:
            fr = res.get(fam)
            if not isinstance(fr, dict):
                rows.append({"estimator": est, "family": fam, "member": "-", "verdict": str(fr)})
                continue
            for m, e in fr.items():
                sm = (res.get("summaries") or {}).get(fam, {}).get(m, {})
                d, q = e.get("difference") or {}, e.get("equivalence") or {}
                ci95, ci90 = d.get("ci95") or sm.get("ci95"), q.get("ci90") or sm.get("ci90")
                sms = sm.get("session_means") or {}
                cond = sm.get("conditional_descriptive") or {}
                rows.append({
                    "estimator": est, "family": fam, "member": m, "family_size": e.get("family_size"),
                    "delta_log": e.get("delta"), "status": e.get("status"), "n_sessions": sm.get("n"),
                    "s1_mean_log": sms.get("1"), "s2_mean_log": sms.get("2"), "s3_mean_log": sms.get("3"),
                    "mean_log": sm.get("mean"), "sd_sessions_log": sm.get("sd"), "se_log": sm.get("se"),
                    "ci95_lo_log": ci95[0] if ci95 else None, "ci95_hi_log": ci95[1] if ci95 else None,
                    "ci90_lo_log": ci90[0] if ci90 else None, "ci90_hi_log": ci90[1] if ci90 else None,
                    "mean_pct": pct(sm.get("mean")), "ci95_lo_pct": pct(ci95[0]) if ci95 else None,
                    "ci95_hi_pct": pct(ci95[1]) if ci95 else None, "ratio": math.exp(sm["mean"]) if sm.get("mean") is not None else None,
                    "p_two_sided": d.get("p"), "p_holm": d.get("p_holm"), "p_beyond_margin": d.get("p_beyond_margin"),
                    "p_beyond_margin_holm": d.get("p_beyond_margin_holm"), "p_tost_unadjusted": q.get("p_tost"),
                    "verdict": e.get("verdict"), "n_blocks": cond.get("n"), "blocks_mean_log": cond.get("mean"),
                    "blocks_ci95_lo_log": (cond.get("ci95") or [None, None])[0],
                    "blocks_ci95_hi_log": (cond.get("ci95") or [None, None])[1]})
    return rows


def write_outputs(out: dict, od: Path, root: Path | None = None, session_dirs: dict | None = None) -> dict:
    od.mkdir(parents=True, exist_ok=True)
    (od / "heldout_analysis.json").write_text(json.dumps(out, indent=1))
    ests = list(out["estimators"])
    urows = []
    for u in out["units"]:
        r = {k: u.get(k) for k in ("launch_id", "session", "block", "arm", "length_s", "status", "R")}
        r["reasons"] = ";".join(u.get("reasons") or [])
        r["exec_key"] = u.get("exec_key")
        for e in ests:
            r[f"J_{e}"] = (u.get("J") or {}).get(e)
            r[f"unavailable_{e}"] = ";".join((u.get("unavailable") or {}).get(e) or [])
        urows.append(r)
    ucols = ["launch_id", "session", "block", "arm", "length_s", "status", "reasons", "R", "exec_key"] + \
        [f"J_{e}" for e in ests] + [f"unavailable_{e}" for e in ests]
    tsv(od / "units.tsv", ucols, urows)
    xrows = [{"kind": "unit", "estimator": "all", "item": u["launch_id"], "session": u["session"], "block": u["block"],
              "reason": ";".join(u.get("reasons") or [])} for u in out["units"] if u["status"] != "USED"]
    for e in ests:
        for u in out["units"]:
            for why in (u.get("unavailable") or {}).get(e) or []:
                xrows.append({"kind": "estimator_unavailable", "estimator": e, "item": u["launch_id"],
                              "session": u["session"], "block": u["block"], "reason": why})
        for fam, m, s, b, why in out["estimators"][e]["excluded"]:
            xrows.append({"kind": "block_contrast_missing", "estimator": e, "item": f"{fam}:{m}", "session": s,
                          "block": b, "reason": why})
    tsv(od / "exclusions.tsv", ["kind", "estimator", "item", "session", "block", "reason"], xrows)
    crows = contrast_rows(out)
    ccols = ["estimator", "family", "member", "family_size", "delta_log", "status", "n_sessions", "s1_mean_log",
             "s2_mean_log", "s3_mean_log", "mean_log", "sd_sessions_log", "se_log", "ci95_lo_log", "ci95_hi_log",
             "ci90_lo_log", "ci90_hi_log", "mean_pct", "ci95_lo_pct", "ci95_hi_pct", "ratio", "p_two_sided", "p_holm",
             "p_beyond_margin", "p_beyond_margin_holm", "p_tost_unadjusted", "verdict", "n_blocks", "blocks_mean_log",
             "blocks_ci95_lo_log", "blocks_ci95_hi_log"]
    tsv(od / "contrasts.tsv", ccols, crows)
    p = out["estimators"]["primary"]
    trows = [{"check": "R3", "key": k, "ok": v.get("ok"), "detail": json.dumps(v.get("diff_c_by_gpu") or v.get("reason"))}
             for k, v in p["r3"].items()]
    trows += [{"check": "control", "key": k, "ok": v.get("valid"), "detail": json.dumps(v.get("differing_lengths"))}
              for k, v in p["control"].items()]
    trows += [{"check": "precision_HD4", "key": "C/A@40", "ok": p["HD4_precision"].get("ok"),
               "detail": json.dumps(p["HD4_precision"])}]
    trows += [{"check": "precision_HD3", "key": k, "ok": v.get("ok"), "detail": json.dumps(v)}
              for k, v in p["HD3_precision"].items()]
    tsv(od / "thermal_control.tsv", ["check", "key", "ok", "detail"], trows)
    arows = [{"scope": k, "status": v["status"], "conditional_statistical": v["conditional_statistical"],
              "unmet_statistical": "; ".join(v["unmet_statistical"]), "unmet_systematic": "; ".join(v["unmet_systematic"])}
             for k, v in out["adequacy"].items()]
    tsv(od / "adequacy.tsv", ["scope", "status", "conditional_statistical", "unmet_statistical", "unmet_systematic"], arows)
    erows = []
    if root is not None:
        for s in (1, 2, 3):
            for rel in (f"prep/heldout_s{s}/run/session_files.sha256", f"protocol/FREEZE_H_S{s}.json",
                        f"code/configs/stage2/heldout_s{s}.yaml", f"prep/heldout_s{s}/allocation/user_authorization.json",
                        f"delivery/heldout_s{s}/heldout_s{s}_evidence_2026-10-14.zip"):
                pth = root / rel
                erows.append({"session": s, "file": rel, "sha256": sha(pth) if pth.exists() else "MISSING"})
            if session_dirs:
                man = root / f"prep/heldout_s{s}/run/session_files.sha256"
                lines = [x for x in man.read_text().splitlines() if x.strip()] if man.exists() else []
                bad = [q for h, q in (x.split("  ", 1) for x in lines) if sha(Path(session_dirs[s]) / q) != h]
                erows.append({"session": s, "file": f"raw session files vs manifest ({len(lines)})",
                              "sha256": "ALL MATCH" if lines and not bad else f"MISMATCH {bad[:3]}"})
        tsv(od / "evidence_manifests.tsv", ["session", "file", "sha256"], erows)
    return {"contrasts": crows, "adequacy": arows, "thermal_control": trows, "evidence": erows}


def main():
    root = ROOT
    repo = root / "code"
    sys.path.insert(0, str(repo / "src"))
    from ncclenergy.v2 import stage2_heldout as H
    dirs = {s: repo / f"data/stage2/v2.5-heldout/session_{s}" for s in (1, 2, 3)}
    frs = {s: root / f"protocol/FREEZE_H_S{s}.json" for s in (1, 2, 3)}
    out = H.analyse_heldout(dirs, frs)                      # strict production path, default trust anchors
    od = root / OUT_NAME
    t = write_outputs(out, od, root, dirs)
    ex = []
    for s in (1, 2, 3):
        p = root / f"prep/heldout_s{s}/run/execution_qc.tsv"
        lines = p.read_text().splitlines()
        if not ex:
            ex.append("session\t" + lines[0])
        ex += [f"{s}\t{x}" for x in lines[1:]]
    (od / "execution.tsv").write_text("\n".join(ex) + "\n")
    print(json.dumps({"provenance_status": out["provenance_status"], "unit_status_counts": out["unit_status_counts"],
                      "analysis_json_sha256": sha(od / "heldout_analysis.json"),
                      "primary_verdicts": {f"{r['family']}:{r['member']}": r["verdict"] for r in t["contrasts"]
                                           if r["estimator"] == "primary"},
                      "adequacy": {r["scope"]: (r["status"], r["conditional_statistical"]) for r in t["adequacy"]}},
                     indent=1))


if __name__ == "__main__":
    main()
