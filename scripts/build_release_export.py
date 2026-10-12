#!/usr/bin/env python3
"""Build the sanitized public-release export of this repository (CPU only; reads the local tree, writes a new
directory outside it). The repository and its data are not modified.

    python scripts/build_release_export.py --out <new dir outside the repo> --inputs-from <release_repro --audit dir>

Contents:
  * every file tracked at git HEAD, except the files in OMIT (with the reason recorded);
  * every untracked file that a published analysis reads (inputs_<step>.tsv of an audited release_repro run);
  * the NCCL ext-tuner example headers and NCCL's license (needed to build and test the guarded tuner).
Text files (also inside .gz) are sanitized by the rules below; a file changed by a rule is a documented derivative
(DERIVATIVES.csv: rules applied, original and export SHA-256). Scientific values are never changed: the rules only
replace path prefixes, the user name, the host name, private IPv4/MAC addresses, GPU UUIDs and serial numbers.
The private tokens are taken from the running environment (repository path, home directory, user, host name), so this
script contains none of them. After copying, the whole export is scanned again; any remaining hit aborts the build.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import getpass
import gzip
import hashlib
import io
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OMIT = [
    ("data/raw/*/metadata_*/nvidia-smi-q.txt",
     "full `nvidia-smi -q` dump (GPU serial numbers, UUIDs, board and module identifiers); not read by any analysis"),
    ("data/raw/*/metadata_*/nvidia-smi-compute-apps.txt",
     "process list at run start, including other users' processes; not read by any analysis"),
    ("data/raw/*/metadata_*/nvidia-smi.txt",
     "`nvidia-smi` summary whose process table lists another user's process; not read by any analysis"),
]
EXTRA = ["third_party/nccl-src/ext-tuner/example/nccl/common.h", "third_party/nccl-src/ext-tuner/example/nccl/err.h",
         "third_party/nccl-src/ext-tuner/example/nccl/tuner.h", "third_party/nccl-src/LICENSE.txt"]
SYNTHETIC_UUID = "GPU-00000000-1111-2222-3333-444444444444"     # used by a test; not a real device
BINARY = {".png", ".pdf", ".so", ".o", ".a", ".bin", ".tar", ".zip"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tokens() -> dict[str, list[str]]:
    """Private strings of this environment, longest first within each class."""
    home = os.path.expanduser("~")
    homes = sorted({home, os.path.realpath(home)}, key=len, reverse=True)
    repos = sorted({str(REPO), str(REPO.resolve()), os.path.realpath(str(REPO))}, key=len, reverse=True)
    host = socket.gethostname()
    hosts = sorted({h for h in (host, host.split(".")[0], socket.getfqdn()) if h and h != "localhost"}, key=len,
                   reverse=True)
    user = getpass.getuser()
    comps = {c for h in homes for c in Path(h).parts[1:] if c not in ("home", "md127", "Users", user)}
    return {"repo": repos, "home": homes, "user": [user], "host": hosts, "home_component": sorted(comps)}


def rules(tok: dict) -> list[tuple[str, re.Pattern, str]]:
    r = []
    for p in tok["repo"]:
        r.append(("repo_path", re.compile(re.escape(p)), "<REPO>"))
    for p in tok["home"]:
        r.append(("home_path", re.compile(re.escape(p)), "<HOME>"))
    for h in tok["host"]:
        r.append(("host_name", re.compile(re.escape(h)), "<host>"))
    for u in tok["user"]:
        r.append(("user_name", re.compile(re.escape(u)), "<user>"))
    # (?<![=.\w]) keeps version pins such as "pkg==10.3.9.90" intact
    r += [("private_ipv4", re.compile(r"(?<![=.\w])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
           "<ip>"),
          ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"), "<mac>"),
          ("gpu_uuid", re.compile(r"GPU-(?!00000000-1111-2222-3333-444444444444)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                                  r"[0-9a-f]{4}-[0-9a-f]{12}"), "GPU-<uuid>"),
          ("serial_number", re.compile(r"(?i)(serial(?:[ _]?number)?\"?\s*[:=]\s*\"?)[A-Za-z0-9-]{6,}"), r"\1<serial>")]
    return r


def scan_patterns(tok: dict) -> list[tuple[str, re.Pattern]]:
    lit = [("repo/home/user/host/home component", re.compile("|".join(
        re.escape(x) for k in ("repo", "home", "user", "host", "home_component") for x in tok[k])))]
    return lit + [(n, p) for n, p, _ in rules(tok)[-4:]] + [
        ("credential", re.compile(r"(?i)passw(?:or)?d\s*[:=]|api[_-]?key|secret[_-]?key|BEGIN [A-Z ]*PRIVATE KEY|"
                                  r"gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12}")),
        ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}"))]


def read_text(raw: bytes, gz: bool) -> str | None:
    try:
        return (gzip.decompress(raw) if gz else raw).decode("utf-8")
    except (UnicodeDecodeError, OSError, EOFError):
        return None


def sanitize(text: str, rr) -> tuple[str, dict]:
    fired = {}
    for name, pat, rep in rr:
        text, n = pat.subn(rep, text)
        if n:
            fired[name] = fired.get(name, 0) + n
    return text, fired


def file_list(inputs_from: Path) -> tuple[list[str], list[tuple[str, str]]]:
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    extra = set()
    for f in inputs_from.glob("inputs_*.tsv"):
        for line in f.read_text().splitlines()[1:]:
            extra.add(line.split("\t")[0])
    files, omitted = [], []
    for p in sorted(set(tracked) | extra | set(EXTRA)):
        why = next((w for pat, w in OMIT if fnmatch.fnmatch(p, pat)), None)
        if why:
            omitted.append((p, why))
        elif (REPO / p).is_file():
            files.append(p)
    return files, omitted


def untracked_not_exported(files: set[str]) -> list[tuple[str, int, int]]:
    """On-disk raw artifacts under data/ that are not in the export, summarized per directory (count, bytes)."""
    out = {}
    for root in ("data/raw", "data/v2/raw", "data/v2.2/raw"):
        for p in (REPO / root).rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(REPO))
                if rel not in files:
                    key = "/".join(rel.split("/")[:4]) if "/blocks/" not in rel else "/".join(rel.split("/")[:5])
                    c, b = out.get(key, (0, 0))
                    out[key] = (c + 1, b + p.stat().st_size)
    return sorted((k, c, b) for k, (c, b) in out.items())


def export_gitignore(out: Path, files: list[str]):
    """The repository ignores per-process data and third_party/, but the export contains exactly the per-process files the
    published analyses read: drop every ignore rule that matches an exported file (git's own matcher), so that a public
    repository built from the export tracks all of it. The dropped rules are listed in the file header."""
    src = REPO / ".gitignore"
    if not src.exists():
        return None
    p = subprocess.run(["git", "check-ignore", "--no-index", "-v", "--stdin"], cwd=REPO, input="\n".join(files),
                       capture_output=True, text=True)
    drop = {int(l.split(":")[1]) for l in p.stdout.splitlines() if l.startswith(".gitignore:")}
    lines = src.read_text().splitlines()
    kept = [l for i, l in enumerate(lines, 1) if i not in drop]
    hdr = ["# Release export: rules of the original .gitignore that matched exported evidence files were removed so the",
           "# evidence is tracked. Removed rules:"] + [f"#   {lines[i - 1]}" for i in sorted(drop)] + [""]
    new = ("\n".join(hdr + kept) + "\n").encode()
    (out / ".gitignore").write_bytes(new)
    return (".gitignore", f"release_gitignore_rules_removed:{len(drop)}", sha256_bytes(src.read_bytes()),
            sha256_bytes(new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs-from", type=Path, required=True)
    ap.add_argument("--tar", action="store_true", help="also write <out>.tar.gz and its .sha256")
    a = ap.parse_args()
    out = a.out.resolve()
    if out == REPO.resolve() or REPO.resolve() in out.parents:
        raise SystemExit("the export must be outside the repository")
    if out.exists():
        raise SystemExit(f"{out} exists; the export is written only into a new directory")
    tok = tokens()
    rr = rules(tok)
    files, omitted = file_list(a.inputs_from)
    out.mkdir(parents=True)
    deriv = []
    for rel in files:
        src = REPO / rel
        raw = src.read_bytes()
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        gz = rel.endswith(".gz")
        text = None if Path(rel).suffix.lower() in BINARY else read_text(raw, gz)
        if text is not None:
            new, fired = sanitize(text, rr)
            if fired:
                nb = new.encode("utf-8")
                if gz:
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
                        g.write(nb)
                    nb = buf.getvalue()
                dst.write_bytes(nb)
                shutil.copymode(src, dst)
                deriv.append((rel, ";".join(f"{k}:{v}" for k, v in sorted(fired.items())), sha256_bytes(raw),
                              sha256_bytes(nb)))
                continue
        shutil.copy2(src, dst)
    gi = export_gitignore(out, files)
    if gi:
        deriv.append(gi)
    # documentation of the export
    with open(out / "DERIVATIVES.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "rules_applied(count)", "sha256_original", "sha256_export"])
        w.writerows(deriv)
    nx = untracked_not_exported(set(files))
    lines = ["# Files not in this export", "",
             "## Omitted for privacy (not read by any published analysis)", "", "| path | reason |", "|---|---|"]
    lines += [f"| `{p}` | {w} |" for p, w in omitted]
    lines += ["", "## Raw artifacts kept only in the original local data (not read by any published analysis)", "",
              "Per-process files that no published analysis reads (determined by auditing every file the analyses "
              "open). They include NCCL debug logs and benchmark stdout that print the host name and private network "
              "addresses, per-rank Phase 10 logs, pilot per-launch directories, probe and calibration runs. The "
              "original manifests in `provenance/` list them with SHA-256 (paths sanitized).", "",
              "| directory | files | bytes |", "|---|---|---|"]
    lines += [f"| `{k}` | {c} | {b} |" for k, c, b in nx]
    (out / "OMITTED.md").write_text("\n".join(lines) + "\n")
    # post-build scan of the complete export
    hits = []
    sp = scan_patterns(tok)
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.suffix.lower() in BINARY:
            continue
        t = read_text(p.read_bytes(), p.suffix == ".gz")
        if t is None:
            continue
        for name, pat in sp:
            for m_ in pat.finditer(t):
                if name == "email" and m_.group(0).endswith(("@anthropic.com",)):
                    continue     # commit attribution address in docs, not personal data
                hits.append((str(p.relative_to(out)), name))
                break
        if any(c in str(p.relative_to(out)) for k in ("user", "host") for c in tok[k]):
            hits.append((str(p.relative_to(out)), "private token in file name"))
    if hits:
        print(f"EXPORT SCAN FAILED: {len(hits)} hits, e.g. {hits[:10]}")
        sys.exit(1)
    # checksums
    with open(out / "SHA256SUMS", "w") as f:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS":
                f.write(f"{sha256_bytes(p.read_bytes())}  {p.relative_to(out)}\n")
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"export: {n} files, {len(deriv)} sanitized derivatives, {len(omitted)} omitted tracked files, "
          f"{sum(c for _, c, _ in nx)} on-disk raw files not exported; post-build scan: 0 hits")
    if a.tar:
        tgz = out.with_suffix(".tar.gz") if out.suffix == "" else Path(str(out) + ".tar.gz")
        with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT) as t:
            for p in sorted(out.rglob("*")):
                ti = t.gettarinfo(str(p), arcname=str(Path(out.name) / p.relative_to(out)))
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mtime = 0
                if p.is_file():
                    with open(p, "rb") as fh:
                        t.addfile(ti, fh)
                else:
                    t.addfile(ti)
        Path(str(tgz) + ".sha256").write_text(f"{sha256_bytes(tgz.read_bytes())}  {tgz.name}\n")
        print(f"archive: {tgz} ({tgz.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
