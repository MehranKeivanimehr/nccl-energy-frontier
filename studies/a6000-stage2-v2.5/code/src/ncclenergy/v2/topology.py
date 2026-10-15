"""Topology fingerprints from `nvidia-smi topo -m` output (parsed, not assumed)."""

from __future__ import annotations

import re

from .common import sha256_bytes

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_topo_matrix(text: str) -> dict:
    """Parse the GPU block of `nvidia-smi topo -m`.

    Returns {"links": {(i, j): "NV4"|"NODE"|"SYS"|...}, "numa": {i: int|None},
    "cpu_affinity": {i: str}}. NIC rows/columns are ignored.
    """
    lines = [_ANSI.sub("", ln).rstrip("\n") for ln in text.splitlines()]
    header = next((ln for ln in lines if re.match(r"^\s*GPU0\b", ln)), None)
    if header is None:
        raise ValueError("no GPU header line in topo output")
    cols = header.split("\t") if "\t" in header else header.split()
    cols = [c.strip() for c in cols if c.strip()]
    gpu_cols = [c for c in cols if re.fullmatch(r"GPU\d+", c)]
    n = len(gpu_cols)
    links, numa, aff = {}, {}, {}
    for ln in lines:
        m = re.match(r"^GPU(\d+)\s", ln)
        if not m:
            continue
        i = int(m[1])
        toks = [t.strip() for t in (ln.split("\t") if "\t" in ln else ln.split())]
        toks = [t for t in toks if t != ""]          # nvidia-smi pads some columns with empty fields
        row = dict(zip(cols, toks[1:]))
        for c, v in row.items():
            if re.fullmatch(r"GPU\d+", c):
                links[(i, int(c[3:]))] = v
        aff[i] = row.get("CPU Affinity")
        na = row.get("NUMA Affinity")
        numa[i] = int(na) if na is not None and na.isdigit() else None
    if any((i, i) not in links for i in range(n)):
        raise ValueError("incomplete topo matrix")
    return {"links": links, "numa": numa, "cpu_affinity": aff, "n": n}


def category(gpus: list[int], topo: dict) -> str:
    pairs = [(a, b) for k, a in enumerate(gpus) for b in gpus[k + 1:]]
    kinds = [topo["links"][(a, b)] for a, b in pairs]
    nv = [k.startswith("NV") for k in kinds]
    if all(nv):
        base = "NVLINK_ALL"
    elif not any(nv):
        base = "PCIE_ONLY"
    else:
        base = "MIXED_NVLINK_PCIE"
    if any(k == "SYS" for k in kinds):
        base += "+CROSS_SOCKET"
    return base


def fingerprint(gpus: list[int], topo: dict, pci: dict[int, str], nvlink_active: dict[int, int] | None = None,
                nccl_version: str | None = None) -> dict:
    """Canonical description of a placement. Index-free link pattern + the concrete PCI ids."""
    gpus = sorted(gpus)
    pairs = [(a, b) for k, a in enumerate(gpus) for b in gpus[k + 1:]]
    link_str = ",".join(f"{gpus.index(a)}-{gpus.index(b)}:{topo['links'][(a, b)]}" for a, b in pairs)
    numa_str = ",".join(str(topo["numa"].get(g)) for g in gpus)
    nvl = ",".join(str((nvlink_active or {}).get(g, "?")) for g in gpus)
    canonical = f"n={len(gpus)};links={link_str};numa={numa_str};nvlinks={nvl}"
    concrete = canonical + ";pci=" + ",".join(pci.get(g, "?") for g in gpus)
    return {"gpus": gpus, "category": category(gpus, topo), "canonical": canonical,
            "fingerprint": sha256_bytes(canonical.encode())[:16],
            "concrete_fingerprint": sha256_bytes(concrete.encode())[:16], "nccl_version": nccl_version}
