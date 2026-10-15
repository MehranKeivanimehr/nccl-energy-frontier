"""Stage 2 runner (prospective measurement-method and thermal-start validation). One owned process at a time.

    python -m ncclenergy.v2.stage2_runner check       --campaign configs/stage2/<cfg>.yaml   (CPU: refusals only)
    python -m ncclenergy.v2.stage2_runner smoke       --campaign configs/stage2/smoke01.yaml   (pre-Freeze-C gate)
    python -m ncclenergy.v2.stage2_runner calibration --campaign configs/stage2/calib01.yaml
    python -m ncclenergy.v2.stage2_runner heldout     --campaign configs/stage2/heldout_sN.yaml

Refuses, before any NVML call, unless the config passes check_config: the prospective identity
(implementation_id / method_id below, never pooled with historical methods), a mode, output root data/stage2, counter
energy source, max_attempts 1, exactly two allocated GPUs in one placement, safe identifiers, the user's allocation
confirmation naming exactly those GPUs with an access window that contains now, an expected scope (GPU model, driver,
pair link, NUMA, UUIDs, power/clock profile), and a freeze record of the matching kind that binds THIS config file,
the code that is loaded, the protocol, the patch, the binary, its build record, the profiler plugin, the loaded NCCL
library (external hash) and, from Freeze-C on, the analysis; with a verified parent chain (S -> C -> H).

Never: retries, replacement units, launching because a wait expired, controller counter reads during a unit or before
the state is validated, privileged operations, signals to processes we do not own.

Known limit: NVML queries are synchronous library calls. A call that blocks inside the driver blocks its thread; the
deadline is enforced at every phase boundary and by the watchdog for the owned benchmark process, but a blocked NVML
call in the controller can delay the controller beyond the deadline. That case is not bounded by this code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import runner as R
from .attempts import AttemptLog
from .common import COLL_FUNC, REPO, redact, sha256_file, utc_now
from .evidence import load_profiler_dir, verify
from .preflight import allocation_problems, light_health, poll_xid, snapshot as _snapshot, xid_event_test
from .procctl import ControllerSignal, SafetyStop, controller_signals
from .qc import QC_METHOD, classify, telemetry_problems
from .region import parse_region_text, region_metrics
from .stage2_budget import Deadline, Design, calibration_design, end_baseline_bound_s, session_tail_reserved_s, \
    unit_bound_s
from .stage2_energy import parse_s2_text, required_channel_problems, s2_metrics
from .stage2_readiness import _row_problems, baseline_problems, clock_event_bit_shares, gate_at, p8_entry, \
    pre_region_check
from .stage2_schedule import DOSES, DOSE_PERIOD_MS
from .telemetry import Sampler, window_summary

IMPLEMENTATION_ID = "stage2-cg1-2026-10-12"            # prospective; replaces stage2-measval-2026-10-12 (never run)
METHOD_ID = "v2.5-stage2-region"                        # never pooled with v2.1/v2.2 historical data or with v2.4
MODES = {"STAGE2_CALIBRATION": "calibration", "STAGE2_HELDOUT": "heldout", "STAGE2_SMOKE": "calibration"}
FREEZE_KIND = {"STAGE2_CALIBRATION": "Freeze-C", "STAGE2_HELDOUT": "Freeze-H", "STAGE2_SMOKE": "Freeze-S"}
FREEZE_SCHEMA = "nccle-stage2-freeze-v1"
MANDATORY_ROLES = {"Freeze-S": ("config", "protocol", "patch", "binary", "build_record", "plugin", "code"),
                   "Freeze-C": ("config", "protocol", "patch", "binary", "build_record", "plugin", "code", "analysis"),
                   "Freeze-H": ("config", "protocol", "patch", "binary", "build_record", "plugin", "code", "analysis")}
MANDATORY_ANALYSIS = {"Freeze-C": ("src/ncclenergy/v2/stage2_calibration.py", "src/ncclenergy/v2/stage2_decisions.py"),
                      "Freeze-H": ("src/ncclenergy/v2/stage2_heldout.py", "src/ncclenergy/v2/stage2_decisions.py")}
PARENT_KIND = {"Freeze-C": "Freeze-S", "Freeze-H": "Freeze-C"}
SHARED_ROLES = ("patch", "binary", "build_record", "plugin", "code")
SMOKE_MAX_UNITS = 4
SMOKE_CEILING_S = 1800.0                 # smoke deadline = min(reservation end, smoke start + 30 min)
OUTPUT_ROOT = "data/stage2"
BIN_DIR_STAGE2 = REPO / "third_party/nccl-tests-stage2-v25/build"
PATCH_V21 = REPO / "patches/nccl-tests-afd59ab-v2.1-region.patch"
PATCH_V25 = REPO / "patches/nccl-tests-afd59ab-v2.5-stage2.patch"
NCCL_LIB_LINK = R.NCCL_LIB / "libnccl.so.2"
ARM_CONFIG = {"A": "auto", "B": "RING/SIMPLE", "C": "RING/LL128"}
ARM_EXPECTED = {"A": None, "B": ("RING", "SIMPLE"), "C": ("RING", "LL128")}
COLL, SIZE_BYTES = "all_gather", 2_097_152
STAGE2_ENV = ("NCCLE_STAGE2_MODE", "NCCLE_S2_FILE", "NCCLE_START_PAUSE_NS", "NCCLE_DOSE_PERIOD_MS", "NCCLE_POST_PROBE")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SCOPE_KEYS = ("gpu_name", "driver", "link", "uuids", "numa", "power_limit_mw", "max_clk_gr_mhz")
UNSAFE_STOP_PREFIXES = ("FOREIGN", "SAFETY", "CLEANUP", "WATCHDOG", "PROCESS_LIST", "power limit", "Xid", "disk",
                        "PAIR_IDENTITY", "SCOPE", "TEMPERATURE", "allocation busy", "INTERRUPTED", "EXCEPTION",
                        "stop condition")


class Stage2Stop(Exception):
    pass


@dataclass(frozen=True)
class Timing:
    """Protocol timing. The CLI always uses these defaults; YAML cannot change them. Tests pass shorter values."""
    cal_fixed_wait_s: float = 120.0
    ho_w_max_s: float = 240.0
    stable_window_s: float = 30.0
    stable_delta_c: float = 1.0
    p8_wait_cap_s: float = 60.0
    baseline_settle_s: float = 10.0
    baseline_measure_s: float = 30.0
    readiness_period_s: float = 1.0
    sampler_period_s: float = 0.05
    baseline_proc_period_s: float = 0.25
    prep_target_s: float = 5.0
    prep_tol: float = 0.10
    length_tol: float = 0.10
    dose_delivery_min: float = 0.95
    pause_tol_ms: float = 2.0
    post_probe_max_s: float = 0.41
    post_probe_min_brackets: int = 3
    temp_stop_c: float = 80.0


# ------------------------------------------------------------------ identity, freeze and scope (no NVML)

def _parse_utc(s) -> datetime:
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("timestamps must carry a UTC offset")
    return d


def loaded_code_files() -> list[str]:
    """Repository-relative paths of every ncclenergy module loaded in this process."""
    out = set()
    for n, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if n.startswith("ncclenergy") and f:
            try:
                out.add(str(Path(f).resolve().relative_to(REPO.resolve())))
            except ValueError:
                out.add(f"OUTSIDE_REPO:{f}")
    return sorted(out)


def verify_freeze(record_path: Path, required_kind: str, config_path: Path | None = None,
                  code_files: list[str] | None = None, bin_path: Path | None = None, _depth: int = 0) -> list[str]:
    """Freeze record (schema nccle-stage2-freeze-v1):
      {"schema", "kind", "utc", "approved_by_user": {...}, "files": {rel: {"sha256", "bytes"}},
       "roles": {"config": rel, "protocol": rel, "patch": rel, "binary": rel, "build_record": rel, "plugin": rel,
                 "code": [rel...], "analysis": [rel...]},
       "external": {"libnccl": {"path": abs, "sha256": ...}}, "parent": {"path": rel, "sha256": ...}}
    Refused unless: every mandatory role for the kind is present and listed in files; every listed file exists with its
    hash; the external NCCL library at the runner's link resolves to the recorded path with the recorded hash; when
    given, the config file being run IS roles.config, the binary being run IS roles.binary and every loaded code file
    is in roles.code; Freeze-C/H have a parent of the previous kind whose record hash matches, which verifies
    recursively, and whose shared roles (patch, binary, build record, plugin, code) have the same hashes."""
    p = []
    try:
        rec = json.loads(Path(record_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"freeze record unreadable: {e}"]
    if rec.get("schema") != FREEZE_SCHEMA:
        p.append(f"freeze record schema must be {FREEZE_SCHEMA}")
    if rec.get("kind") != required_kind:
        p.append(f"freeze record kind {rec.get('kind')!r} != {required_kind}")
    if not rec.get("approved_by_user"):
        p.append("freeze record lacks the user's approval entry")
    files = rec.get("files") or {}
    roles = rec.get("roles") or {}
    for role in MANDATORY_ROLES.get(required_kind, ()):
        v = roles.get(role)
        if not v:
            p.append(f"mandatory role missing: {role}")
            continue
        for rel in (v if isinstance(v, list) else [v]):
            if rel not in files:
                p.append(f"role {role} file not in the manifest: {rel}")
    for rel in MANDATORY_ANALYSIS.get(required_kind, ()):
        if rel not in (roles.get("analysis") or []):
            p.append(f"mandatory analysis file not bound: {rel}")
    for rel, h in files.items():
        want = h.get("sha256") if isinstance(h, dict) else h
        if sha256_file(REPO / rel) != want:
            p.append(f"frozen file changed or missing: {rel}")
    ext = (rec.get("external") or {}).get("libnccl") or {}
    real = str(NCCL_LIB_LINK.resolve()) if NCCL_LIB_LINK.exists() else None
    if not ext.get("path") or not ext.get("sha256"):
        p.append("external libnccl identity missing")
    else:
        if real != ext["path"]:
            p.append(f"loaded NCCL library resolves to {real}, not the frozen {ext['path']}")
        if real is None or sha256_file(Path(real)) != ext["sha256"]:
            p.append("external libnccl hash mismatch")
    if config_path is not None:
        cfg_rel = roles.get("config")
        if not cfg_rel or (REPO / cfg_rel).resolve() != Path(config_path).resolve():
            p.append("the config being run is not the frozen config")
    if bin_path is not None and roles.get("binary") and (REPO / roles["binary"]).resolve() != Path(bin_path).resolve():
        p.append("the binary being run is not the frozen binary")
    if code_files is not None:
        missing = sorted(set(code_files) - set(roles.get("code") or []))
        if missing:
            p.append(f"loaded code not bound by the freeze: {missing}")
    if required_kind in PARENT_KIND:
        par = rec.get("parent") or {}
        if not par.get("path") or not par.get("sha256"):
            p.append(f"{required_kind} needs a {PARENT_KIND[required_kind]} parent")
        elif sha256_file(REPO / par["path"]) != par["sha256"]:
            p.append("parent record hash mismatch")
        elif _depth > 3:
            p.append("parent chain too deep")
        else:
            p += [f"parent: {x}" for x in verify_freeze(REPO / par["path"], PARENT_KIND[required_kind], _depth=_depth + 1)]
            try:
                prec = json.loads((REPO / par["path"]).read_text())
            except (OSError, json.JSONDecodeError):
                p.append("parent unreadable")
            else:
                pf, pr = prec.get("files") or {}, prec.get("roles") or {}

                def hashes(rl, fl):
                    v = rl if isinstance(rl, list) else ([rl] if rl else [])
                    return sorted((x, (fl.get(x) or {}).get("sha256")) for x in v)
                for role in SHARED_ROLES:
                    if hashes(roles.get(role), files) != hashes(pr.get(role), pf):
                        p.append(f"shared role {role} differs from the parent")
    return p


def scope_problems(expected: dict | None, observed: dict) -> list[str]:
    """Scope = GPU model, driver, pair link, UUIDs (rank order), NUMA nodes, power limit and max graphics clock."""
    if not isinstance(expected, dict) or any(k not in expected for k in SCOPE_KEYS):
        return [f"SCOPE_EXPECTED_INCOMPLETE: needs {SCOPE_KEYS}"]
    return [f"SCOPE_MISMATCH {k}: expected {expected[k]!r}, observed {observed.get(k)!r}"
            for k in SCOPE_KEYS if expected[k] != observed.get(k)]


def check_config(cfg: dict, now: datetime | None = None, check_freeze: bool = True, config_path: Path | None = None,
                 bin_path: Path | None = None) -> list[str]:
    """All reasons to refuse this config. Pure (no NVML, no GPU)."""
    p = []
    now = now or datetime.now(timezone.utc)
    if cfg.get("implementation_id") != IMPLEMENTATION_ID:
        p.append(f"implementation_id must be {IMPLEMENTATION_ID} (historical and superseded configs are read-only)")
    if cfg.get("method_id") != METHOD_ID:
        p.append(f"method_id must be {METHOD_ID}")
    mode = MODES.get(cfg.get("mode"))
    if mode is None:
        p.append("mode must be STAGE2_CALIBRATION, STAGE2_HELDOUT or STAGE2_SMOKE")
    if cfg.get("output_root") != OUTPUT_ROOT:
        p.append(f"output_root must be {OUTPUT_ROOT}")
    camp = cfg.get("campaign")
    if not isinstance(camp, str) or not SAFE_ID.match(camp) or ".." in camp:
        p.append("campaign must match [A-Za-z0-9][A-Za-z0-9._-]{0,63} without '..'")
    if cfg.get("energy_source", "counter") != "counter":
        p.append("Stage 2 uses the counter energy source; sampled mode stays disabled")
    if cfg.get("max_attempts") != 1:
        p.append("max_attempts must be 1")
    if any(k in cfg for k in ("timing", "waits", "retry", "retries")):
        p.append("timing and retries are protocol constants, not config")
    alloc = cfg.get("allocation")
    if not isinstance(alloc, list) or len(alloc) != 2 or len(set(alloc)) != 2 or \
            not all(isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in alloc):
        p.append("allocation must be exactly two distinct GPU indices (confirmed by the user)")
        alloc = None
    pl = cfg.get("placements")
    if not isinstance(pl, list) or len(pl) != 1 or (alloc and list(pl[0].get("gpus", [])) != list(alloc)):
        p.append("exactly one placement using exactly the two allocated GPUs, in allocation order")
    if pl and isinstance(pl, list) and pl[0].get("env"):
        p.append("placement env overrides are not allowed in Stage 2")
    if cfg.get("env_extra"):
        p.append("env_extra is not allowed in Stage 2")
    if os.environ.get("LD_PRELOAD"):
        p.append("LD_PRELOAD must be unset or empty in the controller environment")
    conf = cfg.get("allocation_confirmation")
    if not isinstance(conf, dict):
        p.append("allocation_confirmation missing: an idle GPU is not an allocation")
    else:
        if conf.get("confirmed_by") != "user":
            p.append("allocation_confirmation.confirmed_by must be 'user'")
        if alloc and sorted(conf.get("gpus") or []) != sorted(alloc):
            p.append("allocation_confirmation names different GPUs than allocation")
        try:
            w0, w1 = _parse_utc(conf["window_start_utc"]), _parse_utc(conf["window_end_utc"])
            if not (w0 <= now < w1):
                p.append("now is outside the confirmed access window")
        except (KeyError, ValueError, TypeError) as e:
            p.append(f"allocation_confirmation window invalid: {e}")
    sc = cfg.get("scope_expected")
    if not isinstance(sc, dict) or any(k not in sc for k in SCOPE_KEYS):
        p.append(f"scope_expected must give {SCOPE_KEYS} (validated against the pair before any launch)")
    if cfg.get("collective") != COLL or cfg.get("size_bytes") != SIZE_BYTES:
        p.append(f"collective/size must be {COLL}/{SIZE_BYTES}")
    counts = cfg.get("counts") or {}
    for arm in ("A", "B", "C"):
        c = counts.get(arm) or {}
        if not isinstance(c.get("w"), int) or c["w"] <= 0:
            p.append(f"counts.{arm}.w must be a positive integer")
        for L in ("2.5", "10", "20", "40"):
            v = (c.get("R") or {}).get(L)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                p.append(f"counts.{arm}.R[{L}] must be a positive integer")
    units = cfg.get("units")
    if not isinstance(units, list) or not units:
        p.append("units (the frozen schedule for this run) missing")
        units = []
    p += unit_problems(mode, units)
    if cfg.get("mode") == "STAGE2_SMOKE":
        if len(units) > SMOKE_MAX_UNITS:
            p.append(f"smoke runs at most {SMOKE_MAX_UNITS} launches")
        if any(float(u.get("length_s", 0)) != 2.5 for u in units):
            p.append("smoke launches are 2.5 s")
        if not str(cfg.get("campaign", "")).endswith("-smoke"):
            p.append("smoke campaign names end in -smoke (never pooled with calibration)")
    if mode == "heldout":
        if cfg.get("m4d") or cfg.get("total_power_samples"):
            p.append("M4d and TOTAL_POWER samples are calibration-only")
        if not isinstance(cfg.get("session"), int) or cfg["session"] not in (1, 2, 3):
            p.append("held-out config must name session 1, 2 or 3")
        if not cfg.get("readiness_rules"):
            p.append("held-out requires readiness_rules (frozen at Freeze-H)")
    if check_freeze:
        fr = cfg.get("freeze_record")
        if not fr:
            p.append("freeze_record missing: nothing runs before the freeze")
        else:
            p += verify_freeze(REPO / fr, FREEZE_KIND.get(cfg.get("mode"), "Freeze-H"), config_path=config_path,
                               code_files=loaded_code_files(), bin_path=bin_path)
    return p


def unit_problems(mode: str | None, units: list[dict]) -> list[str]:
    """Schedule rules: unique safe ids, valid arm/length/pause; dose and M4d only in calibration-mode (S0b/C2/SMOKE)."""
    p = []
    ids = [u.get("launch_id") for u in units]
    if len(set(ids)) != len(ids):
        p.append("launch ids must be unique")
    for u in units:
        lid = u.get("launch_id")
        if not isinstance(lid, str) or not SAFE_ID.match(lid) or ".." in lid:
            p.append(f"{lid!r}: launch id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}")
        if u.get("arm") not in ARM_CONFIG or str(u.get("length_s")) not in ("2.5", "10.0", "20.0", "40.0", "10", "20", "40"):
            p.append(f"{lid}: arm/length invalid")
        pn = u.get("pause_ns")
        if not isinstance(pn, int) or isinstance(pn, bool) or not 0 <= pn < 100_000_000:
            p.append(f"{lid}: pause_ns must be an integer in [0, 1e8)")
        if mode == "heldout":
            if u.get("dose", "E0") != "E0" or u.get("dose_period_ms", 0) or u.get("post_probe") or u.get("poller_active"):
                p.append(f"{lid}: dose / post-probe are refused in held-out mode")
        elif mode == "calibration":
            if u.get("arm") != "B":
                p.append(f"{lid}: calibration uses arm B only")
            if u.get("dose") not in DOSES or u.get("dose_period_ms") != DOSE_PERIOD_MS.get(u.get("dose")):
                p.append(f"{lid}: dose invalid")
            if u.get("dose") != "E0" and not (u.get("part") in ("S0b", "SMOKE") or str(u.get("part", "")).startswith("C2")):
                p.append(f"{lid}: dose only in S0b, C2 and SMOKE")
            if not isinstance(u.get("post_probe"), bool):
                p.append(f"{lid}: post_probe must be true/false")
    return p


# ------------------------------------------------------------------ context

def pair_identity(snap: dict, alloc: list[int]) -> dict:
    by = {g["index"]: g for g in snap["gpus"]}
    links = (snap.get("topology") or {}).get("links") or {}
    a, b = alloc
    return {"gpus": [{"index": g, "uuid": by[g]["uuid"], "pci": by[g]["pci"], "name": by[g].get("name")} for g in alloc],
            "link": links.get(f"{a}-{b}") or links.get(f"{b}-{a}"),
            "numa": {str(g): ((snap.get("topology") or {}).get("numa") or {}).get(g) for g in alloc},
            "driver": snap.get("driver"), "nvml": snap.get("nvml"), "cuda_driver": snap.get("cuda_driver")}


def observed_scope(identity: dict, health: dict, alloc: list[int]) -> dict:
    by = {g["index"]: g for g in health.get("gpus", [])}
    names = sorted({str(g.get("name")) for g in identity["gpus"]})
    return {"gpu_name": names[0] if len(names) == 1 else names, "driver": identity.get("driver"),
            "link": identity.get("link"), "uuids": [g["uuid"] for g in identity["gpus"]], "numa": identity.get("numa"),
            "power_limit_mw": [by.get(g, {}).get("enforced_limit_mw") for g in alloc],
            "max_clk_gr_mhz": [by.get(g, {}).get("max_clk_gr_mhz") for g in alloc]}


def host_evidence() -> dict:
    """Evidence about the host, not a claim of exclusivity: an idle-looking host can still be shared."""
    try:
        aff = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = None
    try:
        nproc = sum(1 for x in os.listdir("/proc") if x.isdigit())
    except OSError:
        nproc = None
    return {"utc": utc_now(), "cpu_affinity": aff, "cpu_count": os.cpu_count(), "loadavg": list(os.getloadavg()),
            "n_host_processes": nproc, "note": "evidence only; idle state does not establish exclusive use"}


class Stage2Ctx:
    """Allocation, provenance and the Stage 2 binary. snapshot_fn and out_dir (under REPO/logs) are for CPU tests."""

    def __init__(self, cfg: dict, cfg_hash: str, snapshot_fn=None, bin_dir: Path | None = None,
                 out_dir: Path | None = None):
        self.cfg, self.cfg_hash = cfg, cfg_hash
        self.mode = MODES[cfg["mode"]]
        self.alloc = list(cfg["allocation"])
        self.gpus = list(cfg["placements"][0]["gpus"])
        if self.gpus != self.alloc:
            raise SystemExit("placement must list exactly the allocated GPUs in allocation order")
        camp = cfg["campaign"]
        if not SAFE_ID.match(camp) or ".." in camp:
            raise SystemExit("unsafe campaign identifier")
        root = (REPO / OUTPUT_ROOT).resolve()
        if out_dir is None:
            self.out = (REPO / OUTPUT_ROOT / camp).resolve()
            if root not in self.out.parents:
                raise SystemExit("output path escapes data/stage2")
        else:
            self.out = Path(out_dir).resolve()
            if (REPO / "logs").resolve() not in self.out.parents:
                raise SystemExit("test output must be under logs/")
        self.snap = (snapshot_fn or _snapshot)(self.alloc)
        self.by_index = {g["index"]: g for g in self.snap["gpus"]}
        self.bin_dir = Path(bin_dir) if bin_dir is not None else BIN_DIR_STAGE2
        self.identity = pair_identity(self.snap, self.alloc)
        self.versions = {"driver": self.snap.get("driver"), "nvml": self.snap.get("nvml"),
                         "cuda_driver": self.snap.get("cuda_driver"),
                         "nccl_lib": str(NCCL_LIB_LINK.resolve()) if NCCL_LIB_LINK.exists() else None,
                         "nccl_lib_sha256": sha256_file(NCCL_LIB_LINK),
                         "patch_v21_sha256": sha256_file(PATCH_V21), "patch_v25_sha256": sha256_file(PATCH_V25),
                         "plugin_sha256": sha256_file(R.PLUGIN),
                         "binary_sha256": sha256_file(self.bin_dir / f"{COLL}_perf"),
                         "code": {f: sha256_file(REPO / f) for f in loaded_code_files() if not f.startswith("OUTSIDE")}}

    def uuids(self, gpus):
        return [self.by_index[g]["uuid"] for g in gpus]

    def numa(self, gpus):
        nodes = {(self.snap.get("topology") or {}).get("numa", {}).get(g) for g in gpus}
        return nodes.pop() if len(nodes) == 1 and None not in nodes else None

    def check_allocation(self) -> list[str]:
        return allocation_problems(light_health(self.alloc), self.alloc, 16)

    def watch(self, gpus, sampler, wall_limit_s: float, temp_stop_c: float) -> dict:
        return {"gpus": list(gpus), "sampler": sampler, "temp_stop_c": temp_stop_c, "disk_floor_gb": 20,
                "unit_wall_limit_s": wall_limit_s}


# ------------------------------------------------------------------ traces (no counter reads)

def _foreign_at(proc_events, g, t, own: set[int]) -> bool | None:
    """True if a foreign process was listed on GPU g at the last poll at or before t, False if the list was known and
    free of foreign processes, None if unknown (no poll yet, or the query failed)."""
    ev = [e for e in proc_events if e[1] == g and e[0] <= t]
    if not ev or ev[-1][2] is None:
        return None
    return any(p not in own for p in ev[-1][2])


def trace_rows(s: Sampler, t0: float, own: set[int] = frozenset()) -> list[dict]:
    out = [{"t_s": r[0] - t0, "gpu": r[1], "temp_c": float(r[3]), "pstate": int(r[8]), "power_w": float(r[2]),
            "clock_events": int(r[9]), "foreign": _foreign_at(s.proc_events, r[1], r[0], own)} for r in list(s.rows)]
    out += [{"t_s": t - t0, "gpu": g, "error": True, "foreign": None} for t, g, _ in list(s.errors)]
    return sorted(out, key=lambda r: r["t_s"])


def _stop_sampler(s: Sampler, path: Path | None) -> dict:
    s.stop()
    joined = not s._t.is_alive()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        s.write(path)
    return {"joined": joined, "rows": len(s.rows), "errors": len(s.errors)}


def fixed_wait(gpus, seconds: float, d: Path, timing: Timing) -> dict:
    """Calibration: a fixed wait with a 1 Hz trace (temperature, P-state, processes, clock events). A foreign or
    unknown process state after the first poll or the temperature stop ends the session; otherwise the wait always
    runs to its fixed length. The full trace is kept for the readiness derivation."""
    s = Sampler(gpus, period_s=timing.readiness_period_s, proc_period_s=timing.readiness_period_s).start()
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            time.sleep(min(0.5, seconds))
            rows = trace_rows(s, t0)
            settled = [r for r in rows if r["t_s"] > 1.5 * timing.readiness_period_s and not r.get("error")]
            if any(r["foreign"] is not False for r in settled):
                raise Stage2Stop("FOREIGN_OR_UNKNOWN_PROCESS_DURING_WAIT")
            if any(r["temp_c"] >= timing.temp_stop_c for r in rows if not r.get("error")):
                raise Stage2Stop("TEMPERATURE_STOP_DURING_WAIT")
    finally:
        info = _stop_sampler(s, d / "wait_trace.csv.gz")
    rows = [r for r in trace_rows(s, t0) if not r.get("error")]
    with open(d / "wait_trace_rows.json", "w") as f:
        json.dump(rows, f)
    return {"kind": "fixed", "wait_s": seconds, "n_rows": len(rows), "sampler": info,
            "clock_event_bit_shares": clock_event_bit_shares(rows, gpus),
            "trace_rows_file": str((d / "wait_trace_rows.json").resolve().relative_to(REPO.resolve())),
            "t_launch_c": {str(g): [r["temp_c"] for r in rows if r["gpu"] == g][-1] for g in gpus}}


def launch_gate_live(gpus, t_launch_c: float, d: Path, timing: Timing, deadline: Deadline) -> dict:
    """Held-out R1, decided at each 1 Hz tick at the CURRENT time from the explicitly allocated GPUs; W_max expiry
    (or the deadline) is never readiness."""
    s = Sampler(gpus, period_s=timing.readiness_period_s, proc_period_s=timing.readiness_period_s).start()
    t0 = time.perf_counter()
    res = {"status": "NOT_ACHIEVED", "t_ready_s": None, "reason": "READINESS_NOT_ACHIEVED_WITHIN_W_MAX"}
    try:
        while time.perf_counter() - t0 <= timing.ho_w_max_s and deadline.remaining_s() > session_tail_reserved_s(Design()):
            time.sleep(timing.readiness_period_s)
            now = time.perf_counter() - t0
            r = gate_at(trace_rows(s, t0), gpus, now, t_launch_c, timing.stable_window_s, timing.stable_delta_c,
                        max_gap_s=2 * timing.readiness_period_s, max_age_s=2 * timing.readiness_period_s)
            res["last_reasons"] = r["reasons"]
            if r["status"] == "READY":
                res = {"status": "READY", "t_ready_s": now}
                break
    finally:
        res["sampler"] = _stop_sampler(s, d / "wait_trace.csv.gz")
    res["clock_event_bit_shares"] = clock_event_bit_shares(trace_rows(s, t0), gpus)
    res["wait_s"] = time.perf_counter() - t0
    return res


def _counter_read(handles) -> list[tuple]:
    """The only cumulative-counter call site of the Stage 2 controller (session baselines only, after validation)."""
    import pynvml as N
    out = []
    for h in handles:
        ta = time.perf_counter_ns()
        try:
            v, rc = int(N.nvmlDeviceGetTotalEnergyConsumption(h)), 0
        except N.NVMLError as e:
            v, rc = None, int(getattr(e, "value", -1))
        out.append((v, rc, ta, time.perf_counter_ns()))
    return out


def _read_evidence(gpus, reads) -> list[list]:
    """[nvml_index, value_mJ, status, t_call_start_ns, t_call_end_ns] per GPU from _counter_read output (no NVML)."""
    return [[g, v, rc, ta, tb] for g, (v, rc, ta, tb) in zip(gpus, reads)]


def session_baseline(ctx: Stage2Ctx, d: Path, timing: Timing, deadline: Deadline) -> dict:
    """Session P8 idle/context baseline, fail-closed and gated: P8 entry (all allocated GPUs: P8, process list known
    and free of foreign processes, clock events 0, finite temperature) within the cap; the interval from entry must
    stay valid through the settle, the first counter read, the measurement and the second counter read, checked
    continuously; each counter read happens only after a validity check of the interval so far. Any violation ends
    the baseline with no further counter read. The deadline is checked between phases."""
    import pynvml as N
    gpus = ctx.alloc
    hs = [N.nvmlDeviceGetHandleByIndex(g) for g in gpus]
    gap = 4 * timing.sampler_period_s
    out = {"status": "FAILED", "counter_reads": 0, "host": host_evidence()}
    d.mkdir(parents=True, exist_ok=True)
    s = Sampler(gpus, period_s=timing.sampler_period_s, proc_period_s=timing.baseline_proc_period_s).start()
    t0 = time.perf_counter()
    # raw evidence of the two baseline reads (provenance only: filled from the existing reads, no extra NVML call)
    ev = out["counter_read_evidence"] = {
        "gpus": [{"nvml_index": g, "uuid": (ctx.by_index.get(g) or {}).get("uuid"),
                  "pci": (ctx.by_index.get(g) or {}).get("pci")} for g in gpus],
        "clock_domain": {"source": "time.perf_counter_ns", "info": repr(time.get_clock_info("perf_counter")),
                         "t0_perf_counter_ns": int(t0 * 1e9),
                         "note": "same clock as the baseline trace times (time.perf_counter); read-call bounds, not "
                                 "acquisition times"},
        "units": {"value": "mJ (nvmlDeviceGetTotalEnergyConsumption)", "time": "ns", "status": "NVML return code"},
        "columns": ["nvml_index", "value_mJ", "status", "t_call_start_ns", "t_call_end_ns"],
        "e0": None, "e1": None}

    def elapsed():
        return time.perf_counter() - t0

    def check(t_from):
        if deadline.remaining_s() < session_tail_reserved_s(Design()):
            return ["BUDGET_EXHAUSTED"]
        return baseline_problems(trace_rows(s, t0), gpus, t_from, elapsed() - timing.sampler_period_s, gap)
    try:
        entry = None
        while entry is None:
            if elapsed() > timing.p8_wait_cap_s or deadline.remaining_s() < session_tail_reserved_s(Design()):
                rows = trace_rows(s, t0)
                last = {g: [r for r in rows if r.get("gpu") == g][-1:] for g in gpus}
                why = sorted({f"{x}_GPU{g}" for g, rr in last.items() for r in rr for x in _row_problems(r)}
                             | {f"NO_ROWS_GPU{g}" for g, rr in last.items() if not rr})
                out["reason"] = "NO_VALID_P8_ENTRY_WITHIN_CAP: " + ",".join(why)
                return out
            time.sleep(0.1)
            entry = p8_entry(trace_rows(s, t0), gpus, timing.p8_wait_cap_s)
        out["entry_s"] = entry
        while elapsed() < entry + timing.baseline_settle_s:
            time.sleep(0.1)
            why = check(entry) if elapsed() - entry > 2 * gap else []
            if why:
                out["reason"] = "INVALID_DURING_SETTLE: " + ",".join(why)
                return out
        why = check(entry)
        if why:
            out["reason"] = "INVALID_BEFORE_FIRST_READ: " + ",".join(why)
            return out
        e0 = _counter_read(hs)
        out["counter_reads"] += len(hs)
        t_e0 = elapsed()
        ev["e0"] = _read_evidence(gpus, e0)        # stored at once: kept even if the baseline fails later
        while elapsed() < t_e0 + timing.baseline_measure_s:
            time.sleep(0.1)
            why = check(entry)
            if why:
                out["reason"] = "INVALID_DURING_MEASUREMENT (no second read): " + ",".join(why)
                return out
        why = check(entry)
        if why:
            out["reason"] = "INVALID_BEFORE_SECOND_READ: " + ",".join(why)
            return out
        e1 = _counter_read(hs)
        out["counter_reads"] += len(hs)
        t_e1 = elapsed()
        ev["e1"] = _read_evidence(gpus, e1)
        time.sleep(0.3)
        why = check(entry)
        if why:
            out["reason"] = "INVALID_AFTER_SECOND_READ: " + ",".join(why)
            return out
        rows = trace_rows(s, t0)
        per = {}
        for g, a, b in zip(gpus, e0, e1):
            ok = a[1] == 0 and b[1] == 0 and a[0] is not None and b[0] is not None and b[0] >= a[0]
            pw = [r["power_w"] for r in rows if r["gpu"] == g and t_e0 <= r["t_s"] <= t_e1 and not r.get("error")]
            per[str(g)] = {"counter_W": (b[0] - a[0]) / 1000 / ((b[2] + b[3] - a[2] - a[3]) / 2e9) if ok else None,
                           "counter_status": [a[1], b[1]], "sampled_mean_W": float(np.mean(pw)) if pw else None}
        out["per_gpu"] = per
        if any(v["counter_W"] is None for v in per.values()):
            out["reason"] = "COUNTER_READ_FAILED"
            return out
        out.update(status="OK", measure_s=t_e1 - t_e0)
        return out
    finally:
        out["sampler"] = _stop_sampler(s, d / "baseline_trace.csv.gz")
        out["clock_event_bit_shares"] = clock_event_bit_shares(trace_rows(s, t0), gpus)
        out["host_end"] = host_evidence()
        (d / "baseline.json").write_text(json.dumps(out, indent=1, default=str))


# ------------------------------------------------------------------ unit

def stage2_env(ctx: Stage2Ctx, unit: dict, d: Path) -> dict:
    env = R.build_env(ctx.uuids(ctx.gpus), ARM_CONFIG[unit["arm"]], d, energy_source="counter")
    if any(k in env for k in STAGE2_ENV):
        raise Stage2Stop("inherited Stage 2 variables in the environment")
    if env.get("LD_PRELOAD"):
        raise Stage2Stop("LD_PRELOAD is set in the launch environment; it must be unset or empty")
    env["NCCLE_STAGE2_MODE"] = ctx.mode
    env["NCCLE_S2_FILE"] = str(d / "s2.jsonl")
    env["NCCLE_START_PAUSE_NS"] = str(int(unit["pause_ns"]))
    if ctx.mode == "calibration":
        if unit.get("dose_period_ms"):
            env["NCCLE_DOSE_PERIOD_MS"] = str(int(unit["dose_period_ms"]))
        env["NCCLE_POST_PROBE"] = "1" if unit.get("post_probe") else "0"
    elif unit.get("dose_period_ms") or unit.get("post_probe"):
        raise Stage2Stop("dose / post-probe refused in held-out mode")
    return env


def pre_region_state(arr: np.ndarray, gpus, region: dict) -> dict:
    """State immediately before C0 per allocated GPU from the unit's 50 ms trace: mean temperature over
    [C0 - 250 ms, C0), last graphics clock and P-state, OR of clock events in the window, age of the last finite
    POWER_INSTANT row. Missing GPUs stay missing (and fail R2)."""
    out = {}
    for p in (region or {}).get("per_gpu", []):
        g = gpus[p["rank"]]
        t_c0 = p["t_start_ns"] * 1e-9
        sel = arr[(arr[:, 1] == g) & (arr[:, 0] < t_c0)]
        win = sel[sel[:, 0] >= t_c0 - 0.25]
        inst = sel[np.isfinite(sel[:, 13])]
        out[g] = {"t_region_c": float(win[:, 3].mean()) if len(win) else None,
                  "clk_mhz": float(sel[-1, 4]) if len(sel) else None, "pstate": int(sel[-1, 8]) if len(sel) else None,
                  "clock_events": int(np.bitwise_or.reduce(win[:, 9].astype(np.int64))) if len(win) else None,
                  "inst_age_s": float(t_c0 - inst[-1, 0]) if len(inst) else None, "n_rows": int(len(win))}
    return out


def smoke_s5(per: list[dict], unit: dict, timing: Timing) -> tuple[dict, list[str]]:
    """S5 (smoke-1 protocol), per GPU, under the L3 dose rule.
    Active poller (dose_period_ms > 0): delivered / eligible >= 95 % (unchanged threshold); work_end <= stop_request
    <= join_end with all three present; no dose read starts before the end of C0. Straddling and shutdown reads
    (including a read in flight at the stop request) are reported separately, never delivered and never a failure.
    Inactive poller (E0): no dose reads and no poller times; missing stop/join times are expected, not a failure."""
    active = int(unit.get("dose_period_ms") or 0) > 0
    out, why = {"poller": "active" if active else "inactive", "per_gpu": []}, []
    for p in per:
        d, r = p.get("dose") or {}, p["rank"]
        row = {"rank": r}
        if active:
            fr = d.get("delivered_frac")
            row.update({k: d.get(k) for k in ("eligible", "eligible_completed", "delivered", "delivered_frac",
                                              "straddling", "shutdown_reads", "inflight_at_stop", "order")})
            if d.get("poller") != "active" or "order" not in d:
                why.append(f"S5_NO_DOSE_SUMMARY_RANK{r}")
            else:
                if not (d["order"] or {}).get("ok"):
                    why.append(f"S5_ORDER_WORK_END_STOP_JOIN_RANK{r}")
                if fr is None or fr < timing.dose_delivery_min:
                    why.append(f"S5_DELIVERY_BELOW_95PCT_RANK{r}")
                if "DOSE_READ_BEFORE_BASE" in (d.get("flags") or []):
                    why.append(f"S5_READ_BEFORE_C0_RANK{r}")
        else:
            row.update(n_reads=d.get("n_reads"))
            if not d:
                why.append(f"S5_NO_DOSE_SUMMARY_RANK{r}")
            elif d.get("poller") != "inactive" or d.get("n_reads"):
                why.append(f"S5_POLLER_ACTIVE_WITHOUT_DOSE_RANK{r}")
            if "DOSE_POLLER_TIMES_WITHOUT_DOSE_REQUEST" in (p.get("flags") or []):
                why.append(f"S5_POLLER_TIMES_WITHOUT_DOSE_RANK{r}")
        out["per_gpu"].append(row)
    out["problems"] = why
    return out, why


def smoke_problems(rec: dict, timing: Timing, unit: dict) -> list[str]:
    """Smoke criteria beyond unit acceptance: S4 extras (actual vs requested pause, post-probe brackets, span and
    status) and S5 (dose rule, stop/join order; smoke_s5). The per-criterion detail is stored in rec["smoke_checks"].
    An automatic PASS does not replace inspection of every launch."""
    why = []
    per = (rec.get("s2") or {}).get("per_gpu", [])
    if not per:
        rec["smoke_checks"] = {"S4_extra": ["NO_STAGE2_RECORD"], "S5": None}
        return ["NO_STAGE2_RECORD"]
    if any(p.get("pause_dev_ms") is None or abs(p["pause_dev_ms"]) > timing.pause_tol_ms for p in per):
        why.append("PAUSE_DEVIATION_GT_2MS")
    if unit.get("post_probe"):
        for p in per:
            pp = p.get("post_probe") or {}
            if len(pp.get("brackets") or []) < timing.post_probe_min_brackets:
                why.append(f"POST_PROBE_FEW_BRACKETS_RANK{p['rank']}")
            if pp.get("span_s") is None or pp["span_s"] > timing.post_probe_max_s:
                why.append(f"POST_PROBE_TOO_LONG_RANK{p['rank']}")
            if not pp.get("statuses_ok"):
                why.append(f"POST_PROBE_STATUS_RANK{p['rank']}")
    s5, s5_why = smoke_s5(per, unit, timing)
    rec["smoke_checks"] = {"S4_extra": list(why), "S5": s5}
    return why + s5_why


def acceptance(rec: dict, timing: Timing, mode: str, unit: dict, smoke: bool = False) -> list[str]:
    """Pre-specified unit acceptance; returns the reasons a unit is NOT accepted."""
    why = []
    if rec.get("qc_status") != "OK":
        why.append(f"QC_{rec.get('qc_status')}")
    ev = rec.get("evidence") or {}
    if ev.get("status") != "VERIFIED":
        why.append("EVIDENCE_NOT_VERIFIED")
    exp = ARM_EXPECTED[unit["arm"]]
    act = {(x["algo"], x["proto"]) for x in (ev.get("actual") or [])}
    if exp is not None and act != {exp}:
        why.append("EXECUTION_NOT_AS_FORCED")
    if len(act) != 1:
        why.append("NOT_A_SINGLE_EXECUTION")
    if ev.get("n_matches_expected") is not True:
        why.append("OP_COUNT_NOT_R_PLUS_W_PLUS_1")
    reg = rec.get("region") or {}
    T = reg.get("T_region_s")
    if T is None or abs(T / float(unit["length_s"]) - 1) > timing.length_tol:
        why.append("REGION_LENGTH_OFF_TARGET")
    s2 = rec.get("s2") or {}
    if not s2:
        why.append("NO_STAGE2_RECORD")
    else:
        lo, hi = timing.prep_target_s * (1 - timing.prep_tol), timing.prep_target_s * (1 + timing.prep_tol)
        if s2.get("prep_duration_s_min") is None or not (lo <= s2["prep_duration_s_min"] and s2["prep_duration_s_max"] <= hi):
            why.append("PREP_DURATION_OFF_TARGET")
        hard = [f for f in s2.get("flags", []) if not f.startswith("E_PREP_TO_C0_UNAVAILABLE")]
        if hard:
            why.append("STAGE2_RECORD_FLAGS:" + ",".join(hard))
        if unit.get("dose_period_ms"):
            fr = [(p.get("dose") or {}).get("delivered_frac") for p in s2.get("per_gpu", [])]
            if not fr or any(f is None or f < timing.dose_delivery_min for f in fr):
                why.append("DOSE_DELIVERY_BELOW_95PCT")
    if rec.get("telemetry_problems"):
        why.append("TELEMETRY_PROBLEMS")
    if rec.get("required_channel_problems"):
        why.append("REQUIRED_CHANNELS:" + ",".join(rec["required_channel_problems"]))
    if mode == "heldout":
        if (rec.get("readiness") or {}).get("status") != "READY":
            why.append("R1_NOT_PASSED")
        if not (rec.get("pre_region_check") or {}).get("ok"):
            why.append("PRE_REGION_STATE_OUT_OF_BAND")
    if smoke:
        why += smoke_problems(rec, timing, unit)
    return why


def run_stage2_unit(ctx: Stage2Ctx, sdir: Path, unit: dict, readiness: dict, deadline: Deadline, design: Design,
                    timing: Timing) -> dict:
    lid = unit["launch_id"]
    d = (sdir / "units" / lid).resolve()
    if sdir.resolve() not in d.parents:
        raise Stage2Stop("unit path escapes the session directory")
    if d.exists():
        raise Stage2Stop(f"{d} exists; units are never rerun")
    arm, L = unit["arm"], float(unit["length_s"])
    Rn = int(ctx.cfg["counts"][arm]["R"][f"{L:g}"])
    w = int(ctx.cfg["counts"][arm]["w"])
    rec = {"launch_id": lid, "unit_id": lid, "attempt": 0, "implementation_id": IMPLEMENTATION_ID, "method_id": METHOD_ID,
           "campaign": ctx.cfg["campaign"], "mode": ctx.cfg["mode"], "unit": unit, "R": Rn, "w": w,
           "readiness": readiness, "gpus": list(ctx.gpus), "dir": str(d.relative_to(REPO.resolve())),
           "host": host_evidence(), "qc_status": "NOT_STARTED"}
    probs = ctx.check_allocation()
    if probs:
        rec.update(qc_status="FOREIGN_PROCESS", qc_reasons=probs, stop_session=True, accepted=False, unsafe=True)
        return rec
    env = stage2_env(ctx, unit, d)          # refusals happen before anything is created
    d.mkdir(parents=True)
    cmd = R.build_cmd(COLL, 2, SIZE_BYTES, SIZE_BYTES, Rn, w, 1, ctx.numa(ctx.gpus), bin_dir=ctx.bin_dir)
    wall = deadline.unit_wall_limit_s(L, design)
    sampler = Sampler(ctx.gpus, period_s=timing.sampler_period_s, schema_v3=True,
                      total_power_samples=(ctx.mode == "calibration" and bool(ctx.cfg.get("total_power_samples")))).start()
    proc, err0 = None, 0
    try:
        time.sleep(0.2)
        err0 = len(sampler.errors)
        rec["qc_status"] = "RUNNING"
        proc = R.run_process(cmd, env, d, wall, watch=ctx.watch(ctx.gpus, sampler, wall, timing.temp_stop_c))
        time.sleep(0.4)
    except BaseException as e:
        rec.update(qc_status="INTERRUPTED_UNIT", exception=f"{type(e).__name__}: {e}", stop_session=True, accepted=False,
                   unsafe=True, proc=proc)
        raise
    finally:
        rec["sampler_stop"] = _stop_sampler(sampler, d / "telemetry.csv.gz")
        if sampler.total_power_samples:
            sampler.write_total_power_samples(d / "total_power_samples.csv.gz")
        rec["optional_channels"] = {"total_power_samples": {"requested": bool(sampler.total_power_samples),
                                                            "rows": len(sampler.tps_rows), "errors": len(sampler.tps_errors)}}
        if rec["qc_status"] == "INTERRUPTED_UNIT":
            (d / "unit.json").write_text(redact(json.dumps(rec, indent=1, default=str)))
    regs, rflags = parse_region_text((d / "region.jsonl").read_text() if (d / "region.jsonl").exists() else "")
    s2recs, s2flags = parse_s2_text((d / "s2.jsonl").read_text() if (d / "s2.jsonl").exists() else "")
    metrics = None
    pci = [ctx.by_index[g]["pci"] for g in ctx.gpus]
    if len(regs) == 1:
        metrics = region_metrics(regs[0], expected_pci=pci, expected_R=Rn)
    elif len(regs) > 1:
        rflags.append("MULTIPLE_REGIONS")
    s2 = s2_metrics(regs[0], s2recs[0]) if len(regs) == 1 and len(s2recs) == 1 else None
    if len(s2recs) != 1:
        s2flags.append(f"S2_RECORDS_{len(s2recs)}")
    files, pflags = load_profiler_dir(d / "prof")
    algo, proto = R.parse_cfg_label(ARM_CONFIG[arm])
    ev = verify(files, COLL_FUNC[COLL], regs[0]["count"] if regs else -1, 2, algo, proto, min_expected=Rn,
                exact_expected=Rn + w + 1) if regs else {"status": "UNVERIFIED", "reasons": ["NO_REGION"]}
    arr = sampler.array()
    tel, windows = {}, {}
    clock_or, temp_max = 0, None
    if metrics:
        for p in metrics["per_gpu"]:
            g = ctx.gpus[p["rank"]]
            sp = next((x for x in (s2 or {}).get("per_gpu", []) if x["rank"] == p["rank"]), {})
            t_end = (sp.get("t_work_end_ns") or p["t_end_ns"]) * 1e-9
            windows[g] = (p["t_start_ns"] * 1e-9 - 0.25, t_end)
            sm = window_summary(arr, g, p["t_start_ns"] * 1e-9, t_end, sampler.period_s,
                                lag_s=ctx.cfg.get("power_lag_s", 0.75))
            tel[str(g)] = sm
            clock_or |= sm.get("clock_events_or", 0)
            temp_max = max(temp_max or -1, sm.get("temp_max_c", -1))
    foreign = sorted({p for (_, _, pids) in sampler.proc_events if pids is not None for p in pids if p != proc["pid"]})
    plist_unknown = sorted({g for (_, g, pids) in sampler.proc_events if pids is None})
    tprob = telemetry_problems({"errors": len(sampler.errors) - err0, "truncated": sampler.truncated},
                               tel if metrics else None, 2, "counter")
    rows_out = R.parse_stdout_rows((d / "stdout.txt").read_text(errors="replace"))
    hits = R.error_hits(d)
    qc = classify({"exit_code": proc["rc"], "timed_out": proc["timed_out"], "region": metrics,
                   "wrong": rows_out[0]["wrong"] if len(rows_out) == 1 else None, "region_parse_flags": rflags + s2flags,
                   "telemetry_problems": tprob, "evidence_status": ev.get("status"), "foreign_pids": foreign,
                   "clock_events_or": clock_or & ~0x1, "temp_max_c": temp_max, "power_frac_max": None,
                   "temp_stop_c": timing.temp_stop_c}, 2.0, energy_source="counter")
    safety = R.safety_problem(proc)
    stop = bool(qc.get("stop_block")) or bool(proc["timed_out"]) or bool(hits) or bool(safety) or bool(plist_unknown)
    if safety and not proc["timed_out"]:
        qc = {"qc_status": "SAFETY_STOP", "reasons": [safety]}
    if hits and qc["qc_status"] == "OK":
        qc = {"qc_status": "EXECUTION_ERROR", "reasons": ["ERROR_IN_LOGS"]}
    unsafe = bool(foreign) or bool(safety) or bool(plist_unknown) or qc["qc_status"] in ("FOREIGN_PROCESS", "SAFETY_STOP") \
        or not (proc.get("cleanup") or {}).get("ok", False)
    rec.update(proc=proc, region=metrics, region_flags=rflags, s2=s2, s2_flags=s2flags, evidence=ev,
               profiler_flags=pflags, telemetry=tel, telemetry_problems=tprob, foreign_pids=foreign,
               process_list_unknown_gpus=plist_unknown, error_hits=hits,
               required_channel_problems=required_channel_problems(list(sampler.rows), ctx.gpus, windows, sampler.period_s)
               if metrics else ["NO_REGION"],
               qc_status=qc["qc_status"], qc_reasons=qc.get("reasons", []), qc_method=QC_METHOD,
               sampler={"overruns": sampler.overruns, "errors": len(sampler.errors), "n": len(sampler.rows),
                        "schema": "nccle-telemetry-v3"},
               stop_session=stop, unsafe=unsafe, wall_limit_s=wall,
               nccl_env={k: v for k, v in env.items() if k.startswith(("NCCL_", "NCCLE_", "CUDA_"))},
               loader_env={"LD_PRELOAD": env.get("LD_PRELOAD", ""), "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", "")})
    rec["pre_region_state"] = pre_region_state(arr, ctx.gpus, metrics)
    if ctx.mode == "heldout":
        bands = {int(k): v for k, v in ctx.cfg["readiness_rules"]["bands"].items()}
        rec["pre_region_check"] = pre_region_check(rec["pre_region_state"], bands, ctx.gpus)
    rec["not_accepted_reasons"] = acceptance(rec, timing, ctx.mode, unit, smoke=ctx.cfg["mode"] == "STAGE2_SMOKE")
    rec["accepted"] = not rec["not_accepted_reasons"]
    (d / "unit.json").write_text(redact(json.dumps(rec, indent=1, default=str)))
    return rec


# ------------------------------------------------------------------ session

def _unsafe_reason(reason: str) -> bool:
    return any(p in str(reason) for p in UNSAFE_STOP_PREFIXES)


def loader_evidence(bin_path: Path) -> dict:
    """CPU only: LD_PRELOAD of the controller and the dynamic-loader resolution (ldd) of the benchmark binary under
    the Stage 2 library path (the same LD_LIBRARY_PATH prefix every launch gets). Nothing is executed on a GPU."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(R.NCCL_LIB) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    try:
        r = subprocess.run(["ldd", str(bin_path)], capture_output=True, text=True, timeout=30, env=env)
        ldd = r.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired) as e:
        ldd = [f"ldd failed: {e!r}"]
    return {"LD_PRELOAD": os.environ.get("LD_PRELOAD", ""), "ld_library_path_prefix": str(R.NCCL_LIB),
            "binary": str(bin_path), "ldd": ldd, "libnccl_lines": [x for x in ldd if "libnccl" in x]}


def run_session(ctx: Stage2Ctx, units: list[dict], deadline: Deadline, timing: Timing = Timing(),
                label: str | None = None, deadline_info: dict | None = None) -> dict:
    """One smoke run, calibration campaign or held-out session, in schedule order, max_attempts 1.
    execution_status: COMPLETE (every scheduled unit executed), STOPPED (a stop rule or gate), INTERRUPTED,
    STOPPED_SAFETY or EXCEPTION. Scientific acceptance is reported separately (n_accepted, not_accepted; smoke_verdict
    for smoke). COMPLETE does not mean accepted."""
    held = ctx.mode == "heldout"
    smoke = ctx.cfg["mode"] == "STAGE2_SMOKE"
    design = Design() if held else calibration_design(m4d=any(u.get("post_probe") for u in units))
    probs = unit_problems(ctx.mode, units)
    if probs:
        raise SystemExit("REFUSED schedule:\n  " + "\n  ".join(probs))
    default = "smoke" if smoke else ("calibration" if not held else f"session_{ctx.cfg['session']}")
    sdir = (ctx.out / (label or default)).resolve()
    if ctx.out not in sdir.parents:
        raise SystemExit("session path escapes the campaign directory")
    if sdir.exists():
        raise SystemExit(f"{sdir} exists; a session runs once (no replacement sessions without an amendment)")
    sdir.mkdir(parents=True)
    log = AttemptLog(sdir / "attempts.jsonl")
    summary = {"execution_status": "RUNNING", "utc_start": utc_now(), "implementation_id": IMPLEMENTATION_ID,
               "method_id": METHOD_ID, "mode": ctx.cfg["mode"], "config_sha256": ctx.cfg_hash, "versions": ctx.versions,
               "identity": ctx.identity, "n_planned": len(units), "remaining_at_start_s": deadline.remaining_s(),
               "deadline": deadline_info, "host_start": host_evidence(),
               "loader": loader_evidence(Path(ctx.bin_dir) / f"{COLL}_perf")}
    baseline_allowed, current = True, None
    xid = xid_event_test(ctx.alloc)
    es = xid.pop("event_set", None)
    summary["xid"] = xid
    try:
        h0 = light_health(ctx.alloc)
        scope = observed_scope(ctx.identity, h0, ctx.alloc)
        summary["scope_observed"] = scope
        sp = scope_problems(ctx.cfg.get("scope_expected"), scope)
        if sp:
            baseline_allowed = False
            raise Stage2Stop("SCOPE_MISMATCH: " + "; ".join(sp))
        probs = allocation_problems(h0, ctx.alloc, 16)
        if probs:
            baseline_allowed = False
            raise Stage2Stop(f"allocation busy at session start: {probs}")
        policy0 = {g["index"]: (g.get("enforced_limit_mw"), g.get("max_clk_gr_mhz"), g.get("app_clk_gr_mhz"))
                   for g in h0["gpus"]}
        if deadline.remaining_s() < end_baseline_bound_s(design) + session_tail_reserved_s(design):
            raise Stage2Stop("BUDGET_INSUFFICIENT_FOR_START_BASELINE")
        base = session_baseline(ctx, sdir / "baseline_start", timing, deadline)
        summary["baseline_start"] = {"status": base["status"], "reason": base.get("reason"),
                                     "counter_reads": base["counter_reads"]}
        if base["status"] != "OK":
            if any(x in str(base.get("reason")) for x in ("FOREIGN", "SAMPLER_ERROR", "UNKNOWN")):
                baseline_allowed = False
            raise Stage2Stop(f"SESSION_START_BASELINE_FAILED: {base.get('reason')}")
        for i, u in enumerate(units):
            if not deadline.can_start_unit(float(u["length_s"]), design, held):
                summary["incomplete"] = {"reason": "UNIT_START_BUDGET_GATE", "next_unit": u["launch_id"],
                                         "remaining_s": deadline.remaining_s(),
                                         "needed_s": unit_bound_s(float(u["length_s"]), design, held)}
                raise Stage2Stop("UNIT_START_BUDGET_GATE")
            wd = sdir / "waits" / u["launch_id"]
            if held:
                rd = launch_gate_live(ctx.gpus, float(ctx.cfg["readiness_rules"]["t_launch_c"]), wd, timing, deadline)
                if rd["status"] != "READY":
                    log.append({"unit_id": u["launch_id"], "attempt": 0, "qc_status": "NOT_RUN_READINESS_NOT_ACHIEVED",
                                "readiness": rd})
                    if any("FOREIGN" in x or "SAMPLER_ERROR" in x for x in rd.get("last_reasons", [])):
                        baseline_allowed = False
                    raise Stage2Stop("READINESS_NOT_ACHIEVED")
            else:
                rd = fixed_wait(ctx.gpus, timing.cal_fixed_wait_s, wd, timing)
            current = {"unit_id": u["launch_id"], "attempt": 0, "qc_status": "INTERRUPTED_UNIT"}
            rec = run_stage2_unit(ctx, sdir, u, rd, deadline, design, timing)
            current = None
            log.append(rec)
            if rec.get("unsafe"):
                baseline_allowed = False
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(units)} {u['launch_id']}: {rec['qc_status']} "
                  f"accepted={rec.get('accepted')}", flush=True)
            if rec.get("stop_session"):
                raise Stage2Stop(f"stop condition at {u['launch_id']}: {rec['qc_status']} {rec.get('qc_reasons')}")
            if not held and u.get("part") in ("S0a", "SMOKE") and not rec.get("accepted"):
                raise Stage2Stop(f"{u.get('part')} not accepted: {rec.get('not_accepted_reasons')}")
            if es is not None:
                x = poll_xid(es)
                if x:
                    baseline_allowed = False
                    raise Stage2Stop(f"Xid events {x}")
            post = light_health(ctx.alloc)
            for g in post["gpus"]:
                if (g.get("enforced_limit_mw"), g.get("max_clk_gr_mhz"), g.get("app_clk_gr_mhz")) != policy0.get(g["index"]):
                    baseline_allowed = False
                    raise Stage2Stop(f"power limit / clock policy changed on GPU{g['index']}")
            if (post.get("host") or {}).get("disk_free_gb", 0) < 20:
                baseline_allowed = False
                raise Stage2Stop("disk below floor")
        summary["execution_status"] = "COMPLETE"
    except Stage2Stop as e:
        summary["execution_status"], summary["stop_reason"] = "STOPPED", str(e)
        if _unsafe_reason(str(e)):
            baseline_allowed = False
        print(f"SESSION STOPPED: {e}", flush=True)
    except (ControllerSignal, KeyboardInterrupt, SafetyStop) as e:
        summary["execution_status"] = "INTERRUPTED" if not isinstance(e, SafetyStop) else "STOPPED_SAFETY"
        summary["stop_reason"] = f"{type(e).__name__}: {e}"
        baseline_allowed = False
        raise
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised
        summary["execution_status"], summary["stop_reason"] = "EXCEPTION", f"{type(e).__name__}: {e}"
        baseline_allowed = False
        raise
    finally:
        if current is not None:                       # the attempted unit is never lost from the ledger
            try:
                log.append(dict(current, stop_reason=summary.get("stop_reason")))
            except Exception:  # noqa: BLE001
                pass
        recs = log.records()
        launched = any(isinstance(r.get("proc"), dict) or r.get("qc_status") == "INTERRUPTED_UNIT" for r in recs)
        try:
            if not launched:
                summary["baseline_end"] = {"status": "NOT_DONE", "reason": "no unit was launched"}
            elif not baseline_allowed:
                summary["baseline_end"] = {"status": "NOT_DONE", "reason": "contention, unknown ownership, safety, "
                                                                           "interruption or cleanup problem"}
            elif not deadline.can_run_end_baseline(design):
                summary["baseline_end"] = {"status": "NOT_DONE", "reason": "no time left for the end baseline"}
            elif ctx.check_allocation():
                summary["baseline_end"] = {"status": "NOT_DONE", "reason": "allocation not free at session end"}
            else:
                eb = session_baseline(ctx, sdir / "baseline_end", timing, deadline)
                summary["baseline_end"] = {"status": eb["status"], "reason": eb.get("reason"),
                                           "counter_reads": eb["counter_reads"]}
        except Exception as e:  # noqa: BLE001
            summary["baseline_end"] = {"status": "FAILED", "reason": repr(e)}
        try:
            post = light_health(ctx.alloc)
            ours = {r.get("proc", {}).get("pid") for r in recs if isinstance(r.get("proc"), dict)}
            left = [(g["index"], p.get("pid")) for g in post["gpus"] for p in g["processes"] if p.get("pid") in ours]
            summary["postflight"] = {"owned_processes_left_on_gpus": left, "allocation_problems":
                                     allocation_problems(post, ctx.alloc, 16)}
        except Exception as e:  # noqa: BLE001
            summary["postflight"] = {"error": repr(e)}
        attempted = [r for r in recs if r.get("qc_status") != "NOT_RUN_READINESS_NOT_ACHIEVED"]
        summary.update(utc_end=utc_now(), host_end=host_evidence(), n_attempts=len(recs),
                       n_accepted=sum(1 for r in recs if r.get("accepted")),
                       not_accepted={r["unit_id"]: r.get("not_accepted_reasons") or r.get("qc_status")
                                     for r in attempted if not r.get("accepted")},
                       remaining_at_end_s=deadline.remaining_s(), status=summary["execution_status"])
        if smoke:
            summary["smoke_verdict"] = "PASS" if (summary["execution_status"] == "COMPLETE"
                                                  and summary["n_accepted"] == len(units)
                                                  and (summary.get("baseline_start") or {}).get("status") == "OK"
                                                  and (summary.get("baseline_end") or {}).get("status") == "OK") else "FAIL"
        (sdir / "session_summary.json").write_text(redact(json.dumps(summary, indent=1, default=str)))
    return summary


def execution_budget(cfg: dict, start: datetime) -> dict:
    """Execution deadline: the reservation end; for a smoke run the earlier of the reservation end and the smoke start
    (= this command's start, after the user's allocation confirmation) + SMOKE_CEILING_S (30 min)."""
    end = _parse_utc(cfg["allocation_confirmation"]["window_end_utc"])
    window_s = (end - start).total_seconds()
    smoke = cfg.get("mode") == "STAGE2_SMOKE"
    budget_s, binding = window_s, "reservation end"
    if smoke and SMOKE_CEILING_S < window_s:
        budget_s, binding = SMOKE_CEILING_S, "smoke start + 30 min"
    return {"start_utc": start.isoformat(), "window_end_utc": str(cfg["allocation_confirmation"]["window_end_utc"]),
            "smoke_ceiling_s": SMOKE_CEILING_S if smoke else None, "budget_s": budget_s,
            "deadline_utc": (start + timedelta(seconds=budget_s)).isoformat(), "binding": binding}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ncclenergy.v2.stage2_runner")
    ap.add_argument("cmd", choices=["check", "smoke", "calibration", "heldout"])
    ap.add_argument("--campaign", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg, h = R.load_campaign(a.campaign)
    probs = check_config(cfg, config_path=a.campaign, bin_path=BIN_DIR_STAGE2 / f"{COLL}_perf")
    if a.cmd == "check":
        print(json.dumps({"config": str(a.campaign), "refusals": probs}, indent=1))
        raise SystemExit(1 if probs else 0)
    want = {"calibration": "STAGE2_CALIBRATION", "heldout": "STAGE2_HELDOUT", "smoke": "STAGE2_SMOKE"}[a.cmd]
    if cfg.get("mode") != want:
        probs.append(f"command {a.cmd} needs mode {want}")
    if probs:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(probs))
    info = execution_budget(cfg, datetime.now(timezone.utc))
    deadline = Deadline(info["budget_s"])
    with controller_signals():
        ctx = Stage2Ctx(cfg, h)
        s = run_session(ctx, cfg["units"], deadline, deadline_info=info)
    print(json.dumps({k: s.get(k) for k in ("execution_status", "stop_reason", "n_attempts", "n_accepted",
                                            "smoke_verdict")}, default=str))
    if s["execution_status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
