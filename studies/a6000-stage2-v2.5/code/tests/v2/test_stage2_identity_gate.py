"""Stage 2 identity gate, freeze binding, scope, identifiers and historical guards (CPU only; no NVML is touched)."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

from ncclenergy.v2 import runner, stage2_runner as S  # noqa: E402
import ncclenergy.v2.stage2_calibration  # noqa: E402,F401  (loaded code must be bound by the freeze)
from ncclenergy.v2.common import REPO  # noqa: E402

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
(REPO / "logs").mkdir(exist_ok=True)
BIN = S.BIN_DIR_STAGE2 / "all_gather_perf"
SCOPE = {"gpu_name": "NVIDIA RTX A6000", "driver": "580.178.04", "link": "NV4", "uuids": ["GPU-a", "GPU-b"],
         "numa": {"4": 1, "5": 1}, "power_limit_mw": [300000, 300000], "max_clk_gr_mhz": [2100, 2100]}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO.resolve()))


def _units(mode):
    if mode == "STAGE2_HELDOUT":
        return [{"launch_id": "s1-b1-p01", "arm": "A", "length_s": 2.5, "pause_ns": 1}]
    part = "SMOKE" if mode == "STAGE2_SMOKE" else "S0a"
    return [{"launch_id": f"{part}-01", "part": part, "arm": "B", "length_s": 2.5, "dose": "E0", "dose_period_ms": 0,
             "pause_ns": 1, "post_probe": True}]


def _config(tmp: Path, mode: str, freeze_rel: str | None, name: str) -> Path:
    cfg = {"implementation_id": S.IMPLEMENTATION_ID, "method_id": S.METHOD_ID, "mode": mode, "output_root": "data/stage2",
           "campaign": "v2.5-test-smoke" if mode == "STAGE2_SMOKE" else "v2.5-test", "energy_source": "counter",
           "max_attempts": 1, "allocation": [4, 5], "placements": [{"id": "P45", "gpus": [4, 5]}],
           "collective": "all_gather", "size_bytes": 2097152, "scope_expected": SCOPE,
           "allocation_confirmation": {"confirmed_by": "user", "gpus": [4, 5], "confirmed_utc": "2026-10-19T12:00:00Z",
                                       "window_start_utc": "2026-10-20T14:00:00Z", "window_end_utc": "2026-10-20T16:00:00Z"},
           "counts": {a: {"w": 95597, "R": {"2.5": 47799, "10": 191193, "20": 382385, "40": 764769}} for a in "ABC"},
           "units": _units(mode), "freeze_record": freeze_rel}
    if mode == "STAGE2_HELDOUT":
        cfg.update(session=1, readiness_rules={"t_launch_c": 37, "bands": {}})
    p = tmp / name
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def _record(tmp: Path, kind: str, cfg_path: Path, parent: Path | None = None, mutate=None, out: Path | None = None) -> Path:
    protocol = tmp / "PROTOCOL.md"
    if not protocol.exists():
        protocol.write_text("protocol text\n")
    roles = {"config": _rel(cfg_path), "protocol": _rel(protocol), "patch": _rel(S.PATCH_V25), "binary": _rel(BIN),
             "build_record": _rel(BIN.parent.parent / "BUILD_RECORD.txt"), "plugin": _rel(runner.PLUGIN),
             "code": S.loaded_code_files(),
             "analysis": ["src/ncclenergy/v2/stage2_calibration.py", "src/ncclenergy/v2/stage2_decisions.py"]}
    rels = [roles[k] for k in ("config", "protocol", "patch", "binary", "build_record", "plugin")] + roles["code"] + roles["analysis"]
    rec = {"schema": S.FREEZE_SCHEMA, "kind": kind, "utc": "2026-10-19T00:00:00Z", "approved_by_user": {"note": "test"},
           "files": {r: {"sha256": _sha(REPO / r), "bytes": (REPO / r).stat().st_size} for r in rels}, "roles": roles,
           "external": {"libnccl": {"path": str(S.NCCL_LIB_LINK.resolve()), "sha256": _sha(S.NCCL_LIB_LINK.resolve())}}}
    if parent is not None:
        rec["parent"] = {"path": _rel(parent), "sha256": _sha(parent)}
    if mutate:
        mutate(rec)
    out = out or tmp / f"freeze_{kind}_{len(list(tmp.glob('freeze_*')))}.json"
    out.write_text(json.dumps(rec))
    return out


def _chain(tmp: Path):
    """A valid Freeze-S record and a Freeze-C record whose parent is it; returns (cfg_S, rec_S, cfg_C, rec_C). Each
    config names its record by path; each record binds its config by hash (no circularity)."""
    rs_path, rc_path = tmp / "freeze_S.json", tmp / "freeze_C.json"
    cs = _config(tmp, "STAGE2_SMOKE", _rel(rs_path), "smoke.yaml")
    rs = _record(tmp, "Freeze-S", cs, out=rs_path)
    cc = _config(tmp, "STAGE2_CALIBRATION", _rel(rc_path), "calib.yaml")
    rc = _record(tmp, "Freeze-C", cc, parent=rs, out=rc_path)
    return cs, rs, cc, rc


@strict
def test_valid_freeze_chain_binds_config_code_binary_nccl_and_parent():
    if not BIN.exists():
        print("SKIP: v2.5 binary not built")
        return
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        cs, rs, cc, rc = _chain(Path(t))
        assert S.verify_freeze(rs, "Freeze-S", config_path=cs, code_files=S.loaded_code_files(), bin_path=BIN) == []
        assert S.verify_freeze(rc, "Freeze-C", config_path=cc, code_files=S.loaded_code_files(), bin_path=BIN) == []


@strict
def test_freeze_refusals_irrelevant_external_config_parent_code():
    if not BIN.exists():
        print("SKIP: v2.5 binary not built")
        return
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        t = Path(t)
        cs, rs, cc, rc = _chain(t)
        ok = dict(config_path=cc, code_files=S.loaded_code_files(), bin_path=BIN)
        junk = t / "junk.txt"
        junk.write_text("x")
        irrelevant = t / "irrelevant.json"
        irrelevant.write_text(json.dumps({"schema": S.FREEZE_SCHEMA, "kind": "Freeze-C", "approved_by_user": {"x": 1},
                                          "files": {_rel(junk): {"sha256": _sha(junk)}}}))
        p = S.verify_freeze(irrelevant, "Freeze-C", **ok)
        assert any("mandatory role missing" in x for x in p) and any("libnccl identity missing" in x for x in p)
        assert any("needs a Freeze-S parent" in x for x in p)
        wrong_ext = _record(t, "Freeze-C", cc, parent=rs, mutate=lambda r: r["external"]["libnccl"].update(sha256="0" * 64))
        assert "external libnccl hash mismatch" in S.verify_freeze(wrong_ext, "Freeze-C", **ok)
        other_lib = _record(t, "Freeze-C", cc, parent=rs, mutate=lambda r: r["external"]["libnccl"].update(path="/tmp/libnccl.so.2"))
        assert any("loaded NCCL library resolves to" in x for x in S.verify_freeze(other_lib, "Freeze-C", **ok))
        alt = t / "calib_altered.yaml"
        alt.write_text(cc.read_text().replace("max_attempts: 1", "max_attempts: 1\n# edited"))
        assert "the config being run is not the frozen config" in S.verify_freeze(rc, "Freeze-C", **dict(ok, config_path=alt))
        cc.write_text(cc.read_text() + "# edited after the freeze\n")
        assert any("frozen file changed" in x for x in S.verify_freeze(rc, "Freeze-C", **ok))
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        t = Path(t)
        cs, rs, cc, rc = _chain(t)
        ok = dict(config_path=cc, code_files=S.loaded_code_files(), bin_path=BIN)
        rs.write_text(rs.read_text().replace('"test"', '"edited"'))          # parent edited after the child froze it
        assert "parent record hash mismatch" in S.verify_freeze(rc, "Freeze-C", **ok)
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        t = Path(t)
        cs, rs, cc, rc = _chain(t)
        ok = dict(config_path=cc, code_files=S.loaded_code_files(), bin_path=BIN)
        wrong_kind = _record(t, "Freeze-C", cc, parent=rc)                    # parent of the wrong kind
        assert any("parent: freeze record kind" in x for x in S.verify_freeze(wrong_kind, "Freeze-C", **ok))
        other_bin = _record(t, "Freeze-C", cc, parent=rs, mutate=lambda r: (
            r["roles"].update(binary=_rel(S.PATCH_V21)),
            r["files"].update({_rel(S.PATCH_V21): {"sha256": _sha(S.PATCH_V21)}})))
        p = S.verify_freeze(other_bin, "Freeze-C", **ok)
        assert "shared role binary differs from the parent" in p and "the binary being run is not the frozen binary" in p
        unbound = _record(t, "Freeze-C", cc, parent=rs, mutate=lambda r: r["roles"].update(code=r["roles"]["code"][1:]))
        assert any("loaded code not bound" in x for x in S.verify_freeze(unbound, "Freeze-C", **ok))
        no_analysis = _record(t, "Freeze-C", cc, parent=rs, mutate=lambda r: r["roles"].update(analysis=[]))
        assert any("mandatory" in x for x in S.verify_freeze(no_analysis, "Freeze-C", **ok))
        unapproved = _record(t, "Freeze-C", cc, parent=rs, mutate=lambda r: r.pop("approved_by_user"))
        assert "freeze record lacks the user's approval entry" in S.verify_freeze(unapproved, "Freeze-C", **ok)
        assert any("stage2_heldout.py" in x for x in S.verify_freeze(rc, "Freeze-H", **ok))   # Freeze-H not possible yet


@strict
def test_config_rules_scope_and_identifiers():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        t = Path(t)
        cfg = yaml.safe_load(_config(t, "STAGE2_CALIBRATION", "x", "c.yaml").read_text())
        assert S.check_config(cfg, NOW, check_freeze=False) == []
        bad = {"historical id": {"implementation_id": "release-corrections-2026-10-11"},
               "superseded id": {"implementation_id": "stage2-measval-2026-10-12"},
               "superseded method": {"method_id": "v2.4-stage2-region"},
               "mode": {"mode": "TEST"}, "output root": {"output_root": "data/v2.2/raw"},
               "sampled": {"energy_source": "sampled"}, "retries": {"max_attempts": 2},
               "timing": {"timing": {"cal_fixed_wait_s": 1}}, "three gpus": {"allocation": [4, 5, 6]},
               "placement order": {"placements": [{"id": "P", "gpus": [5, 4]}]},
               "env": {"env_extra": {"NCCL_ALGO": "Tree"}}, "no confirmation": {"allocation_confirmation": None},
               "not the user": {"allocation_confirmation": dict(cfg["allocation_confirmation"], confirmed_by="idle-check")},
               "window over": {"allocation_confirmation": dict(cfg["allocation_confirmation"], window_end_utc="2026-10-20T14:30:00Z")},
               "no scope": {"scope_expected": None}, "partial scope": {"scope_expected": {"driver": "580.178.04"}},
               "traversal": {"campaign": "../../etc"}, "slash": {"campaign": "a/b"}, "size": {"size_bytes": 1048576}}
        for name, change in bad.items():
            c = copy.deepcopy(cfg)
            c.update(change)
            assert S.check_config(c, NOW, check_freeze=False), name
        lid = copy.deepcopy(cfg)
        lid["units"][0]["launch_id"] = "../S0a"
        assert any("launch id must match" in x for x in S.check_config(lid, NOW, check_freeze=False))
        assert any("freeze_record missing" in x for x in S.check_config(dict(cfg, freeze_record=None), NOW))
        obs = dict(SCOPE)
        assert S.scope_problems(SCOPE, obs) == []
        assert S.scope_problems(SCOPE, dict(obs, gpu_name="NVIDIA A100-SXM4-40GB"))[0].startswith("SCOPE_MISMATCH gpu_name")
        assert S.scope_problems(SCOPE, dict(obs, link="SYS")) and S.scope_problems(None, obs)
        h = _config(t, "STAGE2_HELDOUT", "x", "h.yaml")
        hc = yaml.safe_load(h.read_text())
        for change in ({"dose": "D100", "dose_period_ms": 100}, {"post_probe": True}, {"poller_active": True}):
            c = copy.deepcopy(hc)
            c["units"][0].update(change)
            assert any("refused in held-out mode" in p for p in S.check_config(c, NOW, check_freeze=False)), change
        sm = yaml.safe_load(_config(t, "STAGE2_SMOKE", "x", "s.yaml").read_text())
        assert S.check_config(sm, NOW, check_freeze=False) == []
        assert any("at most 4" in x for x in S.check_config(dict(sm, units=[dict(sm["units"][0], launch_id=f"SMOKE-{i}")
                                                                           for i in range(5)]), NOW, check_freeze=False))
        assert any("-smoke" in x for x in S.check_config(dict(sm, campaign="v2.5-test"), NOW, check_freeze=False))


@strict
def test_output_path_containment():
    cfg = {"mode": "STAGE2_CALIBRATION", "allocation": [4, 5], "placements": [{"gpus": [4, 5]}], "campaign": "ok-name"}
    snap = lambda a: {"gpus": [{"index": 4, "uuid": "u4", "pci": "p4"}, {"index": 5, "uuid": "u5", "pci": "p5"}]}  # noqa: E731
    for bad in ({"campaign": "../escape"}, {"campaign": "a/../../b"}):
        try:
            S.Stage2Ctx(dict(cfg, **bad), "h", snapshot_fn=snap)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    try:
        S.Stage2Ctx(cfg, "h", snapshot_fn=snap, out_dir=Path("/tmp/elsewhere"))
    except SystemExit as e:
        assert "logs/" in str(e)
    else:
        raise AssertionError("test output outside logs/ accepted")
    try:
        S.Stage2Ctx(dict(cfg, placements=[{"gpus": [5, 4]}]), "h", snapshot_fn=snap)
    except SystemExit:
        pass
    else:
        raise AssertionError("placement in a different order accepted")


@strict
def test_historical_runner_refuses_stage2_config_before_nvml():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        cfg = yaml.safe_load(_config(Path(t), "STAGE2_CALIBRATION", "x", "c.yaml").read_text())
        try:
            runner._run_gpu_command(SimpleNamespace(cmd="block"), cfg, "h")
        except SystemExit as e:
            assert "Historical campaign configs are read-only" in str(e)
        else:
            raise AssertionError("historical runner accepted a Stage 2 config")


@strict
def test_stage2_refuses_every_historical_config():
    paths = sorted(p for p in (REPO / "configs").rglob("*.yaml") if "stage2" not in p.parts)
    assert len(paths) >= 14
    for p in paths:
        try:
            cfg = yaml.safe_load(p.read_text())
        except yaml.YAMLError:                  # configs/v2/replication/template.yaml has placeholders: not loadable
            continue
        assert any("implementation_id must be" in x for x in S.check_config(cfg, NOW)), p


@strict
def test_build_cmd_default_unchanged_and_stage2_binary_separate():
    old = runner.build_cmd("all_gather", 2, 2097152, 2097152, 10, 5, 1, None)
    assert old[3] == str(runner.BIN_DIR / "all_gather_perf")
    new = runner.build_cmd("all_gather", 2, 2097152, 2097152, 10, 5, 1, None, bin_dir=S.BIN_DIR_STAGE2)
    assert new[3] == str(S.BIN_DIR_STAGE2 / "all_gather_perf") and old[4:] == new[4:]
    assert S.BIN_DIR_STAGE2.parent.name == "nccl-tests-stage2-v25"


@strict
def test_stage2_env_sets_only_allowed_variables_and_refuses_holdout_dose():
    ctx = SimpleNamespace(mode="heldout", gpus=[4, 5], uuids=lambda g: ["GPU-a", "GPU-b"])
    d = REPO / "logs/x"
    env = S.stage2_env(ctx, {"arm": "C", "pause_ns": 5}, d)
    assert env["NCCLE_STAGE2_MODE"] == "heldout" and env["NCCLE_START_PAUSE_NS"] == "5"
    assert "NCCLE_DOSE_PERIOD_MS" not in env and "NCCLE_POST_PROBE" not in env and env["NCCL_PROTO"] == "LL128"
    for u in ({"arm": "A", "pause_ns": 5, "dose_period_ms": 100}, {"arm": "A", "pause_ns": 5, "post_probe": True}):
        try:
            S.stage2_env(ctx, u, d)
        except S.Stage2Stop:
            pass
        else:
            raise AssertionError("held-out dose / post-probe accepted")
    cal = SimpleNamespace(mode="calibration", gpus=[4, 5], uuids=lambda g: ["GPU-a", "GPU-b"])
    e2 = S.stage2_env(cal, {"arm": "B", "pause_ns": 0, "dose_period_ms": 500, "post_probe": True}, d)
    assert e2["NCCLE_DOSE_PERIOD_MS"] == "500" and e2["NCCLE_POST_PROBE"] == "1"


@strict
def test_stage2_counter_calls_only_in_the_gated_baseline_helper():
    import ast
    for p in sorted((REPO / "src/ncclenergy/v2").glob("stage2_*.py")):
        tree = ast.parse(p.read_text())
        fns = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for c in ast.walk(node):
                    if isinstance(c, ast.Attribute) and c.attr == "nvmlDeviceGetTotalEnergyConsumption":
                        fns.add(node.name)
        assert fns == ({"_counter_read"} if p.name == "stage2_runner.py" else set()), (p.name, fns)
        assert not [x for x in p.read_text().split() if x.startswith("NVML_FI_") and "ENERGY" in x]


@strict
def test_cli_check_refuses_the_drafts():
    for c in sorted((REPO / "configs/stage2").glob("*.yaml")):
        p = subprocess.run([sys.executable, "-W", "error", "-m", "ncclenergy.v2.stage2_runner", "check", "--campaign", str(c)],
                           cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == 1, (c, p.stdout, p.stderr[-2000:])
        r = json.loads(p.stdout)
        assert any("allocation_confirmation missing" in x for x in r["refusals"]) and \
            any("freeze_record missing" in x for x in r["refusals"]) and any("scope_expected" in x for x in r["refusals"])


@strict
def test_proposed_manifest_is_complete_but_unapproved():
    with tempfile.TemporaryDirectory(dir=REPO / "logs") as t:
        out = Path(t) / "proposal.json"
        p = subprocess.run([sys.executable, "-W", "error", str(REPO / "scripts/stage2_freeze_manifest.py"), "--kind",
                            "Freeze-S", "--out", str(out), "--protocol", str(REPO / "README.md")], cwd=REPO,
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 0 and not p.stderr, p.stderr[-2000:]
        m = json.loads(out.read_text())
        assert m["status"] == "PROPOSED_NOT_FROZEN" and not m["missing"] and "approved_by_user" not in m
        probs = S.verify_freeze(out, "Freeze-S")
        assert probs == ["freeze record lacks the user's approval entry"], probs
