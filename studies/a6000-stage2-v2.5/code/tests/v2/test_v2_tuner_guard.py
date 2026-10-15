"""The guarded NCCL tuner plugin, loaded the way NCCL loads it, on CPU only: a C harness calls init/getCollInfo,
a mock libnvidia-ml answers the read-only NVML queries (and aborts if the energy counter is touched), and the
harness exports ncclGetVersion. Missing, unknown or incompatible scope must leave NCCL's choice untouched."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "tests/v2/native"
PLUGIN_DIR = ROOT / "native/nccl_tuner_guarded"
NCCL_HDR = ROOT / "third_party/nccl-src/ext-tuner/example/nccl"

SCOPE = {"method": "v2.2", "gpu_model": "NVIDIA RTX A6000", "driver_version": "580.178.04",
         "nccl_version_code": "22705", "nranks": "2", "nnodes": "1",
         "pci_bus_ids": "00000000:81:00.0,00000000:A1:00.0", "placement_env": "none",
         "declared_scope": "dtype=float32;ops=all_reduce:sum;in_place=0", "allow_ranges": "0"}
RULES = ["allreduce,524288,524288,ring,ll,-1,1,2,-1,-1", "allreduce,8388608,16777216,tree,simple,-1,1,2,-1,-1"]
GPUS = "GPU-aaa=NVIDIA RTX A6000@00000000:81:00.0;GPU-bbb=NVIDIA RTX A6000@00000000:a1:00.0"


def _build(tmp: Path):
    if not NCCL_HDR.exists():
        raise AssertionError(f"NCCL example headers missing at {NCCL_HDR} (needed to build the plugin)")
    subprocess.run(["make", "-s", "-C", str(PLUGIN_DIR)], check=True, capture_output=True)
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-o", str(tmp / "libmocknvml.so"), str(NATIVE / "mock_nvml.c")],
                   check=True, capture_output=True)
    subprocess.run(["gcc", "-O1", f"-I{NCCL_HDR}", "-rdynamic", "-o", str(tmp / "harness"),
                    str(NATIVE / "tuner_harness.c"), "-ldl"], check=True, capture_output=True)


def _run(tmp, scope=None, rules=RULES, env=None, nranks=2, nbytes=524288, extra_lines=()):
    sc = dict(SCOPE) if scope is None else scope
    conf = tmp / "conf"
    conf.write_text("\n".join(["# test"] + [f"#@scope {k}={v}" for k, v in sc.items()] + list(extra_lines) + rules)
                    + "\n")
    e = {"PATH": os.environ["PATH"], "NCCL_TUNER_CONFIG_FILE": str(conf), "NCCLE_NVML_LIBRARY": str(tmp / "libmocknvml.so"),
         "MOCK_NVML_DRIVER": "580.178.04", "MOCK_NVML_GPUS": GPUS, "CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb",
         "MOCK_NCCL_VERSION": "22705", "NCCLE_TUNER_DECLARED_SCOPE": SCOPE["declared_scope"]}
    for k, v in (env or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    p = subprocess.run([str(tmp / "harness"), str(PLUGIN_DIR / "build/libnccl-tuner-guarded.so"), str(nranks), "1",
                        "4", str(nbytes)], env=e, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert "COUNTER CALLED" not in p.stderr
    return p.stdout.strip(), p.stderr


def test_guarded_tuner_applies_exact_rules_only_when_scope_matches():
    tmp = Path(tempfile.mkdtemp())
    try:
        _build(tmp)
        out, err = _run(tmp)
        assert out == "RULE algo=1 proto=0 nChannels=0" and "ENABLED scope OK; 1 rules, 1 range rules skipped" in err
        out, _ = _run(tmp, nbytes=16777216)                     # inside a range rule: not a measured size
        assert out.startswith("NO_RULE")
        sc = dict(SCOPE, allow_ranges="1")
        assert _run(tmp, scope=sc, nbytes=16777216)[0].startswith("NO_RULE")                 # file opt-in alone
        assert _run(tmp, scope=sc, nbytes=16777216, env={"NCCLE_TUNER_ALLOW_RANGES": "1"})[0].startswith("RULE algo=0 proto=2")
    finally:
        shutil.rmtree(tmp)


def test_guarded_tuner_falls_back_on_missing_unknown_or_incompatible_scope():
    tmp = Path(tempfile.mkdtemp())
    try:
        _build(tmp)
        cases = [
            ({**SCOPE, "declared_scope": "None"}, {}, "SCOPE_MISSING:declared_scope"),
            ({k: v for k, v in SCOPE.items() if k != "gpu_model"}, {}, "SCOPE_MISSING:gpu_model"),
            ({**SCOPE, "method": "v2.1"}, {}, "SCOPE_METHOD_UNKNOWN"),
            (None, {"MOCK_NVML_GPUS": GPUS.replace("RTX A6000", "GeForce GTX 1080 Ti")}, "GPU_MODEL"),
            (None, {"MOCK_NVML_DRIVER": "550.144.03"}, "DRIVER_VERSION"),
            (None, {"MOCK_NCCL_VERSION": "22803"}, "NCCL_VERSION:22803"),
            (None, {"MOCK_NCCL_VERSION": None}, "NCCL_VERSION_UNKNOWN"),
            (None, {"CUDA_VISIBLE_DEVICES": "4,5"}, "CUDA_VISIBLE_DEVICES_NOT_UUIDS"),
            (None, {"CUDA_VISIBLE_DEVICES": None}, "CUDA_VISIBLE_DEVICES_UNSET"),
            (None, {"MOCK_NVML_GPUS": GPUS.replace("a1:00.0", "c1:00.0")}, "PCI_BUS_IDS"),
            (None, {"NCCL_ALGO": "Tree"}, "ENV_SET_NOT_IN_PROFILE:NCCL_ALGO"),
            (None, {"NCCL_P2P_DISABLE": "1"}, "ENV_SET_NOT_IN_PROFILE:NCCL_P2P_DISABLE"),
            ({**SCOPE, "placement_env": "NCCL_P2P_DISABLE=1"}, {}, "ENV_MISMATCH:NCCL_P2P_DISABLE"),
            (None, {"NCCLE_TUNER_DECLARED_SCOPE": None}, "DECLARED_SCOPE_MISSING_OR_DIFFERENT"),
            (None, {"NCCLE_TUNER_DECLARED_SCOPE": "dtype=bfloat16;ops=all_reduce:sum;in_place=0"},
             "DECLARED_SCOPE_MISSING_OR_DIFFERENT"),
            (None, {"MOCK_NVML_FAIL_INIT": "1"}, "NVML_INIT"),
            (None, {"NCCLE_NVML_LIBRARY": "/nonexistent/libnvidia-ml.so.1"}, "NVML_UNAVAILABLE"),
            ({**SCOPE, "gpu_model": "X" * 600}, {}, "SCOPE_LINE_TOO_LONG"),
        ]
        for scope, env, why in cases:
            out, err = _run(tmp, scope=scope, env=env)
            assert out == "NO_RULE nChannels=0", (why, out, err[-500:])
            assert f"DISABLED ({why}" in err, (why, err[-800:])
        out, err = _run(tmp, nranks=4)                                                   # communicator size differs
        assert out == "NO_RULE nChannels=0" and "DISABLED (RANKS_OR_NODES" in err
    finally:
        shutil.rmtree(tmp)


def test_malformed_rule_rejects_the_whole_file_and_real_policy_files_parse():
    """v2.2 release audit: after a passing scope guard, a malformed rule (unknown name, bad number, wrong field count)
    used to be loaded with defaults (allreduce/ring/simple) or skipped; now it disables the plugin. The exported
    v2.2 policy files (exact-size rules) must still load completely."""
    tmp = Path(tempfile.mkdtemp())
    try:
        _build(tmp)
        good = "allreduce,524288,524288,ring,ll,-1,1,2,-1,-1"
        for bad in ("allreduc,524288,524288,ring,ll,-1,1,2,-1,-1",      # unknown collective
                    "allreduce,524288,524288,rng,ll,-1,1,2,-1,-1",      # unknown algorithm
                    "allreduce,524288,524288,ring,ll12,-1,1,2,-1,-1",   # unknown protocol
                    "allreduce,5242x8,524288,ring,ll,-1,1,2,-1,-1",     # not a number
                    "allreduce,524288,524288,ring,ll,-1,1",             # too few fields
                    "allreduce,524288,524288,ring,ll,0,1,2,-1,-1"):     # 0 channels
            out, err = _run(tmp, rules=[good, bad])
            assert out == "NO_RULE nChannels=0" and "DISABLED (invalid rule, line" in err, (bad, out, err)
        rules = [l for l in (ROOT / "data/v2/reanalysis/2026-10-09_v2.2-audit/phase89_v22/nccl_tuner_P2_NVL.conf")
                 .read_text().splitlines() if l and not l.startswith("#")]
        out, err = _run(tmp, rules=rules)
        assert "ENABLED scope OK" in err and f"; {len(rules)} rules, 0 range rules skipped" in err, err[-800:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
