"""Machine and software metadata capture.

Raw command outputs are saved verbatim next to a parsed machine.json so that
nothing in the summary has to be trusted without the source text.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import pynvml as N

COMMANDS = {
    "nvidia-smi.txt": ["nvidia-smi"],
    "nvidia-smi-topo.txt": ["nvidia-smi", "topo", "-m"],
    "nvidia-smi-q.txt": ["nvidia-smi", "-q"],
    "nvidia-smi-nvlink.txt": ["nvidia-smi", "nvlink", "-s"],
    "nvidia-smi-compute-apps.txt": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                    "--format=csv"],
    "nvcc-version.txt": ["{cuda_home}/bin/nvcc", "--version"],
    "lscpu.txt": ["lscpu"],
    "lscpu-e.txt": ["lscpu", "-e"],
    "numactl-H.txt": ["numactl", "-H"],
    "uname.txt": ["uname", "-a"],
    "hostname.txt": ["hostname"],
    "os-release.txt": ["cat", "/etc/os-release"],
    "free.txt": ["free", "-b"],
    "uptime.txt": ["uptime"],
    "git-head.txt": ["git", "rev-parse", "HEAD"],
    "git-status.txt": ["git", "status", "--porcelain"],
    "nccl-tests-git.txt": ["git", "-C", "{nccl_tests_src}", "log", "-1", "--format=%H %cd %s"],
    "pip-freeze.txt": [sys.executable, "-m", "pip", "list", "--format=freeze"],
}


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, f"[failed to run {cmd}: {e}]"


def nccl_library_version(lib_dir: Path) -> int | None:
    try:
        lib = ctypes.CDLL(str(Path(lib_dir) / "libnccl.so.2"))
        v = ctypes.c_int()
        if lib.ncclGetVersion(ctypes.byref(v)) == 0:
            return v.value
    except OSError:
        pass
    return None


def torch_versions(python: str) -> dict:
    code = ("import json,torch;print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
            "'torch_nccl':'.'.join(map(str,torch.cuda.nccl.version()))}))")
    try:
        p = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - metadata must never abort a run
        return {"error": str(e)}


def _nvml_gpu(i: int) -> dict:
    h = N.nvmlDeviceGetHandleByIndex(i)

    def q(fn, *a):
        try:
            v = fn(h, *a)
            return v.decode() if isinstance(v, bytes) else v
        except N.NVMLError as e:
            return f"NVMLError: {e}"

    d = {
        "index": i,
        "name": q(N.nvmlDeviceGetName),
        "uuid": q(N.nvmlDeviceGetUUID),
        "serial": q(N.nvmlDeviceGetSerial),
        "pci_bus_id": q(lambda hh: N.nvmlDeviceGetPciInfo(hh).busId),
        "vbios": q(N.nvmlDeviceGetVbiosVersion),
        "memory_total_bytes": q(lambda hh: N.nvmlDeviceGetMemoryInfo(hh).total),
        "power_limit_mw": q(N.nvmlDeviceGetPowerManagementLimit),
        "power_default_limit_mw": q(N.nvmlDeviceGetPowerManagementDefaultLimit),
        "power_enforced_limit_mw": q(N.nvmlDeviceGetEnforcedPowerLimit),
        "max_clock_graphics_mhz": q(N.nvmlDeviceGetMaxClockInfo, N.NVML_CLOCK_GRAPHICS),
        "max_clock_sm_mhz": q(N.nvmlDeviceGetMaxClockInfo, N.NVML_CLOCK_SM),
        "max_clock_mem_mhz": q(N.nvmlDeviceGetMaxClockInfo, N.NVML_CLOCK_MEM),
        "persistence_mode": q(N.nvmlDeviceGetPersistenceMode),
        "compute_mode": q(N.nvmlDeviceGetComputeMode),
        "ecc_mode_current": q(lambda hh: N.nvmlDeviceGetEccMode(hh)[0]),
        "pcie_link_gen_max": q(N.nvmlDeviceGetMaxPcieLinkGeneration),
        "pcie_link_width_max": q(N.nvmlDeviceGetMaxPcieLinkWidth),
        "energy_counter_supported": None,
        "nvlinks": [],
    }
    try:
        N.nvmlDeviceGetTotalEnergyConsumption(h)
        d["energy_counter_supported"] = True
    except N.NVMLError:
        d["energy_counter_supported"] = False
    for link in range(N.NVML_NVLINK_MAX_LINKS):
        try:
            st = N.nvmlDeviceGetNvLinkState(h, link)
        except N.NVMLError:
            continue
        rem = None
        if st:
            try:
                rem = N.nvmlDeviceGetNvLinkRemotePciInfo(h, link).busId
                rem = rem.decode() if isinstance(rem, bytes) else rem
            except N.NVMLError:
                pass
        d["nvlinks"].append({"link": link, "active": bool(st), "remote_pci": rem})
    procs = []
    for fn in (N.nvmlDeviceGetComputeRunningProcesses, N.nvmlDeviceGetGraphicsRunningProcesses):
        try:
            procs += [{"pid": p.pid, "used_mem": p.usedGpuMemory} for p in fn(h)]
        except N.NVMLError:
            pass
    d["processes_at_capture"] = procs
    return d


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def collect(out_dir: Path, repo_root: Path, cfg: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nt = cfg["nccl_tests"]
    subst = {"cuda_home": nt["cuda_home"], "nccl_tests_src": str(repo_root / nt["src_dir"])}
    rcs = {}
    for fname, cmd in COMMANDS.items():
        cmd = [c.format(**subst) for c in cmd]
        rc, txt = _run(cmd, repo_root)
        (out_dir / fname).write_text(txt, encoding="utf-8")
        rcs[fname] = rc

    N.nvmlInit()
    ngpu = N.nvmlDeviceGetCount()
    drv = N.nvmlSystemGetDriverVersion()
    cuda_drv = N.nvmlSystemGetCudaDriverVersion_v2()
    lib_dir = repo_root / nt["nccl_lib_dir"]
    build = repo_root / nt["build_dir"]
    meta = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvidia_driver": drv.decode() if isinstance(drv, bytes) else drv,
        "cuda_driver_api": cuda_drv,
        "nvml_python": getattr(N, "__version__", None),
        "nccl_library_version_code": nccl_library_version(lib_dir),
        "nccl_library_path": str((lib_dir / "libnccl.so.2").resolve()),
        "nccl_library_sha256": sha256(lib_dir / "libnccl.so.2"),
        "nccl_tests_binaries_sha256": {p.name: sha256(p) for p in sorted(build.glob("*_perf"))},
        "torch": torch_versions(sys.executable),
        "git_head": (out_dir / "git-head.txt").read_text(encoding="utf-8").strip() or None,
        "git_dirty": bool((out_dir / "git-status.txt").read_text(encoding="utf-8").strip()),
        "nccl_tests_commit": (out_dir / "nccl-tests-git.txt").read_text(encoding="utf-8").strip(),
        "load_average": os.getloadavg(),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("CUDA", "NCCL", "LD_LIBRARY_PATH", "OMP_"))},
        "gpus": [_nvml_gpu(i) for i in range(ngpu)],
        "command_return_codes": rcs,
    }
    (out_dir / "machine.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta
