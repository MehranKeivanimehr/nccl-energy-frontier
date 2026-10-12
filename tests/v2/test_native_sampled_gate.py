"""Compile the patch's actual mode/read/probe helpers with CPU-only CUDA stubs.

This is not a CUDA build or an NVML sensor validation. The trivial once_flag
stub is unused here; the tested functions are copied from patch additions.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

from ncclenergy.v2.common import REPO


def test_native_sampled_gate_makes_no_counter_calls():
    cc = shutil.which("gcc")
    if cc is None:
        raise AssertionError("gcc with its C++ frontend is required for native helper tests")
    patch = (REPO / "patches/nccl-tests-afd59ab-v2.3-sampled-gate.patch").read_text()
    additions = "\n".join(line[1:] for line in patch.splitlines()
                          if line.startswith("+") and not line.startswith("+++"))
    helpers = additions.split("// ===== nccl-energy-frontier methodology-v2 region instrumentation (end)")[0]
    prefix = '''#include <stdlib.h>
#include <string.h>
namespace std { struct once_flag {}; }
static const int cudaSuccess = 0;
static int cudaDeviceGetPCIBusId(char* p, int n, int) {
  strncpy(p, "0000:01:00.0", n); return cudaSuccess;
}
'''
    harness = '''
static int calls = 0;
static int fakeEnergy(nccleNvmlDevice_t, unsigned long long* e) { ++calls; *e = 123; return 0; }
int main() {
  NccleGpu g = {}; g.hRc = 0; nccleNvml.energy = fakeEnergy;
  unsigned long long e = 999; int rc = 0; long long a, b;
  setenv("NCCLE_ENERGY_SOURCE", "sampled", 1);
  nccleRead(g, &e, &rc, &a, &b); nccleProbePhase(g);
  if (calls || rc != -2 || e || a > b || g.nEdges) return 1;
  setenv("NCCLE_ENERGY_SOURCE", "counter", 1);
  nccleRead(g, &e, &rc, &a, &b);
  if (calls != 1 || rc || e != 123) return 2;
  setenv("NCCLE_ENERGY_SOURCE", "bad", 1);
  nccleRead(g, &e, &rc, &a, &b); nccleProbePhase(g);
  if (calls != 1 || nccleEnergyMode() != -1) return 3;
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "helper.cc"
        source.write_text(prefix + helpers + harness)
        subprocess.run([cc, "-std=c++11", str(source), "-ldl", "-o", str(root / "check")], check=True)
        subprocess.run([str(root / "check")], check=True)
