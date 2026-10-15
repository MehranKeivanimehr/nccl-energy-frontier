"""v2.5 Stage 2 native patch (CPU only; nothing here runs CUDA, NCCL or NVML). v2.4 is superseded and kept unchanged.

Structure: the v2.1 patch is byte-identical to the historical one; v2.5 only ADDS lines (no v2.1 line is changed or
removed); every BenchTime insertion is guarded by the Stage 2 mode (or is the refusal check); the synchronized work end
is recorded before the dose poller is stopped; the built tree equals afd59ab + v2.1 + v2.5. Behaviour: the patch's own helper code (extracted from the patch, compiled with g++ and CPU
stubs) refuses invalid requests, counts every counter call, paces and stops the dose poller, bounds the post-probe,
sleeps the requested pause and writes parseable JSON. The BenchTime insertions themselves need a GPU (smoke gate).
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helpers"))
from strict import strict  # noqa: E402

from ncclenergy.v2.common import REPO  # noqa: E402
from ncclenergy.v2.stage2_energy import parse_s2_text, s2_metrics  # noqa: E402

P21 = REPO / "patches/nccl-tests-afd59ab-v2.1-region.patch"
P24 = REPO / "patches/nccl-tests-afd59ab-v2.5-stage2.patch"
P24_SUPERSEDED = REPO / "patches/nccl-tests-afd59ab-v2.4-stage2.patch"
V24_SUPERSEDED_SHA256 = "630befb32fef57ee7c13b43d5273e1b0b20c6d5275e1745ca2da6a01eee8c6b2"
TREE = REPO / "third_party/nccl-tests-stage2-v25"
V21_SHA256 = "293527e3ab71ece12ff42088b84f47763ba48e62f08999edbfa4ff04bad2e0b8"
V24_BEGIN = "// ===== nccl-energy-frontier Stage 2 (v2.5) additions (begin) ====="
V24_END = "// ===== nccl-energy-frontier Stage 2 (v2.5) additions (end) ====="


def _additions(patch: Path) -> list[str]:
    return [ln[1:] for ln in patch.read_text().splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def _hunks(patch: Path) -> list[list[str]]:
    out, cur = [], None
    for ln in patch.read_text().splitlines():
        if ln.startswith("@@"):
            cur = []
            out.append(cur)
        elif cur is not None:
            cur.append(ln)
    return out


@strict
def test_v21_unchanged_and_v25_is_additions_only():
    assert hashlib.sha256(P21.read_bytes()).hexdigest() == V21_SHA256
    assert hashlib.sha256(P24_SUPERSEDED.read_bytes()).hexdigest() == V24_SUPERSEDED_SHA256   # provenance, unchanged
    body = [ln for ln in P24.read_text().splitlines() if not ln.startswith(("---", "+++", "diff ", "@@"))]
    assert not [ln for ln in body if ln.startswith("-")], "v2.4 must not remove or change any v2.1/upstream line"
    assert sum(1 for ln in body if ln.startswith("+")) > 0
    files = re.findall(r"^\+\+\+ b/(\S+)", P24.read_text(), re.M)
    assert files == ["src/common.cu"]


@strict
def test_every_benchtime_insertion_is_guarded_by_stage2_mode():
    hunks = _hunks(P24)
    assert any(V24_BEGIN in "\n".join(h) for h in hunks[:1])            # first hunk: the helper block
    groups = []
    for h in hunks[1:]:
        cur = []
        for ln in h:
            if ln.startswith("+"):
                cur.append(ln[1:].strip())
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    first = groups[0]
    assert first[0].startswith("// ---- NCCLE Stage 2 (v2.4)") and first[1] == "std::call_once(nccleS2Once, nccleS2Parse);"
    assert first[2].startswith("if (nccleS2.err != nullptr)") and first[3] == "if (nccleS2.active) {"
    for g in groups[1:]:
        code = [x for x in g if x and not x.startswith("//")]
        assert code[0].startswith("if (nccleS2.active"), g
    text = P24.read_text()                      # v2.5: work end before the poller stop, inside the same hunk
    assert 0 < text.index("tWorkEndNs = nccleNowNs()") < text.index("nccleS2DoseStop(args->thread")


@strict
def test_built_tree_is_afd59ab_plus_v21_plus_v25():
    if not (TREE / ".git").exists():
        print("SKIP: third_party/nccl-tests-stage2 not built (scripts/build_nccl_tests_stage2.sh)")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        subprocess.run(["git", "-C", str(TREE), "worktree", "list"], check=True, capture_output=True)
        src = subprocess.run(["git", "-C", str(TREE), "show", "afd59abab774a6a4b492eb344bd0c845a0b7bb4d:src/common.cu"],
                             check=True, capture_output=True).stdout
        (d / "src").mkdir()
        (d / "src/common.cu").write_bytes(src)
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        for p in (P21, P24):
            subprocess.run(["git", "-C", str(d), "apply", "--check", str(p)], check=True)
            subprocess.run(["git", "-C", str(d), "apply", str(p)], check=True)
        assert (d / "src/common.cu").read_bytes() == (TREE / "src/common.cu").read_bytes()
    rec = dict(ln.split(" ", 1) for ln in (TREE / "BUILD_RECORD.txt").read_text().splitlines() if " " in ln)
    assert rec["patch_v2.1_sha256"] == V21_SHA256
    assert rec["patch_stage2_version"] == "v2.5"
    assert rec["patch_stage2_sha256"] == hashlib.sha256(P24.read_bytes()).hexdigest()
    built = hashlib.sha256((TREE / "build/all_gather_perf").read_bytes()).hexdigest()
    assert f"all_gather_perf {built}" in rec["binary_sha256"] or any(
        ln.endswith(f"all_gather_perf {built}") for ln in (TREE / "BUILD_RECORD.txt").read_text().splitlines())


PREFIX = r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <algorithm>
static const int cudaSuccess = 0;
static int cudaDeviceGetPCIBusId(char* p, int n, int) { strncpy(p, "00000000:81:00.0", n); return cudaSuccess; }
'''

HARNESS = r'''
#include <assert.h>
static long long g_period = 100000000LL, g_slowNs = 0; static int g_fastConst = 0;
static int fakeEnergy(nccleNvmlDevice_t, unsigned long long* e) {
  if (g_slowNs) { struct timespec ts = {0, (long)g_slowNs}; nanosleep(&ts, nullptr); }
  *e = g_fastConst ? 7ULL : (unsigned long long)(nccleNowNs() / g_period) * 10ULL; return 0;
}
static int parseCase(const char* mode, const char* region, const char* file, const char* pause, const char* dose,
                     const char* pp) {
  const char* names[] = {"NCCLE_STAGE2_MODE", "NCCLE_REGION_FILE", "NCCLE_S2_FILE", "NCCLE_START_PAUSE_NS",
                         "NCCLE_DOSE_PERIOD_MS", "NCCLE_POST_PROBE"};
  const char* vals[] = {mode, region, file, pause, dose, pp};
  for (int i = 0; i < 6; i++) { if (vals[i]) setenv(names[i], vals[i], 1); else unsetenv(names[i]); }
  nccleS2 = NccleS2Cfg(); nccleS2Parse();
  return nccleS2.err ? -1 : (nccleS2.active ? 1 : 0);
}
int main(int argc, char** argv) {
  FILE* out = fopen(argv[1], "w");
  // ---- refusal rules
  int r[] = {
    parseCase(0, 0, 0, 0, 0, 0),                                            // 0: v2.1 behaviour
    parseCase(0, 0, 0, 0, "100", 0),                                        // 1: dose without mode
    parseCase("heldout", "r", "s", "5", 0, 0),                              // 2: ok heldout
    parseCase("heldout", "r", "s", "5", "100", 0),                          // 3: dose in heldout
    parseCase("heldout", "r", "s", "5", 0, "1"),                            // 4: M4d in heldout
    parseCase("calibration", "r", "s", "5", "250", 0),                      // 5: bad period
    parseCase("calibration", "r", "s", "5", "100", "1"),                    // 6: ok calibration with dose + M4d
    parseCase("calibration", "r", "s", 0, 0, 0),                            // 7: pause missing
    parseCase("calibration", "r", "s", "-1", 0, 0),                         // 8
    parseCase("calibration", "r", "s", "100000000", 0, 0),                  // 9
    parseCase("calibration", "r", "s", "12x", 0, 0),                        // 10
    parseCase("bogus", "r", "s", "5", 0, 0),                                // 11
    parseCase("calibration", 0, "s", "5", 0, 0),                            // 12: no region file
    parseCase("calibration", "r", 0, "5", 0, 0),                            // 13: no s2 file
    parseCase("calibration", "r", "s", "5", "500", "0"),                    // 14: ok, M4d off
    parseCase("calibration", "r", "s", "5", 0, "yes"),                      // 15
    parseCase(0, 0, "s", 0, 0, 0)};                                         // 16: s2 file without mode
  fprintf(out, "{\"parse\":[");
  for (int i = 0; i < 17; i++) fprintf(out, "%s%d", i ? "," : "", r[i]);
  parseCase("calibration", "r", "s", "5", "100", "1");
  fprintf(out, "],\"cal\":[%d,%d,%lld],", nccleS2.doseMs, nccleS2.postProbe, nccleS2.pauseNs);
  // ---- counting wrapper: every counter call on the GPU is counted; values unchanged
  nccleNvml.energy = fakeEnergy; nccleNvml.loaded = 1; g_slowNs = 1000000LL;   // ~1 ms per read, like NVML
  NccleGpu& G = nccleGpu[0]; G.set = 1; G.hRc = 0; G.h = (nccleNvmlDevice_t)0x1; strcpy(G.pci, "00000000:81:00.0");
  nccleS2InstallShim(); nccleS2InstallShim();                                // idempotent
  nccleS2Reset(0);
  unsigned long long e; int rc; long long a, b;
  nccleRead(G, &e, &rc, &a, &b); nccleRead(G, &e, &rc, &a, &b);
  unsigned long long before = nccleS2Gpu[0].nReads.load();
  nccleProbePhase(G);                                                        // the unchanged v2.1 probe
  unsigned long long probe = nccleS2Gpu[0].nReads.load() - before;
  fprintf(out, "\"shim\":[%llu,%llu,%d,%d],", before, probe, G.nEdges, (int)(nccleNvml.energy == nccleS2EnergyCounted));
  // ---- pause accuracy
  long long t0 = nccleNowNs(), t1 = nccleS2SleepUntil(t0 + 37000000LL);
  fprintf(out, "\"pause_ns\":%lld,", t1 - t0);
  // ---- dose poller: 100 ms, run ~550 ms
  nccleS2.doseMs = 100; nccleS2Reset(0); G.t0b = nccleNowNs();
  nccleS2DoseStart(0); struct timespec w = {0, 550000000}; nanosleep(&w, nullptr);
  NccleS2Gpu& S = nccleS2Gpu[0];
  S.tWorkEndNs = nccleNowNs(); nccleS2DoseStop(0);                       // BenchTime order in v2.5
  fprintf(out, "\"order\":[%d,%d,%d],", (int)(S.tWorkEndNs <= S.doseStopReq), (int)(S.doseStopReq <= S.doseJoined),
          S.doseInflightAtStop);
  long long maxLate = 0; for (int k = 0; k < S.doseN; k++) maxLate = std::max(maxLate, S.dose[k].ta - S.dose[k].req);
  fprintf(out, "\"dose\":[%d,%d,%lld,%lld,%llu],", S.doseN, S.doseSkipped, maxLate, S.doseJoined - S.doseStopReq,
          S.nReads.load());
  // slow reads (150 ms): no burst, missed requests counted
  g_slowNs = 150000000LL; nccleS2Reset(0); G.t0b = nccleNowNs();
  nccleS2DoseStart(0); struct timespec w2 = {0, 600000000}; nanosleep(&w2, nullptr);
  S.tWorkEndNs = nccleNowNs(); nccleS2DoseStop(0);                       // a 150 ms read is in flight (500-650 ms)
  fprintf(out, "\"dose_slow\":[%d,%d,%lld,%d],", S.doseN, S.doseSkipped, S.doseJoined - S.doseStopReq,
          S.doseInflightAtStop);
  g_slowNs = 1000000LL;
  // ---- post-probe (M4d): 4 edges or 4 periods, bounded by NCCLE_PP_MAX
  nccleS2Reset(0); nccleS2PostProbe(0);
  fprintf(out, "\"pp\":[%d,%d,%d,%lld],", S.ppN, S.ppEdges, S.ppOverflow, S.ppEnd - S.ppStart);
  g_fastConst = 1; g_slowNs = 0; nccleS2Reset(0); nccleS2PostProbe(0);
  fprintf(out, "\"pp_overflow\":[%d,%d,%d,%lld],", S.ppN, S.ppEdges, S.ppOverflow, S.ppEnd - S.ppStart);
  g_fastConst = 0; g_slowNs = 1000000LL;
  // ---- JSON writer
  nccleS2Reset(0); G.t0b = nccleNowNs(); nccleS2DoseStart(0); struct timespec w3 = {0, 250000000}; nanosleep(&w3, nullptr);
  nccleS2DoseStop(0); nccleS2PostProbe(0);
  fprintf(out, "\"gpus\":["); nccleS2WriteGpu(out, 0); fprintf(out, "]}\n");
  fclose(out);
  return 0;
}
'''


def _compile_and_run() -> dict:
    cxx = shutil.which("g++")
    assert cxx, "g++ is required for the native helper tests"
    v21 = "\n".join(_additions(P21)).split("// ===== nccl-energy-frontier methodology-v2 region instrumentation (end)")[0]
    add24 = "\n".join(_additions(P24))
    v24 = add24[add24.index(V24_BEGIN):add24.index(V24_END)]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "h.cc").write_text(PREFIX + v21 + "\n" + v24 + HARNESS)
        subprocess.run([cxx, "-std=c++11", "-Wall", "-Werror", "-Wno-unused-function", "-Wno-unused-variable",
                        "-pthread", str(d / "h.cc"), "-ldl", "-o", str(d / "h")], check=True)
        subprocess.run([str(d / "h"), str(d / "out.json")], check=True, timeout=60)
        return json.loads((d / "out.json").read_text())


_RESULT = {}


def _res():
    if not _RESULT:
        _RESULT.update(_compile_and_run())
    return _RESULT


@strict
def test_native_refuses_invalid_and_holdout_dose_or_m4d():
    r = _res()
    assert r["parse"] == [0, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]
    assert r["cal"] == [100, 1, 5]


@strict
def test_native_counts_every_counter_call_including_the_v21_probe():
    before, probe, edges, installed = _res()["shim"]
    assert before == 2 and installed == 1 and probe > 10 and edges >= 1


@strict
def test_native_pause_dose_pacing_stop_and_postprobe_bounds():
    r = _res()
    assert 37e6 <= r["pause_ns"] <= 37e6 + 3e6
    n, skipped, max_late, stop_ns, reads = r["dose"]
    assert n == 5 and skipped == 0 and max_late < 3e6 and stop_ns < 5e6 and reads == 5
    assert r["order"] == [1, 1, 0]                           # work end <= stop request <= joined; idle at stop
    n2, sk2, stop2, inflight = r["dose_slow"]
    assert n2 == 3 and sk2 == 3 and stop2 < 160e6 and stop2 > 30e6   # never bursts; joins after the in-flight read
    assert inflight == 1                                     # the read in flight at the stop request is logged
    ppn, edges, ovf, dur = r["pp"]
    assert edges == 4 and ovf == 0 and ppn < 2048 and dur <= 4 * 100e6 + 5e6
    ppn2, edges2, ovf2, dur2 = r["pp_overflow"]
    assert ppn2 == 2048 and ovf2 == 1 and edges2 == 0 and dur2 <= 4 * 100e6 + 5e6


@strict
def test_native_json_writer_parses():
    g = _res()["gpus"][0]
    assert g["rank"] == 0 and g["pci"] == "00000000:81:00.0"
    assert g["dose"]["n"] == len(g["dose"]["reads"]) == 2 and all(len(x) == 6 for x in g["dose"]["reads"])
    assert g["post_probe"]["n"] == len(g["post_probe"]["reads"]) and g["post_probe"]["edges"] == 4
    assert g["n_counter_reads"] == g["dose"]["n"] + g["post_probe"]["n"]
    assert "t_work_end_ns" in g and "inflight_at_stop" in g["dose"]
    line = json.dumps({"schema": "nccle-s2-v1", "pid": 1, "gpus": [g]})
    recs, flags = parse_s2_text(line)
    assert len(recs) == 1 and not flags
    assert callable(s2_metrics)
