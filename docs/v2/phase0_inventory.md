# Phase 0 — preservation and inventory (methodology v2)

Date: 2026-10-08 (UTC−4). Scope: read-only inspection. **No GPU experiment was
run**: no CUDA context was created, no NCCL process was launched, and the NVML
energy counter was not read. GPU state was observed only through ordinary
`nvidia-smi` queries and read-only NVML calls (versions, power, limits,
thresholds, throttle reasons, remapped rows, process lists).

## 1. Repository state

| item | value |
|---|---|
| path | `research/Overs/nccl-energy-frontier` |
| branch / HEAD | `master` / `f528f34` ("Full sweep results (3 of 5 trials)…") |
| working tree | clean before Phase 0 (0 modified, 0 untracked); no stash, no remotes, no tags |
| `git fsck --no-dangling` | clean |
| git-ignored | `data/raw/{full,smoke}/{launches/,probe/,probe_end/,calibrate/,launches.jsonl}`, `logs/`, `third_party/` |
| code | `src/ncclenergy/` 12 modules (2,642 lines), `tests/` (parser/statistics tests on real captured output), 6 scripts, 2 configs |
| docs | README, `docs/methodology.md`, `docs/limitations.md`, `docs/smoke_test.md` |
| files added in Phase 0 (untracked) | `provenance/pilot_v1_sha256_2026-10-08.txt` (+ `.sha256`), this file |

## 2. Pilot (methodology v1) data — frozen

| dataset | content | size |
|---|---|---|
| `data/raw/full` | 5,113 records in `launches.jsonl` = 72 probe + 72 calibration + 27 idle + 4,942 trial attempts; 3,934 launch directories (3,528 final + 406 superseded short attempts); 72 probe and 56 calibration directories | 145 MB |
| `data/raw/smoke` | smoke run (2 trials, 3 sizes) | 9.9 MB |
| `data/processed/{full,smoke}` | CSV tables, `qc.json`, results summary | 9.3 MB |
| `figures/{full,smoke}` | 36 PNG | 7.7 MB |

**Integrity checks (all read-only):**

* SHA-256 of 27,491 files (raw + processed + figures) written to
  `provenance/pilot_v1_sha256_2026-10-08.txt`; the manifest's own hash is
  `fba0598e…a52f6b`. No earlier manifest existed for the git-ignored raw files,
  so this is the baseline from now on, not a verification against a past one.
* Every launch directory referenced in `launches.jsonl` exists with all six
  artifacts (full: 3,934/3,934; smoke: 180/180).
* Stored `stdout.txt` re-parses to exactly the recorded rows (0 mismatches).
* Every recorded region energy is reproduced exactly from the stored per-chunk
  counter reads in `chunks.json` (0 mismatches, max |Δ| = 0 J).
* Regenerating `data/processed/full/` from raw with the committed code is
  byte-identical for all 12 files (only the source-path line of
  `results_summary.md` differs).
* nccl-tests binaries and `libnccl.so.2` hashes match `metadata_start/machine.json`.

## 3. Software and hardware visible to the user account

| component | now (2026-10-08) | pilot (2026-10-05/06) |
|---|---|---|
| host | `<host>`, 2× EPYC 7713 (256 threads), 944 GB RAM | same |
| kernel | **7.0.0-38-generic** (booted ≈08:55 today) | 6.11.0-25-generic |
| NVIDIA driver | **580.178.04** (CUDA driver API 13.0), NVML 13.580.178.04 | 550.144.03 (CUDA 12.4) |
| GPUs | 8× RTX A6000, 300 W enforced limit, persistence off, ECC off, compute mode Default, 0 remapped rows (GPU 4 checked) | same |
| reported idle board power (P8) | 7.7–17.0 W per GPU (`nvidia-smi`, one reading) | 20.8–29.1 W (counter, 27 windows) |
| topology | NVLink pairs 0–1, 2–3, 4–5, 6–7 (`NV4`, 4 links × 14.06 GB/s); `NODE` within a socket, `SYS` across; GPUs 0–3 on NUMA 0, 4–7 on NUMA 1 | same |
| CUDA toolkit | `/usr/local/cuda-12.4` only (nvcc 12.4.99) | same |
| NCCL | 2.27.5+cuda12.9 (`nvidia-nccl-cu12` wheel in the `ai_scientist` env; no system NCCL); pip index offers up to 2.32.3 | same |
| nccl-tests | v2.21.1 `afd59ab`, built for sm_86; links `libcudart.so.12` (12.4) and the wheel's `libnccl.so.2` | same |
| Python | 3.11.14 (`ai_scientist` env): numpy 1.26.1, pandas 2.3.3, matplotlib 3.10.7, PyYAML 6.0.1, nvidia-ml-py 13.610.43, torch 2.9.1, scipy 1.11.3; no pytest; `venv` available; PyPI reachable | same |
| DCGM | **unavailable**: no `dcgmi`, no running `nv-hostengine`, no `pydcgm` (not installed — not a permission issue) | — |
| scheduler | **none** (no Slurm/PBS/SGE/LSF/HTCondor commands, no allocation environment) | none |
| GPU processes at inspection | none on any GPU; all 8 GPUs in P8, 1 MiB used | GPU 1 occupied by another user at pilot start |
| disk (`/md127`, holds the repo and `$HOME`) | **7.0 TB, 100 % used, 63 GB free** | 828 GB free |

**Permission-limited observability**

| item | status |
|---|---|
| kernel log (Xid, PCIe/NVLink errors via `dmesg`/journal) | `UNAVAILABLE_DUE_TO_PERMISSIONS` (`kernel.dmesg_restrict=1`; account not in `adm`/`systemd-journal`) |
| NVML Xid event registration | not tested yet (Phase 3, read-only) |
| DCGM telemetry | unavailable (not installed / not running) |
| ECC error counters | not applicable while ECC is disabled; remapped-row counters readable |
| throttle / clock-event reasons, power limits, temperature thresholds (slowdown 95 °C, shutdown 98 °C) | readable via NVML |
| energy counter (`nvmlDeviceGetTotalEnergyConsumption`) | **not checked in Phase 0** (deliberately not read); availability and behaviour on driver 580 to be established in Phase 3/4 |
| owner of foreign GPU processes | PID visible via NVML; owner lookup via `/proc` to be checked in Phase 3 |

## 4. Gaps between the pilot method and methodology v2

| v2 requirement | pilot (v1) status | needed |
|---|---|---|
| warm-ups outside the energy region | warm-ups inside (`-w` iterations within the stdout-delimited region) | instrumented region boundaries after warm-up + sync |
| synchronized region boundaries | host-side reads at stdout chunk arrival | counter reads inside the benchmark process after stream sync / barrier |
| exact operation denominator `R` | `n + w`, inferred | `R` = global collectives in region, recorded with every multiplicative factor (`-m` agg, graph launches, cycles) |
| GPU-valid latency | nccl-tests host timer around synchronized loop | CUDA events at region start/end (and separately, if wanted, per-iteration p50/p95/p99 without energy) |
| actual configuration for every timed process | only probe processes (TUNING log) | per-process evidence (profiler plugin or per-process log) or label `UNVERIFIED` |
| calibration separate from trials, frozen `R` | calibration separate, but `R` was re-scaled by short-region retries (406) | frozen `R`; predefined QC; no outcome-dependent re-runs |
| topology controls | 2-GPU NVLink and 4-GPU only (count and topology confounded) | add 2-GPU cross-pair PCIe control |
| idle pre/post per block | start/end/mid-trial idle | fixed-duration pre and post per block, both kept |
| block count justified | 5 planned, 3 run | derived from exploratory variance |

## 5. Risks

1. **System under test changed.** Driver 550 → 580, kernel 6.11 → 7.0,
   and two reboots since the pilot (one unclean on 2026-10-07). Pilot and
   v2 data must not be pooled. On 550 we had observed that reading the energy
   counter can drop the energy of the window in progress; this has to be
   re-measured on 580, not assumed.
2. **Disk space.** `/md127` is at 100 % (63 GB free) and shared. The expected
   v2 data volume is small (estimate 1–3 GB), but other users can fill the
   volume. The plan needs a hard stop floor (proposed 20 GB) and bounded
   telemetry per region.
3. **No scheduler.** There is no formal allocation, and an idle GPU is not
   authorized, so the GPU IDs must come from you.
4. **Energy-region instrumentation.** v2 needs counter reads at synchronized
   boundaries inside the benchmark process. Proposed approach: a minimal,
   version-controlled patch to the pinned nccl-tests. It reads NVML counters
   and records CUDA events after warm-up, sync and barrier, and again after
   the timed loop's sync, and adds an option to skip the in-place pass.
   The patch itself must be validated: warm-up exclusion, energy linear in
   `R`, and timing agreement with unpatched nccl-tests.
5. **Per-process configuration evidence.** Per-call TUNING logging distorts
   small-message latency. Candidates:
   - A small NCCL profiler plugin built in user space against the NCCL 2.27.5
     profiler API. It would record (function, algorithm, protocol, channels)
     per collective as a cheap histogram; its overhead must be measured.
   - Switching to NCCL ≥ 2.28, where nccl-tests `-U` reports the
     configuration. This changes the NCCL under test away from the one
     PyTorch 2.9.1 uses.
6. **4-GPU variance.** In the pilot, the 4-GPU set (NVLink pairs joined
   through host memory) had heavy-tailed block-to-block variation. The SD of
   the paired log-energy difference vs auto was 9 % at the 90th-percentile
   cell. Detecting a 5 % effect there would need about 30 blocks (74 with
   multiplicity control), versus about 5–10 on the NVLink pair. Either the
   variance source must be reduced or identified in exploratory work, or the
   confirmatory endpoints and effect size must be chosen with this in mind.
7. **Runtime on a shared server.** A pilot block took 2.5 h; about half of
   that was the in-place pass, plus process start-up. v2 adds a topology
   control. A confirmatory matrix restricted to auto plus Pareto candidates,
   out-of-place only, is needed to keep blocks short.
8. **Telemetry gaps.** Xid/kernel logs and DCGM are unavailable, so health
   checks rely on NVML (throttle reasons, clocks, temperature, power limit,
   remapped rows, processes) and on application/NCCL logs.
9. **Pilot immutability.** The raw pilot files are git-ignored. Today's
   manifest makes changes detectable but does not prevent them. Option:
   remove write permission on the pilot directories (the account's own files;
   needs your OK).
10. **Deployment realism.** Single-process multi-thread nccl-tests uses
    `P2P/direct pointer`; PyTorch uses one process per GPU. This matters for
    Phase 10 transfer.
11. **NCCL version for the tuner and the workload.** Keeping NCCL 2.27.5
    preserves comparability with PyTorch 2.9.1's NCCL. Moving to a newer
    NCCL would need the workload to load the same library.

## 6. Proposed structure (additive; the pilot is not moved or edited)

```
configs/v2/{calibration,exploratory,confirmatory}.yaml
patches/nccl-tests-afd59ab-v2-region.patch      # synchronized region + NVML reads + OOP-only option
native/nccl_profiler_algo/                      # user-space NCCL profiler plugin (algo/proto evidence)
src/ncclenergy/v2/                              # manifest, accounting, telemetry, qc, topology, attempts, analysis, recommender
tests/v2/                                       # CPU-only tests (no GPU)
data/v2/raw/<campaign>/<block>/<attempt>/       # append-only, never edited
data/v2/processed/<campaign>/
docs/v2/{phase0_inventory,protocol,prereg,phase reports}.md
provenance/                                     # checksum manifests, frozen protocol/code hashes
```

The pilot stays at `data/raw/{full,smoke}` and `data/processed/{full,smoke}`
and is labelled "pilot v1 (driver 550)" in the README.

## 7. Phased plan

| phase | deliverable | GPU use |
|---|---|---|
| 1 | `docs/v2/protocol.md`: region definition, `R` accounting, latency definition, configuration evidence, idle protocol, topology set, QC/stop rules, endpoints and statistical model (with the decisions below filled in) | none |
| 2 | patch + profiler plugin source, `src/ncclenergy/v2/*`, CPU-only tests (unit conversions, warm-up exclusion, exact `R`, participating-GPU sums, inactive-GPU exclusion, missing/irregular samples, counter reversal/non-update/wrap, partial logs, unsupported configs, failed attempts, retry-selection bias); build in user space only | none |
| 3 | read-only preflight report on the GPUs you assign; runtime/disk estimates; **stop for approval** | read-only |
| 4 | minimal smoke: synchronization, `R`, warm-up exclusion, per-process configuration, counter availability/resolution/update period on 580, counter-read cost, sampling intervals, counter vs integration, cleanup | ≤ 4 GPUs, short |
| 5 | bounded exploratory screen (`EXPLORATORY`): supported configs, safe sizes, frozen `R`, candidate Pareto set, block-level variance | ≤ 4 GPUs |
| 6 | frozen confirmatory protocol + hashes; **stop for approval** | none |
| 7–11 | as specified (confirmatory blocks one at a time, analysis, recommender, tuner/workload, replication package) | per approval |

## 8. Decisions needed from you

1. **Which physical GPUs may I use?** There is no scheduler, so I won't
   assume. The pilot used GPUs 4–7, which allow: NVLink pair 4–5,
   cross-pair PCIe control 5–6 (`NODE`), and 4-GPU set 4–7.
2. **Endpoints.** What counts as a meaningful energy effect, and what is the
   latency SLO? Proposed: ≥ 5 % raw joules/collective vs NCCL automatic;
   SLO = latency ≤ 1.05 × NCCL-automatic latency in the same block.
3. **NCCL version.** Keep 2.27.5 (proposed), so microbenchmarks and the
   PyTorch workload use the same library?
4. **Disk.** May v2 data go under the repo on `/md127`, with a 20 GB
   free-space stop floor?
5. **Commits.** May I commit the Phase 0 files (checksum manifest and this
   inventory) to the local repository? No push.
