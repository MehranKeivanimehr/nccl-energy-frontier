# Phase 3 — unprivileged preflight (2026-10-08)

Raw snapshot: `data/v2/raw/preflight/preflight_phase3.json`. All checks used
ordinary NVML reads, `nvidia-smi` query modes and `/proc`; nothing changed GPU
state. One energy-counter read per allocated GPU was made to confirm support.

## Allocation

| item | value |
|---|---|
| scheduler | none; allocation by user instruction "use only idle GPUs, ≤ 6 at once" |
| proposed GPUs | **4, 5, 6, 7** (NUMA node 1); aliases `A6000-b10caa4a`, `A6000-f5f81ac3`, `A6000-cf04a46e`, `A6000-8f2812cc`; PCI `81:00.0`, `a1:00.0`, `c1:00.0`, `e1:00.0` |
| state at preflight | all 8 GPUs: no compute/graphics process, P8, 21–23 °C, 1 MiB used (+601 MiB driver-reserved, identical on all GPUs), 300 W enforced limit, max graphics clock 2100 MHz, application clock 1800 MHz, 0 remapped rows |
| concurrency | ≤ 4 GPUs; one experiment process at a time; no 6-GPU placement planned |

**Correction made during preflight.** The first snapshot flagged 602 MiB
"used" on every GPU. NVML's v2 memory API shows this is 601 MiB of
driver-reserved memory plus 1 MiB used, so it is not a hidden process. The
allocation check now uses the v2 `used` value.

## Telemetry available to the account

| telemetry | status |
|---|---|
| energy counter `nvmlDeviceGetTotalEnergyConsumption` | supported on GPUs 4–7 |
| power, temperature, clocks, utilization, P-state, enforced limit, process list | readable |
| current clock-event (throttle) reasons | readable |
| cumulative clock-event counters (SW power cap, sync boost, SW/HW thermal, HW power brake) | readable via NVML field values 74/76/269/270/271; read pre/post block only |
| NVML Xid critical-error events | registration works on GPUs 4–7; polled after every unit |
| process owner | `/proc/<pid>/status` (owner shown when readable, else `UNAVAILABLE_DUE_TO_PERMISSIONS`) |
| kernel log (Xid / PCIe / NVLink messages) | `UNAVAILABLE_DUE_TO_PERMISSIONS` (`dmesg_restrict=1`) |
| DCGM | not installed (no `dcgmi`, no hostengine) |
| ECC error counters | ECC disabled on these boards |

## Health observations before any v2 GPU work

* **SW power cap flagged at idle.** GPU 6, and occasionally GPU 0, report
  `SW Power Cap: Active` in P8 at ~14 W against a 300 W limit. The cumulative
  SW-power-cap counter is ≈ 31,000 s since boot on GPUs 4 and 6, and
  ≈ 1,450–1,800 s on GPUs 5 and 7. This is treated as a reporting artefact of
  this driver at idle. During a region, SW power cap stops the block only if
  the region's power is ≥ 90 % of the enforced limit; otherwise it is
  recorded as `SW_POWER_CAP_FLAG_BELOW_LIMIT`.
* **SW thermal slowdown since boot** (before any work by this account):
  GPU 4 7.0 s, GPU 6 150.0 s; GPUs 5 and 7 0 s. HW thermal and HW power
  brake are 0 on all four. Each block records the counter difference over the
  block; any HW/SW thermal or power-brake increase is a health event.

## Topology categories (parsed from `nvidia-smi topo -m`)

| placement | GPUs | category | links |
|---|---|---|---|
| `P2_NVL` | 4,5 | `NVLINK_ALL` | NV4 (4 active links each, 14.06 GB/s per link) |
| `P2_PCI` | 5,6 | `PCIE_ONLY` | NODE |
| `P4_MIX` | 4–7 | `MIXED_NVLINK_PCIE` | NV4 within 4–5 and 6–7, NODE across |

## Proposed matrix and cost

* **Exploratory block**: every allowed configuration × 21 sizes × 3
  placements × 3 collectives, about 780 units. The allowed set comes from
  NCCL's tuning table: LL128 is expected only on `P2_NVL`, and Tree only for
  AllReduce. Expected about 60–65 min per block. Two exploratory blocks are
  planned.
* **Calibration**: about 75 short processes, about 15 min. **Phase 4
  validation**: about 15 min. **Confirmatory**: reduced matrix (auto +
  candidates), about 25–35 min per block. The block count is set in Phase 6.
* **Disk**: about 40–50 MB per exploratory block (telemetry about 15 KB/unit
  gz). Free space on `/md127`: 67 GB; stop floor 20 GB.

## Timeout and stability policy

* Per-process timeout `max(120 s, 10 × expected)`. A timeout stops the block.
* A block starts only with all allocated GPUs idle, in P8 (≤ 300 s wait), and
  < 45 °C. A 30 s idle baseline is taken before and after each block.
* Stop on any condition in `protocol.md` §9, with the SW-power-cap refinement
  above.

## Calibration plan

1. Phase 4 validation on driver 580:
   * counter update period and resolution;
   * read-induced loss (dose test with extra counter reads and with
     `POWER_INSTANT` reads during a region);
   * `PowerUsage` averaging lag;
   * energy linear in R;
   * warm-up exclusion;
   * profiler-plugin overhead and agreement with the TUNING log.
2. Choose `T*` and `T_min` from the measured counter granularity.
3. Probes (allowed sets) → calibration processes → frozen schedule
   (`calibration_frozen.json`, hashed).

The user's instruction of 2026-10-08 ("complete all the phases … use only
idle GPUs, ≤ 6 at once") is taken as the approval for this gate.
