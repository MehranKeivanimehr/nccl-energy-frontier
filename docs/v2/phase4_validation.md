# Phase 4 — minimal smoke validation (2026-10-08)

> **v2.2 audit note (2026-10-09).**
> * **Counter behaviour**: the description below ("commits on a fixed
>   100 ms clock", "a read discards …") is the interpretation that these
>   local experiments support. NVML documents no such guarantee, and it
>   was measured only on this host and driver.
> * **Period model**: the native region QC checks observed edges against
>   the assumed 100 ms grid. The Python aligned meter also assumes 100 ms;
>   this is not an independent per-run estimate of under-load cadence.
> * **Read-timing uncertainty**, derived from the recorded read brackets:
>   about 0.85 % of the tail correction, and ≤ 0.10 % of region energy.
>   See [`v22_audit/stage2_report.md`](v22_audit/stage2_report.md) §4.1.
> * **Idle period overstated.** The table row "commits every 100.00 ms idle
>   and under load (10th–90th percentile 99.998–100.002 ms)" overstates the
>   idle evidence.
>   * Under load the raw record (`validation.jsonl`,
>     `load_counter_updates`, 3 runs) supports it.
>   * The single P8-idle run (`idle_counter`) gave median 101.3 ms,
>     p10–p90 93.5–101.5 ms, so the idle period was not resolved.
>   * See [`v22_audit/stage3_plan_rev1.md`](v22_audit/stage3_plan_rev1.md)
>     §2.

GPU work: GPUs 4–7 only (≤ 4 concurrent, one process at a time), all idle
before and after every run. Raw data:
* `data/v2/raw/validation/`: round 1, patch v2.0;
* `data/v2/raw/validation_v21/`: one failed attempt, harness path bug, kept;
* `data/v2/raw/validation_v21b/`: round 2, patch v2.1;
* `data/v2/raw/setup/probe/`: allowed-set probes.

None of this is trial data.

## 1. Results

| check | result | verdict |
|---|---|---|
| region record written, synchronized boundaries | one record per process; event time = host-timer time (ratio 0.99999–1.00001, 42 regions) | PASS |
| exact operation count | profiler plugin counts = `w + R + 1` (warm-up + timed + data check) on every rank in 30/30 runs | PASS |
| warm-up exclusion | J/collective at R = 40k: w = 1k vs 20k → +0.7 % and 0.0 % (v2.1) | PASS |
| actual NCCL selection | plugin VERIFIED 30/30; plugin vs NCCL TUNING log agree in 90/90 (rank × size) entries over 4 KiB–64 MiB on 2 and 4 GPUs | PASS |
| plugin overhead | AllReduce t_gpu with/without plugin (3 ABAB reps): 2 GPU 4 KiB +0.03 %, 1 MiB −0.03 %; 4 GPU 4 KiB +0.26 %, 1 MiB +0.86 % (one rep +3.0 %, others −0.3 %, −0.05 %) | PASS (< 2 %) |
| energy counter available | supported on GPUs 4–7, return code 0 for every read | PASS |
| counter resolution / update | increments of ≥ 3 mJ; commits every **100.00 ms** idle and under load (10th–90th percentile 99.998–100.002 ms) | measured |
| counter read cost | 1.6 ms median, 8.3 ms p90, 16.6 ms max per read (driver 580) | measured |
| **read-induced loss** | 100 Hz counter reads on GPU 4 during 4 s of a 10.6 s region: GPU 4 / GPU 5 energy ratio fell from 1.035–1.049 to 0.699–0.775 (3 reps) | **counter reads are destructive on 580** |
| `POWER_INSTANT` reads | 100 Hz reads: ratio 1.036–1.052, vs control 1.035–1.049 (v2.0); and 1.045–1.051 vs 1.048–1.049 (v2.1) | non-destructive on 580 |
| v2.0 boundary-read bias | linearity intercepts −11.4 J and −16.7 J per 2-GPU region; J/collective rises with R (11.01 → 11.69 mJ) | **FAIL → fixed** |
| v2.1 commit-aligned end read | intercepts **+1.6 J and +3.0 J**; J/collective flat in R (11.76/11.69/11.68 and 11.84/11.79/11.74 mJ at R = 20k/40k/80k) | PASS |
| v2.1 phase/tail diagnostics | 4 commit edges per GPU, brackets ≈ 2 ms, spacing 99.65–100.49 ms; tails 3–95 ms at 77–90 W (median 4.1 J removed); no QC flags in 30 GPU-regions | PASS |
| counter vs sampled power (v2.1) | ∫`POWER_INSTANT` − counter = **−10.5 J per GPU per region and −0.6 % of energy**; ratio 0.987 median for ≥ 5 s regions; ∫`PowerUsage` (lag 0.75 s) ratio 0.973 median | observed disagreement; sensor/sampling lag was a hypothesis, not an established cause (superseded by gate 3b matched-window analysis) |
| sampling intervals | 50 ms median, no gaps > 150 ms in validation regions | PASS |
| cleanup / post-run health | no experiment process left; GPUs back to P8; allocation check clean after every experiment | PASS |

## 2. Method change made in Phase 4 (patch v2.0 → v2.1)

On driver 580 the cumulative counter commits on a fixed 100 ms clock. A read
returns only committed energy and discards what has accumulated in the
current window. Under v2.0 the end-of-region read therefore lost the busy
energy between the last commit and the loop end. That is 0–100 ms, about
60 ms on average: roughly −3 % at 2 s, plus ±1.4 % noise.

Patch v2.1 (`patches/nccl-tests-afd59ab-v2.1-region.patch`) works per GPU:
1. During warm-up, it polls the counter to bracket commit edges and
   estimate the phase. This is outside the region, so the destructive reads
   do no harm.
2. The start read is taken as before. It discards only energy accumulated
   before the region.
3. After the timed loop and sync, it waits for the next commit and takes the
   end read 8 ms later. The delta now contains the region plus an idle tail
   (loop end → commit).
4. One further commit measures the idle-with-context power, and the tail
   energy (`P_ctx × tail`) is subtracted.

All inputs are recorded, so the correction is recomputable offline. If a
phase cannot be established, or is inconsistent or imprecise, the region is
`MEASUREMENT_INVALID`; it is never silently uncorrected.

The remaining intercept of +1.2 J per GPU implies a bias of ≤ +0.5 % at the
chosen region length. It is reported and not corrected.

## 3. Placement change: P2_PCI → P2_PCI_SHM

Under NCCL defaults, GPUs 5–6 (PCIe `NODE`, no NVLink) use `P2P/direct
pointer`, and the first collective never completed: 3/3 auto probes hit the
120 s timeout. `nvidia-smi topo -p2p` still reports P2P read/write "OK"
for the pair. After each kill the GPUs returned to P8 with no process left.
The probe command did not stop after the first timeout; it now does. PCIe P2P
was not exercised again.

The host-path control is therefore run with `NCCL_P2P_DISABLE=1`, which gives
`SHM/direct/direct`, the transport NCCL itself uses for the cross-pair links
inside `P4_MIX`. Its "auto" means NCCL's choice with P2P disabled.

## 4. Allowed configuration sets (from NCCL's init tuning tables)

| placement | AllReduce | AllGather / ReduceScatter |
|---|---|---|
| P2_NVL | auto, Ring/Tree × {Simple, LL, LL128} | auto, Ring × {Simple, LL, LL128} |
| P2_PCI_SHM | auto, Ring/Tree × {Simple, LL} | auto, Ring × {Simple, LL} |
| P4_MIX | auto, Ring/Tree × {Simple, LL} | auto, Ring × {Simple, LL} |

Tree for AllGather/ReduceScatter: rejected by NCCL ("no algorithm/protocol
available") on all placements. LL128 is not enabled by NCCL on P2_PCI_SHM or
P4_MIX and is not forced.

## 5. Settings frozen for Phase 5

`T* = 2.5 s`, `T_min = 2.0 s`, sampler 50 ms with `POWER_INSTANT` and
`PowerUsage`, PowerUsage cross-check lag 0.75 s, patch v2.1, plugin
`nccle-algo-v1`.

## 6. End-to-end smoke block (`configs/v2/smoke.yaml`, `data/v2/raw/smoke/blocks/block_01`)

* **Plan and status.** 48 planned units: 36 executed, all `OK`; 12 recorded
  `NOT_RUN_NOT_ENABLED_BY_NCCL` (Tree for AllGather; LL128 on P4_MIX).
  Block `COMPLETE`.
* **Region quality.** No QC flags or reasons. Region length 2.48–5.90 s;
  maximum temperature 46 °C. ∫`POWER_INSTANT` / counter median 0.981
  (0.915–0.994); idle tail median 12.6 J per region.
* **Idle baselines (pre/post, W).** GPU4 28.74/29.06, GPU5 20.81/20.95,
  GPU6 26.71/26.60, GPU7 21.19/21.07.
* **Evidence-matching bug found and fixed.** The runner first marked all 15
  AllGather units `UNVERIFIED`. NCCL runs AllGather internally as `ncclInt8`
  with `count × 4`, and the verifier compared element counts. Matching is now
  on bytes, and the analysis re-derives evidence from the stored profiler
  files: **36/36 VERIFIED**, one executed `(algo, proto, channels)` per unit.
* **Throttle-counter quirk.** The block's cumulative SW-thermal counter grew
  by 79 s on GPU 4 and 143 s on GPU 6, yet none of about 20,000 samples
  (units and idle baselines, 20 Hz) showed the SW-thermal reason bit.
  A separate 30 s read-only check at pure idle (P8, no process) showed the
  SW-thermal counter growing at 1 s/s on GPUs 4 and 6, and the SW-power-cap
  counter on GPUs 4, 5 and 6. This is an idle reporting artefact of this
  driver/board combination. Block health events therefore use the HW-thermal
  and HW-power-brake counters (unchanged at 0), the per-sample reasons during
  regions, and temperature.
