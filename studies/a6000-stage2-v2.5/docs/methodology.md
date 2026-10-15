# Stage 2 measurement and inference

The implemented method is `v2.5-stage2-region`. The pinned native patch and the retained records, not current upstream defaults or a README summary, define this experiment. No original estimator, operation count, acceptance criterion or frozen decision is changed by the public export. [Stage 2 (2026/10), [native patch](../code/patches/nccl-tests-afd59ab-v2.5-stage2.patch), [held-out protocol](../protocol/STAGE2_HELDOUT1_PROTOCOL.md)]

## Experimental unit and timing

A launch is a fresh nccl-tests process containing one out-of-place AllGather measurement region. The region repeats the collective `R` times on two ranks. Preparation (`w` iterations, targeting approximately five seconds), validation and setup occur outside the timed energy region. The two participating GPUs' energies are summed and divided by `R`, not `2R` or `R+w`. Profiler counts of `R+w+1` include preparation and the validation call; those are not independent statistical replicates. [Stage 2 (2026/10), [region accounting](../code/src/ncclenergy/v2/stage2_energy.py), [execution](../prep/heldout_final/execution.tsv)]

| Arm | Preparation w | R: 2.5 s | R: 10 s | R: 20 s | R: 40 s |
|---|---:|---:|---:|---:|---:|
| A | 95,604 | 47,802 | 191,207 | 382,413 | 764,825 |
| B | 95,597 | 47,799 | 191,193 | 382,385 | 764,769 |
| C | 98,289 | 49,145 | 196,577 | 393,154 | 786,308 |

These exact counts are identical across sessions. Region durations are targets, not fixed wall-clock values; actual timings are recorded. [Stage 2 (2026/10), [protocol counts](../protocol/STAGE2_HELDOUT1_PROTOCOL.md), [session 1 config](../code/configs/stage2/heldout_s1.yaml)]

The patch records CUDA-event elapsed milliseconds for each rank, after synchronizing the relevant device work. The recorded GPU latency is `max(event_ms)/1000/R` seconds per collective. The host seconds-per-collective field and nccl-tests stdout timing are separate quantities. H-D1–H-D4 are energy endpoints; they are not a new confirmatory latency-optimum analysis. [Stage 2 (2026/10), [region_metrics](../code/src/ncclenergy/v2/region.py), [native patch](../code/patches/nccl-tests-afd59ab-v2.5-stage2.patch), [held-out endpoints](../code/src/ncclenergy/v2/stage2_heldout.py)]

NCCL enqueue calls are asynchronous; a CPU enqueue duration alone is not execution latency. CUDA-event timing is expressed in milliseconds. These general contracts explain why synchronization matters; whether it occurred here is established by the pinned patch, not by citing the API. [NVIDIA (accessed 2026/10), [NCCL stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html), [CUDA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)]

## Energy accounting

The native patch calls `nvmlDeviceGetTotalEnergyConsumption` per participating GPU and retains raw integer readings and read-call timestamp brackets. NVIDIA defines this API's output as cumulative millijoules since the driver was reloaded; the documented interface does not establish this project's absolute accuracy, counter-update cadence or a read-reset mechanism. [NVIDIA NVML (accessed 2026/10), [Device Queries, TotalEnergyConsumption](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html); Stage 2 (2026/10), [native patch](../code/patches/nccl-tests-afd59ab-v2.5-stage2.patch)]

For GPU `g`, the exact implemented equations are:

```text
C0, C1, Cpost                    raw counter readings [mJ]
t1mid = (t_end_a + t_end_b)/2     end-read midpoint [ns, monotonic]
Twork                            synchronized work-end timestamp [ns]
Tcommit                          predicted counter commit timestamp [ns]
P                                period used by this estimator [ns]

Eraw_g    = (C1_g - C0_g)/1000                                [J]
post_s_g  = (Tcommit_g + P_g - t1mid_g)/1e9                   [s]
Pctx_g    = ((Cpost_g - C1_g)/1000)/post_s_g                  [W]
tail_s_g  = (Tcommit_g - Twork_g)/1e9                         [s]
Ealign_g  = Eraw_g - Pctx_g * tail_s_g                        [J]
Ealign    = sum_g Ealign_g                                   [J]
J/op      = Ealign/R                                         [J/op]
```

Primary uses the per-GPU guarded period estimate `P_hat`; the nominal sensitivity uses 100 ms. `Tcommit` remains the native code's phase-grid prediction, not a directly observed work-end update. The primary period changes the post-window denominator, not the recorded prediction. A failed API status, decreasing counter, missing period or invalid window is not repaired by clipping or by promoting a sensitivity estimator. [Stage 2 (2026/10), [e_aligned_gpu_J](../code/src/ncclenergy/v2/stage2_energy.py), [unit_energies](../code/src/ncclenergy/v2/stage2_calibration.py), [period guard](../code/src/ncclenergy/v2/stage2_phase.py)]

The tail subtraction is a model-based accounting correction. `Pctx × tail_s` has units W×s=J, but dimensional consistency does not establish physical accuracy. The native pre-region probe and boundary reads, the seeded 0–100 ms pause and their recorded timing remain part of the method. A separate counter before preparation reports preparation-to-C0 energy; it does not prove that reading is non-perturbing. Held-out mode has no additional in-region dose reader and no post-region M4d probe. [Stage 2 (2026/10), [protocol](../protocol/STAGE2_HELDOUT1_PROTOCOL.md), [s2_metrics](../code/src/ncclenergy/v2/stage2_energy.py)]

## Sampled-power and idle measurements

The sampler's nominal query interval is 50 ms. The sampled sensitivity trapezoidally integrates `POWER_INSTANT` per participating GPU, from the C0 read midpoint to that GPU's synchronized work end, using each power query's monotonic bracket midpoint. Repeated values are retained; only the two window edges are linearly interpolated between bracketing samples. Missing coverage or an in-window gap greater than 150 ms makes that estimate unavailable, rather than extrapolating. [Stage 2 (2026/10), [telemetry sampler](../code/src/ncclenergy/v2/telemetry.py), [unit_energies](../code/src/ncclenergy/v2/stage2_calibration.py), [sampled_energy_trapezoid_J](../code/src/ncclenergy/v2/stage2_energy.py)]

NVML field timestamps are retained in their reported realtime domain; they are not subtracted from monotonic execution timestamps or assumed to be physical sensor acquisition times. Query frequency, observed value-change frequency and sensor-update frequency are different quantities. [Stage 2 (2026/10), [clock-domain types](../code/src/ncclenergy/v2/stage2_energy.py), [telemetry fields](../code/src/ncclenergy/v2/telemetry.py)]

P8 baselines are recorded at session start and end, with a 10 s settle plus 30 s measurement and a maximum 60 s entry wait. They are not subtracted from the primary held-out endpoint. Thus the endpoint includes background board consumption during communication; it is neither pure dynamic communication energy, physical-link energy nor whole-system energy. [Stage 2 (2026/10), [runner baseline](../code/src/ncclenergy/v2/stage2_runner.py), [session baselines](../code/data/stage2/v2.5-heldout), [primary analysis](../code/src/ncclenergy/v2/stage2_heldout.py)]

## Statistics and checks

Within each block, form the prespecified log ratio. Average available block contrasts within each session, then summarize the three session means using a t distribution with df=2. No iteration, rank or telemetry sample is treated as an independent session. Randomization reduces a fixed order association but does not establish independence across sessions or shared-host conditions. [Stage 2 (2026/10), [session_summary](../code/src/ncclenergy/v2/stage2_decisions.py); Kalibera and Jones (2013/06), [benchmark repetition levels](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf), §3]

H-D1 has nine contrasts and H-D3 three; their difference and minimum-effect tests use Holm adjustment with the original family size retained when a member is unavailable. Descriptive 95% CIs are not simultaneous family-wise intervals. TOST uses prespecified log margins and a 90% CI at alpha=0.05; nonsignificance of a difference test is not equivalence. The margins are project choices, not physical error bounds supplied by a paper. [Stage 2 (2026/10), [family and evidence](../code/src/ncclenergy/v2/stage2_decisions.py); Lakens (2017/05), [TOST primer](https://pure.tue.nl/ws/portalfiles/portal/80918653/lakeequi2017.pdf), Figure 1]

The live runner applies process, temperature, telemetry, storage, owned-process cleanup and readiness guards. Offline analysis verifies retained acceptance and scope records and recomputes R2/R3; R1's `READY` is checked as a recorded decision, not fully reconstructed from its trace. These mechanisms and successful execution checks support retention of the data, not proof of absolute physical accuracy. [Stage 2 (2026/10), [runner](../code/src/ncclenergy/v2/stage2_runner.py), [held-out validation](../code/src/ncclenergy/v2/stage2_heldout.py)]
