# Stage 2 results

These are conditional comparisons of NVML-reported participating-GPU board energy on one RTX A6000 NVLink pair and host. They do not replace the older Gate 3b results and are not pooled with the original pilot, v2 exploration/confirmation, smoke or calibration launches. Calibration supplies the prespecified readiness rules and Q1 status, not additional held-out replicates. [Stage 2 (2026/10), [held-out protocol](../protocol/STAGE2_HELDOUT1_PROTOCOL.md), [analysis code](../code/src/ncclenergy/v2/stage2_heldout.py)]

## Execution and evidence

| Stage | Completed launches | Statistical role |
|---|---:|---|
| Smoke | 3 | Pipeline and dose-logging check; not a sensor-accuracy experiment |
| Calibration | 23 | One session; preparation, read-dose, period/phase and readiness characterization |
| Held-out session 1 | 24 | 2026-10-14, 00:42:51–01:26:03 UTC |
| Held-out session 2 | 24 | 2026-10-14, 01:39:12–02:20:25 UTC |
| Held-out session 3 | 24 | 2026-10-14, 02:28:30–03:10:35 UTC |

The three held-out sessions are consecutive same-evening runs on the same pair, not demonstrated independent samples across days, hosts or GPU pairs. All 72 were accepted, with distinct launch PIDs and no recorded stop, retry or replacement. GPU query snapshots did not report a competing process on the measured pair; other jobs were recorded on GPUs 0–2. Those checks do not establish exclusive host access or rule out every event between polls. [Stage 2 (2026/10), [session summaries](../code/data/stage2/v2.5-heldout), [execution](../prep/heldout_final/execution.tsv), [protocol](../protocol/STAGE2_HELDOUT1_PROTOCOL.md)]

## Principal comparisons

All percentages below are `100 × (exp(mean log ratio) − 1)`. The 95% intervals are descriptive and unadjusted. Difference and minimum-effect decisions use the frozen family rules; each TOST p is unadjusted. [Stage 2 (2026/10), [contrasts](../prep/heldout_final/contrasts.tsv), [decision code](../code/src/ncclenergy/v2/stage2_decisions.py)]

| Endpoint | Primary effect, 95% CI | Recorded outcome |
|---|---|---|
| H-D3, B/A at 10 s | −0.1909% [−0.4497%, +0.0685%] | `COMPATIBLE_WITH_EQUIVALENCE`; 90% CI [−0.3666%, −0.0149%], TOST p=0.01819 |
| H-D3, B/A at 20 s | +0.2180% [−0.5439%, +0.9857%] | `THERMAL_COMPARABILITY_NOT_ESTABLISHED` |
| H-D3, B/A at 40 s | +0.1698% [−0.9539%, +1.3062%] | `INSUFFICIENT_PRECISION`; log-scale half-width 1.128% |
| H-D4, C/A at 40 s | +1.6454% [+1.0047%, +2.2901%] | `RESOLVED_MEANINGFUL_DIFFERENCE`; precision target nevertheless missed |

For H-D4, session effects were +1.456%, +1.940% and +1.540%; their log-scale SD was 0.2545 percentage points. The 95% half-width was 0.6323 log-percentage points, above the prespecified limit `100 × ln(1.004)` (approximately 0.40). All four estimators give a positive effect of approximately +1.62% to +1.65%. They share the same NVML infrastructure and launches; their agreement is a sensitivity result, not four independent physical calibrations. [Stage 2 (2026/10), [all contrasts](../prep/heldout_final/contrasts.tsv), [precision and adequacy](../prep/heldout_final/heldout_analysis.json)]

For H-D3(10 s), the margin is exactly `[1/1.005, 1.005]` on the ratio scale. Compatibility concerns the conditional mean ratio under the session-level model; it does not bound the error of each energy reading. At 20 s, the recorded B/A pre-region temperature imbalance reaches 1.0°C, so the frozen thermal override takes precedence over a numerical effect estimate. [Stage 2 (2026/10), [thermal checks](../prep/heldout_final/thermal_control.tsv), [readiness code](../code/src/ncclenergy/v2/stage2_readiness.py)]

## Outcomes that remain unresolved

H-D1 tests nine duration contrasts against 40 s. Eight are `INSUFFICIENT_PRECISION`. B at 2.5 s is approximately −0.37% [−0.42%, −0.31%]; its difference test survives Holm (p≈0.011), but its minimum-effect test does not (p≈0.295). The recorded conclusion is `DIFFERENCE_MAGNITUDE_UNRESOLVED`, not proof of no length effect. H-D2 is approximately +0.20% [−0.47%, +0.88%] and remains `INSUFFICIENT_PRECISION`. [Stage 2 (2026/10), [contrasts](../prep/heldout_final/contrasts.tsv)]

All five aggregate scopes have conditional statistical status `NOT_MET` and overall status `NOT_ESTABLISHED`. Calibration Q1 was already unbounded, and other held-out criteria also failed. Every systematic-error component remains unknown; no hard physical error bound or external energy reference was established. The primary unavailability, precision targets and thermal rules were not relaxed after collection. [Stage 2 (2026/10), [adequacy](../prep/heldout_final/adequacy.tsv), [calibration analysis](../prep/calib1/run/calibration_analysis.json)]

Primary energy is unavailable for `s2-b2-p08` (B, 2.5 s) and `s3-b2-p11` (B, 20 s), owing to the frozen period-estimate guard. Three primary block contrasts therefore use the remaining block in their respective session, as prespecified; no value is imputed. The other three estimators remain available for both launches. [Stage 2 (2026/10), [exclusions](../prep/heldout_final/exclusions.tsv), [unit estimates](../prep/heldout_final/units.tsv)]

The sampled-power sensitivity reports approximately −3.8% to −5.0% at 2.5 s versus 40 s, unlike the counter estimators. This is observed estimator disagreement. Thermal evolution, clock behavior, window alignment and reporting dynamics have not been experimentally separated, so a sensor-only mechanism is not established. The sampled estimator is not promoted to the primary after observing the results. [Stage 2 (2026/10), [sensitivity contrasts](../prep/heldout_final/contrasts.tsv), [energy code](../code/src/ncclenergy/v2/stage2_calibration.py)]

The frozen thermal check flags one difference of −0.5000000000000036°C at the nominal 0.5°C boundary. This floating-point artifact affects no principal H-D3/H-D4 verdict; it remains disclosed rather than being repaired in frozen code. Historical D1–D4 replay, secondary covariate analysis, sampled-hold sensitivity, delay-shifted estimation and systematic bounds are explicitly `NOT_COMPUTED`. [Stage 2 (2026/10), [thermal table](../prep/heldout_final/thermal_control.tsv), [analysis disclosures](../prep/heldout_final/heldout_analysis.json)]

## Calibration observation

Additional in-region reads at 10 Hz and 2 Hz changed raw-counter reported energy relative to E0 by −10.2874% [−12.4011%, −8.1228%] and −2.1154% [−2.9935%, −1.2293%], respectively. These are descriptive geometric changes over three paired rounds in one calibration session. They are not physical energy savings: E0 itself includes probe and boundary reads, sampled-power responses are inconclusive, and the primary Q1 comparison has only two complete rounds (`INSUFFICIENT_ROUNDS`). The observation establishes a local read-dose dependence in reported counter energy, not a documented counter-reset mechanism. [Calibration (2026/10), [raw analysis](../prep/calib1/run/calibration_analysis.json), [frozen Q1 calculation](../code/src/ncclenergy/v2/stage2_decisions.py)]
