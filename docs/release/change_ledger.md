# Change ledger — v2.2 pilot release

**What this lists**: every number or statement that changed relative to an earlier report or document, and every reproduction difference.

**Sources**:
* Current candidate corrections: [corrections_2026-10-11.md](corrections_2026-10-11.md).
* Stage 2 entries summarize `data/v2/reanalysis/2026-10-09_v2.2-audit/ledger.csv` and the Stage 2 report.
* Machine-readable copy: [`change_ledger.csv`](change_ledger.csv).

**Kinds of change**:

| kind | meaning |
|---|---|
| `NUMBER_CORRECTION` | a reported number was wrong |
| `WORDING` / `INTERPRETATION` | the claim changed, not the number |
| `NEW_*` | an added analysis |
| `CODE_CHANGE` | code behaviour for future runs; no published number changed |
| `REPRODUCTION_NOTE` / `KNOWN_UNREPRODUCED` | how the reproduction differs from frozen files |

**Unchanged**: no pre-specified decision, confirmatory number or frozen output changed. The full release reproduction compares every published output.

| id | earlier_report | item | earlier | now | kind | reason_or_evidence |
|---|---|---|---|---|---|---|
| L01 | README / final_report (v2) | RQ1 cells with >= 5 % saving | 25 cells | 25 cells (pre-registered point-estimate rule, unchanged); 21 cells supported by a separately identified threshold test | NEW_SEPARATE_ANALYSIS | Stage 2 (data/v2/reanalysis/2026-10-09_v2.2-audit/ledger.csv); rq1_threshold.csv |
| L02 | README / final_report (v2) | RQ2 AllGather 2 MiB LL128 energy penalty | DIFFER (counter estimator) | preliminary, estimator-sensitive: counter 1.012 [1.003, 1.022]; instant 1.005 [0.991, 1.019]; lag 1.009 [0.999, 1.019] | WORDING_AND_SENSITIVITY | Stage 2; rq2_estimators.csv |
| L03 | phase8_results (v2) | order/temperature model intervals | iid OLS | block-, cell- and two-way-clustered fits + wild cluster bootstrap; same point estimates (order p 0.353, temperature p 0.0029) | REANALYSIS | Stage 2; order_temperature.json |
| L04 | phase9_recommender (v2) | exported tuner files | range rules, no scope header | one rule per measured size, #@scope header; ranges only as labeled experimental opt-in | CODE_CHANGE | Stage 2; phase89_v22/nccl_tuner_*.conf |
| L05 | phase4_validation (v2) | counter update period at idle | presented as established at idle | under load 100.000 ms; P8 idle median 101.3 ms (p10-p90 93.5-101.5 ms); idle evidence weaker than stated | CORRECTION | Stage 2 addendum (a81c938) |
| L06 | stage2_report first version | boundary uncertainty "0.10 %" | 0.10 % total | 0.10 % covers only timing/phase components A1+A2+B1; tail-correction magnitude median 1.34 % (max 2.41 %) and tail-model bound median 0.41 % (max 1.39 %) reported separately; sensor accuracy uncharacterized | CORRECTION | Stage 2 addendum; uncertainty_components.json |
| L07 | README / final_report (v2) | order effect | no order effect | no statistically resolved order association in the tested model | WORDING | Stage 2 amendment |
| L08 | README / protocol (v2) | counter read semantics | stated as behaviour | local interpretation of Phase 4 experiments on driver 580, not an NVML guarantee | WORDING | Stage 2 amendment |
| L09 | README (v2) | energy definition | energy | GPU-board energy of the participating GPUs; not communication-only, not whole-system | WORDING | Stage 2; this release |
| L10 | gate3a_smoke_report | instant/counter ratios 0.98 / 0.96 / 0.94 (single 2.5 s runs) | possible protocol dependence, to be measured by D2 | D2 LL128 term −0.12 % [−0.48, +0.25] (INCONCLUSIVE); the 2.5 s ratios are dominated by a per-region gap between estimators; not evidence of protocol dependence | INTERPRETATION | gate3b report §2 D2 |
| L11 | gate3b report first version (042625d) | D2 intercept, instant | +0.03 % [−0.11, +0.16] | +0.02 % [−0.11, +0.16] | NUMBER_CORRECTION | hand rounding of +0.0245 %; decisions.json |
| L12 | gate3b report first version (042625d) and its commit message | D2 LL128 term, instant, lower bound | −0.12 % [−0.49, +0.25] | −0.12 % [−0.48, +0.25] | NUMBER_CORRECTION | hand rounding of -0.4847 %; decision unchanged (outside +-0.3 % either way) |
| L13 | gate3b report first version | per-pair B/A agreement between estimators | "agree to within about 0.1 %" at 10-40 s | max difference from the counter: lag 0.05−0.19 pp, instant 0.17-1.16 pp (40 / 20 / 10 s) | NUMBER_CORRECTION | decisions.json D3 per_block |
| L14 | gate3b report first version | counter path | "real differences in the GPUs' power draw, not counter error"; "not disagreement between the counter and the sampled power" | the B/A spread is present in every estimator; estimator agreement is consistency, not accuracy; the per-region estimator gap is unexplained | WORDING | release review |
| L15 | gate3b report first version | temperature | "GPU temperature explains almost all of the between-process variation" | post-hoc model associates temperature with energy variation (0.43 [0.38, 0.48] %/C); adjustment reduces residual SD under that model; not established as sole cause | WORDING | release review; posthoc_temperature.json |
| L16 | gate3b report first version | D3 20 s result | "it is not a consistent offset ... compatible with chance" | formal decision SYSTEMATIC_OFFSET kept; four competing explanations listed separately, none established | WORDING | release review |
| L17 | gate3b report first version | estimator gap | "fixed ~28 J per region below the counter ... about one update period" | matched windows: +13.5 [12.2, 14.8] J (instant) / +13.1 [12.9, 13.3] J (lag) per GPU per region (~27 J per 2-GPU region); pre-start-window explanation not supported; unexplained | NEW_POSTHOC_ANALYSIS | posthoc_temperature.json window_gap |
| L18 | gate3b report first version | steady-state agreement | "agrees with the counter within about 0.1 % at steady state" (D2 intercept) | D2 intercepts unchanged (+0.02 % instant, +0.11 % lag); matched-window steady-state power difference per GPU −0.016 % [−0.062, +0.030] (instant), −0.031 % [−0.043, −0.019] (lag) | REFINEMENT | posthoc_temperature.json window_gap |
| L19 | gate3b report first version | accepted count wording | "70 processes OK and VERIFIED" | 70 of 72 processes OK and VERIFIED (accepted); unchanged count | WORDING | counts.json |
| L20 | README (v2) | AllGather 2 MiB counter penalty format | +1.2 % [0.3, 2.2] | 1.012 [1.003, 1.022] (same values, ratio format) | FORMAT | rq2_estimators.csv |
| L21 | release reproduction | phase89 outputs from the current code vs frozen v2.1 outputs | frozen data/v2/processed/confirmatory (v2.1 code) | 6 tuner files and placement_meta.json differ by design (scope headers, exact-size rules, campaign-derived scope); identical to the Stage 2 phase89_v22 outputs (placement_meta.json equal except the recorded repository path) | REPRODUCTION_NOTE | release_repro v2_phase89 |
| L22 | release reproduction | phase10 summary from the current code vs frozen v2.1 summary | frozen data/v2/processed/phase10 | adds all_rank_execution and numerics_comparisons (NOT_RECORDED); identical to Stage 2 phase10_v22; numbers unchanged | REPRODUCTION_NOTE | release_repro v2_phase10 |
| L23 | release reproduction | figures/full/method_example_trace.png | pilot figure | not reproducible: made once by plots.example_trace() for an unrecorded launch; illustrative, no number depends on it | KNOWN_UNREPRODUCED | release_repro pilot_plot |
| L24 | release code audit | process cleanup ok semantics | ok = no survivors | ok = no survivors and no failed cleanup step; affects future runs only | CODE_CHANGE | 3b499b5 |
| L25 | release code audit | QC / sampler / meters / workload / tuner parsing | fail-open paths (see limitations §4) | fail closed; workload default data per_rank; affects future runs only; all published outputs reproduce identically | CODE_CHANGE | 4975780 |
