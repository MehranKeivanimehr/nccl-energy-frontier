# Scientific status and the two unresolved questions

Date: 2026-10-11. Scope: the retained RTX A6000 pilot, not a new experiment.
Literature was checked against primary papers and version-matched documentation.
Project percentages require project evidence; similarity to a paper does not
validate them. Paper citation counts were not retrieved.

## Contents

1. [Evidence and permitted claims](#1-evidence-and-permitted-claims)
2. [Training numerical validity](#2-training-numerical-validity)
3. [Different latency and energy optima](#3-different-latency-and-energy-optima)
4. [Literature comparison](#4-literature-comparison)
5. [Sources and unresolved limits](#5-sources-and-unresolved-limits)

## 1. Evidence and permitted claims

| Claim | Evidence | Status and boundary |
|---|---|---|
| Selected configurations improve reported board energy versus auto in 36/39 cells | `data/v2/raw/confirmatory/blocks/*/attempts.jsonl`; `data/v2/processed/confirmatory/rq1.csv` | Supported within this host/session and measurement method, not a global optimum |
| 25 cells show ≥5% point-estimate savings; 21 pass the separate corrected threshold test | `rq1.csv`; `data/v2/reanalysis/2026-10-09_v2.2-audit/rq1_threshold.csv` | Distinct rules; neither number may replace the other |
| P4_MIX AllReduce 512 KiB has energy ratio .312 and latency ratio .316 | Same confirmatory raw attempts and `rq1.csv` | Large local comparison, not topology-independent scaling or transferable industry savings |
| Synthetic DDP batch128 energy/step ratio .791, time/step ratio .706 | `data/v2/raw/phase10/workload/runs.jsonl` and referenced `result.json`; reanalysis `phase10_v22/phase10_summary.json` | Descriptive synthetic workload observation; numerical equivalence of training unverified |
| LL128 versus auto AllGather 2 MiB has a resolved energy penalty | `rq2_estimators.csv`; `data/v2.2/raw/v2.2-measval-01/blocks/*/attempts.jsonl`; corresponding `decisions.json` | NOT established: estimators disagree on inference and the precision gate failed |
| Post-hoc temperature adjustment confirms a tradeoff | `posthoc_temperature.json` | Unsupported confirmatory interpretation: nonrandomized covariate and conditional intervals |
| `no_sync` subtraction isolates communication energy | Phase10 records and power regimes | Unsupported; intervention also changes execution, overlap and power state |
| Absolute energy accuracy is calibrated | No external reference retained | Not assessed |

The export is an analysis-input subset. Some per-launch raw logs/probes were
omitted; see [OMITTED.md](../../OMITTED.md). Frozen processed results reproduce
reported arithmetic but do not replace missing measurement or training evidence.
See [claims.md](claims.md) for all individual empirical claims and status labels.

## 2. Training numerical validity

**Historical training equivalence cannot be recovered from timing/energy logs.**
The 50 historical workload runs did not save loss trajectories, gradients or
parameter vectors needed for this comparison. Later tests of corrected code do
not demonstrate what those processes computed. Their energy and time remain
descriptive observations, not verified energy savings for equivalent training.
Evidence: historical `result.json` files and `phase10_summary.json` numerics
status; current `phase10.py::prospective_acceptance` rejects missing evidence.

DDP synchronizes gradients, while users are responsible for partitioning input
data. This is explicitly documented, not an assumption inferred from performance.

```text
DistributedDataParallel does not chunk or otherwise shard the input
```
[PyTorch DDP (2.9), class description](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)

**Proposed future validation, not an executed result:**

1. Freeze GPU/software/topology, actual policy execution, model, precision,
   optimizer, loss normalization, global batch, warmup/steps and seeds. Model
   initialization is identical between policies. Each rank receives a distinct
   deterministic shard; the same shard/step is reused for that rank across policies.
2. In a separate correctness run, compare collective outputs with known inputs
   and a higher-precision reference, then compare distributed averaged gradients
   with a correctly normalized global-batch reference. Include an intentionally
   missing/wrong reduction as a negative control: it must fail.
3. Save per-rank losses, gradient vectors and parameter-update vectors at declared
   steps, checking finiteness, shapes, missing gradients, reduction counts and
   rank consistency. Compare componentwise absolute/relative error, normalized
   L2 error and maximum error. Equal parameter norms alone are not sufficient:
   different vectors can have identical norm.
4. Prespecify numerical tolerances using precision, reduction conditioning and
   same-policy reproducibility controls. Do not use a universal arbitrary
   tolerance or require bitwise equality for different summation orders.
5. Failures/missing evidence exclude the paired comparison and are retained.
   Do not collect performance while extensive correctness snapshots alter the
   timed region. Keep correctness and minimally instrumented performance runs
   separate, with common fixed execution semantics and raw boundary provenance.
6. After correctness passes, collect randomized, matched policy/auto comparisons
   in multiple sessions. For practical training claims, use a specified real
   model and held-out evaluation target, paired seeds and equal training budget;
   measure energy/time to that quality, not just fixed-step throughput.

Floating-point arithmetic is not associative, so mathematically identical
computations need not be bitwise equal. [PyTorch numerical accuracy (2025/07),
opening](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)

```text
the order of the operations affects the results
```
[PyTorch numerical accuracy (2025/07), opening](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)

Steps above are this project's proposed acceptance design, not claimed results
of the cited sources. A short numerical check validates the tested execution;
it cannot alone establish long-horizon convergence or real-model transfer.
The pilot release disables CUDA workload collection until this protocol and
the guarded plugin have separate validation. CPU tests remain available.

## 3. Different latency and energy optima

The current confirmatory design compared selected candidates with auto; it did
not confirm all eligible configurations. AllGather 2 MiB energy inference
depends on the estimator. The fresh 40 s C/A comparison is 1.020242 with 95% CI
[.980426, 1.061675], five complete within-session pairs; its log half-width is
3.9808%, not the required ≤.4%. Evidence: `rq2_estimators.csv`, Gate3b raw
attempts and `data/v2.2/processed/v2.2-measval-01/decisions.json`. Therefore the
optimum-difference claim is unresolved, not refuted or confirmed.

The D3 label `SYSTEMATIC_OFFSET` is the frozen protocol's decision category,
not proof of a persistent physical bias. The 20 s B/A contrast is −2.4353%,
unadjusted p=.0494; a separate Holm correction across three tested durations
gives p=.1482. Preserve the prespecified decision while keeping that exploratory
multiplicity sensitivity separate.

**Proposed inferential target:** for one declared hardware/software/topology/
collective/size/semantics cell, let C be the finite set of compatible, verified
effective configurations. Define T(c) and E(c) as the expected latency and board
energy per collective under a specified operating regime. The target is whether
the sets argmin T and argmin E share any member. Requested aliases of the same
effective configuration are not independent alternatives. This mathematical
definition restricts the claim to C, not every possible NCCL tuning parameter.

**Proposed future procedure, frozen before new data:**

1. First validate the measurement method: identical-execution control,
   start-phase/window diagnostics, counter/sampled disagreement and drift.
   Both readouts share board instrumentation; their agreement is not external
   calibration. Bound systematic uncertainty below the claimed effect, or limit
   the estimand explicitly to an estimator. No guarantee is supplied by merely
   extending every region to 40 seconds.
2. Use thermally matched/balanced interleaving and multiple independently
   scheduled sessions; retain temperature, clocks, contention and all exclusions.
   Freeze durations, the temperature policy and any covariate model beforehand.
   Prefer pre-run thermal state for adjustment. Mean temperature during a
   policy's measured region can be affected by that policy; treating it as a
   nuisance may remove part of the effect or introduce post-treatment bias.
   Fresh processes alone do not establish session independence.
3. Measure all compatible candidates needed for the stated optimum claim in
   held-out confirmation, or use a prespecified valid selection/elimination
   design. Candidate discovery and confirmation must remain separate.
4. Prespecify meaningful margins δT and δE, confidence level, multiplicity
   treatment and a precision/sample-size plan from a separate redesigned pilot.
   Use paired configuration contrasts and block/session-aware uncertainty;
   thousands of internal iterations do not become independent trials.
   If a fitted covariate adjustment is used, propagate its fitted-parameter
   uncertainty (for example, refit the entire prespecified model inside a
   valid session/block resampling procedure). A CI treating the learned slope
   as known is not sufficient; few sessions also limit resampling inference.
5. One sufficient route is to establish A as the unique latency optimum within
   C with simultaneous comparisons, then establish some B has strictly lower
   energy than A. If latency has ties, every latency minimizer must be excluded
   from the energy-minimizer set; showing just one is energy-inferior is not
   enough. To report B as the energy optimum, its energy ranking against the
   remaining C must also be established. Practical-equivalence sets require
   their own declared margins, not arbitrary tie-breaking.
6. With simultaneous confidence bounds L/U for every configuration, conservative
   possible-minimizer sets are C_T={c:L_T(c)≤min U_T} and analogously C_E.
   Disjoint sets support different optima at the joint coverage level, assuming
   bounds validly cover the target and systematic uncertainty is addressed.
   Overlap means unresolved; it does not prove a common optimum.

A nonsignificant difference is not evidence of equivalence. A practical tie
requires prespecified equivalence bounds and an equivalence procedure, not
choosing the smallest point estimate. [Lakens (2017/05), abstract and Testing
for Equivalence](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/)

```text
an upper and lower equivalence bound is specified
```
[Lakens (2017/05), abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/)

Do not simply append two blocks to the old method or collect until significant.
An unresolved result is a legitimate scientific outcome. If validated energy
and latency optimum sets overlap within useful margins, publish that bounded
result rather than forcing a tradeoff claim.

## 4. Literature comparison

| Primary source | What it establishes | What it does not validate here |
|---|---|---|
| NVIDIA NVML API, R550, 2024/03 | Cumulative energy API reports mJ since driver reload on supported devices | A universal 100 ms cadence, a read-reset guarantee or absolute A6000 accuracy |
| Yang, Adámek, Armour, Oxford, arXiv v3, 2024/12 | Investigates sensor/polling/averaging behavior across GPUs including A6000 and 1080 Ti, using an external reference | Our per-region gap's physical cause or counter as ground truth |
| Xu et al., USTC/Microsoft Research, AutoCCL, NSDI 2025 | Configuration optimization accounts for topology and computation/communication interference; real-model evaluation | Our exact energy percentages or universal transfer of the policy |
| Jia, Bhuyan, Wong, UC Riverside, PCCL, ICCD 2024 | Energy-aware collectives using DVFS, with T4/PCIe and training evaluation | Our algorithm/protocol-only intervention, or permission to change shared-server clocks |
| Kalibera/Jones, Kent, ISMM 2013 | Benchmark uncertainty has multiple levels; repetition allocation should follow variation | Calling consecutive blocks independent sessions |
| Lakens, Eindhoven, 2017/05 | Prespecified equivalence margins distinguish equivalence from nonsignificance | A universally correct 1%/3% margin for this application |

Methodological inference: the project direction is consistent with prior work,
but literature cannot certify missing observations or resolve its small effect.
Large local gains are supported by local evidence, not by matching someone
else's percentage. Absolute accuracy, equivalent real-model training and the
global optimum claim remain unearned.

## 5. Sources and unresolved limits

Sources accessed 2026-10-11; documentation version/date is given where known.

- [NVIDIA (2024/03)] NVIDIA. NVML API Reference R550.
  https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html
- [Yang et al. (2024/12)] Yang, Z.; Adámek, K.; Armour, W. Part-time Power
  Measurements: nvidia-smi's Lack of Attention. arXiv 2312.02741v3.
  https://arxiv.org/html/2312.02741v3
- [AutoCCL (2025)] Xu, G. et al. AutoCCL. NSDI 2025.
  https://www.usenix.org/system/files/nsdi25-xu-guanbin.pdf
- [PCCL (2024)] Jia, Z.; Bhuyan, L.N.; Wong, D. PCCL: Energy-efficient LLM
  Training with Power-aware Collective Communication. ICCD 2024 author PDF.
  https://danielwong.org/files/PCCL-ICCD2024.pdf
- [Kalibera/Jones (2013)] Rigorous Benchmarking in Reasonable Time. ISMM 2013.
  https://kar.kent.ac.uk/33611/
- [Lakens (2017/05)] Equivalence Tests: A Practical Primer for t Tests,
  Correlations, and Meta-Analyses. Social Psychological and Personality Science.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/
- [PyTorch DDP (2.9)] PyTorch contributors. DistributedDataParallel.
  https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html
- [PyTorch numerical accuracy (2025/07)] PyTorch contributors. Numerical accuracy.
  https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html
- [statsmodels (accessed 2026/10)] Partial-regression plotting: regress both axes
  on the remaining regressors before plotting.
  https://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.plot_partregress.html

**Known:** retained arithmetic and raw-derived counts are reproducible; the
precision gate failed as recorded. **Open:** physical estimator bias, historical
training equivalence, cross-session inference and real guarded/Pascal behavior.
**Next experiment:** no GPU run for presentation/release corrections; for
stronger claims, measurement validation precedes held-out optimum confirmation,
and a separate numerical protocol precedes training-benefit collection.
