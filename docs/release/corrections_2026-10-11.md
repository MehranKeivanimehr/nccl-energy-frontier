# Release-candidate corrections

Date: 2026-10-11. This candidate was reconstructed from the checksum-verified
sanitized archive. The delivered expanded folders were not used as the source.
No GPU experiment, Git commit, branch integration or push was performed here.

## Preserved evidence

Historical raw inputs, configs, pre-registration, frozen schedules, native v2.1
patch and frozen processed results/figures are retained byte-for-byte. The old
release presentations remain under `results/v2.2-pilot-release_2026-10-09/`.
Corrected presentations are separate:
`results/release-corrections_2026-10-11/`.
Its `figures/` contains the corrected temperature, control and estimator-gap
plots, plus v2 RQ1, estimator comparison, exploratory regret and descriptive
historical workload plots. Use these current captions; frozen older figures
remain historical artifacts, not silently revised figures.

`DERIVATIVES.csv` describes the original export's sanitization, not these new
code/document edits. `provenance/export_original_SHA256SUMS` preserves its
original checksum manifest; root `SHA256SUMS` verifies this candidate's published
files. Generated logs, local compiler environments and native binaries are not
release contents. The public export remains an analysis-input subset; see
[OMITTED.md](../../OMITTED.md).

## Interpretation and presentation

| Item | Correction | Numerical consequence |
|---|---|---|
| LL128 AllGather 2 MiB latency | 2.8% lower latency relative to auto; auto is 2.9% slower relative to LL128 | Denominator/rounding correction only; original time ratio unchanged |
| Temperature scatter | Residualize both temperature and log-energy on arm×length; overlay the partial-regression slope | Plot coordinates changed; original fitted coefficient and intervals unchanged |
| LOBO energy-regret heatmap | Negative differences receive a visible marker rather than an undisclosed clipped color | Stored values unchanged |
| `no_sync` subtraction | Descriptive workload contrast, not an isolated communication-energy fraction or upper bound | Historical arithmetic unchanged |
| Sample/counter disagreement | Earlier lag explanation is a hypothesis, not established cause | Historical numbers unchanged |
| NCCL evidence | Initial cost-table probes versus timed profiler execution records are distinguished | No invented timed TUNING logs |
| Idle-adjusted energy | Uses P8 block baselines, not post-loop context power | Historical arithmetic unchanged |
| Inference | Within-session blocks, estimator sensitivity and conditional post-hoc intervals explicit | No strengthened tradeoff, training or deployment claim |

Detailed empirical claims, raw references, literature and prospective inference
designs are in [scientific_resolution.md](scientific_resolution.md).

## Forward code and safety changes

These change future acceptance/execution, not historical measurements:

- Aligned-meter validation rejects nonfinite counters, malformed/reversed read
  brackets and implausible energy. Sampled-meter/QC checks reject nonfinite or
  nonmonotonic original telemetry. Method IDs are now
  `v2.3-aligned-meter-logged-qc`, `v2.3-sampled-meter-qc` and
  `v2.3-finite-order-qc`.
- The watchdog inspects all new telemetry rows between polls, including brief
  recorded health events; this is not continuous foreign-process or Xid coverage.
- Failed/malformed calibrations cannot seed a schedule. Allocation/placement
  lists must be nonempty and unique. Extra environment settings cannot replace
  the selected GPUs, requested algorithm/protocol, profiler or measurement paths.
- Native tuner parsing rejects extra/empty/truncated fields, duplicate scope
  keys and duplicate GPU identities. Re-entrant parsing replaces shared tokenizer
  state. Real NCCL deployment remains unvalidated.
- A **new** native sampled-gate patch is provided without changing the historical
  patch. It avoids counter reads in sampled mode and labels its mode/method in
  raw output. Patch applicability and actual helper code are CPU-tested; full
  CUDA/Pascal behavior is not validated. Native sampled collection stays disabled.
- Historical v1 and Phase4 launchers and CUDA training collection are disabled.
  Newly approved counter-mode campaigns must declare a new implementation identity
  and use new output directories. These guards must not be removed just to resume
  old commands.
- Optional Phase10 `strict-sanity` analysis rejects missing numerical, all-rank
  execution and per-GPU meter evidence. All 50 historical runs lack the numerical
  evidence and fail this gate. The default historical-descriptive analysis remains
  reproducible; passing sanity would still not prove equivalent training.
- Build concurrency defaults to two CPU jobs. Test dependencies are separated
  from analysis-only dependencies; native Windows reproduction is not advertised.

## Reproduction comparator

CSV columns containing JSON numerical arrays are compared numerically with the
declared tolerance rather than as exact strings. Ordinary strings, booleans and
paths remain exact; byte-identity metadata is not forged. This affects the
reproduction check, not an experimental statistic. Figures are regenerated, not
treated as independently validated just because they render successfully.

The 14-step reproduction compares corrected release presentations with the new
versioned directory and historical analyses with their retained references.
Cross-platform differences and known unreproduced illustrative figures must be
reported explicitly. Frozen data are never rewritten to obtain a PASS.

## Outstanding scientific gates

Historical training equivalence cannot be repaired retrospectively. Equivalent
training needs new correctness evidence with distinct per-rank data and saved
vectors, followed by separately measured performance. Distinct optima need a
validated small-effect measurement method and held-out simultaneous ranking of
the compatible candidate set, with ties and session variability accounted for.
Neither is required to publish this bounded pilot; both are required for the
stronger claims. No such future collection has been authorized or run here.
