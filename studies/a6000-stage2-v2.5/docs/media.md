# README media: retained evidence, new presentation

The new-study assets live in `media/`, separate from the historical GIFs in
the repository's `docs/media/`. No original animation or scientific output
was replaced. [Release (2026/10), [renderer](../scripts/make_readme_media.py),
[historical renderer](../../../scripts/make_readme_media.py)]

| Asset | What it shows | Evidence |
|---|---|---|
| `study_overview.png` | Smoke, calibration, held-out counts; requested and observed arm configurations | Smoke summary, calibration analysis, held-out execution table |
| `heldout_comparisons.gif` / `.png` | All three H-D3 controls and the H-D4 protocol comparison, session means and descriptive 95% intervals | Retained primary summaries and decisions |
| `estimator_duration.gif` / `.png` | H-D1 duration contrasts for all three arms and all four estimators, including descriptive 95% intervals | Retained estimator-specific summaries |
| `media_values.json` | Exact displayed values, retained verdicts, missingness and source SHA-256 hashes | The four source records listed below |

The sources are [held-out analysis](../prep/heldout_final/heldout_analysis.json),
[execution](../prep/heldout_final/execution.tsv),
[calibration analysis](../prep/calib1/run/calibration_analysis.json) and
[smoke summary](../code/data/stage2/v2.5-s2-smoke/smoke/session_summary.json).
The renderer records their hashes before drawing. [Stage 2 (2026/10),
[media value ledger](../media/media_values.json)]

## What the animation does and does not mean

Only visual emphasis changes between frames. Every mean and confidence
interval always uses the complete retained three-session summary; there
is no sequential testing, refitted interval, interpolated measurement or
new observation. Percentages are `100 * expm1(retained_log_ratio)`, not
`100 * retained_log_ratio`. The renderer imports no acquisition or frozen
analysis module. [Media (2026/10), [renderer](../scripts/make_readme_media.py)]

The H-D3 shaded margin is exactly `[1/1.005, 1.005]` on the ratio scale.
The interval drawings are descriptive, unadjusted 95% intervals. The
retained equivalence verdict instead uses its frozen, unadjusted TOST rule
and 90% interval; it is not recomputed from the drawing. Session markers
are transformed session means, not extra independent replicates. Thermal
failure at 20 s, insufficient precision at 40 s, and the H-D4 precision
failure remain visible. [Stage 2 (2026/10),
[decisions](../prep/heldout_final/heldout_analysis.json),
[results](results.md#principal-comparisons)]

The duration plot covers all nine H-D1 contrasts per estimator, without
joining the measured durations by interpolated curves. It is a sensitivity
view, not a new decision table. All estimators share the same launches and
NVML infrastructure; their disagreement does not establish a sensor-only
mechanism, and agreement does not externally calibrate physical energy.
[Stage 2 (2026/10), [sensitivity limits](results.md#outcomes-that-remain-unresolved)]

The three sessions were on one host during the same evening. All 72
launches were accepted, but two primary estimates remain unavailable.
All aggregate adequacy scopes remain unmet/unestablished. None of these
assets proves a global latency/energy optimum difference or real-training
benefit. [Stage 2 (2026/10), [adequacy](../prep/heldout_final/adequacy.tsv),
[claim limits](claims.md)]

## Regenerate without acquiring data

From the repository root, with the analysis dependencies installed:

```sh
python studies/a6000-stage2-v2.5/scripts/make_readme_media.py
python -m unittest discover -s studies/a6000-stage2-v2.5/tests -p 'test_readme_media.py' -v
```

Pillow is needed for GIF output and is a dependency of the installed
Matplotlib environment. The generator defaults to the fresh directory
`reproduced/stage2-readme-media/`; `--output` selects another fresh
directory. It refuses a nonempty destination. Check the plots and source
ledger before replacing tracked presentation assets, then refresh the
current public manifests. The original private freeze records are not
rewritten. [Media (2026/10), [renderer](../scripts/make_readme_media.py),
[provenance scopes](../../../provenance/README.md)]

## Sources

[Stage 2, 2026/10] Retained A6000 study records and decisions. Links above.

[Media, 2026/10] Presentation renderer, value ledger and
[regression tests](../tests/test_readme_media.py).

[Release, 2026/10] Public repository layout and historical media. Links above.
