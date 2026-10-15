# Publication and provenance status

The A6000 data-collection scope is closed: smoke, calibration and the three held-out sessions are retained, with no added rounds, replacement launches or proposed reruns to force a desired verdict. The work is a pilot characterization, not a peer-reviewed publication or externally calibrated energy study. A 1080 Ti/Titan replication is not part of the completed study. [Stage 2 (2026/10), [final outcomes](../prep/heldout_final/heldout_analysis.json), [frozen held-out scope](../protocol/STAGE2_HELDOUT1_PROTOCOL.md)]

## What this export preserves

The study directory retains the final acquisition and analysis implementation, native patch, configurations, schedules, calibration inputs, held-out raw records, numerical outputs and methodological limitations. Earlier campaigns remain in their historical repository locations. Stage 2 does not replace the old Gate 3b's 70 accepted launches with the new 72 accepted launches; these are different campaigns. [Release (2026/10), [study index](../README.md), [export provenance](../provenance), [historical release methods](../../../docs/release/methodology.md)]

Original private evidence and freeze records are preserved separately. Public records may replace private paths, device identities or correspondence with documented aliases or summaries. Numerical readings, timestamps, counts, statuses and decisions are not anonymized by changing their values. The export ledger records original and public digests and explains omitted material; an omitted original is not represented as available in this repository. [Release (2026/10), [provenance records](../provenance)]

Authorization summaries preserve the distinction between explicit user permission, interpreted pair/window choices and availability checks. A delegated selection is not rewritten as a directly quoted user choice. No record is promoted to an administrator-confirmed allocation or scheduler reservation. Publication editing does not establish exclusive or unaided authorship, and third-party source licenses and attribution remain applicable. [Release (2026/10), [provenance policy](../provenance), [recorded allocation notes](../code/configs/stage2/heldout_s1.yaml)]

## Two different reproducibility claims

Freeze dates and hash chains are locally recorded project provenance, not an
independent timestamp or a claim of registration with an external registry.
No pre-collection Git commit is invented for this release.

The original production analysis checks approved private Freeze-H records against their bound bytes. The public export changes some of those bytes for privacy and therefore uses a separate manifest-aware, offline replay. Its status is `PUBLIC_DERIVATIVE_REPLAY`, not an assertion that public derivatives are the unchanged original freezes. [Stage 2 (2026/10), [production entry point](../code/src/ncclenergy/v2/stage2_heldout.py); Release (2026/10), [public adapter](../scripts/reproduce_public.py)]

From the repository root:

```sh
# Validate the public manifest and replay without creating result files.
python studies/a6000-stage2-v2.5/scripts/reproduce_public.py --check

# Write a new result directory; an existing nonempty directory is refused.
python studies/a6000-stage2-v2.5/scripts/reproduce_public.py

# Check the public adapter on CPU.
python -m unittest discover -s studies/a6000-stage2-v2.5/tests -p 'test_public_replay.py' -v
```

The generated JSON/TSV tables are reproducible outputs. The explanatory Markdown report is separately authored from those tables; the historical `run_final_analysis.py` does not generate that report despite its docstring. Compiled binaries, profiler plugins and external server libraries are not silently treated as verified when absent locally. [Stage 2 (2026/10), [historical report driver](../prep/heldout_final/run_final_analysis.py), [production offline checks](../code/src/ncclenergy/v2/stage2_heldout.py)]

The frozen private-production tests require original private calibration
and freeze bytes; they cannot be used to demand that privacy derivatives
pass the original verifier. The public-export guards and numerical replay
are a separate test scope. Tests that skip unavailable protected inputs or
compiled artifacts must be reported as not run, not as additional passes.
No frozen test or hash anchor is weakened merely to make a public derivative
look like the original input.

The frozen calibration JSON contains one positive `Infinity` readiness-wait
diagnostic (a tried threshold was never reached). It is retained as the
original Python-readable JSON value, not converted into a finite wait or
valid energy. The calibration replay checks that exact diagnostic separately;
its finite-energy checks are not relaxed. Strict RFC JSON consumers must
account for this original non-standard diagnostic serialization.

## Remaining limitations

All five aggregate scopes remain unmet/unestablished; no external physical-error bound exists. The three session means have uncertain variability and same-evening dependence is not excluded. Two primary estimates are missing, thermal balance fails for some comparisons, sampled power disagrees on duration effects, and the C/A precision target was missed. Historical training correctness and global optimum differences remain unestablished. Keeping these limitations is part of publishing the result, not a reason to rewrite or suppress it. [Stage 2 (2026/10), [results](results.md), [claims](claims.md), [adequacy](../prep/heldout_final/adequacy.tsv)]

Future work would require a separately justified scope, protocol and authorization. It is not needed to finish publication of this bounded A6000 pilot. [Release recommendation (2026/10), based on [the completed evidence](../prep/heldout_final/heldout_analysis.json)]
