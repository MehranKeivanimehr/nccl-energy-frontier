# A6000 branch: presentation refresh

This update changes the landing page and presentation assets after the
completed A6000 release commit `8ac44d45`. It introduces no GPU run,
measurement, inferential analysis or stronger scientific conclusion.
[Media (2026/10), [renderer](../../studies/a6000-stage2-v2.5/scripts/make_readme_media.py),
[source ledger](../../studies/a6000-stage2-v2.5/media/media_values.json)]

## Changes

- The root README leads with the completed A6000 study, its evidence and
  limitations. Historical first-extension findings are separately labelled
  and collapsed; old release notices no longer look like the current scope.
- Two new GIFs show held-out comparisons and duration/estimator sensitivity.
  There are PNG alternatives and a new overview graphic. Every GIF frame
  retains the study scope, uncertainty and limitations.
- The study README, media documentation and reproduction instructions link
  directly to these assets and their CPU-only generator.
- Historical GIFs, raw data, frozen acquisition/analysis code and numerical
  outputs are unchanged. The current public manifests are refreshed;
  original private freeze records and historical export digests are not.

These changes are inspectable in the diff and in the
[media-method document](../../studies/a6000-stage2-v2.5/docs/media.md).

The media regression tests check copied estimates/intervals, exact source
hashes, three-session units, missingness, execution labels and retained
thermal/precision failures. The animations change emphasis only; the
renderer never fits a new statistic. [Media (2026/10),
[tests](../../studies/a6000-stage2-v2.5/tests/test_readme_media.py)]

## Validation of this refresh

The 18 public guard tests and 12 new media tests pass (30/30). A separate
check of all 55 retained contrasts found zero numeric discrepancies at its
`1e-10` tolerance. Six regenerated media/value files match byte-for-byte;
all 12 GIF frames were visually inspected. Offline desktop/mobile previews
have no missing images or horizontal overflow. The 132 checked local links
resolve. [Validation (2026/10),
[machine-readable record](../../studies/a6000-stage2-v2.5/provenance/validation/readme_media.json)]

A full raw-to-result replay was attempted but not newly completed in this
refresh: native Windows lacks the frozen imports' Linux `pwd` module, and
the existing WSL environment lacks SciPy. The [earlier complete replay
record](../../studies/a6000-stage2-v2.5/provenance/validation/public_replay.json)
is retained, not presented as a new pass. The frozen source/data/results
remain byte-identical, and the independent statistical check above passed.
No frozen code was modified to bypass a platform or measurement guard.
This does not change any study conclusion or the disclosed archival-test
limitations. [Validation (2026/10),
[current scope](../../studies/a6000-stage2-v2.5/provenance/validation/readme_media.json),
[original local verification](a6000_stage2_release_notes.md#local-verification)]

## Publish this update

From the same prepared working copy, review the diff, then:

```sh
git branch --show-current
git diff --check
git add .
git commit -m "Distinguish A6000 held-out study with reproducible visual summaries"
git push origin a6000-heldout-release
```

Expected branch: `a6000-heldout-release`. No new branch, force-push, raw-data
edit, merge or GPU campaign is needed for this presentation update. The
earlier content ZIP remains an older snapshot; use the current Git working
copy to publish the refresh.

## Sources

[Media, 2026/10] A6000 presentation renderer, source ledger and regression
tests. Links above.

[Validation, 2026/10] Presentation-refresh checks, independent statistics
and preserved original replay record. Links above.
