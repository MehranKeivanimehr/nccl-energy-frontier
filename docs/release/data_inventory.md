# Data inventory and checksums — v2.2 pilot release

**Source**: generated from the local tree (`results/v2.2-pilot-release_2026-10-09/data_inventory.csv`).

**Preservation**:
* All original raw evidence is kept locally and unmodified.
* On 2026-10-09 every file in the historical manifests was re-verified with 0 failures:
  * pilot: 27,491 files;
  * v2: 24,541 files.

**Export columns**:
* The export columns come from the export builder's own rules (`scripts/build_release_export.py`).
* "Unused raw" means per-process files that no published analysis reads, determined by auditing every file the analyses open.

| category | path | files | bytes | git-tracked | on disk only | exported | sanitized derivatives | omitted (privacy) | not exported (unused raw) | manifest |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 historical pilot (v1) | `data/raw/full` | 25666 | 73215182 | 45 | 25621 | 40 | 10 | 6 | 25620 | provenance/pilot_v1_sha256_2026-10-08.txt |
| 1 historical pilot (v1) | `data/raw/smoke` | 1764 | 4634632 | 43 | 1721 | 37 | 9 | 6 | 1721 | provenance/pilot_v1_sha256_2026-10-08.txt |
| 1 historical pilot (v1) | `data/processed/full` | 12 | 9048226 | 12 | 0 | 12 | 0 | 0 | 0 | provenance/pilot_v1_sha256_2026-10-08.txt |
| 1 historical pilot (v1) | `data/processed/smoke` | 13 | 547463 | 13 | 0 | 13 | 0 | 0 | 0 | provenance/pilot_v1_sha256_2026-10-08.txt |
| 1 historical pilot (v1) | `figures/full` | 18 | 4549456 | 18 | 0 | 18 | 0 | 0 | 0 | provenance/pilot_v1_sha256_2026-10-08.txt |
| 1 historical pilot (v1) | `figures/smoke` | 18 | 3346820 | 18 | 0 | 18 | 0 | 0 | 0 | provenance/pilot_v1_sha256_2026-10-08.txt |
| 2 v2 exploratory/confirmatory/Phase 10 (v2.1) | `data/v2/raw` | 24494 | 84379900 | 599 | 23895 | 12565 | 3119 | 0 | 11929 | provenance/v2_raw_sha256_2026-10-09.txt |
| 2 v2 exploratory/confirmatory/Phase 10 (v2.1) | `data/v2/processed` | 28 | 1933974 | 28 | 0 | 28 | 0 | 0 | 0 | provenance/v2_raw_sha256_2026-10-09.txt |
| 2 v2 exploratory/confirmatory/Phase 10 (v2.1) | `figures/v2` | 8 | 1646164 | 8 | 0 | 8 | 0 | 0 | 0 | provenance/v2_raw_sha256_2026-10-09.txt |
| 4 post-hoc: Stage 2 reanalysis of v2 | `data/v2/reanalysis` | 42 | 2253318 | 42 | 0 | 42 | 1 | 0 | 0 | git (tracked) |
| 3 v2.2 smoke + measurement validation (+4 gate 3b post hoc) | `data/v2.2/raw` | 804 | 7467281 | 92 | 712 | 319 | 9 | 0 | 485 | provenance/v22_sha256_2026-10-09.txt |
| 3 v2.2 smoke + measurement validation (+4 gate 3b post hoc) | `data/v2.2/processed` | 6 | 84128 | 6 | 0 | 6 | 0 | 0 | 0 | provenance/v22_sha256_2026-10-09.txt |
| release tables and figures | `results/v2.2-pilot-release_2026-10-09` | 8 | 217104 | 8 | 0 | 8 | 0 | 0 | 0 | export SHA256SUMS |

## Manifests (SHA-256 of the manifest file)

| manifest | SHA-256 |
|---|---|
| `provenance/pilot_v1_sha256_2026-10-08.txt` | `fba0598e16e648a92a8fde208ead5b78e18e0be833bb40cd702aca6255a52f6b` |
| `provenance/v2_raw_sha256_2026-10-09.txt` | `aeff3e7c41d99ee7eef837fa29dbc9cdca2a9cd6078bd46c1dca3653efe6c50f` |
| `provenance/v22_sha256_2026-10-09.txt` | `83486f67276a778503c9a8bbd0b364204288be4856db4692027df3beb96cb3d3` |
