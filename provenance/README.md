# Provenance scopes

`SHA256SUMS` and `DERIVATIVES.csv` describe the earlier public export at
`2b73580e848986905c45bc8ee10706383dc5cd72`. They are retained as historical
records, not silently rewritten to pretend the current README was frozen
before collection. Some current presentation files have changed, so those
old hashes are not a checksum manifest of this entire new working tree.

Use `RELEASE_MANIFEST_2026-10-14.sha256` for current public release bytes.
It excludes itself, Git internals, caches, local test/build artifacts and
reproduced outputs. Stage 2 has a separate manifest, original/public digest
mapping, file disposition list, source bindings and omissions under
[`studies/a6000-stage2-v2.5/provenance/`](../studies/a6000-stage2-v2.5/provenance/).
These are integrity and reproducibility records, not an independent
timestamp, authorship certificate or external physical-energy calibration.
