# Files not in this export

This is an **analysis-input subset**, not the complete forensic dataset.
It retains the inputs consumed by the published analyses, but omits original
stdout/NCCL logs, some boundary/probe/calibration evidence and per-rank
training logs. Consequently a reader can reproduce retained analyses but
cannot independently audit every historical measurement from this export.
The manifests record omitted-file hashes; hashes do not reconstruct files
and are not a promise of public availability. The omission counts below are
the original export inventory, not an inventory of later candidate edits.

## Omitted for privacy (not read by any published analysis)

| path | reason |
|---|---|
| `data/raw/full/metadata_end/nvidia-smi-compute-apps.txt` | process list at run start, including other users' processes; not read by any analysis |
| `data/raw/full/metadata_end/nvidia-smi-q.txt` | full `nvidia-smi -q` dump (GPU serial numbers, UUIDs, board and module identifiers); not read by any analysis |
| `data/raw/full/metadata_end/nvidia-smi.txt` | `nvidia-smi` summary whose process table lists another user's process; not read by any analysis |
| `data/raw/full/metadata_start/nvidia-smi-compute-apps.txt` | process list at run start, including other users' processes; not read by any analysis |
| `data/raw/full/metadata_start/nvidia-smi-q.txt` | full `nvidia-smi -q` dump (GPU serial numbers, UUIDs, board and module identifiers); not read by any analysis |
| `data/raw/full/metadata_start/nvidia-smi.txt` | `nvidia-smi` summary whose process table lists another user's process; not read by any analysis |
| `data/raw/smoke/metadata_end/nvidia-smi-compute-apps.txt` | process list at run start, including other users' processes; not read by any analysis |
| `data/raw/smoke/metadata_end/nvidia-smi-q.txt` | full `nvidia-smi -q` dump (GPU serial numbers, UUIDs, board and module identifiers); not read by any analysis |
| `data/raw/smoke/metadata_end/nvidia-smi.txt` | `nvidia-smi` summary whose process table lists another user's process; not read by any analysis |
| `data/raw/smoke/metadata_start/nvidia-smi-compute-apps.txt` | process list at run start, including other users' processes; not read by any analysis |
| `data/raw/smoke/metadata_start/nvidia-smi-q.txt` | full `nvidia-smi -q` dump (GPU serial numbers, UUIDs, board and module identifiers); not read by any analysis |
| `data/raw/smoke/metadata_start/nvidia-smi.txt` | `nvidia-smi` summary whose process table lists another user's process; not read by any analysis |

## Raw artifacts kept only in the original local data (not read by any published analysis)

Per-process files that no published analysis reads (determined by auditing every file the analyses open). They include NCCL debug logs and benchmark stdout that print the host name and private network addresses, per-rank Phase 10 logs, pilot per-launch directories, probe and calibration runs. The original manifests in `provenance/` list them with SHA-256 (paths sanitized).

| directory | files | bytes |
|---|---|---|
| `data/raw/full/calibrate` | 448 | 596727 |
| `data/raw/full/launches` | 23604 | 49835148 |
| `data/raw/full/metadata_end` | 3 | 80662 |
| `data/raw/full/metadata_start` | 3 | 81020 |
| `data/raw/full/probe` | 1568 | 2448946 |
| `data/raw/smoke/calibrate` | 224 | 229151 |
| `data/raw/smoke/launches` | 1080 | 2326920 |
| `data/raw/smoke/launches.jsonl` | 1 | 1069223 |
| `data/raw/smoke/metadata_end` | 3 | 81017 |
| `data/raw/smoke/metadata_start` | 3 | 81019 |
| `data/raw/smoke/probe` | 208 | 349949 |
| `data/raw/smoke/probe_end` | 208 | 353925 |
| `data/v2.2/raw/v2.2-measval-01/blocks` | 432 | 1500413 |
| `data/v2.2/raw/v2.2-smoke-01/blocks` | 27 | 56025 |
| `data/v2.2/raw/v2.2-smoke-02-s3/blocks` | 5 | 11621 |
| `data/v2.2/raw/v2.2-smoke-02-s4/blocks` | 3 | 6288 |
| `data/v2.2/raw/v2.2-smoke-02/blocks` | 18 | 48701 |
| `data/v2/raw/confirmatory/blocks` | 984 | 0 |
| `data/v2/raw/exploratory/blocks` | 9324 | 33184292 |
| `data/v2/raw/phase10` | 701 | 3900607 |
| `data/v2/raw/preflight` | 1 | 8724 |
| `data/v2/raw/setup` | 581 | 2996704 |
| `data/v2/raw/smoke/blocks` | 318 | 845105 |
