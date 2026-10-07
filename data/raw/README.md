# Retained raw-run records

This directory contains the public, run-level portion of the experiment record:

- the exact captured YAML configurations;
- randomized condition-order files;
- GPU-set status and event logs; and
- start/end system metadata with private identifiers redacted.

The following files exist in the private experiment archive but are intentionally omitted from the public repository:

- `launches.jsonl`;
- per-launch stdout, stderr, NCCL debug logs, result JSON, and power traces;
- probe, closing-probe, and calibration process directories.

The omitted full-run and smoke-run directories occupy about 70 MiB and 4 MiB locally but contain machine names, device identifiers, private network details, user paths, and tens of thousands of small files. Representative NCCL and nccl-tests output needed by parser tests is retained in `tests/fixtures/` after replacing nonmeasurement identifiers.

No measurement values in the retained run-level records were changed. Public metadata substitutions are limited to host names, personal filesystem paths, and GPU UUIDs/serial numbers. The complete raw archive is not claimed to be available from this repository.

The processed CSV/JSON tables in `data/processed/` are sufficient to regenerate the checked-in reports and figures. They are not sufficient to reproduce the raw-to-processed transformation without the omitted private archive.
