"""Methodology v2: synchronized energy regions, exact operation accounting,
per-process configuration evidence, append-only attempts, block design.

Pure, CPU-testable logic lives in common/topology/region/evidence/qc/attempts;
anything that touches NVML or launches processes lives in telemetry/preflight/
runner. The pilot (v1) code in the parent package is not used for v2 data.
"""
