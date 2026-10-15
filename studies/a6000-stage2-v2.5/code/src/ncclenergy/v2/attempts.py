"""Append-only attempt tracking and the predefined, outcome-blind retry rule."""

from __future__ import annotations

from pathlib import Path

from .common import append_jsonl, read_jsonl

MAX_ATTEMPTS = 2   # one retry, only for INFRA_FAIL


class AttemptLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def records(self) -> list[dict]:
        recs, _ = read_jsonl(self.path)
        return recs

    def append(self, rec: dict) -> None:
        for k in ("unit_id", "attempt", "qc_status"):
            if k not in rec:
                raise ValueError(f"attempt record missing {k}")
        append_jsonl(self.path, rec)

    def attempts_for(self, unit_id: str) -> list[dict]:
        return [r for r in self.records() if r.get("unit_id") == unit_id]


def should_retry(attempts: list[dict]) -> bool:
    """Retry only if every attempt so far is an infrastructure failure and the cap is not reached.

    Uses only qc_status/retryable — never energy or latency values.
    """
    if not attempts or len(attempts) >= MAX_ATTEMPTS:
        return False
    return all(a.get("qc_status") == "INFRA_FAIL" and a.get("retryable") for a in attempts)


def final_attempt(attempts: list[dict]) -> dict | None:
    """The attempt that represents the unit: the first non-INFRA attempt, else the last attempt.

    Deterministic in attempt order; independent of measured values.
    """
    if not attempts:
        return None
    ordered = sorted(attempts, key=lambda a: a["attempt"])
    for a in ordered:
        if a.get("qc_status") != "INFRA_FAIL":
            return a
    return ordered[-1]
