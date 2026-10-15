# Retained final analysis and report

These JSON/TSV values and the original Markdown report are retained from the
completed analysis, with identifier-only privacy transforms recorded in the
export ledger. The current [result interpretation](../../docs/results.md)
and [claim ledger](../../docs/claims.md) take precedence over shorthand in
the original report or execution messages.

In particular, the sampled-power duration discrepancy is observed estimator
disagreement, not a experimentally isolated sensor-only mechanism. The
same-execution result concerns compatibility of the conditional mean, not
an absolute or individual-reading physical accuracy guarantee. All
adequacy limits and unavailable estimates remain unchanged.

`run_final_analysis.py` is the original numerical-table writer, retained
byte-for-byte despite its overbroad docstring. It does not generate the
Markdown report. Use the separate public replay adapter for this export,
not the private production verifier against transformed freeze bytes.
