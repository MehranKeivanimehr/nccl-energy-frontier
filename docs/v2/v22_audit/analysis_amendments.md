# v2.2 analysis-amendment record (gate 3b, `v2.2-measval-01`)

This record lists every change made to the code or scripts of gate 3b
after data collection began. It gives each change's effect and the
check that demonstrates it.

**Collection timeline**:
* The analysis and the campaign script were frozen in commit `f4735b1`
  (2026-10-09 19:30:44 UTC).
* Collection ran from 19:30:56 to 20:05:21 UTC.
* The raw data were committed unmodified in `e963e12` (20:12:37 UTC),
  before any analysis output was written.
* No raw file changed after `e963e12`: `git diff --stat e963e12 HEAD --
  data/v2.2/raw` is empty.

**Frozen copies**:
* Kept byte-for-byte in `configs/v2.2/frozen_code/`, so the checks also
  run in an export without git history.
* Their git blob IDs equal those in `f4735b1`:

| file | git blob (`git hash-object`) | source |
|---|---|---|
| `measval_f4735b1.py` | `fb378505edcfbdae12cc424ffab5a839ff29e0cb` | `f4735b1:src/ncclenergy/v2/measval.py` |
| `run_confirmatory_v2_f4735b1.sh` | `20c7e53209803e28465610cd1cb3ac8cda33a0ec` | `f4735b1:scripts/run_confirmatory_v2.sh` |

## AM-1 — `measval.descriptive()` crash (commit `0be9c1d`)

**Defect**: `df.flags` resolves to pandas' `DataFrame.flags` attribute,
not the `flags` column. The frozen `main()` computed acceptance, control
status and D1–D4, then raised `KeyError: 0` in the descriptive section
before writing any output.

**Change**: one line, `df.flags` → `df["flags"]`, inside
`descriptive()`.

**No decision rule, threshold, acceptance criterion, estimator or frozen
schedule changed.** This is demonstrated by
`python -m ncclenergy.v2.amendment_check` (reproduction step
`gate3b_amendment_check`), which on the same raw data shows:
* the frozen copy's git blob matches `f4735b1`;
* the source diff between the frozen copy and the current `measval.py`
  is exactly the two lines quoted above (one removed, one added);
* the frozen `descriptive()` fails with `KeyError: 0`;
* the processes table (acceptance and exclusions) is identical;
* every pre-specified output is identical: n_processes 72, n_accepted
  70, control ESTABLISHED, and D1 INCONCLUSIVE, D2 INCONCLUSIVE, D3
  SYSTEMATIC_OFFSET, D4 NOT_SUFFICIENT with all their numbers.

**Regression test**:
`tests/v2/test_v2_measval.py::test_descriptive_runs_on_a_frame_with_a_flags_column`
fails on `f4735b1` and passes now.

## AM-2 — campaign script exit status (commit `4438d07`)

**Defect**: the last statement in the block loop of
`scripts/run_confirmatory_v2.sh` was `[ "$rc" -ne 0 ] && break`. After
the final successful block the test is false, so the script exited 1
even though every block's runner returned 0. The bug was present since
v2 (`63c5332`).

**Change**:
* `if [ "$rc" -ne 0 ]; then break; fi`;
* `rc=0` before the loop;
* `exit "$rc"` after it.

The signal (130) and wall-budget (3, 130) paths are unchanged.

**No collected measurement is affected**:
* The gate 3b data were collected with the frozen script. The faulty
  status was produced only after the last block had finished.
* No raw file changed in `4438d07`.
* `tests/v2/test_v2_deps_and_script.py::test_exit_status_fix_does_not_change_runner_invocations`
  runs the frozen and the fixed script with a recording stub runner. It
  shows that both start the runner identically: argument vector, working
  directory, environment (`PYTHONPATH`, `CAMPAIGN`, `CUDA_HOME`, NCCL
  variables) and number of blocks. Only the script's own exit status
  differs: 1 for the frozen script and 0 for the fixed one.
* `test_campaign_script_exit_status_reflects_the_blocks` checks the new
  behaviour on success and on a failing block.

## Additions that are not amendments

None of these modules changes a pre-specified output:

| module | role | labeled |
|---|---|---|
| `src/ncclenergy/v2/measval_posthoc.py` | temperature association and matched-window estimator gap | **POST-HOC EXPLORATORY** in code, output JSON and report §5 |
| `src/ncclenergy/v2/amendment_check.py` | the AM-1 demonstration above | check |
| `src/ncclenergy/v2/release_repro.py`, `scripts/open_audit/` | release reproduction and input audit | tooling |

Gate 3b's thresholds, acceptance rules, block count and exclusions are
unchanged. No unit was rerun and no block was added.
