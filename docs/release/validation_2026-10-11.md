# Local verification of the corrected pilot candidate

Date: 2026-10-11. No GPU run, original-data edit, Git mutation or publication.

| Check | Result | Boundary |
|---|---|---|
| Full CPU suite | 124/124 passed | Linux/WSL, CPU PyTorch, fake NVML and actual C plugin harnesses; not real CUDA/NCCL validation |
| Analysis reproduction | 14/14 steps passed | Historical references preserved; numeric tolerance rel 1e-9 / abs 1e-12 where parsing is needed |
| Frozen outputs | 145 hashed; 0 changed | `data/processed`, v2/v2.2 processed/reanalysis outputs and historical figures |
| Input-open audit | 0 undeclared outside reads | Reproduction's instrumented file-open audit, not a network or all-syscall audit |
| Required rounded numerical strings | 17/17 README, 29/29 claims, 45/45 Gate3b report | 65 key-number rows; this checks consistency, not scientific truth |
| Historical strict workload gate | 50/50 rejected | Missing historical numerical evidence; default descriptive results retained |
| New native patch | Applies to pinned `afd59ab`; actual mode/read/probe helpers pass CPU compilation/test | Full CUDA build and real Pascal behavior untested; collection disabled |

Environment: Ubuntu 22.04 WSL, Python 3.10.12, NumPy 1.26.1, pandas 2.3.3,
SciPy 1.11.3, Matplotlib 3.10.7, PyYAML 6.0.1, CPU PyTorch 2.9.1+cpu,
GCC 11.2.0/Make 4.3. The producer originally used Python 3.11; this check
demonstrates reproduction in the stated local environment, not an exact
environment clone. Dependencies/toolchain were isolated outside publication
contents. Later changes were figure captions/presentation and documentation;
the final presentation step was regenerated and compared separately.

Per-step reproduction includes byte identity, numerical equality, repository-path
normalization and regenerated figures. Differences from the earlier v2.1 analysis
are explicitly contextual, not silently compared with the wrong code version.
`figures/full/method_example_trace.png` remains a declared illustrative,
unreproduced figure: its launch identity was not recorded and no empirical number
depends on it. Its preservation does not make it independently verified evidence.

New figures were inspected for axes, titles, scope labels and visible uncertainty.
All historical numerical conclusions and Gate3b decisions remain unchanged.
The measurement precision gate still fails; post-hoc adjustment does not rescue
it. Training equivalence, distinct optima, external energy calibration, real
guarded-plugin behavior and replication on Pascal remain unresolved.

The original export manifest and current root manifest serve different purposes.
Checksums detect changes, not correctness. This release is suitable for review
as a bounded pilot with explicit omissions; the exact public overlay must pass
its own tests/privacy review before any commit or push.
