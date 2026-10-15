# Completed A6000 pilot: release preparation

Prepared 2026-10-14 from public baseline
`2b73580e848986905c45bc8ee10706383dc5cd72` (`extension-pilot-reviewed`).
Local branch: `a6000-heldout-release`. No commit, push or publication was
performed during preparation. Public `main` remains `8868457`.

## Changes and preservation

Seven existing presentation/configuration files changed: `.gitignore`,
`README.md`, `configs/v2/replication/template.yaml`,
`docs/release/{claims,methodology,reproduction}.md`, and
`docs/v2/replication_1080ti_titan.md`. The unused replication placeholder
received a YAML quoting fix and an archived-status label; it was never an
executed condition.

Added `studies/a6000-stage2-v2.5/`: final matched source, native patches,
build identities, tests, protocols, schedules, privacy derivatives of all
new scientific raw records, results, two reproducible figures, numerical
validation and file-level provenance. The scientific records comprise:

| Campaign | Launches | Retained scientific files | Role |
|---|---:|---:|---|
| Smoke | 3 | 45 | Pipeline check |
| Calibration | 23 | 305 | One-session characterization, not an absolute calibration |
| Held-out | 72 | 810 | Three same-evening sessions, 24 launches each |

Added the 50-page Persian PDF, editable standalone HTML and renderer under
`docs/teaching/`, and a new current-release manifest under `provenance/`.
The original local educational edition is preserved. Its public edition
changes only a transient workflow sentence and the public/private evidence
appendix; numerical content and embedded figures are unchanged.

No original/private or previously tracked file was deleted. Historical
source, raw/processed data, figures and numerical outputs remain unchanged.
An archival Stage 2 build record was placed in `code/build_records/` rather
than occupying the frozen build script's future output directory. Source
and build-script hashes are not changed for that relocation.

The [file dispositions](../../studies/a6000-stage2-v2.5/provenance/FILE_DISPOSITIONS.csv),
[original/public digest mapping](../../studies/a6000-stage2-v2.5/provenance/DERIVATIVES_STAGE2.csv)
and [packet-member inventory](../../studies/a6000-stage2-v2.5/provenance/PACKET_MEMBERS.csv)
explain retained, derived and omitted items. The 1,475 delivered packet
members were inventoried. Duplicate ZIPs, private correspondence, superseded
preparation material and compiled/external runtimes are not supplied as
though they were public scientific inputs. Older public-export omissions
remain documented; this addition does not claim to recover missing old
training numerics or every omitted v1 forensic trace.

## Scientific wording corrections

The current documents distinguish observed NVML-reported board energy from
externally calibrated physical energy. Read-frequency dependence is
observed; a read-reset/discard mechanism remains a hypothesis, not an NVML
guarantee. New held-out results do not replace old Gate 3b decisions. No
latency/energy global optimum, real-training correctness, exclusive host
access, topology-independent scaling, five completed v1 trials or 8-GPU
measurement is claimed. Ti/Titan replication is explicitly unexecuted and
outside the closed scope.

The selected C/A 40 s comparison is +1.6454% reported energy, with descriptive
95% CI [+1.0047%, +2.2901%]. Its precision target was missed. The B/A 10 s
mean-ratio control is compatible with the prespecified equivalence margin,
not a physical-error bound. All five aggregate adequacy scopes remain
unmet/unestablished. See [results](../../studies/a6000-stage2-v2.5/docs/results.md)
and [claims](../../studies/a6000-stage2-v2.5/docs/claims.md).

## Privacy, provenance and numerical behavior

Necessary public derivatives alias real GPU UUIDs and replace personal home
paths, hostnames and private network addresses. Numeric readings, time
values, operation counts, missingness and decisions are not changed to
anonymize records. The original private archives/freezes remain immutable.
Published freeze views retain original digest identities and are not
misrepresented as their original byte strings.

No frozen numerical kernel or original inference rule changed. New public
adapters add manifest/identifier checks and separate output paths, not new
estimators. The independent formula checker changes only the existing
audit's I/O layout. Its exploratory latency and floating-point diagnostic
fields do not change any prespecified decision.

Irrelevant current workflow prose was removed, not genuine provenance,
material allocation/delegation facts, historical environment inventories or
third-party notices. There is no assertion of exclusive or unaided
authorship, and this package is not a certificate that all tool-name strings
were erased. Git history and authorship evidence were not manufactured or
rewritten.

## Local verification

| Check | Result and scope |
|---|---|
| Historical analysis replay | 14/14 steps pass; 145 frozen files hashed, 0 changed; 0 undeclared outside data reads |
| Public held-out raw-to-result replay | 3,557 numeric fields checked, max absolute difference 2.45e-15; all decisions/missingness unchanged; six scientific TSVs identical text |
| Calibration raw-to-result replay | 23/23 launches; 1,918 finite numeric fields checked, max absolute difference 4.40e-13; original infinite readiness-wait diagnostic preserved |
| Independent statistical formulas | 55 contrasts checked with closed-form df=2 t distribution, TOST and Holm; no differences at 1e-10 tolerance |
| New public-export guards | 18/18 pass on CPU |
| Existing public root tests | 118/124 pass locally; 6 compiler-dependent tests cannot complete because GCC/G++/Make are absent; runner exit 1, not an all-green suite |
| Archival Stage 2 tests in public layout | 91 pass, 8 fail, 12 not run; includes original-private compiled-fixture requirements, a stale frozen draft test and explicitly counted early-return skips; not advertised as a green public suite |
| PDF | 50/50 pages rendered/inspected; A4; 0 layout overflows/replacement characters/private path hits; 17 external source links |

The archival test failures are not hidden or repaired by weakening frozen
identity checks: five production fixtures need omitted compiled artifacts;
one private-byte/window regression cannot pass privacy derivatives; one
manifest builder needs the omitted external NCCL library; one frozen test
incorrectly expects already executed configs to remain drafts. Those tests
are distinct from the manifest-aware public replay. Native paths are not
called newly hardware-validated because CPU mocks passed.

Machine-readable checks are in the study's
[`provenance/validation/`](../../studies/a6000-stage2-v2.5/provenance/validation/).
One historical illustrative trace figure remains explicitly
`KNOWN_UNREPRODUCED`; regenerated image bytes are not all required to match
across platforms. Neither exception changes a scientific number.

## Review and publish

Open the prepared working copy (it contains the real public Git history),
not a new `git init` inside the ZIP. The ZIP is a content archive without
`.git`, caches, venvs, temporary logs or compiled outputs.

After reviewing the report and diff, from that working copy:

```sh
git branch --show-current
git status --short
git diff --check
git add .
git update-index --chmod=+x studies/a6000-stage2-v2.5/code/scripts/*.sh
git diff --cached --stat
git commit -m "Add completed A6000 measurement-characterization pilot"
git push -u origin a6000-heldout-release
```

Expected branch is `a6000-heldout-release`. Do not force-push, rewrite old
history, or format raw logs/patches to remove their original whitespace.
The commands above are recommendations, not commands executed in this
preparation. A normal PR can target `extension-pilot-reviewed`; no new
repository is scientifically necessary.

Suggested repository description (under 120 characters):

> Pilot study of NCCL latency and NVML-reported GPU-board energy on RTX A6000, with reproducible measurement evidence.

Suggested topics: `nccl`, `gpu-energy`, `rtx-a6000`,
`collective-communication`, `hpc`, `benchmarking`, `reproducibility`,
`pilot-study`.

The next step is human review and publication of this bounded pilot, not
another GPU campaign to force a favorable or conclusive result.
