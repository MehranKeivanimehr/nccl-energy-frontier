# Public integration review

Use the corrected sanitized candidate, not the private repository history or the
damaged delivered expansions. Keep the existing public repository; the extension
belongs on a reviewed content-only branch. No branch/commit/push was made by this
review. The source remains an analysis-input subset, not a complete raw archive.

1. Verify the candidate archive's `.sha256`, extract into a newly created empty
   directory, then run `sha256sum -c --quiet SHA256SUMS` inside the extracted tree.
   Check the frozen original export manifest separately only against that original
   export; it is not the current candidate manifest.
2. Install the documented analysis/test environments and reproduce from this
   exact tree. Inspect numeric/provenance differences, not only process exit codes.
3. Begin only in a clean clone of the existing public repository. Preserve any
   unrelated edits instead of resetting them:

   ```bash
   cd /path/to/public-clone
   test -z "$(git status --porcelain)" || { echo "Review existing changes first"; exit 1; }
   git switch -c extension-pilot-reviewed
   ```

4. Compare the candidate with the clone before copying. Do not use automatic
   deletion. Review old files that may otherwise remain stale, including GPU
   launchers. Keep original pilot numerical records; sanitize private strings
   only with a derivative ledger if additional sanitization is needed.
5. Copy only candidate publication files. Exclude generated `logs/`, caches,
   environments, native build outputs and any `.git`. Never merge/cherry-pick the
   producer's private branch. Repeat tests and analysis in the prospective public
   checkout; a pass in a standalone candidate is not a pass in an untested overlay.
6. Review staged privacy and the full diff before an authentic new commit:

   ```bash
   git add .
   git update-index --chmod=+x scripts/*.sh
   git diff --cached --stat
   git diff --cached --check
   git status --short
   ```

   Do not rewrite raw files just to silence trailing-whitespace warnings. Verify
   that no required manifest-listed evidence is excluded by `.gitignore`, that
   the retained raw data and frozen results have their expected hashes, and that
   new files are included in the current manifest.

   ```bash
   git commit -m "Add reviewed NCCL board-energy pilot extension"
   ```

These are recommended review commands, not commands executed here. Choose normal
current authorship; do not backdate or invent previous work. There is deliberately
no push command. Publication needs the owner's final review.

Suggested description (under 120 characters):

NCCL latency and GPU-board energy pilot on RTX A6000: verified configurations, raw-derived analysis and limitations.

Suggested topics: `nccl`, `cuda`, `gpu`, `energy-efficiency`, `collective-communication`,
`distributed-training`, `benchmarking`, `reproducible-research`.
