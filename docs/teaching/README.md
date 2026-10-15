# Persian educational guide

[Read the 50-page PDF](NCCL_Energy_Frontier_FA.pdf) or open the [editable HTML](NCCL_Energy_Frontier_FA.html). It explains the original pilot, v2 extension, measurement characterization, calibration and held-out outcomes. It is an educational account, not an additional experiment or peer-reviewed paper.

The original local edition remains unchanged. This public edition makes two editorial changes: a transient workflow sentence is made tool-neutral, and the evidence appendix explains the public/private file mapping. Numerical content and figures are unchanged.

The appendix retains identifiers of the original private evidence packets and local review reports; these names do not imply that their original ZIPs or every review file is published. Public scientific records, source, results and protocols are in [`studies/a6000-stage2-v2.5/`](../../studies/a6000-stage2-v2.5/). The [disposition inventory](../../studies/a6000-stage2-v2.5/provenance/FILE_DISPOSITIONS.csv) and [omission note](../../studies/a6000-stage2-v2.5/provenance/OMITTED_STAGE2.md) describe the export.

Page 40 describes the original private strict-verification layout. For this
public export use the [public replay adapter](../../studies/a6000-stage2-v2.5/scripts/reproduce_public.py),
not the original private command against transformed freezes.

The HTML embeds the figures, styles and bibliography, so it is standalone. To re-render the same educational content, install Node.js and Playwright, then use a browser with Persian fonts:

```sh
npm install --no-save playwright
npx playwright install chromium
node docs/teaching/render_pdf.cjs
```

Alternatively set `PDF_BROWSER` to an installed Chromium/Edge executable. Tahoma/Arial-compatible Persian fonts affect layout; inspect the resulting PDF and `layout_checks.json` before replacing it. Rendering is document work, not GPU benchmarking. The original scientific figure values must not be changed when editing presentation.
