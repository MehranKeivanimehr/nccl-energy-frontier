// Render the editable Persian HTML with a browser that supports RTL shaping.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    ...(process.env.PDF_BROWSER ? { executablePath: process.env.PDF_BROWSER } : {}),
    headless: true, args: ['--disable-gpu', '--no-first-run'],
  });
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(path.join(__dirname, 'NCCL_Energy_Frontier_FA.html')).href);
    await page.emulateMedia({media: 'print'});
    await page.evaluate(() => document.fonts.ready);
    const layout = await page.evaluate(() => [...document.querySelectorAll('.page')].map(section => {
      const content = section.querySelector('.body-content').getBoundingClientRect();
      const footer = section.querySelector('.footer').getBoundingClientRect();
      const horizontal = [...section.querySelectorAll('table,pre,.formula,img,.bib')].filter(el => {
        const r = el.getBoundingClientRect(), s = section.getBoundingClientRect();
        return r.right > s.right - 20 || r.left < s.left + 20;
      }).map(el => el.tagName);
      return {page: Number(section.dataset.page), bottomGapPx: Math.round(footer.top-content.bottom), horizontal};
    }));
    const problems = layout.filter(p => p.bottomGapPx < 9 || p.horizontal.length);
    if (problems.length) throw new Error(JSON.stringify(problems));
    await page.pdf({path: path.join(__dirname, 'NCCL_Energy_Frontier_FA.pdf'), format: 'A4',
      printBackground: true, preferCSSPageSize: true, tagged: true, outline: true});
    fs.writeFileSync(path.join(__dirname, 'layout_checks.json'), JSON.stringify({pages:layout.length,problems},null,2)+'\n');
    console.log(JSON.stringify({pages: layout.length, layoutProblems: problems.length}));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
