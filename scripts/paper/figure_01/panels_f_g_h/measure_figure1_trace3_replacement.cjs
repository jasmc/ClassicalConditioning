// Render only the exact temporary Figure 1 candidate; no network or state writes.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require('playwright');
const digest = text => crypto.createHash('sha256').update(text).digest('hex');

(async () => {
  const dir = process.argv[2];
  const browser = await chromium.launch({ executablePath: process.argv[3], headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1000, height: 1400 }, deviceScaleFactor: 2 });
    const html = fs.readFileSync(path.join(dir, 'candidate.html'), 'utf8');
    // An intercepted in-memory localhost document supplies a secure context for
    // the existing collector's SHA-256 API. No server or outbound fetch is used.
    await page.route('**/*', route => route.request().url() === 'http://127.0.0.1/figure1-candidate'
      ? route.fulfill({ body: html, contentType: 'text/html' }) : route.abort());
    await page.goto('http://127.0.0.1/figure1-candidate');
    await page.evaluate(() => document.fonts.ready);
    const collectorPath = path.join(process.cwd(), 'src/classical_conditioning/figures/layout_frames_browser.js');
    const collector = fs.readFileSync(collectorPath, 'utf8');
    await page.addScriptTag({ content: collector });
    const svg = fs.readFileSync(path.join(dir, 'candidate.svg'), 'utf8');
    const report = await page.evaluate(async ({ svg, hash }) => collectLayoutFrames(svg, hash),
      { svg, hash: digest(collector) });
    report.render_helper = { path: __filename, sha256: digest(fs.readFileSync(__filename)) };
    report.visible_label = await page.locator('main>svg text').allTextContents().then(v => v.filter(x => /^202\d{5}_\d\d$/.test(x)));
    report.associated_legend = await page.locator('.legend').innerText();
    report.exact_embedded_svg = await page.locator('main>svg').evaluate(node => node.getAttribute('width') === '136mm');
    fs.writeFileSync(path.join(dir, 'browser.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
    await page.locator('main>svg').screenshot({ path: path.join(dir, 'preview.png') });
    await page.addStyleTag({ content: 'main{width:136mm;margin:0;padding:0}main>svg{width:136mm}.legend{font-size:12px}' });
    await page.screenshot({ path: path.join(dir, 'final-scale.png'), fullPage: true });
    console.log(JSON.stringify({ artifact_sha256: report.artifact_sha256, issues: report.issues,
      fonts_ready: report.fonts_ready, images_ready: report.images_ready, labels: report.visible_label }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
