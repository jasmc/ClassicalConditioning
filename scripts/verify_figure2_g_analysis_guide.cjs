// Browser checks for the guide we build. Uses a fresh context, never user notes.
// node script.cjs guide-url browser-executable new-qa-directory
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const {chromium} = require('playwright');
const assert = require('assert/strict');
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
(async () => {
  const [url, executablePath, out] = process.argv.slice(2);
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({executablePath,headless:true});
  try {
    const context = await browser.newContext({viewport:{width:1350,height:1000},acceptDownloads:true});
    const page = await context.newPage(), errors=[], badResponses=[];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('response', r => {if (r.status() >= 400) badResponses.push({url:r.url(),status:r.status()});});
    await page.goto(url, {waitUntil:'networkidle'});
    await page.evaluate(() => document.fonts.ready);
    const structure=await page.evaluate(() => ({
      title:document.title, chapters:document.querySelectorAll('.chapter').length,
      notebooks:document.querySelectorAll('.notebook').length,
      unresolved:/\{\{[^{}]+\}\}/.test(document.body.textContent),
      deadAnchors:[...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.getElementById(decodeURIComponent(a.hash.slice(1)))).map(a=>a.hash),
      duplicateIds:[...document.querySelectorAll('[id]')].map(n=>n.id).filter((x,i,a)=>a.indexOf(x)!==i),
      currentG:document.querySelector('#figure-current-g svg').getAttribute('viewBox'),
      otherPanelsInG:document.querySelector('#figure-current-g').querySelectorAll('#panel-H,#panel-I,#panel-D').length,
      currentLegend:document.querySelector('#figure-current-g + details').textContent,
    }));
    assert.equal(structure.chapters,14);assert.equal(structure.notebooks,13);
    assert.equal(structure.unresolved,false);assert.deepEqual(structure.deadAnchors,[]);
    assert.deepEqual(structure.duplicateIds,[]);assert.equal(structure.otherPanelsInG,0);
    assert.ok(structure.currentLegend.includes('(g) Delay experiment'));
    assert.ok(!structure.currentLegend.includes('(h)'));
    // Open all galleries for asset decoding and capture the exact current excerpt.
    await page.evaluate(() => {for(const el of document.querySelectorAll('details'))el.open=true;for(const img of document.images)img.loading='eager';});
    const decoded=await page.evaluate(async()=>{await Promise.all([...document.images].map(i=>i.decode()));return [...document.images].map(i=>({src:i.getAttribute('src'),width:i.naturalWidth,height:i.naturalHeight}));});
    assert.ok(decoded.every(i=>i.width>0&&i.height>0));
    await page.evaluate(() => {for(const el of document.querySelectorAll('details'))el.open=false;document.getElementById('chapter-menu').open=true;});
    await page.goto(url+'#04-exactly-what-selected-figure-2-g-shows');
    await page.locator('#figure-frozen-g').screenshot({path:path.join(out,'frozen-g.png')});
    await page.locator('#figure-current-g').screenshot({path:path.join(out,'current-g.png')});
    await page.goto(url+'#06-the-independent-analysis-and-its-numerical-results');
    const resultTable = page.locator('[id="06-the-independent-analysis-and-its-numerical-results"] table').first();
    await resultTable.screenshot({path:path.join(out,'phase-results.png')});
    const phaseText=await resultTable.innerText();
    for(const value of ['0.0947 [0.0117, 0.1778]','0.1029 [0.0386, 0.1673]','0.0978 [−0.0477, 0.2434]','989 / 999'])assert.ok(phaseText.includes(value),value);
    // Notebook persistence and real JSON download.
    const notebook=page.locator('.notebook').first();await notebook.locator('summary').click();
    const sample='I can distinguish model means from observed medians. <script>window.bad=1</script>';
    await notebook.locator('[data-field="understanding"]').fill(sample);
    await notebook.locator('[data-field="understood"]').check();
    await notebook.locator('[data-field="position"]').selectOption('questions');
    await page.reload({waitUntil:'networkidle'});
    assert.equal(await notebook.locator('[data-field="understanding"]').inputValue(),sample);
    assert.equal(await notebook.locator('[data-field="understood"]').isChecked(),true);
    const downloading=page.waitForEvent('download');await page.locator('#export-notes').click();
    const download=await downloading;await download.saveAs(path.join(out,'test-notebook-export.json'));
    const exported=JSON.parse(fs.readFileSync(path.join(out,'test-notebook-export.json')));
    const first=Object.keys(exported.sections)[0];assert.equal(exported.sections[first].understanding,sample);
    assert.equal(exported.schema,'figure2-g-review-notebook/1');
    // Version mismatch must be exposed, and import must not apply before confirm.
    const imported=structuredClone(exported);imported.guide_id='another-version';imported.source_fingerprint='different-evidence';
    imported.sections[first].understanding='Imported text: provisional only';
    await page.locator('#import-file').setInputFiles({name:'notes.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(imported))});
    await page.locator('#import-dialog').waitFor({state:'visible'});
    assert.ok(await page.locator('#import-dialog').isVisible());
    const warnings=await page.locator('#import-warning').innerText();
    assert.ok(warnings.includes('different guide version'));assert.ok(warnings.includes('fingerprint differs'));
    assert.equal(await notebook.locator('[data-field="understanding"]').inputValue(),sample);
    await page.locator('#cancel-import').click();
    assert.equal(await notebook.locator('[data-field="understanding"]').inputValue(),sample);
    await page.locator('#import-file').setInputFiles({name:'notes.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(imported))});
    await page.locator('#import-dialog').waitFor({state:'visible'});
    const backupPending=page.waitForEvent('download');await page.locator('#confirm-import').click();
    await (await backupPending).saveAs(path.join(out,'test-notebook-backup.json'));
    assert.equal(await notebook.locator('[data-field="understanding"]').inputValue(),'Imported text: provisional only');
    assert.equal(JSON.parse(fs.readFileSync(path.join(out,'test-notebook-backup.json'))).sections[first].understanding,sample);
    await page.locator('#import-file').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from('{bad')});
    await page.waitForFunction(()=>document.getElementById('import-message').textContent.includes('Import rejected'));
    assert.ok((await page.locator('#import-message').innerText()).includes('Import rejected'));
    assert.equal(await notebook.locator('[data-field="understanding"]').inputValue(),'Imported text: provisional only');
    // Input text never becomes HTML.
    assert.equal(await page.evaluate(()=>window.bad),undefined);
    const search=page.locator('#guide-search');await search.fill('WIP');
    assert.ok((await page.locator('#search-results').innerText()).includes('Legacy learner'));
    await search.fill('');
    await page.evaluate(()=>localStorage.removeItem(GuideNotebook.storageKey));
    await page.reload({waitUntil:'networkidle'});
    await page.screenshot({path:path.join(out,'guide-desktop.png')});
    // Narrow browser: main content must fit, tables scroll within their wrappers.
    await page.setViewportSize({width:390,height:844});await page.reload({waitUntil:'networkidle'});
    const mobile=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,menuOpen:document.getElementById('chapter-menu').open}));
    assert.ok(mobile.scrollWidth<=mobile.width);assert.equal(mobile.menuOpen,false);
    await page.screenshot({path:path.join(out,'guide-narrow.png')});
    await page.goto(url+'#11-presentation-alternatives-asterisks-and-neuroscience-examples');
    await page.locator('#figure-bracket-g').screenshot({path:path.join(out,'bracket-g.png')});
    await page.setViewportSize({width:1350,height:1000});
    await page.goto(url);
    await page.pdf({path:path.join(out,'guide-print-check.pdf'),preferCSSPageSize:true,printBackground:true});
    // Browser saving unavailable must be explicit, with export still usable.
    const blocked=await browser.newContext({viewport:{width:1100,height:800},acceptDownloads:true});
    await blocked.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw new Error('storage disabled for test');}}));
    const off=await blocked.newPage();await off.goto(url,{waitUntil:'networkidle'});
    assert.ok((await off.locator('#save-status').innerText()).includes('unavailable'));
    const offnote=off.locator('.notebook').first();await offnote.locator('summary').click();
    await offnote.locator('[data-field="questions"]').fill('Keep this note through export.');
    const offdownload=off.waitForEvent('download');await off.locator('#export-notes').click();
    await (await offdownload).saveAs(path.join(out,'test-no-storage-export.json'));
    const noStorage=JSON.parse(fs.readFileSync(path.join(out,'test-no-storage-export.json')));
    assert.equal(Object.values(noStorage.sections)[0].questions,'Keep this note through export.');
    await blocked.close();
    assert.deepEqual(errors,[]);assert.deepEqual(badResponses,[]);
    const report={url,checked_at:new Date().toISOString(),structure,decoded_images:decoded,mobile,errors,badResponses,
      tests:['source anchors and unique IDs','G-only current excerpt and matching legend','all artwork decoded','saved phase values','note persistence after reload','real export download','import mismatch preview','cancel preserves notes','confirmed import preserves downloadable backup','malformed import preserves notes','text treated as text','chapter search','narrow viewport','print rendering','storage-disabled export'],
      artifacts:fs.readdirSync(out).filter(n=>fs.statSync(path.join(out,n)).isFile()).map(n=>({path:n,sha256:sha(fs.readFileSync(path.join(out,n)))}))};
    fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({passed:true,tests:report.tests.length,images:decoded.length,mobile,report:path.join(out,'browser-verification.json')},null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
