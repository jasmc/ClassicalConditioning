// Verify collection screen/responsive/print layout and capture standalone comparisons.
// Usage: node this.cjs repository-root browser-executable [temporary-output-directory]
// Requires Playwright on NODE_PATH and the external reviews directory served at port 8766.
const fs=require('fs');const {chromium}=require('playwright');
const path=require('path'); const output=process.argv[4]||require('os').tmpdir();
(async()=>{const root=process.argv[2];const browser=await chromium.launch({executablePath:process.argv[3],headless:true});const page=await browser.newPage({viewport:{width:1000,height:1000},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(String(e)));await page.goto('http://127.0.0.1:8766/main-paper-figures.html');await page.evaluate(()=>document.fonts.ready);await page.screenshot({path:path.join(output,'main-paper-screen.png'),fullPage:true});let reports=[];
for(const n of [1,2]){const figure=page.locator('#fig'+n);reports.push(await figure.evaluate(el=>({id:el.id,artwork:el.querySelector('.artwork').getBoundingClientRect().toJSON(),legend:el.querySelector('figcaption').getBoundingClientRect().toJSON(),fonts:[...el.querySelector('.artwork').shadowRoot.querySelectorAll('text')].slice(0,3).map(n=>getComputedStyle(n).fontFamily)})));}
await page.pdf({path:path.join(output,'main-paper-print.pdf'),preferCSSPageSize:true,printBackground:true});
// Compare the namespaced embedded artwork with its exact standalone selected SVG.
for(const n of [1,2]){await page.goto('http://127.0.0.1:8766/main-paper-figures.html');await page.evaluate(n=>{for(const f of document.querySelectorAll('figure'))if(f.id!=='fig'+n)f.remove();document.querySelector('main').style.margin='0';document.querySelector('main').style.maxWidth='none'},n);await page.evaluate(()=>document.fonts.ready);await page.locator('.artwork').screenshot({path:path.join(output,'collection-fig'+n+'.png')});
const s=JSON.parse(fs.readFileSync(root+`/docs/paper/figures/figure-0${n}/current-state.json`));const p=await browser.newPage();await p.goto('file://'+s.draft.working_svg.path);await p.evaluate(()=>document.fonts.ready);await p.locator('svg').first().screenshot({path:path.join(output,'standalone-fig'+n+'.png')});await p.close();}
// Responsive narrow view remains inside the viewport.
await page.setViewportSize({width:360,height:900});await page.goto('http://127.0.0.1:8766/main-paper-figures.html');await page.evaluate(()=>document.fonts.ready);const mobile=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,figures:document.querySelectorAll('figure').length}));
if(errors.length||mobile.scrollWidth>mobile.width||mobile.figures!==2||reports.some(r=>Math.abs(r.artwork.width-183*96/25.4)>.1||r.legend.top<r.artwork.bottom))throw Error('Collection rendering check failed');
const collection=JSON.parse(fs.readFileSync(root+'/docs/paper/figures/main-paper-current-state.json'));
fs.writeFileSync(path.join(output,'paper-html-browser-check.json'),JSON.stringify({html_sha256:collection.html.sha256,reports,mobile,errors},null,2));await browser.close()})().catch(e=>{console.error(e);process.exit(1)});
