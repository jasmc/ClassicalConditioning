// Saved Figure 2 exploration: exact frame collection and scoped column checks.
// Uses the maintained collector; never mutates the SVG or presentation state.
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const {chromium} = require('playwright');
const digest = b => crypto.createHash('sha256').update(b).digest('hex');
(async () => {
  const [directory, temporary, edge, url] = process.argv.slice(2);
  fs.mkdirSync(temporary, {recursive:true});
  const svg = fs.readFileSync(path.join(directory, 'figure2.svg'), 'utf8');
  const collectorFile = path.join(process.cwd(), 'src/classical_conditioning/figures/layout_frames_browser.js');
  const collector = fs.readFileSync(collectorFile, 'utf8');
  const browser = await chromium.launch({executablePath:edge, headless:true});
  try {
    const p = await browser.newPage({viewport:{width:1100,height:1000}, deviceScaleFactor:2});
    const errors=[];
    p.on('pageerror', e => errors.push(String(e)));
    await p.goto(url, {waitUntil:'networkidle'});
    await p.evaluate(() => document.fonts.ready);
    await p.addScriptTag({content:collector});
    const report = await p.evaluate(async ({svg, hash}) => await collectLayoutFrames(svg, hash),
                                    {svg, hash:digest(collector)});
    fs.writeFileSync(path.join(temporary,'browser.json'),JSON.stringify(report,null,2));
    await p.locator('.artwork').screenshot({path:path.join(temporary,'preview.png')});
    await p.screenshot({path:path.join(temporary,'review.png'),fullPage:true});
    const visible = await p.evaluate(() => {
      const artworks=[...document.querySelectorAll('.artwork')];
      const svgs=artworks.map(n=>n.shadowRoot?.querySelector('svg'));
      return {figure_count:artworks.length, svg_count:svgs.filter(Boolean).length,
              legend:document.querySelector('figcaption')?.innerText,
              viewport_width:innerWidth, artwork_width_px:artworks[0]?.getBoundingClientRect().width};
    });
    // Use a native-size import of the exact saved SVG for letter and text checks.
    const text = await p.evaluate(async svg => {
      const host=document.createElement('div');host.style.cssText='position:absolute;left:0;top:0';
      const root=document.importNode(new DOMParser().parseFromString(svg,'image/svg+xml').documentElement,true);
      host.attachShadow({mode:'open'}).append(root);document.body.append(host);await document.fonts.ready;
      const mm=25.4/96;
      const bounds=node=>{const b=node.getBBox(),m=node.getScreenCTM();
        const q=[[b.x,b.y],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));
        return [q[0].x*mm,q[0].y*mm,(q[1].x-q[0].x)*mm,(q[1].y-q[0].y)*mm]};
      const letters=Object.fromEntries('ABCDEFGHI'.split('').map(k=>[k,bounds(root.querySelector('[id="'+k+'-letter"]'))]));
      const fonts=[...root.querySelectorAll('text')].map(n=>{const m=n.getScreenCTM();return {id:n.parentNode.id,text:n.textContent,
        family:getComputedStyle(n).fontFamily,size_pt:parseFloat(getComputedStyle(n).fontSize)*Math.hypot(m.c,m.d)*72/96}});
      host.remove();return {letters,fonts};
    },svg);
    const columns = [['A-pair','D-pair','G-observed'],['B-pair','E-pair','H-observed'],['C-pair','F-pair','I-observed']]
      .map((ids,i)=>{const centres=ids.map(k=>{const r=report.data_axes[k];if(!r)throw Error('Unmeasured '+k);return r[0]+r[2]/2});
        const letters=['ADG','BEH','CFI'][i].split('').map(k=>text.letters[k][0]);
        return {ids,centres_mm:centres,centre_spread_mm:Math.max(...centres)-Math.min(...centres),
          panel_letter_lefts_mm:letters,panel_letter_spread_mm:Math.max(...letters)-Math.min(...letters)};});
    const alignment={artifact_sha256:digest(svg),status:columns.every(c=>c.centre_spread_mm<.02&&c.panel_letter_spread_mm<.02)&&!errors.length?'passed':'failed',
      tolerance_mm:.02,columns,visible_scope:visible,page_errors:errors,
      measurement_implementation:{path:__filename,sha256:digest(fs.readFileSync(__filename))},
      collector:{path:collectorFile,sha256:digest(collector)},text_measurements:text};
    fs.writeFileSync(path.join(temporary,'column-measurements.json'),JSON.stringify(alignment,null,2));
    console.log(JSON.stringify({issues:report.issues,allocations:report.allocations,columns,status:alignment.status,
      visible_scope:{figure_count:visible.figure_count,svg_count:visible.svg_count},page_errors:errors},null,2));
  } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
