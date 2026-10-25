const repositoryRoot=(()=>{const f=require('fs'),p=require('path');let d=__dirname;while(!f.existsSync(p.join(d,'pyproject.toml'))){const parent=p.dirname(d);if(parent===d)throw Error('Repository root unavailable');d=parent;}return d;})();
// Exact-SVG browser measurements and a static preview; no figure state is changed.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('playwright');
(async()=>{
  const [directory,executable,preview,baseURL]=process.argv.slice(2);
  const browser=await chromium.launch({executablePath:executable,headless:true});
  const page=await browser.newPage({viewport:{width:1100,height:1200}});
  const data=fs.readFileSync(path.join(directory,'figure3.svg'),'utf8');
  await page.goto(baseURL+'/review.html');
  await page.evaluate(()=>document.fonts.ready);
  const text=await page.locator('svg').first().textContent();
  if(!text.includes('Individual evidence')||!text.includes('Larger acquisition contrast')||!text.includes('Intermediate contrast')||!text.includes('Near-zero contrast'))throw Error('Wrong figure scope');
  if(/WIP|Nominal|Improved|Legacy rule|learner flag|Rules disagree/.test(text))throw Error('Multiple-method content in single-path figure');
  const actualIDs=await page.locator('svg [id]').evaluateAll(nodes=>nodes.map(e=>e.id));
  if(actualIDs.length!==new Set(actualIDs).size)throw Error('Duplicate SVG ID');
  const loadedRoot=await page.locator('svg').first().evaluate(e=>({width:e.getAttribute('width'),height:e.getAttribute('height')}));
  if(loadedRoot.width!=='183mm'||loadedRoot.height!=='158mm')throw Error('Wrong selected artwork geometry');
  await page.locator('svg').first().screenshot({path:preview});
  const source=fs.readFileSync(path.join(repositoryRoot, 'src/classical_conditioning/figures/layout_frames_browser.js'),'utf8');
  const digest=crypto.createHash('sha256').update(source).digest('hex');
  await page.addScriptTag({content:source});
  const report=await page.evaluate(async({data,digest})=>await collectLayoutFrames(data,digest),{data,digest});
  fs.writeFileSync(path.join(directory,'layout-browser.json'),JSON.stringify(report));
  console.log(JSON.stringify({allocations:Object.keys(report.allocations),issues:report.issues}));
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
