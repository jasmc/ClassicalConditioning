const repositoryRoot=(()=>{const f=require('fs'),p=require('path');let d=__dirname;while(!f.existsSync(p.join(d,'pyproject.toml'))){const parent=p.dirname(d);if(parent===d)throw Error('Repository root unavailable');d=parent;}return d;})();
// Verify exact embedded SVGs and interactive inventory; collect 600 dpi frame evidence.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {chromium} = require('playwright');

(async () => {
  const [directory, executable, baseURL, screenshotDir] = process.argv.slice(2);
  const browser = await chromium.launch({executablePath: executable, headless: true});
  const page = await browser.newPage({viewport:{width:1200,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(baseURL+'/review.html');
  await page.evaluate(()=>document.fonts.ready);
  const count=await page.locator('article.fish').count();
  if(count!==57)throw Error('Incomplete HTML fish inventory: '+count);
  const duplicatePaintIds=await page.evaluate(()=>{
    const seen=new Set(),duplicates=[];
    for(const n of document.querySelectorAll('svg [id]')){
      if(n.tagName==='metadata')continue;
      if(seen.has(n.id))duplicates.push(n.id);seen.add(n.id);
    }
    return duplicates;
  });
  if(duplicatePaintIds.length)throw Error('Duplicate SVG painted IDs: '+duplicatePaintIds.slice(0,4));
  await page.locator('#search').fill('20240618_08');
  if(await page.locator('article.fish:visible').count()!==1)throw Error('Search failed');
  await page.locator('#search').fill('');
  await page.locator('#search').fill('20240618_08');
  await page.locator('article.fish:visible').screenshot({path:path.join(screenshotDir,'figure3-selected-learner.png')});
  await page.locator('#search').fill('');
  await page.locator('#rule').selectOption('legacy-wip');
  await page.locator('#condition').selectOption('delay');
  await page.locator('#decision').selectOption('unclassified');
  if(await page.locator('article.fish:visible').count()!==3)throw Error('Unclassified fish filter failed');
  await page.locator('#condition').selectOption('all');
  await page.locator('#decision').selectOption('all');
  if(await page.locator('article.fish:visible').count()!==57)throw Error('Reset lost fish');
  await page.locator('#examples').screenshot({path:path.join(screenshotDir,'figure3-examples.png')});
  await page.locator('article.fish').first().screenshot({path:path.join(screenshotDir,'figure3-full-card.png')});
  await page.locator('#search').fill('20221122_03');
  await page.locator('article.fish:visible').screenshot({path:path.join(screenshotDir,'figure3-protocol-card.png')});
  await page.locator('#search').fill('');
  const collectorPath=path.join(repositoryRoot, 'src/classical_conditioning/figures/layout_frames_browser.js');
  const collector=fs.readFileSync(collectorPath,'utf8');
  const collectorHash=crypto.createHash('sha256').update(collector).digest('hex');
  await page.addScriptTag({content:collector});
  const cases={overview:fs.readFileSync(path.join(directory,'overview.svg'),'utf8')};
  const cards=JSON.parse(fs.readFileSync(path.join(directory,'card-svg-data.json'),'utf8'));
  for(const [fish,encoded] of Object.entries(cards))cases[fish]=Buffer.from(encoded,'base64').toString('utf8');
  // Review exact artwork without sticky HTML controls covering screenshot regions.
  const artwork=await browser.newPage({viewport:{width:1000,height:1200}});
  for(const [name,file] of [['overview','figure3-exact-overview.png'],
      ['20221115_09','figure3-exact-control.png'],['20240618_08','figure3-exact-learner.png']]){
    await artwork.setContent('<!doctype html><body style="margin:0;background:white">'+cases[name]+'</body>');
    await artwork.evaluate(async()=>{await document.fonts.ready;await Promise.all([...document.images].map(x=>x.decode()));});
    await artwork.locator('svg').first().screenshot({path:path.join(screenshotDir,file)});
  }
  await artwork.close();
  const measurements={};
  for(const [name,svg] of Object.entries(cases)) {
    measurements[name]=await page.evaluate(async ({svg,hash})=>await collectLayoutFrames(svg,hash),{svg,hash:collectorHash});
    if(measurements[name].issues.length)console.log(name,measurements[name].issues);
    fs.writeFileSync(path.join(directory,'layout-browser.json'),JSON.stringify(measurements));
    if(Object.keys(measurements).length%5===0)console.log('Measured '+Object.keys(measurements).length+'/58 SVGs');
  }
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(directory,'browser-ui-checks.json'),JSON.stringify({
    fish_count:count,search_passed:true,unclassified_filter_passed:true,reset_passed:true,
    page_errors:errors,duplicate_svg_paint_ids:duplicatePaintIds,
    overview_and_full_cards_visually_captured:true,url:baseURL+'/review.html'},null,2));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
