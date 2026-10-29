// Render and measure only the saved 10sTrace draft outputs.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('playwright');
(async()=>{
 const [directory,name,preview]=process.argv.slice(2);
 const htmlName=name==='atlas'?'all-fish-heatmap-atlas.html':name+'.html';
 const server=require('http').createServer((req,res)=>{res.setHeader('Content-Type','text/html; charset=utf-8');res.end(fs.readFileSync(path.join(directory,htmlName)));});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 try {
  const page=await browser.newPage({viewport:{width:1100,height:1500}});
  await page.goto('http://127.0.0.1:'+server.address().port+'/');
  await page.evaluate(()=>document.fonts.ready);
  if(name==='atlas'){
   const cards=await page.locator('article').count();if(cards!==63)throw Error('Atlas coverage mismatch');
   const samples=['20230306_02','20250715_06','20250715_10','20230315_04','20250715_08'];
   for(const fish of samples)await page.locator('article').filter({has:page.locator('h3',{hasText:fish})}).screenshot({path:preview.replace('.png','-'+fish+'.png')});
   fs.writeFileSync(path.join(directory,'atlas-browser.json'),JSON.stringify({cards,sampled_fish:samples,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(directory,htmlName))).digest('hex'),fonts_ready:true,inspection_scope:'representative card renders; numerical inventory checked separately'},null,2),{flag:'wx'});
   console.log(JSON.stringify({cards,samples}));return;
  }
  const svg=fs.readFileSync(path.join(directory,name+'.svg'),'utf8');
  const ids=await page.locator('svg [id]').evaluateAll(nodes=>nodes.map(e=>e.id));
  if(new Set(ids).size!==ids.length)throw Error('Duplicate SVG IDs');
  const texts=await page.locator('svg text').allTextContents();
  if(!texts.length)throw Error('No editable text');
  await page.locator('svg').screenshot({path:preview});
  const source=fs.readFileSync(path.resolve('src/classical_conditioning/figures/layout_frames_browser.js'),'utf8');
  await page.addScriptTag({content:source});
  const report=await page.evaluate(async({svg,digest})=>await collectLayoutFrames(svg,digest),{svg,digest:crypto.createHash('sha256').update(source).digest('hex')});
  fs.writeFileSync(path.join(directory,name+'-browser.json'),JSON.stringify(report,null,2),{flag:'wx'});
  console.log(JSON.stringify({name,texts:texts.length,issues:report.issues,allocations:Object.keys(report.allocations)}));
 } finally {await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exit(1)});
