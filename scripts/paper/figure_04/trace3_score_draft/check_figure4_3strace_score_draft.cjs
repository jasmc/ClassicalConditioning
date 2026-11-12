const fs=require('fs'),path=require('path'),crypto=require('crypto'),{chromium}=require('playwright');
(async()=>{
 const out=process.argv[2],browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:1100},deviceScaleFactor:300/96});
  await page.route('http://127.0.0.1:8899/**',r=>r.fulfill({status:200,contentType:'text/html',body:'<!doctype html><html><body></body></html>'}));
  const collector=fs.readFileSync('src/classical_conditioning/figures/layout_frames_browser.js','utf8'),results=[];
  for(const name of ['figure4','block-profiles-1','block-profiles-2','individual-catches','coverage']){
   const svg=fs.readFileSync(path.join(out,name+'.svg'),'utf8');
   await page.goto('http://127.0.0.1:8899/render');await page.addScriptTag({content:collector});
   const report=await page.evaluate(async({svg,hash})=>await collectLayoutFrames(svg,hash),{svg,hash:crypto.createHash('sha256').update(collector).digest('hex')});
   fs.writeFileSync(path.join(out,name+'-browser.json'),JSON.stringify(report,null,2),{flag:'wx'});
   await page.setContent(fs.readFileSync(path.join(out,name+'.html'),'utf8'));await page.evaluate(()=>document.fonts.ready);
   if(await page.locator('body > svg').count()!==1||await page.locator('p').count()!==1)throw Error('Wrong review scope or missing legend: '+name);
   await page.locator('body > svg').screenshot({path:path.join(out,name+'.png')});
   results.push({name,issues:report.issues,svg_sha256:report.artifact_sha256,png_dpi:300});
  }
  fs.writeFileSync(path.join(out,'browser-summary.json'),JSON.stringify(results,null,2),{flag:'wx'});console.log(JSON.stringify(results));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
