const fs=require('fs'),path=require('path'),crypto=require('crypto'),{chromium}=require('playwright');
(async()=>{
 const out=process.argv[2],b=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 try{
  const page=await b.newPage({viewport:{width:1000,height:1100},deviceScaleFactor:300/96});
  await page.route('http://127.0.0.1:8899/**',r=>r.fulfill({status:200,contentType:'text/html',body:'<!doctype html><html><body></body></html>'}));
  await page.goto('http://127.0.0.1:8899/render');
  const collector=fs.readFileSync('src/classical_conditioning/figures/layout_frames_browser.js','utf8'),svg=fs.readFileSync(path.join(out,'figure3.svg'),'utf8');
  await page.addScriptTag({content:collector});
  const report=await page.evaluate(async({svg,hash})=>await collectLayoutFrames(svg,hash),{svg,hash:crypto.createHash('sha256').update(collector).digest('hex')});
  const evidence=path.join(out,'browser.json');
  if(fs.existsSync(evidence)){if(JSON.parse(fs.readFileSync(evidence,'utf8')).artifact_sha256!==report.artifact_sha256)throw Error('Changed source SVG');}
  else fs.writeFileSync(evidence,JSON.stringify(report,null,2),{flag:'wx'});
  await page.setContent(fs.readFileSync(path.join(out,'review.html'),'utf8'));await page.evaluate(()=>document.fonts.ready);
  if(await page.locator('body > svg').count()!==1||await page.locator('p').count()!==1)throw Error('Review scope or legend missing');
  await page.locator('body > svg').screenshot({path:path.join(out,'figure3.png')});
  console.log(JSON.stringify({issues:report.issues,allocations:report.allocations,png_dpi:300}));
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exit(1)});
