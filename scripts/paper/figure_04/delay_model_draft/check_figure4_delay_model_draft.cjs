// Exact saved-artifact frame evidence and a temporary visual-review image.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{chromium}=require('playwright');
(async()=>{
 const [out,preview]=process.argv.slice(2);
 if(!out||!preview)throw Error('Usage: checker <external draft directory> <temporary preview.png>');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:1100},deviceScaleFactor:2});
  await page.route('http://127.0.0.1:8899/**',r=>r.fulfill({status:200,contentType:'text/html',body:'<!doctype html><html><body></body></html>'}));
  const collector=fs.readFileSync('src/classical_conditioning/figures/layout_frames_browser.js','utf8');
  const svg=fs.readFileSync(path.join(out,'figure4-delay.svg'),'utf8');
  await page.goto('http://127.0.0.1:8899/render');await page.addScriptTag({content:collector});
  const report=await page.evaluate(async({svg,hash})=>await collectLayoutFrames(svg,hash),{svg,hash:crypto.createHash('sha256').update(collector).digest('hex')});
  const evidence=path.join(out,'layout-browser.json');
  if(fs.existsSync(evidence)){
   const prior=JSON.parse(fs.readFileSync(evidence,'utf8'));
   const history=path.join(out,'history','browser-'+prior.artifact_sha256.slice(0,12));
   fs.mkdirSync(history,{recursive:true});
   const target=path.join(history,'layout-browser.json');
   if(!fs.existsSync(target))fs.copyFileSync(evidence,target,fs.constants.COPYFILE_EXCL);
  }
  fs.writeFileSync(evidence,JSON.stringify(report,null,2));
  await page.setContent(fs.readFileSync(path.join(out,'figure4-delay.html'),'utf8'));
  await page.evaluate(()=>document.fonts.ready);
  if(await page.locator('main > svg').count()!==1||await page.locator('.legend p').count()!==1)throw Error('Wrong review scope or missing legend');
  await page.screenshot({path:preview,fullPage:true});
  console.log(JSON.stringify({svg_sha256:report.artifact_sha256,issues:report.issues,preview}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
