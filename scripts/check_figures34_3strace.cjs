// Inspect exact saved SVGs in Edge; no shared figure state changes.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('playwright');
(async()=>{
 const out=process.argv[2],edge=process.argv[3];
 const dir=path.join(out,'browser-verification');fs.mkdirSync(dir,{recursive:true});
 const browser=await chromium.launch({executablePath:edge,headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1200,height:1200}});
  await page.route('http://127.0.0.1:8899/**',route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><html><head></head><body></body></html>'}));
  const collector=fs.readFileSync('src/classical_conditioning/figures/layout_frames_browser.js','utf8');
  const names=['figure3-score','figure4-condition-profiles','figure4-individual-catches'];
  if(process.argv[4]!=='--summaries-only')names.push('individual-atlas/atlas_20230227_01','individual-atlas/atlas_20230307_12','individual-atlas/atlas_20230227_03');
  const evidence=[];
  for(const name of names){
   const svgPath=path.join(out,name+'.svg'),svg=fs.readFileSync(svgPath,'utf8');
   const safe=path.basename(name),reportPath=path.join(dir,safe+'.json');
   if(fs.existsSync(reportPath)){
    const prior=JSON.parse(fs.readFileSync(reportPath,'utf8'));
    if(prior.artifact_sha256!==crypto.createHash('sha256').update(svg).digest('hex')||!fs.existsSync(path.join(dir,safe+'.png')))throw Error('Changed or incomplete existing browser record: '+name);
    evidence.push({name,svg_sha256:prior.artifact_sha256,issues:prior.issues});continue;
   }
   await page.goto('http://127.0.0.1:8899/render');await page.addScriptTag({content:collector});
   const report=await page.evaluate(async({svg,hash})=>await collectLayoutFrames(svg,hash),{
    svg,hash:crypto.createHash('sha256').update(collector).digest('hex')});
   fs.writeFileSync(reportPath,JSON.stringify(report,null,2),{flag:'wx'});
   await page.setContent(svg);await page.evaluate(()=>document.fonts.ready);
   await page.screenshot({path:path.join(dir,safe+'.png'),fullPage:true});
   evidence.push({name,svg_sha256:crypto.createHash('sha256').update(svg).digest('hex'),issues:report.issues});
  }
  if(process.argv[4]!=='--summaries-only')fs.writeFileSync(path.join(dir,'summary.json'),JSON.stringify(evidence,null,2),{flag:'wx'});
  console.log(JSON.stringify(evidence));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
