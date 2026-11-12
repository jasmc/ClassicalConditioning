// Read exact temporary candidate bytes; collect browser evidence and a temporary preview.
const fs=require('fs'),path=require('path');const {chromium}=require('playwright');
(async()=>{const dir=process.argv[2],browser=await chromium.launch({executablePath:process.argv[3],headless:true});
try{const p=await browser.newPage({viewport:{width:820,height:1000}});
await p.goto('http://127.0.0.1:8874/single-fish.html');
const collector=fs.readFileSync(path.join(process.cwd(),'src/classical_conditioning/figures/layout_frames_browser.js'),'utf8');
await p.addScriptTag({content:collector});
const crypto=require('crypto');const svg=fs.readFileSync(path.join(dir,'candidate.svg'),'utf8');
const report=await p.evaluate(async({svg,hash})=>await collectLayoutFrames(svg,hash),{svg,hash:crypto.createHash('sha256').update(collector).digest('hex')});
fs.writeFileSync(path.join(dir,'browser.json'),JSON.stringify(report,null,2),{flag:'wx'});
await p.setContent(fs.readFileSync(path.join(dir,'candidate.html'),'utf8'));await p.evaluate(()=>document.fonts.ready);
await p.screenshot({path:path.join(dir,'preview.png'),fullPage:true});
console.log(JSON.stringify({issues:report.issues,allocations:report.allocations}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
