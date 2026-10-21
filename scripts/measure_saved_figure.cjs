// Exact saved-SVG browser measurement for scientific draft review.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('playwright');
(async()=>{const [svgFile,htmlFile,outDir,edge]=process.argv.slice(2);
fs.mkdirSync(outDir,{recursive:true});
const browser=await chromium.launch({executablePath:edge,headless:true});
try{const p=await browser.newPage({viewport:{width:1050,height:1100}});
await p.goto('http://127.0.0.1:8874/single-fish.html');
const collector=fs.readFileSync(path.join(process.cwd(),'src/classical_conditioning/figures/layout_frames_browser.js'),'utf8');
await p.addScriptTag({content:collector});
const svg=fs.readFileSync(svgFile,'utf8');
const report=await p.evaluate(async({svg,hash})=>await collectLayoutFrames(svg,hash),{svg,hash:crypto.createHash('sha256').update(collector).digest('hex')});
fs.writeFileSync(path.join(outDir,'browser.json'),JSON.stringify(report,null,2),{flag:'wx'});
await p.setContent(fs.readFileSync(htmlFile,'utf8'));await p.evaluate(()=>document.fonts.ready);
await p.screenshot({path:path.join(outDir,'preview.png'),fullPage:true});
console.log(JSON.stringify({issues:report.issues,allocations:report.allocations}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
