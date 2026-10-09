const {chromium}=require('C:/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
const page=await browser.newPage({viewport:{width:1280,height:900}});
await page.goto('file:///'+path.join(__dirname,'index.html').replaceAll('\\','/'));
await page.locator('img').evaluate(el=>el.decode());
const report=await page.evaluate(()=>({imageLoaded:document.images[0].naturalWidth===2700,pdf:document.querySelectorAll('a[href^="data:application/pdf"]').length,svg:document.querySelectorAll('a[href^="data:image/svg+xml"]').length,desktopOverflow:document.documentElement.scrollWidth>innerWidth}));
await page.setViewportSize({width:390,height:844});
report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
if(!report.imageLoaded||report.pdf!==1||report.svg!==1||report.desktopOverflow||report.mobileOverflow)throw Error(JSON.stringify(report));
fs.writeFileSync(path.join(__dirname,'html_validation.json'),JSON.stringify(report,null,2));
await browser.close();console.log(report);
})();
