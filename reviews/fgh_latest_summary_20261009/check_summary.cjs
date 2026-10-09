const {chromium} = require('C:/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
(async () => {
  const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page = await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
  const errors=[];page.on('pageerror', e=>errors.push(e.message));
  await page.goto('file:///'+path.join(__dirname,'index.html').replaceAll('\\','/'));
  await page.locator('img').last().waitFor();
  const report=await page.evaluate(()=>({
    figures:[...document.images].map(im=>({loaded:im.complete&&im.naturalWidth>0,width:im.naturalWidth})),
    pdfDownloads:document.querySelectorAll('a[href^="data:application/pdf"]').length,
    sectionIds:['analysis','comparison','c-samples','d-samples','c-bins','median-bins','verification'].map(id=>({id,present:!!document.getElementById(id)})),
    desktopOverflow:document.documentElement.scrollWidth>innerWidth,
    traceFish:document.querySelector('.fish-grid').innerText.includes('20230310_08')
  }));
  if(report.figures.length!==4||report.figures.some(im=>!im.loaded)||report.pdfDownloads!==4||report.desktopOverflow||report.sectionIds.some(s=>!s.present))throw Error('HTML verification failed');
  await page.screenshot({path:path.join(__dirname,'header_preview.png')});
  await page.locator('#analysis').screenshot({path:path.join(__dirname,'analysis_preview.png')});
  for(const id of ['c-samples','d-samples','c-bins','median-bins'])await page.locator('#'+id).screenshot({path:path.join(__dirname,id+'_preview.png')});
  await page.setViewportSize({width:390,height:844});
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if(report.mobileOverflow||errors.length)throw Error('Mobile or browser error');
  report.pageErrors=errors;
  fs.writeFileSync(path.join(__dirname,'html_validation.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
