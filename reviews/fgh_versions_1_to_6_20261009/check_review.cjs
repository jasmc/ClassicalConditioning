const {chromium}=require('C:/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1280,height:900},acceptDownloads:true});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('file:///'+path.join(__dirname,'index.html').replaceAll('\\','/'));
 const report=await page.evaluate(()=>({
  numberedSections:[...document.querySelectorAll('.version')].map(el=>el.id),
  images:[...document.images].map(el=>({loaded:el.complete&&el.naturalWidth===2700})),
  choices:document.querySelectorAll('[data-choice]').length,
  pdfDownloads:document.querySelectorAll('a[href^="data:application/pdf"]').length,
  svgDownloads:document.querySelectorAll('a[href^="data:image/svg+xml"]').length,
  meanCentredOptionsRemoved:!document.querySelector('#version-4')&&!document.querySelector('#version-5')&&!document.querySelector('[data-choice="v5-linear"]'),
  version6IsOneSecond:document.querySelector('#version-6').textContent.includes('1 s means'),
  fourColourNotSelectable:!document.querySelector('[data-choice="v6-four"]'),
  desktopOverflow:document.documentElement.scrollWidth>innerWidth
 }));
 if(report.numberedSections.join(',')!=='version-1,version-2,version-6'||report.images.length!==4||report.images.some(el=>!el.loaded)||report.choices!==4||report.pdfDownloads!==4||report.svgDownloads!==4||!report.meanCentredOptionsRemoved||!report.version6IsOneSecond||!report.fourColourNotSelectable||report.desktopOverflow)throw Error(JSON.stringify(report));
 await page.screenshot({path:path.join(__dirname,'header_preview.png')});
 await page.locator('#overview').screenshot({path:path.join(__dirname,'overview_preview.png')});
 for(const n of [1,2,6])await page.locator('#version-'+n).screenshot({path:path.join(__dirname,'version'+n+'_preview.png')});
 await page.locator('#choice-v2').selectOption('Keep');
 await page.locator('#choice-v1-d').selectOption('Discard');
 await page.locator('#choice-v6-five').selectOption('Keep');
 await page.reload();
 if(await page.locator('#choice-v2').inputValue()!=='Keep'||await page.locator('#choice-v1-d').inputValue()!=='Discard')throw Error('Choice persistence failed');
 const downloaded=page.waitForEvent('download');await page.locator('#export-choices').click();
 const download=await downloaded;const tmp=path.join(__dirname,'tmp');fs.mkdirSync(tmp,{recursive:true});
 const saved=path.join(tmp,'test_choice_export.txt');await download.saveAs(saved);
 const text=fs.readFileSync(saved,'utf8');
 if(!text.includes('Version 2 C: Keep')||!text.includes('Version 1 D: Discard')||!text.includes('Version 6 five bands: Keep')||text.split('\n').length<10)throw Error('Choice export is invalid: '+JSON.stringify(text));
 report.choicePersistenceAndExportVerified=true;
 await page.setViewportSize({width:390,height:844});
 report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
 await page.screenshot({path:path.join(__dirname,'mobile_preview.png')});
 if(report.mobileOverflow||errors.length)throw Error(JSON.stringify({mobileOverflow:report.mobileOverflow,errors}));
 report.pageErrors=errors;report.htmlSha256=crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname,'index.html'))).digest('hex');
 fs.writeFileSync(path.join(__dirname,'html_validation.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
