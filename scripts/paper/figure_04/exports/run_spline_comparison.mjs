// Local WebR executes the standard mgcv/nlme implementations without installing
// a system-wide R. Only package binaries are downloaded; fish data stay local.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [directory,runtime,codePath,...options]=process.argv.slice(2);
const {WebR}=await import(pathToFileURL(path.join(runtime,'package/dist/webr.mjs')).href);
const r=new WebR({interactive:false});
await r.init();
try {
  console.log('Installing standard R analysis packages into the local virtual runtime.');
  await r.installPackages(['mgcv','nlme']);
  await r.FS.mkdir('/input');
  for(const assay of ['delay','3strace'])await r.FS.writeFile('/input/'+assay+'.csv',fs.readFileSync(path.join(directory,assay+'-model-data.csv')));
  const shelter=await new r.Shelter();
  let result=await shelter.captureR(fs.readFileSync(codePath,'utf8'));
  for(const o of result.output)if(['stdout','stderr'].includes(o.type))console.log(o.data);
  await shelter.purge();
  const candidates=['LMM6','LMM12','LMM20','GAMM20','GAMM30','GAMM20_AR1'];
  const known=new Set();
  async function flush(){
    const names=await r.evalRRaw('list.files("/output")','string[]');
    for(const name of names)if(!known.has(name)){
      const target=path.join(directory,'fits',name);
      fs.writeFileSync(target,await r.FS.readFile('/output/'+name),{flag:'wx'});known.add(name);
    }
  }
  const smoke=options.includes('--smoke');
  for(const assay of ['delay','3strace'])for(const candidate of candidates)for(const fold of (smoke?[0]:[0,1,2,3,4,5])) {
    const before=Date.now();
    result=await shelter.captureR(`run_fit(${JSON.stringify(assay)},${JSON.stringify(candidate)},${fold})`);
    for(const o of result.output)if(['stdout','stderr'].includes(o.type))console.log(o.data);
    await shelter.purge();await flush();
    console.log(`${assay} ${candidate} fold ${fold}: ${((Date.now()-before)/1000).toFixed(1)} s`);
  }
}finally{await r.close();}
