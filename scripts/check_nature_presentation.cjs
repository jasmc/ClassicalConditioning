// Measure exact saved presentation leaves; edits only compensate stroke widths.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{chromium}=require('playwright');
(async()=>{const dir=process.argv[2],verify=process.argv[4]==='--verify',b=await chromium.launch({executablePath:process.argv[3],headless:true});try{const p=await b.newPage({viewport:{width:1000,height:1000}});
for(const n of [1,2]){const f=path.join(dir,`figure-0${n}-layout.svg`),raw=fs.readFileSync(f,'utf8'),mapping=JSON.parse(fs.readFileSync(path.join(dir,`figure-0${n}-presentation-map.json`)));
await p.setContent('<style>body{margin:0}svg{height:auto}</style>');await p.evaluate(raw=>document.body.appendChild(document.importNode(new DOMParser().parseFromString(raw,'image/svg+xml').documentElement,true)),raw);await p.evaluate(()=>document.fonts.ready);
const input=JSON.parse(fs.readFileSync(path.join(dir,`figure-0${n}-input.json`)));
const result=await p.evaluate(({mapping,verify,input})=>{const root=document.querySelector('svg'),pt=96/72,changes=[],strokes=[],issues=[];
const address=node=>{const p=[];while(node!==root){p.unshift([...node.parentNode.children].indexOf(node));node=node.parentNode}return p};
const at=p=>p.reduce((node,i)=>node.children[i],root);
const ticks=new Set(input.axes.flatMap(a=>a.axis_marks.filter(r=>r.scientific_role==='axis.tick'||(a.panel==='B'&&r.svg_type==='line'&&at(r.xml_index_path).getAttribute('x1')===at(r.xml_index_path).getAttribute('x2'))).map(r=>at(r.xml_index_path))));
const spines=new Set(input.axes.flatMap(a=>(a.explicit_spines||[]).flatMap(r=>{const node=at(r.xml_index_path);return node.matches('path,line,polyline,polygon,rect,circle,ellipse,use')?[node]:[...node.querySelectorAll('path,line,polyline,polygon,rect,circle,ellipse,use')]})));
const widths=mapping.stroke_styles||{spine:.5,trace:.6,summary:1,stimulus:.7,separator:.8};
function rendered(node){for(let a=node;a&&a!==root.parentNode;a=a.parentElement){if(['defs','clipPath','marker','symbol','metadata'].includes(a.localName)||getComputedStyle(a).display==='none')return false}return true}
const texts=mapping.text.filter(t=>rendered(at(t.xml_index_path))).map(t=>{const node=at(t.xml_index_path),s=getComputedStyle(node),m=node.getScreenCTM(),size=parseFloat(s.fontSize)*Math.hypot(m.c,m.d)/pt;const record={...t,size_pt_measured:size,font_family:s.fontFamily,font_weight:s.fontWeight,fill:s.fill};if(node.textContent!==t.display_text)issues.push('Text alias mismatch: '+JSON.stringify(t.xml_index_path));if(Math.abs(size-t.size_pt)>.01||s.fontFamily.split(',')[0].trim().replace(/["']/g,'')!=='Arial'||s.fill!=='rgb(0, 0, 0)')issues.push('Text role style: '+JSON.stringify(t.xml_index_path));return record});
for(const node of root.querySelectorAll('path,line,polyline,polygon,rect,circle,ellipse,use')){if(!rendered(node))continue;let s=getComputedStyle(node);if(s.stroke==='none'||Number(s.strokeOpacity)===0||Number(s.opacity)===0)continue;const m=node.getScreenCTM(),sx=Math.hypot(m.a,m.b),sy=Math.hypot(m.c,m.d);let scale=(sx+sy)/2;
// Axis ticks keep their measured normal transform; their dedicated checker
// resolves length and width. Other strokes use physical viewport widths so
// historical nested axis transforms cannot change their visible thickness.
if(ticks.has(node)){strokes.push({xml_index_path:address(node),element_id:node.id,status:'registered axis collector'});continue}
const ancestry=[];for(let a=node;a&&a!==root;a=a.parentElement)ancestry.push(a.id||'');const names=ancestry.join(' ');
let role=spines.has(node)||/spine|time-axis-/.test(names)?'spine':/cs-onset|cs-offset|us-expected|us-onset|continuous-cs|trial-\d+-event/.test(names)?'stimulus':/median|iqr-/.test(names)?'summary':/fish-.*trajectory|trial-\d+-trace/.test(names)?'trace':/boundary|phase-separator/.test(names)?'separator':'unclassified.presentation.stroke';
let width=s.vectorEffect==='non-scaling-stroke'?parseFloat(s.strokeWidth)/pt:parseFloat(s.strokeWidth)*scale/pt;
const target=widths[role]||Math.max(.25,Math.min(1,width));
if(!verify){changes.push({xml_index_path:address(node),element_id:node.id,role,before_width_pt:width,after_width_pt:target});node.style.setProperty('vector-effect','non-scaling-stroke','important');node.style.setProperty('stroke-width',target*pt+'px','important');width=target;s=getComputedStyle(node)}
if(width<.25-.01||width>1+.01||(widths[role]&&Math.abs(width-target)>.01))issues.push('Stroke outside role/range: '+node.id);
strokes.push({xml_index_path:address(node),element_id:node.id,role,vector_effect:s.vectorEffect,width_pt:width,stroke:s.stroke,opacity:s.opacity,dasharray:s.strokeDasharray,linecap:s.strokeLinecap});}
return {svg:new XMLSerializer().serializeToString(root),texts,strokes,changes,issues};},{mapping,verify,input});
if(!verify)fs.writeFileSync(f,result.svg);delete result.svg;result.artifact_sha256=crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');fs.writeFileSync(path.join(dir,`figure-0${n}-nature-${verify?'verification':'styles'}.json`),JSON.stringify(result,null,2));console.log(n,'text',result.texts.length,'strokes',result.strokes.length,'issues',result.issues);if(result.issues.length)throw Error('Nature presentation check failed');}
}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
