// Right-align registered color keys and center condition headings on their own
// data rectangles. --verify reads the exact saved SVG without editing it.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{chromium}=require('playwright');
(async()=>{const dir=process.argv[2],verify=process.argv[4]==='--verify',b=await chromium.launch({executablePath:process.argv[3],headless:true});try{const p=await b.newPage({viewport:{width:1000,height:1000}});
for(const n of [1,2]){
 const file=path.join(dir,`figure-0${n}-layout.svg`),raw=fs.readFileSync(file,'utf8'),manifest=JSON.parse(fs.readFileSync(path.join(dir,`figure-0${n}-annotation-changes.json`)));
 await p.setContent('<style>body{margin:0}svg{width:183mm;height:auto}</style>');await p.evaluate(raw=>document.body.appendChild(document.importNode(new DOMParser().parseFromString(raw,'image/svg+xml').documentElement,true)),raw);await p.evaluate(()=>document.fonts.ready);
 const report=await p.evaluate(({manifest,verify,n})=>{
  const root=document.querySelector('svg'),mm=25.4/96,pt=96/72,changes=[];
  const find=id=>{const ns=[...root.querySelectorAll('[id]')].filter(n=>n.id===id);if(ns.length!==1)throw Error('Missing/ambiguous annotation '+id);return ns[0]};
  function address(node){const out=[];while(node!==root){out.unshift([...node.parentNode.children].indexOf(node));node=node.parentNode}return out}
  function remember(node){const p=address(node);changes.push({xml_index_path:p,element_id:node.id,before_attributes:[...node.attributes].map(a=>[a.name,a.value])});if(!node.id)node.id=`fig${n}-legend-element-${p.join('-')}`}
  function box(node){const q=node.getBBox(),m=node.getScreenCTM(),ps=[[q.x,q.y],[q.x+q.width,q.y+q.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));return [Math.min(...ps.map(p=>p.x))*mm,Math.min(...ps.map(p=>p.y))*mm,Math.max(...ps.map(p=>p.x))*mm,Math.max(...ps.map(p=>p.y))*mm]}
  function shift(node,dx,dy=0){remember(node);const m=node.getScreenCTM(),det=m.a*m.d-m.b*m.c;node.setAttribute('transform',(node.getAttribute('transform')||'')+` translate(${(m.d*dx-m.c*dy)/mm/det} ${(-m.b*dx+m.a*dy)/mm/det})`)}
  function font(node,size){remember(node);const m=node.getScreenCTM();node.setAttribute('style',(node.getAttribute('style')||'')+`;font-size:${size*pt/Math.hypot(m.c,m.d)}px !important;font-family:"DejaVu Sans",sans-serif !important;font-weight:normal !important;font-style:normal !important`)}
  function edge(group){const ref=group.alignment_reference;if(ref.element_id)return box(find(ref.element_id))[2];const parent=find(group.element_id).parentNode,m=parent.getScreenCTM();return new DOMPoint(ref.local_right_edge,0).matrixTransform(m).x*mm}
  for(const group of manifest.legend_groups){const node=find(group.element_id);if(!verify){for(const text of node.querySelectorAll('text'))font(text,7);shift(node,edge(group)-box(node)[2])}}
  for(const h of manifest.condition_headings){const group=find(h.element_id),ref=box(find(h.alignment_reference));if(!verify){for(const text of group.querySelectorAll('text'))font(text,8);const q=box(group);shift(group,(ref[0]+ref[2]-q[0]-q[2])/2)}}
  if(n===2&&!verify&&!manifest.stat_row_layout_done){
   const labels=[...find('edit-G-axis__g-stats__y').children],first=box(labels[0]),start=(first[1]+first[3])/2;
   const prefixes=['D-','mean_p_fdr-','slope_p_raw-','slope_p_fdr-','trial-'];
   for(let i=0;i<labels.length;i++){const q=box(labels[i]),dy=start+3.2*i-(q[1]+q[3])/2;shift(labels[i],0,dy);for(const node of find('edit-G-axes__g-stats__main').children)if(node.id.startsWith('edit-G-fig2__g__stats__'+prefixes[i]))shift(node,0,dy)}
   shift(find('fig2__g__native__axes'),0,12);
  }
  const at=p=>p.reduce((node,i)=>node.children[i],root);
  const visible=node=>{for(let p=node;p&&p!==root.parentNode;p=p.parentNode)if(p.nodeType===1&&getComputedStyle(p).display==='none')return false;return true},texts=[...root.querySelectorAll('text')].filter(visible);
  const naming={obsolete_trace_labels:texts.filter(t=>/\b(3|10)\s+s\s+Trace\b/.test(t.textContent)).map(t=>t.textContent),visible_sample_counts:n===2?texts.filter(t=>/\bn\s*=\s*\d+/.test(t.textContent)).map(t=>t.textContent):[]};
  if(n===2){const g=find('fig2-g-native-figure');naming.ambiguous_G_heading_visible=[...g.querySelectorAll('text')].some(t=>visible(t)&&t.textContent==='Delay'&&!find('legend_1').contains(t));naming.E_local_spine_offset_mm=box(find('fig2-E-visible-y-spine'))[0]-box(find('edit-E-fig2__e__control__background'))[0]}
  function metric(text){const m=text.getScreenCTM(),p=new DOMPoint(Number(text.getAttribute('x')||0),Number(text.getAttribute('y')||0)).matrixTransform(m);return {element_id:text.id,label:text.textContent,size_pt:parseFloat(getComputedStyle(text).fontSize)*Math.hypot(m.c,m.d)/pt,family:getComputedStyle(text).fontFamily,weight:getComputedStyle(text).fontWeight,baseline_mm:p.y*mm}}
  function entry(text,group){let artist=text;while(artist.parentNode!==group)artist=artist.parentNode;const glyph=artist.previousElementSibling;return {...metric(text),scientific_role:'legend.key',glyph_artist:glyph?{element_id:glyph.id,xml_index_path:address(glyph)}:null}}
  return {svg:new XMLSerializer().serializeToString(root),changes:changes.map(c=>({...c,after_attributes:[...at(c.xml_index_path).attributes].map(a=>[a.name,a.value])})),naming,legend_groups:manifest.legend_groups.map(g=>{const node=find(g.element_id);return {...g,right_edge_delta_mm:box(node)[2]-edge(g),experiment:n===2?(g.panel==='G'?'Delay':'3sTrace'):null,entries:[...node.querySelectorAll('text')].map(text=>entry(text,node))}}),condition_headings:manifest.condition_headings.map(h=>{const node=find(h.element_id),a=box(node),r=box(find(h.alignment_reference));return {...h,center_delta_mm:(a[0]+a[2]-r[0]-r[2])/2,text_metrics:[...node.querySelectorAll('text')].map(metric)}})};
 },{manifest,verify,n});
 if(!verify){fs.writeFileSync(file,report.svg);if(n===2){manifest.stat_row_layout_done=true;fs.writeFileSync(path.join(dir,`figure-0${n}-annotation-changes.json`),JSON.stringify(manifest,null,2))}}delete report.svg;report.artifact_sha256=crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');fs.writeFileSync(path.join(dir,`figure-0${n}-annotation-${verify?'verification':'alignment'}.json`),JSON.stringify(report,null,2));console.log(n,report.legend_groups.map(g=>[g.panel,g.right_edge_delta_mm]));
} }finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
