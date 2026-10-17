// Right-align registered color keys and center condition headings on their own
// data rectangles. --verify reads the exact saved SVG without editing it.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),{chromium}=require('playwright');
(async()=>{const dir=process.argv[2],verify=process.argv[4]==='--verify',b=await chromium.launch({executablePath:process.argv[3],headless:true});try{const p=await b.newPage({viewport:{width:1000,height:1000}});
for(const n of [1,2]){
 const file=path.join(dir,`figure-0${n}-layout.svg`),raw=fs.readFileSync(file,'utf8'),manifest=JSON.parse(fs.readFileSync(path.join(dir,`figure-0${n}-annotation-changes.json`)));
 await p.setContent('<style>body{margin:0}svg{height:auto}</style>');await p.evaluate(raw=>document.body.appendChild(document.importNode(new DOMParser().parseFromString(raw,'image/svg+xml').documentElement,true)),raw);await p.evaluate(()=>document.fonts.ready);
 const report=await p.evaluate(({manifest,verify,n})=>{
  const root=document.querySelector('svg'),mm=25.4/96,pt=96/72,changes=[];
  const style=manifest.annotation_style||{family:'DejaVu Sans',legend_size_pt:7,heading_size_pt:8};
  const find=id=>{const ns=[...root.querySelectorAll('[id]')].filter(n=>n.id===id);if(ns.length!==1)throw Error('Missing/ambiguous annotation '+id);return ns[0]};
  function address(node){const out=[];while(node!==root){out.unshift([...node.parentNode.children].indexOf(node));node=node.parentNode}return out}
  function remember(node){const p=address(node);changes.push({xml_index_path:p,element_id:node.id,before_attributes:[...node.attributes].map(a=>[a.name,a.value])});if(!node.id)node.id=`fig${n}-legend-element-${p.join('-')}`}
  function box(node){const q=node.getBBox(),m=node.getScreenCTM(),ps=[[q.x,q.y],[q.x+q.width,q.y+q.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));return [Math.min(...ps.map(p=>p.x))*mm,Math.min(...ps.map(p=>p.y))*mm,Math.max(...ps.map(p=>p.x))*mm,Math.max(...ps.map(p=>p.y))*mm]}
  function shift(node,dx,dy=0){remember(node);const m=node.getScreenCTM(),det=m.a*m.d-m.b*m.c;node.setAttribute('transform',(node.getAttribute('transform')||'')+` translate(${(m.d*dx-m.c*dy)/mm/det} ${(-m.b*dx+m.a*dy)/mm/det})`)}
  function font(node,size){remember(node);const m=node.getScreenCTM();node.setAttribute('style',(node.getAttribute('style')||'')+`;font-size:${size*pt/Math.hypot(m.c,m.d)}px !important;font-family:"${style.family}",sans-serif !important;font-weight:normal !important;font-style:normal !important`)}
  function edge(group){const ref=group.alignment_reference;if(ref.element_id)return box(find(ref.element_id))[2];const parent=find(group.element_id).parentNode,m=parent.getScreenCTM();return new DOMPoint(ref.local_right_edge,0).matrixTransform(m).x*mm}
  for(const group of manifest.legend_groups){const node=find(group.element_id);if(!verify){for(const text of node.querySelectorAll('text'))font(text,style.legend_size_pt);shift(node,edge(group)-box(node)[2])}}
  for(const h of manifest.condition_headings){const group=find(h.element_id),ref=box(find(h.alignment_reference));if(!verify){for(const text of group.querySelectorAll('text'))font(text,style.heading_size_pt);const q=box(group);shift(group,(ref[0]+ref[2]-q[0]-q[2])/2)}}
  if(n===2&&!verify&&manifest.annotation_style){
   for(const alias of manifest.stat_text_aliases||[]){const node=find(alias.element_id);if(node.textContent!==alias.source_text&&node.textContent!==alias.display_text)throw Error('Changed statistical alias source');node.textContent=alias.display_text;}
   // Reflow the vertical H key using whole text/handle pairs. G is a
   // horizontal two-column key and retains its distinct arrangement.
   const legend=find('edit-H-legend_1'),entries=[...legend.querySelectorAll('text')];
   if(entries.length!==2)throw Error('Expected two identified H legend entries');
   const a=box(entries[0]),z=box(entries[1]),target=(a[1]+a[3])/2+Math.max(a[3]-a[1],z[3]-z[1])+.6,dy=target-(z[1]+z[3])/2;
   let artist=entries[1];while(artist.parentNode!==legend)artist=artist.parentNode;
   const glyph=artist.previousElementSibling;shift(artist,0,dy);if(glyph)shift(glyph,0,dy);
   // Separate the statistical phase heading from the first hypothesis row;
   // all row glyphs move together without changing their X/value mapping.
   const labels=[...find('edit-G-axis__g-stats__y').querySelectorAll('text')],headers=(manifest.stat_phase_headers||[]).map(h=>find(h.element_id));
   if(headers.length!==9)throw Error('Missing explicit phase-header mapping');
   const envelope=nodes=>{const bs=nodes.map(box);return[Math.min(...bs.map(b=>b[0])),Math.min(...bs.map(b=>b[1])),Math.max(...bs.map(b=>b[2])),Math.max(...bs.map(b=>b[3]))]};
   const panelLetter=box(find('fig2__g__compact-row__panel-letter')),oldHeader=envelope(headers);
   for(const text of headers)shift(text,0,panelLetter[3]+2-oldHeader[1]);
   const header=envelope(headers);
   const prefixes=['D-','mean_p_fdr-','slope_p_raw-','slope_p_fdr-','trial-'];
   if(labels.length!==5)throw Error('Incomplete statistical row identities');
   for(let i=0;i<labels.length;i++){
    const q=box(labels[i]),center=header[3]+1+(q[3]-q[1])/2+3*i;shift(labels[i],0,center-(q[1]+q[3])/2);
    const marks=[...find('edit-G-axes__g-stats__main').children].filter(node=>node.id.startsWith('edit-G-fig2__g__stats__'+prefixes[i]));
    if(!marks.length)throw Error('Missing statistical row '+prefixes[i]);
    for(const node of marks){const texts=[...node.querySelectorAll('text')];const members=texts.length?texts:[...node.querySelectorAll('path,line,circle,ellipse,use')].filter(n=>!n.closest('defs'));
     const boxes=members.map(box);if(!boxes.length)throw Error('Unmeasurable statistical mark '+node.id);const top=Math.min(...boxes.map(b=>b[1])),bottom=Math.max(...boxes.map(b=>b[3]));shift(node,0,center-(top+bottom)/2);}
   }
  }
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
  const dense_layout=n===2?{phase_headers:(manifest.stat_phase_headers||[]).map(h=>find(h.element_id)).map(t=>({text:t.textContent,box_mm:box(t)})),stat_row_labels:[...find('edit-G-axis__g-stats__y').querySelectorAll('text')].map(t=>({text:t.textContent,box_mm:box(t)})),first_row_marks:[...find('edit-G-axes__g-stats__main').children].filter(t=>t.id.startsWith('edit-G-fig2__g__stats__D-')).flatMap(g=>[...g.querySelectorAll('text')].map(t=>({text:t.textContent,box_mm:box(t)}))),H_key:[...find('edit-H-legend_1').querySelectorAll('text')].map(t=>({text:t.textContent,box_mm:box(t)}))}:null;
  return {dense_layout,svg:new XMLSerializer().serializeToString(root),changes:changes.map(c=>({...c,after_attributes:[...at(c.xml_index_path).attributes].map(a=>[a.name,a.value])})),naming,legend_groups:manifest.legend_groups.map(g=>{const node=find(g.element_id);return {...g,right_edge_delta_mm:box(node)[2]-edge(g),experiment:n===2?(g.panel==='G'?'Delay':'3sTrace'):null,entries:[...node.querySelectorAll('text')].map(text=>entry(text,node))}}),condition_headings:manifest.condition_headings.map(h=>{const node=find(h.element_id),a=box(node),r=box(find(h.alignment_reference));return {...h,center_delta_mm:(a[0]+a[2]-r[0]-r[2])/2,text_metrics:[...node.querySelectorAll('text')].map(metric)}})};
 },{manifest,verify,n});
 if(!verify){fs.writeFileSync(file,report.svg);if(n===2){manifest.stat_row_layout_done=true;fs.writeFileSync(path.join(dir,`figure-0${n}-annotation-changes.json`),JSON.stringify(manifest,null,2))}}delete report.svg;report.artifact_sha256=crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');fs.writeFileSync(path.join(dir,`figure-0${n}-annotation-${verify?'verification':'alignment'}.json`),JSON.stringify(report,null,2));console.log(n,report.legend_groups.map(g=>[g.panel,g.right_edge_delta_mm]));
} }finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
