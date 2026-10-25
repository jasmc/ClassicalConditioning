const repositoryRoot=(()=>{const f=require('fs'),p=require('path');let d=__dirname;while(!f.existsSync(p.join(d,'pyproject.toml'))){const parent=p.dirname(d);if(parent===d)throw Error('Repository root unavailable');d=parent;}return d;})();
if(process.argv.includes('--help')){console.log('Usage: node '+require('path').basename(__filename)+' candidate-directory browser-executable [--verify|--apply|--center-labels]');process.exit(0);}
const commandMode=require(require('path').join(repositoryRoot,'src/classical_conditioning/figures/browser/command_options.cjs')).parseMode(process.argv.slice(4),{allowCenter:true});
// Read-only SVG verification by default; use --apply for explicit candidate edits.
// Final-scale presentation edits to explicitly registered axis leaves only.
// Usage: node style_axis_geometry.cjs external-directory browser-executable
const fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const dir=process.argv[2],centerOnly=commandMode.centerOnly,verify=commandMode.verify,browser=await chromium.launch({executablePath:process.argv[3],headless:true});
 const p=await browser.newPage({viewport:{width:1000,height:1000}});
 for(const n of [1,2]){
  const extra=path.join(dir,`figure-0${n}-axis-input.json`),input=JSON.parse(fs.readFileSync(verify&&fs.existsSync(extra)?extra:path.join(dir,`figure-0${n}-input.json`))),svg=fs.readFileSync(verify?path.join(dir,`figure-0${n}-layout.svg`):input.svg.path,'utf8');
  await p.setContent('<html><head><style>body{margin:0}svg{height:auto;display:block}</style></head><body></body></html>');
  await p.evaluate(svg=>{const xml=new DOMParser().parseFromString(svg,'image/svg+xml');if(xml.querySelector('parsererror'))throw Error('Invalid SVG XML');document.body.appendChild(document.importNode(xml.documentElement,true))},svg);
  await p.evaluate(()=>document.fonts.ready);
  const result=await p.evaluate(({input,n,verify,centerOnly})=>{
   const root=document.querySelector('svg'),mm=25.4/96,pt=96/72,changes=new Map();
   const style={tick_length_pt:2,tick_stroke_width_pt:.5,tick_color:'#000000',tick_linecap:'butt',tick_label_edge_gap_pt:3,title_label_edge_gap_pt:3,...(input.axis_style||{family:'DejaVu Sans',tick_label_size_pt:7,axis_title_size_pt:8,block_size_pt:8,trial_size_pt:7})};
   const at=a=>a.xml_index_path.reduce((node,i)=>node.children[i],root);
   const original=new Map([...root.querySelectorAll('*'),root].map(node=>[node,{attrs:[...node.attributes].map(a=>[a.name,a.value]),text:node.textContent}]));
   const nodes=[root,...root.querySelectorAll('*')];
   const address=node=>{const list=[];while(node!==root){list.unshift([...node.parentNode.children].indexOf(node));node=node.parentNode}return list};
   function touch(node,role){if(!changes.has(node))changes.set(node,{xml_index_path:address(node),roles:[],before_attributes:original.get(node).attrs});const r=changes.get(node);if(!r.roles.includes(role))r.roles.push(role);if(!node.id)node.id=`fig${n}-axis-registered-${nodes.indexOf(node)}`;return node}
   const ctm=node=>node.getScreenCTM();
   function box(node){const b=node.getBBox(),m=ctm(node),v=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));return [Math.min(...v.map(p=>p.x))*mm,Math.min(...v.map(p=>p.y))*mm,Math.max(...v.map(p=>p.x))*mm,Math.max(...v.map(p=>p.y))*mm]}
   const union=bs=>[Math.min(...bs.map(b=>b[0])),Math.min(...bs.map(b=>b[1])),Math.max(...bs.map(b=>b[2])),Math.max(...bs.map(b=>b[3]))];
   function shift(node,dx,dy){touch(node,'presentation-offset');const m=ctm(node),det=m.a*m.d-m.b*m.c,x=(m.d*dx-m.c*dy)/mm/det,y=(-m.b*dx+m.a*dy)/mm/det;node.setAttribute('transform',(node.getAttribute('transform')||'')+` translate(${x} ${y})`)}
   function applyStyle(node,properties){node.setAttribute('style',(node.getAttribute('style')||'')+';'+Object.entries(properties).map(([k,v])=>k+':'+v+' !important').join(';'))}
   function font(node,size,role){touch(node,role);const m=ctm(node),local=size*pt/Math.hypot(m.c,m.d);applyStyle(node,{'font-size':local+'px','font-family':'"'+style.family+'",sans-serif','font-weight':'normal','font-style':'normal'})}
   function ends(node){let x1,y1,x2,y2,geom=node;
    if(node.localName==='use'){geom=root.querySelector('[id="'+(node.getAttribute('href')||node.getAttributeNS('http://www.w3.org/1999/xlink','href')).slice(1)+'"]')}
    if(geom.localName==='line'){[x1,y1,x2,y2]=['x1','y1','x2','y2'].map(k=>Number(geom.getAttribute(k)))}
    else {const d=geom.getAttribute('d')||'',v=d.match(/[-+]?(?:\d*\.)?\d+(?:e[-+]?\d+)?/gi)?.map(Number);if(!/^\s*M[^MLQCZA]*L[^MLQCZA]*\s*$/i.test(d)||v?.length!==4)throw Error('Unresolved tick path '+node.id);[x1,y1,x2,y2]=v}
    if(node.localName==='use'){const x=Number(node.getAttribute('x')||0),y=Number(node.getAttribute('y')||0);x1+=x;x2+=x;y1+=y;y2+=y}
    return [x1,y1,x2,y2]}
   const measured=[],axes=input.axes;
   for(const a of axes){
    const orientation=a.orientation==='color'?'y':a.orientation;
    a.tick_nodes=a.axis_marks.filter(r=>r.scientific_role==='axis.tick').map(at);
    // Protocol axes are explicit renderer line artists: the baseline is horizontal,
    // and each other line connects a labelled time to its outward tick endpoint.
    if(a.panel==='B'&&n===1)a.tick_nodes=a.axis_marks.map(at).filter(node=>node.localName==='line'&&node.getAttribute('x1')===node.getAttribute('x2'));
    a.label_nodes=a.tick_labels.map(at);a.title_nodes=a.titles.map(at);
    if(!verify)for(const node of a.tick_nodes){touch(node,'axis.tick');const [x1,y1,x2,y2]=ends(node),m=ctm(node),v=new DOMPoint(x2-x1,y2-y1).matrixTransform(new DOMMatrix([m.a,m.b,m.c,m.d,0,0])),length=Math.hypot(v.x,v.y),ratio=style.tick_length_pt*pt/length;
     const vertical=Math.abs(x2-x1)<1e-8,sx=vertical?1:ratio,sy=vertical?ratio:1;
     node.setAttribute('transform',(node.getAttribute('transform')||'')+` translate(${x1} ${y1}) scale(${sx} ${sy}) translate(${-x1} ${-y1})`);
     const cross=vertical?Math.hypot(m.a,m.b):Math.hypot(m.c,m.d);
     applyStyle(node,{'stroke-width':style.tick_stroke_width_pt*pt/cross,'stroke':style.tick_color,'stroke-linecap':style.tick_linecap,'stroke-linejoin':'miter','stroke-dasharray':'none','fill':'none','opacity':'1'});
    }
    if(!verify&&input.axis_style)for(const node of a.tick_nodes)if(node.localName==='use'){
     const href=node.getAttribute('href')||node.getAttributeNS('http://www.w3.org/1999/xlink','href');
     const target=root.querySelector('[id="'+href.slice(1)+'"]');
     if(!target)throw Error('Unresolved tick definition '+href);
     // A specified width on a referenced path overrides its <use>'s inherited
     // width. Delegate that one property to each registered tick instance.
     touch(target,'axis.tick_definition');applyStyle(target,{'stroke-width':'inherit'});
    }
    if(!verify)for(const node of a.label_nodes)font(node,style.tick_label_size_pt,'axis.tick_label');
    if(!verify)for(const node of a.title_nodes)font(node,style.axis_title_size_pt,'axis.label');
   }
   if(!verify&&n===1)for(const panel of ['D','E'])for(const node of root.querySelectorAll(`[id^="edit-${panel}-trial-"]`))if(node.localName==='text'&&/-((name)|(block))$/.test(node.id))font(node,node.id.endsWith('-block')?style.block_size_pt:style.trial_size_pt,'example-trial-label');
   // Use geometric glyph edges, not baseline/anchor padding. Move only normal
   // to the axis, preserving the data coordinate of each tick and label.
   for(const a of axes){if(verify||!a.tick_nodes.length)continue;const normal=a.orientation==='x'?1:0,tangent=1-normal;
    for(const label of a.label_nodes){const b=box(label),center=(b[tangent]+b[tangent+2])/2;
     const tick=[...a.tick_nodes].sort((u,v)=>{const U=box(u),V=box(v);return Math.abs((U[tangent]+U[tangent+2])/2-center)-Math.abs((V[tangent]+V[tangent+2])/2-center)})[0];
     const [x1,y1,x2,y2]=ends(tick),m=ctm(tick),start=new DOMPoint(x1,y1).matrixTransform(m),end=new DOMPoint(x2,y2).matrixTransform(m),sign=Math.sign(normal===1?end.y-start.y:end.x-start.x),target=(normal===1?end.y:end.x)*mm+sign*style.tick_label_edge_gap_pt*25.4/72,delta=target-(sign>0?b[normal]:b[normal+2]);
     shift(label,normal===0?delta:0,normal===1?delta:0);
    }
   }
   const titleGroups=new Map();
   for(const a of axes)for(const node of a.title_nodes){const key=a.titles.map(t=>t.xml_index_path.join('.')).join('|');if(!titleGroups.has(key))titleGroups.set(key,{nodes:a.title_nodes,axes:[]});const g=titleGroups.get(key);if(!g.axes.includes(a))g.axes.push(a)}
   for(const g of titleGroups.values()){
    if(verify)continue;
    const labels=[...new Set(g.axes.flatMap(a=>a.label_nodes))];if(!labels.length)continue;
    const a=g.axes[0],normal=a.orientation==='x'?1:0,lb=union(labels.map(box)),tb=union(g.nodes.map(box));
    let sign=1;if(normal===0){const ticks=g.axes.flatMap(a=>a.tick_nodes);sign=ticks.length&&box(ticks[0])[0]>lb[2]?-1:1}
    const delta=(sign>0?lb[normal+2]+style.title_label_edge_gap_pt*25.4/72:lb[normal]-style.title_label_edge_gap_pt*25.4/72)-(sign>0?tb[normal]:tb[normal+2]);
    for(const node of g.nodes)shift(node,normal===0?delta:0,normal===1?delta:0);
   }
   if(centerOnly)for(const a of axes){if(!a.tick_nodes.length)continue;const tangent=a.orientation==='x'?0:1;
    for(const label of a.label_nodes){const lb=box(label),center=(lb[tangent]+lb[tangent+2])/2;const tb=a.tick_nodes.map(box).sort((u,v)=>Math.abs((u[tangent]+u[tangent+2])/2-center)-Math.abs((v[tangent]+v[tangent+2])/2-center))[0];const delta=(tb[tangent]+tb[tangent+2])/2-center;shift(label,tangent===0?delta:0,tangent===1?delta:0)}
   }
   if(centerOnly&&input.axis_style)for(const g of titleGroups.values()){
    const a=g.axes[0],tangent=a.orientation==='x'?0:1;
    const references=g.axes.flatMap(a=>[...a.axis_marks,...(a.explicit_spines||[])]).map(at);
    if(!references.length)continue;
    const rb=union(references.map(box)),tb=union(g.nodes.map(box)),delta=(rb[tangent]+rb[tangent+2]-tb[tangent]-tb[tangent+2])/2;
    for(const node of g.nodes)shift(node,tangent===0?delta:0,tangent===1?delta:0);
   }
   // Example row descriptions sit outside the Y-title column. They must not
   // occupy the now explicitly reserved tick-label-to-axis-title strip.
   const exampleLabels=[];
   if(n===1)for(const panel of ['D','E']){
    const titleNodes=[...new Set(axes.filter(a=>a.panel===panel&&a.orientation==='y').flatMap(a=>a.title_nodes))];
    const edge=union(titleNodes.map(box))[0]-3*25.4/72;
    for(const node of root.querySelectorAll(`[id^="edit-${panel}-trial-"]`))if(node.localName==='text'&&/-((name)|(block))$/.test(node.id)){
     if(!verify)shift(node,edge-box(node)[2],0);
     exampleLabels.push({element_id:node.id,size_pt:parseFloat(getComputedStyle(node).fontSize)*Math.hypot(ctm(node).c,ctm(node).d)/pt,gap_to_y_title_mm:union(titleNodes.map(box))[0]-box(node)[2]});
    }
   }
   for(const a of axes){const ticks=a.tick_nodes.map(node=>{const [x1,y1,x2,y2]=ends(node),m=ctm(node),v=new DOMPoint(x2-x1,y2-y1).matrixTransform(new DOMMatrix([m.a,m.b,m.c,m.d,0,0]));const vertical=Math.abs(x2-x1)<1e-8;return {element_id:node.id,length_pt:Math.hypot(v.x,v.y)/pt,stroke_width_pt:parseFloat(getComputedStyle(node).strokeWidth)*(vertical?Math.hypot(m.a,m.b):Math.hypot(m.c,m.d))/pt,stroke:getComputedStyle(node).stroke,linecap:getComputedStyle(node).strokeLinecap}});
    const normal=a.orientation==='x'?1:0,tangent=1-normal;
    const gaps=a.label_nodes.map(label=>{if(!a.tick_nodes.length)return null;const lb=box(label),center=(lb[tangent]+lb[tangent+2])/2;const tick=[...a.tick_nodes].sort((u,v)=>{const U=box(u),V=box(v);return Math.abs((U[tangent]+U[tangent+2])/2-center)-Math.abs((V[tangent]+V[tangent+2])/2-center)})[0];const [x1,y1,x2,y2]=ends(tick),m=ctm(tick),start=new DOMPoint(x1,y1).matrixTransform(m),end=new DOMPoint(x2,y2).matrixTransform(m),sign=Math.sign(normal===1?end.y-start.y:end.x-start.x);return sign*((sign>0?lb[normal]:lb[normal+2])-(normal===1?end.y:end.x)*mm)}).filter(v=>v!==null);
    const key=a.titles.map(t=>t.xml_index_path.join('.')).join('|'),g=titleGroups.get(key);let titleGap=null;
    if(g){const labels=[...new Set(g.axes.flatMap(a=>a.label_nodes))];if(labels.length){const lb=union(labels.map(box)),tb=union(g.nodes.map(box));titleGap=Math.max(tb[normal]-lb[normal+2],lb[normal]-tb[normal+2])}}
    const textRecord=node=>({element_id:node.id,box_mm:box(node),size_pt:parseFloat(getComputedStyle(node).fontSize)*Math.hypot(ctm(node).c,ctm(node).d)/pt,font_family:getComputedStyle(node).fontFamily,font_weight:getComputedStyle(node).fontWeight});
    measured.push({id:a.id,ticks,tick_label_gaps_mm:gaps,title_gap_mm:titleGap,tick_labels:a.label_nodes.map(textRecord),titles:a.title_nodes.map(textRecord),tickless:!ticks.length});
   }
   return {svg:new XMLSerializer().serializeToString(root),changes:[...changes].map(([node,r])=>({...r,after_attributes:[...node.attributes].map(a=>[a.name,a.value])})),axes:measured,example_labels:exampleLabels};
  },{input,n,verify,centerOnly});
  if(!verify||centerOnly)fs.writeFileSync(path.join(dir,`figure-0${n}-layout.svg`),result.svg);delete result.svg;
  result.artifact_sha256=require('crypto').createHash('sha256').update(verify?svg:fs.readFileSync(path.join(dir,`figure-0${n}-layout.svg`))).digest('hex');
  fs.writeFileSync(path.join(dir,`figure-0${n}-axis-geometry${centerOnly?'-centering':verify?'-verification':''}.json`),JSON.stringify(result,null,2));
  console.log(n,'registered modified leaves',result.changes.length,'ticks',result.axes.reduce((s,a)=>s+a.ticks.length,0));
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
