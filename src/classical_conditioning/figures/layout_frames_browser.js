/* Browser-only collector. No state writer, drawing overlay or network resources.
   Loaded by the calibration/check review; exact source hash accompanies evidence. */
globalThis.collectLayoutFrames = async function(svgText, collectorHash) {
  const digest = async text => [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)))].map(v=>v.toString(16).padStart(2,'0')).join('');
  const source = new DOMParser().parseFromString(svgText, 'image/svg+xml');
  if(source.querySelector('parsererror')) throw Error('Invalid SVG');
  const root=source.documentElement, metadata=root.querySelector('metadata#layout-frames');
  if(!metadata) throw Error('Missing frame metadata');
  const spec=JSON.parse(metadata.textContent), size=spec.final_size_mm;
  const report={artifact_sha256:await digest(svgText),spec_sha256:await digest(metadata.textContent),collector_sha256:collectorHash,
    final_size_mm:size,scope:spec.scope,user_agent:navigator.userAgent,fonts_ready:false,images_ready:false,
    allocations:{},data_axes:{},issues:[],raster_dpi:600};
  const issue = text => {if(!report.issues.includes(text)) report.issues.push(text);};
  if(root.querySelector('script,foreignObject,animate,animateTransform,set')) issue('Unsupported active/foreign SVG content');
  for(const node of root.querySelectorAll('*')) for(const attr of node.attributes) {
    if(/^on/i.test(attr.name)) issue('Unsupported event attributes');
    if(/href$/.test(attr.name) && !/^(#|data:)/.test(attr.value)) issue('External resource: '+attr.value);
  }
  const cssText=[...root.querySelectorAll('style')].map(n=>n.textContent).join('');
  if(/@import/i.test(cssText)) issue('External CSS resource');
  for(const match of cssText.matchAll(/url\(([^)]+)\)/g)) if(!/^(data:|#)/.test(match[1].trim().replace(/^["']|["']$/g,''))) issue('External CSS resource');
  if(report.issues.length) return report;
  const host=document.createElement('div');host.style.cssText='position:absolute;left:-100000px;top:0;visibility:hidden';
  // Chromium does not register @font-face rules declared inside a shadow root.
  // Register only the SVG's self-contained faces in the document while measuring.
  const fontStyle=document.createElement('style');
  fontStyle.textContent=[...cssText.matchAll(/@font-face\s*\{[^}]*\}/g)].map(m=>m[0]).join('\n');
  document.head.append(fontStyle);
  const live=document.importNode(root,true);host.style.font='initial';host.attachShadow({mode:'open'}).append(live);document.body.append(host);
  try {
    await document.fonts.ready;
    for(const text of live.querySelectorAll('text')) {
      const style=getComputedStyle(text); await document.fonts.load(`${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`,text.textContent);
    }
    await document.fonts.ready;report.fonts_ready=true;
    for(const n of live.querySelectorAll('image')) {
      const href=n.getAttribute('href')||n.getAttributeNS('http://www.w3.org/1999/xlink','href');
      if(!href||!href.startsWith('data:')) {issue('Unsupported image resource');continue;}
      await new Promise((resolve,reject)=>{const image=new Image();image.onload=resolve;image.onerror=()=>reject(Error('Embedded image decode failed'));image.src=href;});
    }
    const vb=live.viewBox.baseVal;
    if(!vb.width || !vb.height || Math.abs(vb.width/vb.height-size[0]/size[1])>1e-7) issue('Final physical size/viewBox aspect mismatch');
    for(const n of live.querySelectorAll('*')) {
      if(n.closest('defs,clipPath,mask,marker,pattern,symbol'))continue;
      const s=getComputedStyle(n);
      if(s.filter!=='none'||(s.maskImage&&s.maskImage!=='none')||s.mixBlendMode!=='normal')issue('Unsupported ancestor effect: '+(n.id||n.localName));
    }
    const drawable='path,rect,line,polyline,polygon,circle,ellipse,text,image,use';
    const leaves=[...live.querySelectorAll(drawable)].filter(n=>!n.closest('defs,clipPath,mask,marker,pattern,symbol'));
    const backgrounds=new Set(spec.background_ids||[]), owners=new Map();
    for(const a of spec.allocations) for(const id of a.group_ids) {
      const matches=[...live.querySelectorAll('[id]')].filter(n=>n.id===id);
      if(matches.length!==1) issue('Missing/duplicate owner group: '+id);else owners.set(matches[0],a.id);
    }
    const ownerOf=node=>{const found=new Set();for(let p=node;p&&p!==live;p=p.parentElement)if(owners.has(p))found.add(owners.get(p));
      if(found.size>1)issue('Overlapping content ownership: '+node.id);return [...found][0];};
    const ignored=node=>{for(let p=node;p;p=p.parentElement)if(backgrounds.has(p.id))return true;return false;};
    for(const id of backgrounds) if(![...live.querySelectorAll('[id]')].some(n=>n.id===id))issue('Unknown explicit background: '+id);
    const visible=node=>{for(let p=node;p&&p!==host;p=p.parentElement){const s=getComputedStyle(p);if(s.display==='none'||s.visibility==='hidden'||+s.opacity===0)return false;}return true;};
    // Host opacity/visibility must not change the SVG's computed visibility.
    host.style.visibility='visible';
    const rootInverse=live.getScreenCTM().inverse();
    const geomBounds=node=>{
      const b=node.getBBox(), m=rootInverse.multiply(node.getScreenCTM());
      const pts=[[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y])=>new DOMPoint(x,y).matrixTransform(m));
      const s=getComputedStyle(node), line=s.stroke!=='none'?parseFloat(s.strokeWidth)||0:0;
      // Frobenius norm bounds any affine normal; include miter joins conservatively.
      const pad=line/2*Math.hypot(m.a,m.b,m.c,m.d)*(s.strokeLinejoin==='miter'?Math.max(1,+s.strokeMiterlimit):1);
      const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
      return [Math.min(...xs)-pad,Math.min(...ys)-pad,Math.max(...xs)+pad,Math.max(...ys)+pad];
    };
    const allBounds=[];
    for(const n of leaves) {
      const s=getComputedStyle(n);
      if(!visible(n)||ignored(n))continue;
      if(!ownerOf(n))issue('Unowned visible content: '+(n.id||n.localName));
      if(s.mask&&s.mask!=='none')issue('Unsupported mask: '+n.id);
      if(s.filter!=='none'||s.maskImage&&s.maskImage!=='none'||s.mixBlendMode!=='normal')issue('Unsupported filter/mask/blending: '+n.id);
      if(s.vectorEffect&&s.vectorEffect!=='none')issue('Unsupported vector effect: '+n.id);
      try{allBounds.push(geomBounds(n));}catch(e){issue('Unmeasurable geometry: '+n.id);}
    }
    // Expand the raster canvas to include pre-clip geometry, preventing a crop
    // at the root canvas from hiding overflow. Extra 5 mm captures marker ends.
    const sx=size[0]/vb.width,sy=size[1]/vb.height,pxPerMM=600/25.4;
    const left=Math.min(vb.x,...allBounds.map(b=>b[0]))-5/sx,top=Math.min(vb.y,...allBounds.map(b=>b[1]))-5/sy;
    const right=Math.max(vb.x+vb.width,...allBounds.map(b=>b[2]))+5/sx,bottom=Math.max(vb.y+vb.height,...allBounds.map(b=>b[3]))+5/sy;
    const pixelW=Math.ceil((right-left)*sx*pxPerMM),pixelH=Math.ceil((bottom-top)*sy*pxPerMM);
    if(pixelW*pixelH>50000000){issue('Raster bounds exceed safe measurement capacity');return report;}
    const render=async(owner,unclip)=>{
      const clone=live.cloneNode(true), originals=[...live.querySelectorAll(drawable)],copies=[...clone.querySelectorAll(drawable)];
      for(let i=0;i<originals.length;i++) {
        const n=originals[i];if(n.closest('defs,clipPath,mask,marker,pattern,symbol'))continue;
        if(ignored(n)||ownerOf(n)!==owner)copies[i].remove();
      }
      const allowedClips=new Set((spec.scientific_clips||[]).map(c=>c.id));
      for(const n of clone.querySelectorAll('*')) {
        if(unclip) {
          const original=n.id?[...live.querySelectorAll('[id]')].find(e=>e.id===n.id):null;
          const value=original?getComputedStyle(original).clipPath:(n.getAttribute('clip-path')||n.style?.clipPath);
          if(value&&value!=='none'&&!allowedClips.has(value.match(/#([^)'"\s]+)/)?.[1])) {n.removeAttribute('clip-path');n.style?.setProperty('clip-path','none','important');}
          if(n.localName==='svg') {n.setAttribute('overflow','visible');n.style.overflow='visible';}
        }
      }
      clone.setAttribute('viewBox',`${left} ${top} ${right-left} ${bottom-top}`);
      clone.setAttribute('width',pixelW);clone.setAttribute('height',pixelH);
      const image=new Image(),url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'}));
      try{
        await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error('SVG/image decode failed'));image.src=url;});
        const canvas=document.createElement('canvas');canvas.width=pixelW;canvas.height=pixelH;
        const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(image,0,0);
        const pixels=ctx.getImageData(0,0,pixelW,pixelH).data;
        let x0=pixelW,y0=pixelH,x1=-1,y1=-1;
        for(let y=0;y<pixelH;y++)for(let x=0;x<pixelW;x++)if(pixels[(y*pixelW+x)*4+3]>0){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}
        if(x1<0)return null;
        if(x0===0||y0===0||x1===pixelW-1||y1===pixelH-1)issue('Paint reaches expanded raster boundary: '+owner);
        const unitX=(right-left)/pixelW,unitY=(bottom-top)/pixelH;
        return [(left+x0*unitX-vb.x)*sx,(top+y0*unitY-vb.y)*sy,(x1-x0+1)*unitX*sx,(y1-y0+1)*unitY*sy];
      }finally{URL.revokeObjectURL(url);}
    };
    for(const a of spec.allocations) report.allocations[a.id]={painted_bounds_mm:await render(a.id,false),unclipped_bounds_mm:await render(a.id,true)};
    for(const mapping of spec.data_axes||[]) {
      const ids=mapping.element_ids||[mapping.element_id], bounds=[];
      for(const id of ids){const n=[...live.querySelectorAll('[id]')].find(n=>n.id===id);
        if(!n){issue('Missing data axis: '+mapping.id+'/'+id);continue;}bounds.push(geomBounds(n));}
      if(bounds.length!==ids.length)continue;
      const b=[Math.min(...bounds.map(v=>v[0])),Math.min(...bounds.map(v=>v[1])),Math.max(...bounds.map(v=>v[2])),Math.max(...bounds.map(v=>v[3]))];
      report.data_axes[mapping.id]=[(b[0]-vb.x)*sx,(b[1]-vb.y)*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy];
    }
    report.images_ready=true;
    return report;
  } catch(error){issue(String(error));return report;}finally{host.remove();fontStyle.remove();}
};
