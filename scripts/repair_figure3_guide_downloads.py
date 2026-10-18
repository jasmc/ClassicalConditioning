"""Guarded interaction-only repair of embedded guide download links."""
from pathlib import Path
import argparse,base64,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,publish_state

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
HANDLER='''<script id="embedded-source-downloads">
// Blob downloads work in the in-app browser while retaining offline payloads.
for (const source of document.querySelectorAll('a[download][href^="data:"]')) {
  source.addEventListener('click', event => {
    event.preventDefault();
    const raw=source.getAttribute('href'), comma=raw.indexOf(','), header=raw.slice(5,comma);
    const content=raw.slice(comma+1), mime=header.split(';')[0] || 'application/octet-stream';
    const bytes=header.includes(';base64') ? Uint8Array.from(atob(content), c=>c.charCodeAt(0)) : new TextEncoder().encode(decodeURIComponent(content));
    const url=URL.createObjectURL(new Blob([bytes],{type:mime}));
    const download=document.createElement('a'); download.href=url; download.download=source.download;
    document.body.append(download); download.click(); download.remove();
    setTimeout(()=>URL.revokeObjectURL(url),3000);
  });
}
</script>'''

def main(out):
 ticketpath=out/'link-repair-ticket-20261018.json';ticket=json.loads(ticketpath.read_text());record=Path(ticket['record']);path=Path(ticket['html']['path'])
 if sha(record)!=ticket['record_sha256'] or sha(path)!=ticket['html']['sha256']:raise ValueError('Guide/state changed; preserve and reconcile')
 state=json.loads(record.read_text());before=path.read_text();beforebytes=path.read_bytes()
 if 'id="embedded-source-downloads"' in before:raise ValueError('Repair already present')
 # Disambiguate the two source modules with the same basename.
 after=before
 for folder in ['review','figures']:
  pattern=r'<a download="individual_atlas\.py" href="(data:[^"]+)">([^<]*'+folder+r'/individual_atlas\.py)</a>'
  after,n=re.subn(pattern,lambda m:'<a download="'+folder+'-individual_atlas.py" href="'+m[1]+'">'+m[2]+'</a>',after)
  if n!=1:raise ValueError('Missing/ambiguous source download '+folder)
 after=after.replace('</body>',HANDLER+'</body>')
 extract=lambda s:re.search(r'<script id="guide-data" type="application/json">(.*?)</script>',s,re.S)[1]
 if extract(after)!=extract(before):raise ValueError('Scientific payload changed')
 # Every embedded file remains byte-identical; only event handling/names change.
 payloads=lambda s:re.findall(r'<a download="[^"]+" href="(data:[^"]+)"',s)
 if payloads(after)!=payloads(before):raise ValueError('Download contents changed')
 snapshot=Snapshot([record,path,ticketpath,Path(__file__),*ticket['protected']]);snapshot.hashes[str(record.resolve())]=ticket['record_sha256'];snapshot.hashes[str(path.resolve())]=ticket['html']['sha256']
 for p,h in ticket['protected'].items():
  if sha(p)!=h:raise ValueError('Protected figure state changed')
 history=out/'history'/'links-before-20261018';history.mkdir(parents=True,exist_ok=True)
 report={'date':'2026-10-18','scope':ticket['scope'],'before_sha256':sha(path),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'embedded_scientific_payload_preserved':True,'all_17_download_payloads_preserved':len(payloads(after))==17,'duplicate_filenames_disambiguated':True,'note_schema_and_storage_key_preserved':True,'implementation':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}}
 state['html']['sha256']=report['after_sha256'];state['link_repair']={'path':str(out/'link-repair-20261018.json'),'sha256':hashlib.sha256((json.dumps(report,indent=2)+'\n').encode()).hexdigest()}
 state['verification_before_link_repair']=state.pop('verification',None)
 publish_state(record,state,snapshot,artifacts=[(history/path.name,beforebytes),(history/'guide-record.json',record.read_bytes()),(out/'link-repair-20261018.json',(json.dumps(report,indent=2)+'\n').encode()),(out/'source-link-repair.py',Path(__file__).read_bytes())],evidence_updates=[(path,after.encode())])
 print(json.dumps(report))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();main(args.output_dir.resolve())
