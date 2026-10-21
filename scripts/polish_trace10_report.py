"""Render the saved scientific Markdown as readable HTML; preserve prior text."""
from pathlib import Path
import html,re,sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from analyze_trace10_learners import OUT,STATE,load,ref,check
from classical_conditioning.figure_workspace import Snapshot,publish_state,sha

def inline(value):
    value=html.escape(value)
    value=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    return re.sub(r'`([^`]+)`',r'<code>\1</code>',value)

def main():
    state=load(STATE);refs=state['draft']['report'];check(list(refs.values()))
    md=Path(refs['markdown']['path']);page=Path(refs['html']['path'])
    snapshot=Snapshot([STATE,md,page,Path(__file__),REPO/'docs/paper/figures/figure-01/current-state.json',REPO/'docs/paper/figures/figure-02/current-state.json'])
    text=md.read_text(encoding='utf-8');text=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text=re.sub(r'(?<=[A-Za-z])(?=\[)', ' ', text);text=re.sub(r'(?<=[A-Za-z])(?=[+−]\d)', ' ', text)
    body=[]
    for block in text.strip().split('\n\n'):
        if block.startswith('# '):body.append('<h1>'+inline(block[2:])+'</h1>')
        elif block.startswith('## '):body.append('<h2>'+inline(block[3:])+'</h2>')
        elif block.startswith('|'):
            lines=block.splitlines();head=[x.strip() for x in lines[0].strip('|').split('|')]
            rows=[[x.strip() for x in line.strip('|').split('|')] for line in lines[2:]]
            body.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
        else:body.append('<p>'+inline(block.replace('\n',' '))+'</p>')
    links='<p><a href="candidate-comparison.html">Matched editable comparison</a> · <a href="preferred-single-fish.html">Preferred single-fish heatmap</a> · <a href="full-score-coverage-exclusion-table-verified.csv">Complete score/coverage/exclusion table</a></p>'
    data=('<!doctype html><html><meta charset="utf-8"><title>10sTrace candidate recommendation</title><style>body{max-width:1050px;margin:36px auto;padding:0 24px;font:16px/1.55 Arial,sans-serif;color:#171717;background:white}h1{font-size:29px;line-height:1.2}h2{font-size:21px;margin-top:30px}p{margin:14px 0}.table-scroll{overflow-x:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}code{font-size:14px}a{color:#0645ad}</style>'+''.join(body)+links+'</html>').encode()
    state['draft']['report']={'markdown':{'path':str(md),'sha256':sha(text.encode())},'html':{'path':str(page),'sha256':sha(data)}}
    state['verification']['report_formatting']={'status':'readable semantic headings, paragraphs and table; no numerical or scientific changes','generator':ref(Path(__file__))}
    history=OUT/'history'
    publish_state(STATE,state,snapshot,artifacts=[(history/'selection-report-before-formatting.md',md.read_bytes()),(history/'selection-report-before-formatting.html',page.read_bytes())],evidence_updates=[(md,text.encode()),(page,data)])
    print(page)

if __name__=='__main__':main()
