"""Write PGFPlots coordinates directly from the frozen experiment records."""
from pathlib import Path
import json, math, statistics as st, hashlib
P=Path(__file__).resolve().parent;S=P.parent/'experiments/rigorous'
groups={}
for e in json.loads((S/'final_evaluations.json').read_text()):
 c=e['config']
 if c['budget']=='parameters' and c['context']==128:
  groups.setdefault((c['phase'],c['kind']),[]).append(st.mean(b['nats'] for b in e['test'].values()))
style={'full':'black,mark=*,mark options={fill=black}','localglobal':'black!65,mark=square*,mark options={fill=black!65},dashed','striped':'black,mark=triangle*,mark options={fill=black},line width=.9pt','localonly':'black!65,mark=diamond*,mark options={fill=black!65}'}
def plotquality(rows):
 s=''
 for y,key in rows:
  v=groups[key];mu=st.mean(v);sd=st.stdev(v);k=key[1]
  s+=r'\addplot['+style[k]+r',only marks,mark size=2pt,error bars/.cd,x dir=both,x explicit] coordinates {'+f'({mu:.10f},{y}) +- ({sd:.10f},0)'+'};\n'
  s+=r'\addplot[only marks,black!60,mark=*,mark size=.7pt] coordinates {'+' '.join(f'({x:.10f},{y+.13})' for x in v)+'};\n'
 return s
q=r'''\begin{tikzpicture}
\begin{axis}[name=left,width=7.0cm,height=4.65cm,xmin=1.62,xmax=1.83,ymin=-.5,ymax=3.5,
xtick={1.65,1.70,1.75,1.80},ytick={0,1,2,3},yticklabels={ModalStripe,Local-only$^*$,Local + global,Full attention},xlabel={Loss (nats/character)},title={(a) Equal training characters},ytick style={draw=none},axis y line*=left,y axis line style={draw=none}]
'''+plotquality([(3,('main','full')),(2,('main','localglobal')),(1,('posthoc_local','localonly')),(0,('main','striped'))])+r'''\end{axis}
\begin{axis}[at={(left.south east)},anchor=south west,xshift=3.25cm,width=7.0cm,height=4.65cm,xmin=1.62,xmax=1.83,ymin=-.5,ymax=3.5,
xtick={1.65,1.70,1.75,1.80},ytick={0,1,2,3},yticklabels={ModalStripe: 1{,}800,Full: elapsed time$^\dagger$,Full: 2{,}439,Full: 1{,}800},xlabel={Loss (nats/character)},title={(b) Training budgets},ytick style={draw=none},axis y line*=left,y axis line style={draw=none}]
'''+plotquality([(3,('main','full')),(2,('calibrated_time','full')),(1,('equal_time','full')),(0,('main','striped'))])+r'\end{axis}\end{tikzpicture}'
(P/'plots/quality.tex').write_text(q)
rows=json.loads((S/'paired_timing.json').read_text())['rows'];timing={}
for k in ('full','localglobal','striped'):
 timing[k]={L:{m:[st.median(r[m] for r in rows if r['kind']==k and r['length']==L and r['seed']==seed) for seed in range(3)] for m in ('decode_ms','cache_bytes')} for L in (128,512,2048,8192)}
r=r'''\begin{tikzpicture}
\begin{axis}[name=left,width=8.1cm,height=5.0cm,xmode=log,log basis x=2,xmin=110,xmax=9500,xtick={128,512,2048,8192},xticklabels={128,512,2{,}048,8{,}192},ymin=.1,ymax=.36,ytick={.1,.2,.3},xlabel={Prefix length (characters; log scale)},ylabel={Decode (ms/character)},title={(a) CPU decoding},legend to name=models,legend columns=3]
'''
for k,label in [('full','Full attention'),('localglobal','Local + global'),('striped','ModalStripe')]:
 r+=r'\addplot['+style[k]+r',mark size=2pt] coordinates {'+' '.join(f'({L},{st.mean(v["decode_ms"]):.10f})' for L,v in timing[k].items())+r'};\addlegendentry{'+label+'}\n'
 r+=r'\addplot[only marks,black!55,mark=*,mark size=.65pt,forget plot] coordinates {'+' '.join(f'({L},{x:.10f})' for L,v in timing[k].items() for x in v['decode_ms'])+'};\n'
r+=r'''\end{axis}
\begin{axis}[at={(left.south east)},anchor=south west,xshift=2.75cm,width=6.4cm,height=5.0cm,xmin=0,xmax=10,xtick={0,4,8},ymin=-.55,ymax=2.55,ytick={0,1,2},yticklabels={ModalStripe,Local + global,Full attention},xlabel={Retained cache (MiB)},title={(b) Cache at prefix 8{,}192},axis y line*=left,y axis line style={draw=none},ytick style={draw=none}]
'''
for y,k in [(2,'full'),(1,'localglobal'),(0,'striped')]:
 v=st.mean(timing[k][8192]['cache_bytes'])/1048576
 r+=r'\addplot['+style[k]+r',only marks,mark size=2pt] coordinates {'+f'({v:.9f},{y})'+r'};\node[anchor=west,font=\fontsize{10}{12}\selectfont] at (axis cs:'+f'{v+.12},{y}'+'){'+f'{v:.2f}'+'};\n'
r+=r'''\end{axis}
\node[anchor=south] at ($(left.north east)+(.7,.48)$) {\ref{models}};
\end{tikzpicture}'''
(P/'plots/resources.tex').write_text(r)
# A companion context plot has the complete model comparison, in separate training-context panels.
rows=json.loads((S/'exact_window_audit.json').read_text())['rows'];out=r'\begin{tikzpicture}'
for i,ctx in enumerate((128,512)):
 out+=r'\begin{axis}['+(r'name=left,legend to name=contextmodels,legend columns=3,' if i==0 else r'at={(left.south east)},anchor=south west,xshift=1.8cm,')+f'width=8cm,height=5.4cm,title={{({"ab"[i]}) Training context {ctx}}},'+r'xmode=log,log basis x=2,xmin=110,xmax=9500,xtick={128,512,2048,8192},xticklabels={128,512,2{,}048,8{,}192},ymin=4,ymax=16,ytick={4,8,12,16},xlabel={History (characters; log scale)},ylabel={Perplexity}]'+'\n'
 for k in ('full','localglobal','striped'):
  row=next(r for r in rows if r['kind']==k and r['train_context']==ctx)
  coords=[(L,math.exp(st.mean(b[str(L)]['nats'] for b in row['scores'].values()))) for L in (128,512,2048,8192)]
  out+=r'\addplot['+style[k]+r',mark size=2pt] coordinates {'+' '.join(f'({L},{v:.9f})' for L,v in coords)+'};\n'
 
  if i==0:
   out+=r'\addlegendentry{'+{'full':'Full attention','localglobal':'Local + global','striped':'ModalStripe'}[k]+'}\n'
 out+=r'\end{axis}'
out+=r'\node[anchor=south] at ($(left.north east)+(.7,.5)$) {\ref{contextmodels}};\end{tikzpicture}'
(P/'plots/context.tex').write_text(out)
(P/'plots/provenance.json').write_text(json.dumps({n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ('final_evaluations.json','paired_timing.json','exact_window_audit.json')},indent=2))

print("Wrote quality, resource and context PGFPlots from frozen JSON.")
