"""Build manuscript tables directly from the frozen study JSON records."""
import json,math,statistics,hashlib
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent
S=P.parent/'experiments'/'rigorous'
es=json.loads((S/'final_evaluations.json').read_text());groups=defaultdict(list)
for e in es:
 c=e['config'];r=json.loads((S/e['run']/'result.json').read_text())
 groups[(c['phase'],c['budget'],c['context'],c['kind'])].append((e,r))
labels={'full':'Full attention','localglobal':'Local + global','striped':'ModalStripe','localonly':'Local-only'}
def mean(v):return statistics.mean(v)
def sd(v):return statistics.stdev(v)
def vals(rows):return [mean(v['nats'] for v in e['test'].values()) for e,r in rows]
def cell(rows):
 v=vals(rows);return f'{mean(v):.4f} $\\pm$ {sd(v):.4f}',f'{math.exp(mean(v)):.3f}'
def write(name,rows):
 (P/'tables'/f'{name}.tex').write_text('\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n')
write('primary',[[labels[k],f'{groups[("main","parameters",128,k)][0][1]["parameters"]:,}',*cell(groups[('main','parameters',128,k)])] for k in ('full','localglobal','striped')])
write('fixedff',[[labels[k],f'{groups[("main","fixed_ff",128,k)][0][1]["parameters"]:,}',*cell(groups[('main','fixed_ff',128,k)])] for k in ('full','localglobal','striped')])
rows=[]
for phase,kind,label in [('main','full','Equal characters: attention'),('main','striped','Equal characters: ModalStripe'),('calibrated_time','full','Calibrated budget: attention'),('equal_time','full','Literal elapsed time: attention')]:
 rs=groups[(phase,'parameters',128,kind)];updates=[r['steps'] for e,r in rs]
 update=f'{mean(updates):,.0f}'+(r' (mean)' if len(set(updates))>1 else '')
 rows.append([label,update,*cell(rs)])
write('budgets',rows)
rows=[]
for n in ('hamlet','alice','moby'):
 meta=es[0]['config']['provenance']['data'][n];N=meta['characters'];a=int(.8*N);b=int(.9*N)
 rows.append([{'hamlet':r'\emph{Hamlet}','alice':r'\emph{Alice}','moby':r'\emph{Moby-Dick}'}[n],str(meta['gutenberg']),f'{N:,}',f'{a:,}',f'{b-a:,}','--'])
for n in ('frankenstein','sherlock'):
 meta=es[0]['test_metadata'][n];N=meta['characters']
 rows.append([{'frankenstein':r'\emph{Frankenstein}','sherlock':r'\emph{Sherlock Holmes}'}[n],str(meta['gutenberg']),f'{N:,}','--','--',f'{N:,}'])
write('datasets',rows)
rows=json.loads((S/'paired_timing.json').read_text())['rows'];out=[]
for length in (128,8192):
 for k in ('full','localglobal','striped'):
  med=[]
  for seed in range(3):
   rs=[r for r in rows if r['length']==length and r['kind']==k and r['seed']==seed]
   med.append({x:statistics.median(r[x] for r in rs) for x in ('prefill_ms','decode_ms','cache_bytes')})
  out.append([f'{length:,}',labels[k],f'{mean(r["prefill_ms"] for r in med):.3f}',f'{mean(r["decode_ms"] for r in med):.4f}',f'{mean(r["cache_bytes"] for r in med)/1024:,.2f}'])
write('timing',out)
rows=json.loads((S/'paired_training_timing.json').read_text())['rows'];out=[]
for length in (128,512):
 for k in ('full','localglobal','striped'):
  med=[statistics.median(r['step_ms'] for r in rows if r['context']==length and r['kind']==k and r['seed']==seed) for seed in range(3)]
  out.append([str(length),labels[k],f'{mean(med):.3f}'])
write('train_timing',out)
rows=json.loads((S/'exact_window_audit.json').read_text())['rows'];out=[]
for r in rows:
 ns={n:mean(book[n]['nats'] for book in r['scores'].values()) for n in ('128','8192')}
 out.append([str(r['train_context']),labels[r['kind']],f'{math.exp(ns["128"]):.3f}',f'{math.exp(ns["8192"]):.3f}'])
write('exact',out)
out=[]
for phase,k in [('main','full'),('main','localglobal'),('main','striped'),('calibrated_time','full'),('posthoc_local','localonly')]:
 rs=groups[(phase,'parameters',128,k)];label=labels[k]+(' (calibrated)' if phase=='calibrated_time' else '')
 out.append([label,*[f'{math.exp(mean(e["test"][n]["nats"] for e,r in rs)):.3f}' for n in ('frankenstein','sherlock')]])
write('perbook',out)
out=[]
for context in (128,512):
 for k in ('full','localglobal','striped'):
  rs=groups[('main','parameters',context,k)]
  ns={L:mean(mean(book[L]['nats'] for book in e['long_test'].values()) for e,r in rs) for L in ('128','8192')}
  out.append([str(context),labels[k],f'{math.exp(ns["128"]):.3f}',f'{math.exp(ns["8192"]):.3f}'])
write('continuation',out)
rs=groups[('posthoc_local','parameters',128,'localonly')]
write('localquality',[[labels['localonly'],f'{rs[0][1]["parameters"]:,}',*cell(rs)]])
local=json.loads((S/'local_control_timing.json').read_text())['rows'];out=[]
for k in ('full','striped','localonly'):
 rs=[r for r in local if r['kind']==k and r['length']==8192]
 out.append([labels[k],f'{statistics.median(r["decode_ms"] for r in rs):.4f}',f'{rs[0]["cache_bytes"]/1024:,.2f}'])
write('localtiming',out)
write('longtrain',[[labels[k],*cell(groups[('main','parameters',512,k)])] for k in ('full','localglobal','striped')])
write('learning',[[labels[k],str(step),f'{mean(mean(v["nats"] for v in h["dev"].values()) for e,r in groups[("main","parameters",128,k)] for h in r["history"] if h["step"]==step):.4f}'] for k in ('full','localglobal','striped') for step in (300,600,1200,1800)])
manifest={name:hashlib.sha256((S/name).read_bytes()).hexdigest() for name in ('final_evaluations.json','paired_timing.json','paired_training_timing.json','exact_window_audit.json','trained_checks.json','data_overlap_audit.json','selection.json','calibration_decision.json','local_control_timing.json')}
manifest['run_ids']=[e['run'] for e in es]
(P/'manuscript_evidence.json').write_text(json.dumps(manifest,indent=2))
print('Generated 13 tables from',len(es),'evaluated models.')
print('literal updates:',[r['steps'] for e,r in groups[('equal_time','parameters',128,'full')]])

# Include complete tabular environments so file hooks run outside alignment rows.
layouts=json.loads((P/'table_layouts.json').read_text())
for name,layout in layouts.items():
 path=P/'tables'/f'{name}.tex'
 body=path.read_text()
 path.write_text('\\begin{tabular}{'+layout['columns']+'}\\toprule\n'+layout['header']+'\\midrule\n'+body+'\\bottomrule\\end{tabular}\n')
# Compact comparison for the revised main text; exploratory controls stay marked.
comparison=[]
for phase,kind,label in [('main','full','Full attention'),('main','localglobal','Local + global'),('main','striped','ModalStripe'),('calibrated_time','full','Full, calibrated budget'),('equal_time','full','Full, elapsed-time budget'),('posthoc_local','localonly',r'Local-only$^{\dagger}$')]:
 rs=groups[(phase,'parameters',128,kind)];updates=[r['steps'] for e,r in rs]
 n=f'{mean(updates):,.0f}'+(' avg.' if len(set(updates))>1 else '')
 comparison.append([label,n,*cell(rs)])
write('comparison',comparison)
p=P/'tables/comparison.tex';p.write_text('\\begin{tabular}{lrrr}\\toprule\nModel and budget & Updates & Test loss (nats) & PPL\\\\\\midrule\n'+p.read_text()+'\\bottomrule\\end{tabular}\n')
rows=json.loads((S/'exact_window_audit.json').read_text())['rows'];short=[]
for r in rows:
 if r['kind']=='striped':
  short.append([str(r['train_context']),*[f'{math.exp(mean(book[L]["nats"] for book in r["scores"].values())):.3f}' for L in ('128','8192')]])
write('length_summary',short)
p=P/'tables/length_summary.tex';p.write_text('\\begin{tabular}{rrr}\\toprule\nTraining context & PPL at 128 & PPL at 8,192\\\\\\midrule\n'+p.read_text()+'\\bottomrule\\end{tabular}\n')
