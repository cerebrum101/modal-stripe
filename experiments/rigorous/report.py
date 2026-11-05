"""Generate descriptive report from every frozen run, never select by test score."""
import json,math,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def mean(v):return statistics.mean(v)
def sd(v):return statistics.stdev(v) if len(v)>1 else 0.
def main():
 evaluations=json.loads((ROOT/'final_evaluations.json').read_text())
 groups=defaultdict(list)
 for e in evaluations:
  c=e['config'];r=json.loads((ROOT/e['run']/'result.json').read_text())
  groups[(c['phase'],c['budget'],c['context'],c['kind'])].append((e,r))
 lines=['# Controlled spectral-hybrid study','',
 'Primary model settings were frozen before fetching the final books. The calibrated-time baseline follows a rule specified before final-score inspection; the local-only control and corrective fixed-window diagnostic are explicitly exploratory additions. Final texts are Frankenstein and Sherlock Holmes. The first table always evaluates reset128-character windows, including for models trained with context512; use the identical-target context table for matched long-context comparisons. Three seeds describe optimization spread, not independent datasets. Quality uses fixed ASCII128 next-character prediction. No claims about useful-scale LLMs or GPU performance follow from this CPU study.','',
 '## Final book results with reset128 evaluation','',
 '| Phase | Budget | Train context | Model | Parameters | Updates | Training seconds | Mean test nats ± seed SD | Geometric test PPL |','|---|---|---:|---|---:|---:|---:|---:|---:|']
 for key,rs in sorted(groups.items()):
  phase,budget,context,kind=key
  ns=[mean([v['nats'] for v in e['test'].values()]) for e,r in rs]
  lines.append(f'| {phase} | {budget} | {context} | {kind} | {rs[0][1]["parameters"]} | {mean([r["steps"] for e,r in rs]):.0f} | {mean([r["train_seconds"] for e,r in rs]):.1f} | {mean(ns):.4f} ± {sd(ns):.4f} | {math.exp(mean(ns)):.3f} |')
 lines+=['','## Spectral attribution','',
 'Differences below are striped minus localglobal cross-entropy on the same seeds. Negative favors adding spectral memory. Both parameter budgets are reported; fixed_ff permits extra parameters in striped.','',
 '| Budget | Train context | Paired nats differences by seed | Mean difference |','|---|---:|---|---:|']
 for budget,context in (('parameters',128),('fixed_ff',128),('parameters',512)):
  a=sorted(groups[('main',budget,context,'striped')],key=lambda x:x[0]['config']['seed'])
  b=sorted(groups[('main',budget,context,'localglobal')],key=lambda x:x[0]['config']['seed'])
  diffs=[mean(v['nats'] for v in x[0]['test'].values())-mean(v['nats'] for v in y[0]['test'].values()) for x,y in zip(a,b)]
  lines.append(f'| {budget} | {context} | '+', '.join(f'{d:+.4f}' for d in diffs)+f' | {mean(diffs):+.4f} |')
 lines+=['','## Prefix-plus-continuation extrapolation','',
 'The same eight 128-character continuation blocks per final book are evaluated after each prefix. The scored positions run from prefix-1 through prefix+126; the 128-prefix row therefore already exceeds the position range of context128 training. This is a continuation experiment, not an exact fixed-window baseline. This sample is limited; it is not a broad long-context benchmark.','',
 '| Training context | Model | Prefix | Mean test nats | Geometric PPL |','|---:|---|---:|---:|---:|']
 for context in (128,512):
  for kind in ('full','localglobal','striped'):
   rs=groups[('main','parameters',context,kind)]
   for length in ('128','512','2048','8192'):
    ns=[mean(book[length]['nats'] for book in e['long_test'].values()) for e,r in rs]
    lines.append(f'| {context} | {kind} | {length} | {mean(ns):.4f} | {math.exp(mean(ns)):.3f} |')
 lines+=['','## Development learning curves','',
 '| Budget | Context | Model | Update | Mean development nats |','|---|---:|---|---:|---:|']
 for key,rs in sorted(groups.items()):
  if key[0]!='main':continue
  for step in (300,600,1200,1800):
   vals=[mean(v['nats'] for v in h['dev'].values()) for e,r in rs for h in r['history'] if h['step']==step]
   lines.append(f'| {key[1]} | {key[2]} | {key[3]} | {step} | {mean(vals):.4f} |')
 timing=ROOT/'paired_timing.json'
 if timing.exists():
  rows=json.loads(timing.read_text())['rows']
  lines+=['','## Paired randomized CPU/CUDA timing','',
  'All raw repetitions are retained in paired_timing.json. Tables use the median over repeats per seed, then the mean across three seeds. Context128 main models only. Cache is tensor payload, not process peak memory.','',
  '| Prefix | Model | Prefill ms | Decode ms/character | Cache KiB |','|---:|---|---:|---:|---:|']
  for length in (128,512,2048,8192):
   for kind in ('full','localglobal','striped'):
    medians=[]
    for seed in range(3):
     rs=[r for r in rows if r['length']==length and r['kind']==kind and r['seed']==seed]
     medians.append({m:statistics.median(r[m] for r in rs) for m in ('prefill_ms','decode_ms','cache_bytes')})
    lines.append(f'| {length} | {kind} | {mean([r["prefill_ms"] for r in medians]):.3f} | {mean([r["decode_ms"] for r in medians]):.4f} | {mean([r["cache_bytes"] for r in medians])/1024:.2f} |')
 if (ROOT/'exact_window_audit.json').exists():
  audit=json.loads((ROOT/'exact_window_audit.json').read_text())
  lines+=['','## Corrective exact-window diagnostic','','Exploratory, seed0 only, 64 fixed target characters per final book. Each target is predicted from exactly the named number of preceding characters; position of the last input is window-1. Targets are identical across models/windows. The small target sample is descriptive, not a reliable broad quality estimate.','','| Training context | Model | Input window | Mean nats | Geometric PPL |','|---:|---|---:|---:|---:|']
  for row in audit['rows']:
   for length in ('128','512','2048','8192'):
    value=mean(book[length]['nats'] for book in row['scores'].values())
    lines.append(f'| {row["train_context"]} | {row["kind"]} | {length} | {value:.4f} | {math.exp(value):.3f} |')
 local_group=groups.get(('posthoc_local','parameters',128,'localonly'))
 if local_group:
  lines+=['','## Exploratory local-only efficiency control','','This baseline was added after primary test outcomes were inspected. Its learning rate was selected only on development data, but the architecture comparison is explicitly post hoc. It is a diagnostic efficiency baseline, not an untouched confirmatory test.','','| Model | Reset128 whole-book PPL | Prefix128 target-block PPL | Prefix8192 target-block PPL |','|---|---:|---:|---:|']
  for kind,rs in [('full',groups[('main','parameters',128,'full')]),('striped',groups[('main','parameters',128,'striped')]),('localonly',local_group)]:
   short=math.exp(mean([mean(v['nats'] for v in e['test'].values()) for e,r in rs]))
   p128=math.exp(mean([mean(v['128']['nats'] for v in e['long_test'].values()) for e,r in rs]))
   p8192=math.exp(mean([mean(v['8192']['nats'] for v in e['long_test'].values()) for e,r in rs]))
   lines.append(f'| {kind} | {short:.3f} | {p128:.3f} | {p8192:.3f} |')
  if (ROOT/'local_control_timing.json').exists():
   raw=json.loads((ROOT/'local_control_timing.json').read_text())['rows']
   lines+=['','Exploratory randomized paired timing, seed0 only:','','| Prefix | Model | Median decode ms | Cache KiB |','|---:|---|---:|---:|']
   for length in (128,8192):
    for kind in ('full','striped','localonly'):
     rs=[r for r in raw if r['length']==length and r['kind']==kind]
     lines.append(f'| {length} | {kind} | {statistics.median(r["decode_ms"] for r in rs):.4f} | {rs[0]["cache_bytes"]/1024:.2f} |')
 train_timing=ROOT/'paired_training_timing.json' 
 if train_timing.exists():
  rows=json.loads(train_timing.read_text())['rows']
  lines+=['','## Interleaved training-step timing','','Three warmup updates, then twenty raw repeats per model/seed. Data sampling is excluded; forward, loss, backward, clipping and AdamW are included. Checkpoint files are unchanged.','','| Context | Model | Mean of seed median step ms |','|---:|---|---:|']
  for length in (128,512):
   for kind in ('full','localglobal','striped'):
    ms=[statistics.median(r['step_ms'] for r in rows if r['context']==length and r['kind']==kind and r['seed']==seed) for seed in range(3)]
    lines.append(f'| {length} | {kind} | {mean(ms):.3f} |')
 if (ROOT/'trained_checks.json').exists():
  checks=json.loads((ROOT/'trained_checks.json').read_text())
  lines+=['','## Verification','',f"All {len(checks)} final checkpoints passed the trained-model causality and cached-generation assertions, including an 8192-character prefix. Worst long-prefix logit difference: {max(r['long_cached_max_error'] for r in checks):.3g}."]
 if (ROOT/'data_overlap_audit.json').exists():
  overlap=json.loads((ROOT/'data_overlap_audit.json').read_text())
  lines.append(f"Exact-block audit: {sum(r['nonoverlapping_128_character_blocks_checked'] for r in overlap)} sampled final-book blocks checked; {sum(len(r['exact_training_matches']) for r in overlap)} matched a training substring. This is not semantic deduplication.")
 lines+=['','## Interpretation limits','',
 'Striped retains one full global-attention block: its KV cache still grows linearly and its full-sequence attention work remains quadratic. The cache reduction is a constant-factor saving, not a bounded-memory architecture or an overall O(T log T) model. Local-only attention has a bounded receptive field, so its resource advantage does not establish distant retrieval ability. The legacy recall task used separately trained models. 1800 updates is a finite-budget comparison, not verified convergence. A seed0 learning-rate pilot is limited tuning. All selected rates were 0.002, the upper tested endpoint, so an optimum was not bracketed. Equal-time runs use noisy host wall time with a time-based learning-rate schedule, rather than FLOP matching. Final books supply two transfer domains. Development sources informed the architecture before this study. Long-context losses may expose severe extrapolation failures; always read them alongside speed. Loss values are not directly comparable with legacy experiments because the alphabet, splits and recipe differ. These ablations test the whole modal branch, not the Fourier basis against every alternative mixer. A half-cache result may come from replacing global attention alone; use localglobal to assess spectral attribution.','']
 (ROOT/'REPORT.md').write_text('\n'.join(lines))
 print(ROOT/'REPORT.md')
if __name__=='__main__':main()
