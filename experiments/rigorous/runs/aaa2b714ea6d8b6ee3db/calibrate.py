"""Pre-specified baseline using paired training-step timing, not test quality."""
import json,math,statistics
from pathlib import Path
import torch
from study import ROOT,DEVICE,make,train,sources,SOURCES,FINAL,evaluate,long_eval,atomic_json,digest

def main():
 raw_path=ROOT/'paired_training_timing.json';rows=json.loads(raw_path.read_text())['rows']
 pairs={(r['kind'],r['seed'],r['repeat']):r['step_ms'] for r in rows if r['context']==128}
 ratios=[pairs[('striped',seed,rep)]/pairs[('full',seed,rep)] for seed in range(3) for rep in range(20)]
 steps=math.ceil(1800*statistics.median(ratios))
 selection=json.loads((ROOT/'selection.json').read_text())['selected']
 frozen=json.loads((ROOT/'frozen_selection.json').read_text())
 docs,_=sources(SOURCES);parts=[d[:int(.8*len(d))] for d in docs.values()]
 dev={n:d[int(.8*len(d)):int(.9*len(d))] for n,d in docs.items()}
 decision={'steps':steps,'median_time_ratio':statistics.median(ratios),'training_timing_sha256':digest(raw_path.read_bytes()),'controller_sha256':digest(Path(__file__).read_bytes()),'plan_sha256':digest((ROOT/'CALIBRATION_PLAN.md').read_bytes()),'rule':'ceil(1800 * median paired striped/full step-ms ratio), context128'}
 atomic_json(ROOT/'calibration_decision.json',decision)
 configs=[]
 for seed in range(3):
  cfg={'phase':'calibrated_time','kind':'full','budget':'parameters','seed':seed,'lr':selection['parameters_128_full'],'steps':steps,'context':128,'batch':16,'calibration':decision}
  r,f=train(cfg,parts,dev,frozen['provenance']);(f/'calibrate.py').write_bytes(Path(__file__).read_bytes());configs.append((r,f))
 atomic_json(ROOT/'calibrated_selection.json',{'decision':decision,'runs':[str(f.relative_to(ROOT)) for r,f in configs]})
 tests,meta=sources(FINAL);extra=[]
 audits=json.loads((ROOT/'trained_checks.json').read_text())
 for r,f in configs:
  c=r['config'];m=make('full','parameters',c['seed']);m.load_state_dict(torch.load(f/'weights.pt',map_location=DEVICE,weights_only=True))
  e={'run':str(f.relative_to(ROOT)),'config':c,'test':evaluate(m,tests),'long_test':long_eval(m,tests),'test_metadata':meta}
  atomic_json(f/'final_test.json',e);extra.append(e)
  with torch.no_grad():
   m.eval();x=tests['frankenstein'][:8193][None].to(DEVICE)
   expected=m(x)[:,-1];_,cache=m.prefill(x[:,:-1]);actual,_=m.step(x[:,-1],cache)
   error=float((expected-actual).abs().max());assert error<1e-3
   short=x[:,:193];base=m(short);changed=short.clone();changed[:,97:]=changed[:,97:].flip(1)
   future=float((base[:,:97]-m(changed)[:,:97]).abs().max());assert future<1e-4
   _,cache=m.prefill(short[:,:32]);ys=[]
   for t in short[:,32:].unbind(1):a,cache=m.step(t,cache);ys.append(a)
   cached=float((base[:,32:]-torch.stack(ys,1)).abs().max());assert cached<1e-4
  audits=[a for a in audits if a['run']!=e['run']]
  audits.append({'run':e['run'],'future_max_error':future,'cached_max_error':cached,'long_cached_max_error':error})
  print('calibrated evaluation',f.name,flush=True)
 path=ROOT/'final_evaluations.json';original=json.loads(path.read_text())
 if not (ROOT/'primary_final_evaluations.json').exists():atomic_json(ROOT/'primary_final_evaluations.json',original)
 original=[e for e in original if e['config']['phase']!='calibrated_time']
 atomic_json(path,original+extra);atomic_json(ROOT/'trained_checks.json',audits)
 print('CALIBRATED COMPARISON COMPLETE',flush=True)
if __name__=='__main__':main()
