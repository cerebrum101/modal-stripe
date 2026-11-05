"""Explicitly exploratory strong efficiency baseline; see LOCAL_CONTROL_PLAN.md."""
import json,statistics,random,time
from pathlib import Path
import torch
import study
from study import ROOT,DEVICE,LM,make,sources,SOURCES,FINAL,train,evaluate,long_eval,atomic_json,digest,sync
from measure import nbytes

original_make=study.make
def factory(kind,budget,seed):
 if kind!='localonly':return original_make(kind,budget,seed)
 torch.manual_seed(seed);return LM('local',128).to(DEVICE)

def main():
 study.make=factory
 frozen=json.loads((ROOT/'frozen_selection.json').read_text())
 docs,_=sources(SOURCES);parts=[d[:int(.8*len(d))] for d in docs.values()]
 dev={n:d[int(.8*len(d)):int(.9*len(d))] for n,d in docs.items()}
 extra={'controller_sha256':digest(Path(__file__).read_bytes()),'plan_sha256':digest((ROOT/'LOCAL_CONTROL_PLAN.md').read_bytes()),'posthoc':True}
 pilots=[]
 for lr in study.LRS:
  r,f=train({'phase':'posthoc_local_pilot','kind':'localonly','budget':'parameters','seed':0,'lr':lr,'steps':400,'context':128,'batch':16,**extra},parts,dev,frozen['provenance'])
  (f/'local_control.py').write_bytes(Path(__file__).read_bytes())
  pilots.append({'lr':lr,'nats':statistics.mean(v['nats'] for v in r['dev'].values())})
 lr=min(pilots,key=lambda p:p['nats'])['lr'];runs=[]
 atomic_json(ROOT/'local_control_selection.json',{'pilots':pilots,'selected_lr':lr,'posthoc':True})
 for seed in range(3):
  r,f=train({'phase':'posthoc_local','kind':'localonly','budget':'parameters','seed':seed,'lr':lr,'steps':1800,'context':128,'batch':16,**extra},parts,dev,frozen['provenance'])
  (f/'local_control.py').write_bytes(Path(__file__).read_bytes());runs.append((r,f))
 tests,meta=sources(FINAL);evaluations=[];audits=json.loads((ROOT/'trained_checks.json').read_text())
 for r,f in runs:
  c=r['config'];m=factory('localonly','parameters',c['seed']).eval();m.load_state_dict(torch.load(f/'weights.pt',map_location=DEVICE,weights_only=True))
  e={'run':str(f.relative_to(ROOT)),'config':c,'test':evaluate(m,tests),'long_test':long_eval(m,tests),'test_metadata':meta}
  atomic_json(f/'final_test.json',e);evaluations.append(e)
  with torch.no_grad():
   m.eval();x=tests['frankenstein'][:8193][None].to(DEVICE)
   expected=m(x)[:,-1];_,cache=m.prefill(x[:,:-1]);actual,_=m.step(x[:,-1],cache);long_error=float((expected-actual).abs().max())
   short=x[:,:193];base=m(short);changed=short.clone();changed[:,97:]=changed[:,97:].flip(1);future=float((base[:,:97]-m(changed)[:,:97]).abs().max())
   _,cache=m.prefill(short[:,:32]);ys=[]
   for t in short[:,32:].unbind(1):y,cache=m.step(t,cache);ys.append(y)
   cached=float((base[:,32:]-torch.stack(ys,1)).abs().max())
   assert max(future,cached)<1e-4 and long_error<1e-3
  audits=[a for a in audits if a['run']!=e['run']];audits.append({'run':e['run'],'future_max_error':future,'cached_max_error':cached,'long_cached_max_error':long_error})
  print('local evaluation',f.name,flush=True)
 existing=json.loads((ROOT/'final_evaluations.json').read_text());existing=[e for e in existing if e['config']['phase']!='posthoc_local']
 atomic_json(ROOT/'final_evaluations.json',existing+evaluations);atomic_json(ROOT/'trained_checks.json',audits)
 # Paired diagnostic timing, seed0 only; keep distinct from primary timing.
 models={}
 paths={'localonly':runs[0][1]}
 for folder in frozen['runs']:
  r=json.loads((ROOT/folder/'result.json').read_text());c=r['config']
  if c['phase']=='main' and c['budget']=='parameters' and c['context']==128 and c['seed']==0 and c['kind'] in ('full','striped'):paths[c['kind']]=ROOT/folder
 for kind,p in paths.items():
  m=factory(kind,'parameters',0).eval();m.load_state_dict(torch.load(p/'weights.pt',map_location=DEVICE,weights_only=True));models[kind]=m
 prefix=next(iter(dev.values()));rng=random.Random(123);raw=[]
 with torch.inference_mode():
  for length in (128,512,2048,8192):
   x=prefix[:length][None].to(DEVICE)
   for m in models.values():
    logits,cache=m.prefill(x);token=logits[:,-1].argmax(-1)
    for _ in range(32):logits,cache=m.step(token,cache);token=logits.argmax(-1)
   for repeat in range(9):
    keys=list(models);rng.shuffle(keys)
    for kind in keys:
     m=models[kind];sync();start=time.perf_counter();logits,cache=m.prefill(x);sync();pre=1000*(time.perf_counter()-start)
     size=nbytes(cache);token=logits[:,-1].argmax(-1);sync();start=time.perf_counter()
     for _ in range(32):logits,cache=m.step(token,cache);token=logits.argmax(-1)
     sync();dec=1000*(time.perf_counter()-start)/32
     raw.append({'kind':kind,'length':length,'repeat':repeat,'prefill_ms':pre,'decode_ms':dec,'cache_bytes':size})
 atomic_json(ROOT/'local_control_timing.json',{'rows':raw,'seed':0,'posthoc':True,'controller_sha256':extra['controller_sha256']})
 print('LOCAL CONTROL COMPLETE',flush=True)
if __name__=='__main__':main()
