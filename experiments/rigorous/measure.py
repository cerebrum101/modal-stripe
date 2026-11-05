"""Paired randomized raw timing and trained checkpoint checks; run after study.py."""
import json,random,time,platform
from pathlib import Path
import torch
from torch.nn import functional as F
from study import ROOT,OUT,DEVICE,make,sources,SOURCES,atomic_json,sync,digest,sample

def nbytes(x):
 if torch.is_tensor(x):return x.numel()*x.element_size()
 if isinstance(x,dict):return sum(nbytes(v) for v in x.values())
 if isinstance(x,(tuple,list)):return sum(nbytes(v) for v in x)
 return 0

def main():
 frozen=json.loads((ROOT/'frozen_selection.json').read_text())
 docs,meta=sources(SOURCES)
 prefix=docs['hamlet'][int(.8*len(docs['hamlet'])):int(.9*len(docs['hamlet']))]
 audit=[];models={};model_ids={}
 with torch.inference_mode():
  for folder in frozen['runs']:
   p=ROOT/folder;r=json.loads((p/'result.json').read_text());c=r['config']
   assert digest((p/'weights.pt').read_bytes())==r['checkpoint_sha256']
   m=make(c['kind'],c['budget'],c['seed']).eval()
   m.load_state_dict(torch.load(p/'weights.pt',map_location=DEVICE,weights_only=True))
   x=prefix[:193][None].to(DEVICE);base=m(x);changed=x.clone();changed[:,97:]=changed[:,97:].flip(1)
   future=float((base[:,:97]-m(changed)[:,:97]).abs().max())
   _,cache=m.prefill(x[:,:32]);ys=[]
   for t in x[:,32:].unbind(1):
    y,cache=m.step(t,cache);ys.append(y)
   cached=float((base[:,32:]-torch.stack(ys,1)).abs().max())
   assert max(future,cached)<1e-4,(folder,future,cached)
   long_x=prefix[:8193][None].to(DEVICE)
   expected=m(long_x)[:,-1]
   _,long_cache=m.prefill(long_x[:,:-1])
   actual,_=m.step(long_x[:,-1],long_cache)
   long_error=float((expected-actual).abs().max())
   assert long_error<1e-3,(folder,'8192-prefix cache mismatch',long_error)
   audit.append({'run':folder,'future_max_error':future,'cached_max_error':cached,'long_cached_max_error':long_error})
   if c['phase']=='main' and c['budget']=='parameters' and c['context']==128:
    models[(c['kind'],c['seed'])]=m
    model_ids[f"{c['kind']}_{c['seed']}"]={'run':folder,'checkpoint_sha256':r['checkpoint_sha256']}
 atomic_json(ROOT/'trained_checks.json',audit)
 raw=[];rng=random.Random(10001)
 with torch.inference_mode():
  for length in (128,512,2048,8192):
   x=prefix[:length][None].to(DEVICE)
   for m in models.values():
    logits,cache=m.prefill(x);token=logits[:,-1].argmax(-1)
    for _ in range(32):logits,cache=m.step(token,cache);token=logits.argmax(-1)
   for repeat in range(9):
    keys=list(models);rng.shuffle(keys)
    for kind,seed in keys:
     m=models[(kind,seed)]
     sync();t=time.perf_counter();logits,cache=m.prefill(x);sync();prefill=(time.perf_counter()-t)*1000
     payload=nbytes(cache);token=logits[:,-1].argmax(-1)
     sync();t=time.perf_counter()
     for _ in range(32):logits,cache=m.step(token,cache);token=logits.argmax(-1)
     sync();decode=(time.perf_counter()-t)*1000/32
     raw.append({'kind':kind,'seed':seed,'length':length,'repeat':repeat,'prefill_ms':prefill,'decode_ms':decode,'cache_bytes':payload})
    print('timing',length,repeat,flush=True)
 atomic_json(ROOT/'paired_timing.json',{'rows':raw,'models':model_ids,'torch':torch.__version__,'platform':platform.platform(),'threads':torch.get_num_threads(),'measurement_code_sha256':digest(Path(__file__).read_bytes()),'prefix_sha256':digest(prefix.numpy().tobytes()),'device':DEVICE,'warmup':32,'continuation':32,'ordering':'seeded random architecture/seed order within each repeat'})
 # Fresh optimizer state, warmed independently; changes only in-memory copies.
 models={key:make(key[0],'parameters',key[1]) for key in models}
 for key,m in models.items():
  folder=ROOT/model_ids[f'{key[0]}_{key[1]}']['run']
  m.load_state_dict(torch.load(folder/'weights.pt',map_location=DEVICE,weights_only=True))
 training_rows=[];gen=torch.Generator().manual_seed(444)
 parts=[d[:int(.8*len(d))] for d in docs.values()]
 def update(m,opt,x,y):
  opt.zero_grad(set_to_none=True)
  loss=F.cross_entropy(m(x).flatten(0,1),y.flatten())
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 for length,batch in ((128,16),(512,4)):
  optimizers={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01) for k,m in models.items()}
  x,y=sample(parts,gen,batch=batch,length=length)
  for key,m in models.items():
   m.train()
   for _ in range(3):update(m,optimizers[key],x,y)
  for repeat in range(20):
   keys=list(models);rng.shuffle(keys)
   x,y=sample(parts,gen,batch=batch,length=length)
   for key in keys:
    sync();start=time.perf_counter();update(models[key],optimizers[key],x,y);sync()
    training_rows.append({'kind':key[0],'seed':key[1],'context':length,'batch':batch,'repeat':repeat,'step_ms':1000*(time.perf_counter()-start)})
 atomic_json(ROOT/'paired_training_timing.json',{'rows':training_rows,'device':DEVICE,'warmup_steps':3,'measurement_code_sha256':digest(Path(__file__).read_bytes()),'note':'model-only forward/loss/backward/clip/AdamW; data sampling excluded; checkpoints unchanged'})
 print('MEASUREMENTS COMPLETE',flush=True)
if __name__=='__main__':main()
