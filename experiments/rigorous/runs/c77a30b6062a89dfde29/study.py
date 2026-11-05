"""Frozen, bounded CPU/CUDA study. See PROTOCOL.md before interpreting results."""
import argparse, hashlib, json, math, os, platform, random, statistics, time, urllib.request
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model import LM, Block, Attention, Spectral

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs'; OUT.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)
SOURCES={'hamlet':1524,'alice':11,'moby':2701}
FINAL={'frankenstein':84,'sherlock':1661}
KINDS=('full','localglobal','striped')
LRS=(.0005,.001,.002)

def digest(x): return hashlib.sha256(x).hexdigest()
def atomic_json(path,obj):
 tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)
def sync():
 if DEVICE=='cuda': torch.cuda.synchronize()
def clean(raw):
 lines=raw.decode('utf-8-sig').splitlines()
 starts=[i for i,s in enumerate(lines) if '*** START OF' in s.upper() and 'GUTENBERG' in s.upper()]
 ends=[i for i,s in enumerate(lines) if '*** END OF' in s.upper() and 'GUTENBERG' in s.upper()]
 assert len(starts)==1 and len(ends)==1 and starts[0]<ends[0], 'invalid Gutenberg wrapper'
 s='\n'.join(lines[starts[0]+1:ends[0]]).strip()
 for a,b in {'\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u2014':'--','\u2013':'-','\u2026':'...','\u00a0':' '}.items(): s=s.replace(a,b)
 return torch.tensor(list(s.encode('ascii',errors='ignore')),dtype=torch.long)
def sources(ids):
 result={};meta={}
 for name,number in ids.items():
  path=DATA/f'pg{number}.txt'
  if not path.exists():
   old=ROOT.parent/'data'/f'pg{number}.txt'
   raw=old.read_bytes() if old.exists() else urllib.request.urlopen(f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt',timeout=60).read()
   path.write_bytes(raw)
  raw=path.read_bytes(); result[name]=clean(raw)
  assert len(result[name])>20000
  meta[name]={'raw_sha256':digest(raw),'characters':len(result[name]),'gutenberg':number}
 return result,meta

def sample(parts,gen,batch=16,length=128):
 counts=[len(x)-length for x in parts]
 assert min(counts)>0
 bounds=torch.tensor(counts).cumsum(0)
 picks=torch.randint(int(bounds[-1]),(batch,),generator=gen)
 rows=[]
 for p in picks.tolist():
  i=int(torch.searchsorted(bounds,torch.tensor(p),right=True))
  offset=p-(int(bounds[i-1]) if i else 0)
  rows.append(parts[i][offset:offset+length+1])
 seq=torch.stack(rows).to(DEVICE)
 return seq[:,:-1],seq[:,1:]
def make(kind,budget,seed):
 torch.manual_seed(seed)
 m=LM('full' if kind=='full' else 'striped',128)
 if kind=='localglobal': m.blocks[0]=Block('local',64,8,8)
 if budget=='fixed_ff':
  for b in m.blocks: b.ff=nn.Sequential(nn.Linear(64,256,bias=False),nn.GELU(),nn.Linear(256,64,bias=False))
 return m.to(DEVICE)
@torch.no_grad()
def evaluate(m,parts):
 m.eval();result={}
 for name,data in parts.items():
  total=0.;count=0
  for pos in range(0,len(data)-1,128):
   seq=data[pos:pos+129].to(DEVICE)
   loss=F.cross_entropy(m(seq[:-1][None])[0],seq[1:],reduction='sum')
   total+=float(loss);count+=len(seq)-1
  result[name]={'nats':total/count,'ppl':math.exp(total/count),'targets':count}
 m.train();return result
@torch.no_grad()
def long_eval(m,parts):
 m.eval();out={}
 for name,data in parts.items():
  # Identical target blocks and positions at every context length.
  positions=torch.linspace(8192,len(data)-129,8).long().tolist()
  scores={}
  for length in (128,512,2048,8192):
   loss=0.;n=0
   for p in positions:
    seq=data[p-length:p+128].to(DEVICE)
    logits=m(seq[:-1][None])[0,-128:]
    target=data[p:p+128].to(DEVICE)
    loss+=float(F.cross_entropy(logits,target,reduction='sum'));n+=len(target)
   scores[str(length)]={'nats':loss/n,'ppl':math.exp(loss/n),'targets':n}
  out[name]=scores
 m.train();return out

def provenance(meta):
 return {'code':{p.name:digest(p.read_bytes()) for p in (ROOT/'study.py',ROOT/'model.py')},'data':meta,'torch':torch.__version__,'python':platform.python_version(),'device':DEVICE,'threads':2,'platform':platform.platform()}
def train(cfg,parts,dev,prov):
 config=cfg|{'provenance':prov}
 key=digest(json.dumps(config,sort_keys=True).encode())[:20]; folder=OUT/key;folder.mkdir(exist_ok=True)
 result_path=folder/'result.json'; checkpoint=folder/'weights.pt'
 for source in ('study.py','model.py'):
  (folder/source).write_bytes((ROOT/source).read_bytes())
 if result_path.exists():
  r=json.loads(result_path.read_text());assert digest(checkpoint.read_bytes())==r['checkpoint_sha256'];return r,folder
 m=make(cfg['kind'],cfg['budget'],cfg['seed']);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=.01)
 gen=torch.Generator().manual_seed(1000+cfg['seed']);elapsed=0.;step=0;history=[]
 while True:
  sync();start=time.perf_counter()
  fraction=(elapsed/cfg['seconds']) if cfg.get('seconds') else step/cfg['steps']
  opt.param_groups[0]['lr']=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*min(fraction,1))))
  x,y=sample(parts,gen,length=cfg.get('context',128),batch=cfg.get('batch',16))
  opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x).flatten(0,1),y.flatten())
  assert torch.isfinite(loss)
  loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();sync()
  elapsed+=time.perf_counter()-start;step+=1
  done=(elapsed>=cfg['seconds']) if cfg.get('seconds') else step>=cfg['steps']
  if step in (300,600,1200,1800) or done:
   history.append({'step':step,'train_seconds':elapsed,'dev':evaluate(m,dev)})
  if step%300==0 or done: print(json.dumps({'run':key,**cfg,'step':step,'seconds_elapsed':round(elapsed,1)}),flush=True)
  if done: break
 torch.save(m.state_dict(),checkpoint)
 r={'config':config,'steps':step,'tokens':step*cfg.get('context',128)*cfg.get('batch',16),'train_seconds':elapsed,'parameters':sum(p.numel() for p in m.parameters()),'history':history,'dev':history[-1]['dev'],'checkpoint_sha256':digest(checkpoint.read_bytes())}
 atomic_json(result_path,r);return r,folder

def checks():
 # The corrected sampler must reach the last possible window.
 d=torch.arange(132);gen=torch.Generator().manual_seed(3)
 x,y=sample([d],gen,batch=256,length=128)
 assert x[:,0].max()==3 and torch.equal(x[:,1:],y[:,:-1])
 spectral=Spectral(4,3).double()
 assert torch.autograd.gradcheck(spectral,(torch.randn(1,7,4,dtype=torch.double,requires_grad=True),),eps=1e-6,atol=1e-4)
 errors=[]
 for kind in KINDS:
  for budget in ('parameters','fixed_ff'):
   m=make(kind,budget,42).eval();tokens=torch.randint(128,(1,97),device=DEVICE)
   with torch.no_grad():
    base=m(tokens);changed=tokens.clone();changed[:,48:]=changed[:,48:].flip(1)
    causal=float((base[:,:48]-m(changed)[:,:48]).abs().max())
    _,cache=m.prefill(tokens[:,:16]);out=[]
    for t in tokens[:,16:].unbind(1):
     a,cache=m.step(t,cache);out.append(a)
    cached=float((base[:,16:]-torch.stack(out,1)).abs().max())
   assert max(causal,cached)<3e-5,(kind,budget,causal,cached)
   # Check gradients cannot flow from a past output to future input embeddings.
   emb=m.embed(tokens).detach().requires_grad_();z=emb
   for b in m.blocks:z=b(z)
   grad=torch.autograd.grad(z[:,30].sum(),emb)[0]
   assert grad[:,31:].abs().max()<1e-6
   errors.append({'kind':kind,'budget':budget,'causal':causal,'cached':cached})
 # Local attention agrees with an independently masked dense reference.
 a=Attention(16,8).to(DEVICE);x=torch.randn(2,17,16,device=DEVICE)
 q,k,v=a.qkv(x).reshape(2,17,3,4,4).permute(2,0,3,1,4)
 p=torch.arange(17,device=DEVICE);lag=p[:,None]-p[None,:]
 ref=a.out(F.scaled_dot_product_attention(q,k,v,attn_mask=(lag>=0)&(lag<8)).transpose(1,2).reshape(2,17,16))
 assert (ref-a(x)).abs().max()<1e-6
 atomic_json(ROOT/'checks.json',errors);print('framework checks passed',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--checks-only',action='store_true');args=parser.parse_args()
 checks()
 if args.checks_only:return
 docs,meta=sources(SOURCES)
 train_parts=[d[:int(.8*len(d))] for d in docs.values()]
 dev={n:d[int(.8*len(d)):int(.9*len(d))] for n,d in docs.items()}
 prov=provenance(meta)
 pilots=[];selected={}
 for budget,context,batch in (('parameters',128,16),('fixed_ff',128,16),('parameters',512,4)):
  for kind in KINDS:
   candidates=[]
   for lr in LRS:
    r,f=train({'phase':'pilot','kind':kind,'budget':budget,'seed':0,'lr':lr,'steps':400,'context':context,'batch':batch},train_parts,dev,prov)
    row={'kind':kind,'budget':budget,'context':context,'lr':lr,'score':statistics.mean(v['nats'] for v in r['dev'].values())}
    pilots.append(row);candidates.append(row)
   selected[f'{budget}_{context}_{kind}']=min(candidates,key=lambda r:r['score'])['lr']
 atomic_json(ROOT/'selection.json',{'pilots':pilots,'selected':selected,'rule':'lowest equally weighted development nats; seed-0 pilot, 400 steps'})
 finals=[]
 for budget,context,batch in (('parameters',128,16),('fixed_ff',128,16),('parameters',512,4)):
  for seed in range(3):
   for kind in KINDS:
    r,f=train({'phase':'main','kind':kind,'budget':budget,'seed':seed,'lr':selected[f'{budget}_{context}_{kind}'],'steps':1800,'context':context,'batch':batch},train_parts,dev,prov)
    finals.append((r,f))
 # Match accumulated training-step wall time; evaluation time is excluded.
 for seed in range(3):
  striped=next(r for r,f in finals if r['config']['kind']=='striped' and r['config']['budget']=='parameters' and r['config']['context']==128 and r['config']['seed']==seed)
  r,f=train({'phase':'equal_time','kind':'full','budget':'parameters','seed':seed,'lr':selected['parameters_128_full'],'seconds':striped['train_seconds'],'context':128,'batch':16},train_parts,dev,prov)
  finals.append((r,f))
 atomic_json(ROOT/'frozen_selection.json',{'selected':selected,'runs':[str(f.relative_to(ROOT)) for r,f in finals],'provenance':prov})
 # Final books are fetched and inspected only after every training configuration is frozen.
 test,testmeta=sources(FINAL)
 evaluations=[]
 for r,f in finals:
  cfg=r['config'];m=make(cfg['kind'],cfg['budget'],cfg['seed'])
  m.load_state_dict(torch.load(f/'weights.pt',map_location=DEVICE,weights_only=True))
  e={'run':str(f.relative_to(ROOT)),'config':cfg,'test':evaluate(m,test),'long_test':long_eval(m,test),'test_metadata':testmeta}
  atomic_json(f/'final_test.json',e);evaluations.append(e)
  print('final evaluation',f.name,flush=True)
 atomic_json(ROOT/'final_evaluations.json',evaluations)
 print('STUDY COMPLETE',flush=True)
if __name__=='__main__':main()
