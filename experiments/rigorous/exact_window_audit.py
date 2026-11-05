"""Corrective, exploratory fixed-window diagnostic, seed0 and 64 targets/book."""
import json,math
import torch
from study import ROOT,DEVICE,make,sources,FINAL,atomic_json,digest
from model import LM

@torch.no_grad()
def score(m,docs):
 m.eval();out={}
 for name,data in docs.items():
  positions=torch.linspace(8192,len(data)-1,64).long()
  target=data[positions].to(DEVICE)
  scores={}
  for length in (128,512,2048,8192):
   loss=0.
   for start in range(0,len(positions),2):
    picks=positions[start:start+2].tolist()
    x=torch.stack([data[p-length:p] for p in picks]).to(DEVICE)
    assert x.shape[1]==length
    logits=m(x)[:,-1]
    loss+=float(torch.nn.functional.cross_entropy(logits,target[start:start+2],reduction='sum'))
   scores[str(length)]={'nats':loss/len(positions),'ppl':math.exp(loss/len(positions)),'targets':len(positions)}
  out[name]=scores
 return out

def main():
 evaluations=json.loads((ROOT/'final_evaluations.json').read_text());docs,meta=sources(FINAL)
 result=[]
 for e in evaluations:
  c=e['config']
  if c['seed']!=0:continue
  primary=c['phase']=='main' and c['budget']=='parameters'
  local=c['phase']=='posthoc_local'
  if not(primary or local):continue
  m=LM('local',128).to(DEVICE) if local else make(c['kind'],c['budget'],0)
  p=ROOT/e['run'];m.load_state_dict(torch.load(p/'weights.pt',map_location=DEVICE,weights_only=True))
  result.append({'run':e['run'],'kind':c['kind'],'train_context':c['context'],'seed':0,'scores':score(m,docs)})
  print('exact-window audit',c['kind'],c['context'],flush=True)
 atomic_json(ROOT/'exact_window_audit.json',{'rows':result,'targets_per_book':64,'seeds':[0],'test_metadata':meta,'code_sha256':digest(__import__('pathlib').Path(__file__).read_bytes()),'scope':'corrective exploratory diagnostic; fixed single-target windows, not the prefix-plus-continuation test'})
if __name__=='__main__':main()
