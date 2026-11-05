"""Independent alignment, segmentation, provenance and source-parser checks."""
import json
import torch
from torch import nn
from study import DEVICE,ROOT,sample,evaluate,long_eval,clean,digest
from exact_window_audit import score as exact_score

class NextInteger(nn.Module):
 def forward(self,x):
  logits=torch.full((*x.shape,128),-30.,device=x.device)
  return logits.scatter_(-1,((x+1)%128).unsqueeze(-1),30.)

# An exactly predictable sequence must score essentially zero at every
# context; catches shifted targets and incorrect trailing-block indexing.
data=torch.arange(10003)%128
m=NextInteger().to(DEVICE)
r=evaluate(m,{'probe':data})['probe']
assert r['targets']==len(data)-1 and r['nats']<1e-6
for row in long_eval(m,{'probe':data})['probe'].values():
 assert row['targets']==1024 and row['nats']<1e-6
for row in exact_score(m,{'probe':data})['probe'].values():
 assert row['targets']==64 and row['nats']<1e-6

# Different constant documents must never be joined into a training window.
x,y=sample([torch.full((256,),7),torch.full((300,),19)],torch.Generator().manual_seed(8),batch=512)
assert (x==x[:,:1]).all() and (y==x).all() and set(x[:,0].tolist())=={7,19}
try:clean(b'not a valid Gutenberg document')
except AssertionError:pass
else:raise AssertionError('invalid source accepted')

# Verify copied code and all completed checkpoint identities.
count=0
for path in (ROOT/'runs').glob('*/result.json'):
 result=json.loads(path.read_text());p=path.parent
 for name,expected in result['config']['provenance']['code'].items():assert digest((p/name).read_bytes())==expected
 assert digest((p/'weights.pt').read_bytes())==result['checkpoint_sha256']
 count+=1
print('independent alignment/parser/provenance checks passed;',count,'completed checkpoints verified')
