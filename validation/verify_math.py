"""Check the redesigned matrix-state abstraction and block compositions."""
import sys,json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;S=P.parent/'experiments/rigorous'
sys.path.insert(0,str(S))
from model import Spectral,LM,Block

torch.manual_seed(11);torch.set_num_threads(2)
checks={}
def close(name,a,b):
 e=float((a-b).abs().max());checks[name]=e;assert e<1e-5,(name,e)
with torch.no_grad():
 m=Spectral(64,8).eval();A=torch.randn(2,23,64)
 value,write,read=m.proj(A).chunk(3,-1)
 U=value*write.sigmoid();G=read.sigmoid()
 poles=torch.exp(-m.log_decay.exp().to(torch.complex64)+1j*m.omega)
 transition=torch.diag(poles);C=torch.view_as_complex(m.residue.contiguous())
 state=torch.zeros(2,8,64,dtype=torch.complex64);ys=[]
 for t in range(23):
  broadcast=torch.ones(8,1)@U[:,t,None,:]
  assert broadcast.shape==state.shape
  state=transition@state+broadcast
  z=(C[None]*state).sum(1).real
  ys.append(m.out(z*G[:,t]))
 close('matrix_feedback_cell_vs_fft',torch.stack(ys,1),m(A))
 close('state_buffer_vs_prefill',state,m.cache(A))
 assert state[0].numel()*state.element_size()==4096
 for kind in ['full','localglobal','striped']:
  lm=LM('full' if kind=='full' else 'striped',128).eval()
  if kind=='localglobal':lm.blocks[0]=Block('local',64,8,8)
  b=lm.blocks[0];H=torch.randn(2,13,64);x=H
  for mix in b.mixers:x=x+mix(b.norm(x))
  x=x+b.ff(b.norm2(x));close('B1_composition_'+kind,x,b(H))
  assert len(lm.blocks[1].mixers)==1 and lm.blocks[1].mixers[0].window is None
source_hash=hashlib.sha256((S/'model.py').read_bytes()).hexdigest()
assert source_hash=='afa4b39a20cac0503caa680ae5edcd56577813ba30794939ec0f5cc51e3b406d'
result={'checks':checks,'source_model_sha256':source_hash,'scope':'Deterministic operator checks only; no training or new scientific benchmark.'}
(P/'numerical_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
