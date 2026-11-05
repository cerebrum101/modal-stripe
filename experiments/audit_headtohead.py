"""Independent deterministic holdout scoring and trained-checkpoint causality audit."""
import json, math
import torch
from corpus_headtohead import get_text, SOURCES, encode, OUT
from run import LM

torch.set_num_threads(2)
texts={n:get_text(n)[0] for n in SOURCES}
lookup={c:i for i,c in enumerate(sorted(set(''.join(texts.values()))))}
rows=[]
with torch.no_grad():
 for corpus in ('hamlet','mixed3'):
  for seed in range(3):
   for kind in ('full','striped'):
    path=OUT/f'headtohead_{corpus}_{kind}_s{seed}.json'
    model=LM(kind,len(lookup)).eval()
    model.load_state_dict(torch.load(path.with_suffix('.pt'),weights_only=True))
    scores={}
    for name in (('hamlet',) if corpus=='hamlet' else SOURCES):
     text=texts[name]; data=encode(text[int(.9*len(text)):],lookup)
     # Nonoverlapping targets; reset context every 128 characters.
     total_loss=0.; count=0
     for start in range(0,len(data)-1,128):
      seq=data[start:min(start+129,len(data))]
      logits=model(seq[:-1][None])
      total_loss+=torch.nn.functional.cross_entropy(logits[0],seq[1:],reduction='sum').item()
      count+=len(seq)-1
     scores[name]={'ppl':math.exp(total_loss/count),'targets':count}
    tokens=encode(texts['hamlet'][:257],lookup)[None]
    original=model(tokens)
    changed=tokens.clone(); changed[:,128:]=changed[:,128:].flip(1)
    causal=(original[:,:128]-model(changed)[:,:128]).abs().max().item()
    logits,cache=model.prefill(tokens[:,:128]); streamed=[]
    for token in tokens[:,128:].unbind(1):
     out,cache=model.step(token,cache); streamed.append(out)
    cached=(original[:,128:]-torch.stack(streamed,1)).abs().max().item()
    assert causal<2e-5 and cached<2e-4,(corpus,kind,seed,causal,cached)
    row={'corpus':corpus,'kind':kind,'seed':seed,'deterministic_holdout':scores,'causality_max_error':causal,'cached_max_error':cached}
    rows.append(row); print(json.dumps(row),flush=True)
(OUT/'headtohead_audit.json').write_text(json.dumps(rows,indent=2))
