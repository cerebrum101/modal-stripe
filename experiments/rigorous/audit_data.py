"""Post-selection exact-block overlap check on final books."""
import json
from study import ROOT,SOURCES,FINAL,sources,atomic_json
assert (ROOT/'final_evaluations.json').exists(),'run only after frozen final evaluation'
train,_=sources(SOURCES);tests,_=sources(FINAL)
train_bytes={n:bytes(d[:int(.8*len(d))].tolist()) for n,d in train.items()}
rows=[]
for name,data in tests.items():
 text=bytes(data.tolist());matches=[];blocks=0
 for pos in range(0,len(text)-127,128):
  block=text[pos:pos+128];blocks+=1
  for source,body in train_bytes.items():
   offset=body.find(block)
   if offset>=0:matches.append({'test_offset':pos,'training_source':source,'training_offset':offset});break
 rows.append({'book':name,'nonoverlapping_128_character_blocks_checked':blocks,'exact_training_matches':matches})
atomic_json(ROOT/'data_overlap_audit.json',rows)
print(json.dumps(rows,indent=2))
