"""Cross-check saved evidence before reporting or packaging it."""
import json,math,statistics
from study import ROOT,digest

def main():
 evaluations=json.loads((ROOT/'final_evaluations.json').read_text())
 ids=[e['run'] for e in evaluations];assert len(ids)==36 and len(set(ids))==36
 checks=json.loads((ROOT/'trained_checks.json').read_text());assert {r['run'] for r in checks}==set(ids)
 for e in evaluations:
  p=ROOT/e['run'];r=json.loads((p/'result.json').read_text())
  assert e['config']==r['config']
  assert p.name==digest(json.dumps(r['config'],sort_keys=True).encode())[:20]
  assert digest((p/'weights.pt').read_bytes())==r['checkpoint_sha256']
  c=r['config'];assert r['tokens']==r['steps']*c['context']*c['batch']
  for name,sha in c['provenance']['code'].items():assert digest((p/name).read_bytes())==sha
  if 'calibration' in c:assert digest((p/'calibrate.py').read_bytes())==c['calibration']['controller_sha256']
  if 'controller_sha256' in c:assert digest((p/'local_control.py').read_bytes())==c['controller_sha256']
  for score in e['test'].values():assert abs(math.log(score['ppl'])-score['nats'])<1e-10
  for name,meta in e['test_metadata'].items():
   assert digest((ROOT/'data'/f"pg{meta['gutenberg']}.txt").read_bytes())==meta['raw_sha256']
 timing=json.loads((ROOT/'paired_timing.json').read_text());assert len(timing['rows'])==324
 training=json.loads((ROOT/'paired_training_timing.json').read_text());assert len(training['rows'])==360
 pairs={(r['kind'],r['seed'],r['repeat']):r['step_ms'] for r in training['rows'] if r['context']==128}
 ratios=[pairs[('striped',s,i)]/pairs[('full',s,i)] for s in range(3) for i in range(20)]
 decision=json.loads((ROOT/'calibration_decision.json').read_text())
 assert decision['steps']==math.ceil(1800*statistics.median(ratios))
 assert decision['training_timing_sha256']==digest((ROOT/'paired_training_timing.json').read_bytes())
 exact=json.loads((ROOT/'exact_window_audit.json').read_text());assert len(exact['rows'])==7
 assert all(r['run'] in ids and all(v['targets']==64 for book in r['scores'].values() for v in book.values()) for r in exact['rows'])
 print('final consistency verified: 36 evaluated models, checkpoints/configurations/data, timing rows, calibrated budget, and exact-window audit')
if __name__=='__main__':main()
