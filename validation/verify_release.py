"""Read-only verification of the published artifact, using only the standard library."""
from pathlib import Path
import hashlib,json,math,re,statistics
R=Path(__file__).resolve().parents[1];S=R/'experiments/rigorous'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 manifest=json.loads((R/'RELEASE_MANIFEST.json').read_text())
 for name,expected in manifest['files'].items():
  p=R/name;assert p.is_file(),name;assert sha(p)==expected,('release hash mismatch',name)
 runs=list((S/'runs').glob('*/result.json'));assert len(runs)==66,len(runs)
 for p in runs:
  r=json.loads(p.read_text());d=p.parent;c=r['config']
  assert d.name==hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:20]
  assert sha(d/'weights.pt')==r['checkpoint_sha256'],str(d)
  for n,h in c['provenance']['code'].items():assert sha(d/n)==h,(d,n)
  assert r['tokens']==r['steps']*c['context']*c['batch']
 es=json.loads((S/'final_evaluations.json').read_text());ids={e['run'] for e in es};assert len(es)==len(ids)==36
 assert {x['run'] for x in json.loads((S/'trained_checks.json').read_text())}==ids
 for e in es:
  r=json.loads((S/e['run']/'result.json').read_text());assert e['config']==r['config']
  for meta in e['test_metadata'].values():assert sha(S/'data'/f"pg{meta['gutenberg']}.txt")==meta['raw_sha256']
  for score in e['test'].values():assert abs(math.log(score['ppl'])-score['nats'])<1e-10
 timing=json.loads((S/'paired_timing.json').read_text());training=json.loads((S/'paired_training_timing.json').read_text())
 assert len(timing['rows'])==324 and len(training['rows'])==360
 paired={(x['kind'],x['seed'],x['repeat']):x['step_ms'] for x in training['rows'] if x['context']==128}
 ratio=statistics.median(paired['striped',s,i]/paired['full',s,i] for s in range(3) for i in range(20))
 dec=json.loads((S/'calibration_decision.json').read_text());assert dec['steps']==math.ceil(1800*ratio)==2439
 assert dec['training_timing_sha256']==sha(S/'paired_training_timing.json')
 exact=json.loads((S/'exact_window_audit.json').read_text())['rows'];assert len(exact)==7
 assert all(x['run'] in ids and all(v['targets']==64 for book in x['scores'].values() for v in book.values()) for x in exact)
 source=(R/'main.tex').read_text();assert not re.search(r'\\(?:input|include|includegraphics|bibliography)\s*[{[]',source)
 defined=set(re.findall(r'\\bibitem\{([^}]+)\}',source));used={k for m in re.findall(r'\\cite\{([^}]+)\}',source) for k in m.split(',')};assert defined==used
 labels=re.findall(r'\\label\{([^}]+)\}',source);assert len(labels)==len(set(labels))
 assert {k for k in re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source) if k not in ('models','contextmodels')}<=set(labels)
 print(f"Verified {len(manifest['files'])} release files, 66 training runs, 36 final models, all raw timing, calibration, context audit and standalone TeX.")
if __name__=='__main__':verify()
