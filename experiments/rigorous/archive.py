"""Preserve immutable evidence before timing/summary files are regenerated."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files=[p for p in ROOT.rglob('*') if p.is_file() and 'archives' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.suffix not in ('.log','.tmp','.pyc')]
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
blob=json.dumps(manifest,sort_keys=True,indent=2).encode();key=hashlib.sha256(blob).hexdigest()[:20]
out=ROOT/'archives'/f'{key}.zip';out.parent.mkdir(exist_ok=True)
if not out.exists():
 temp=out.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('ARCHIVE_MANIFEST.json',blob)
  for p in files:z.write(p,str(p.relative_to(ROOT)))
 temp.replace(out)
print(out)
