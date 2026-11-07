"""Check release-file SHA-256 values without network access."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'07_Provenance/files_sha256.json').read_text('utf-8'))
bad=[]
for row in manifest['files']:
 p=ROOT/row['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:bad.append(row['path'])
print(f"File integrity: {len(manifest['files'])-len(bad)}/{len(manifest['files'])} matched")
for p in bad:print('MISMATCH:',p)
sys.exit(1 if bad else 0)
