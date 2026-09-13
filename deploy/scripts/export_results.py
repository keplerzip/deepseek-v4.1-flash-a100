"""Collect bounded operational reports; omit keys, caches and client histories."""
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

root=Path('/state');target=Path(sys.argv[1])
secret=(root/'api-key.txt').read_text().strip() if (root/'api-key.txt').exists() else ''
rows=[]
with tarfile.open(target,'w:gz') as archive:
    deploy=Path('/deploy')
    release={'baseline_sha256':hashlib.sha256((deploy/'manifests/baseline.json').read_bytes()).hexdigest(),
             'component_manifest_sha256':hashlib.sha256((deploy/'manifests/deploy.sha256').read_bytes()).hexdigest(),
             'deployment_policy':'fixed native DSpark k=5'}
    data=json.dumps(release,indent=2).encode()
    info=tarfile.TarInfo('results/release-state.json');info.size=len(data);info.mode=0o600
    archive.addfile(info,io.BytesIO(data));rows.append({'path':'release-state.json','bytes':len(data)})
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.is_symlink() or path==target:continue
        rel=path.relative_to(root)
        if any(part.startswith(('cache','client')) or part in ['hf','triton','inductor','hotfixes'] for part in rel.parts):continue
        if path.suffix not in ['.json','.jsonl','.log','.txt','.csv','.xml']:continue
        if any(word in path.name.lower() for word in ['key','secret','token','auth']):continue
        if path.stat().st_size>128*1024**2:continue
        text=path.read_text(errors='replace')
        if secret:text=text.replace(secret,'[REDACTED]')
        text=re.sub(r'(?i)(Bearer\s+)[A-Za-z0-9_.-]+',r'\1[REDACTED]',text)
        text=re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b','[REDACTED]',text)
        data=text.encode();info=tarfile.TarInfo('results/'+rel.as_posix());info.size=len(data);info.mode=0o600
        archive.addfile(info,io.BytesIO(data));rows.append({'path':rel.as_posix(),'bytes':len(data)})
    data=json.dumps({'files':rows,'scope':'operational reports; no model/cache/key/client history'},indent=2).encode()
    info=tarfile.TarInfo('results/manifest.json');info.size=len(data);info.mode=0o600;archive.addfile(info,io.BytesIO(data))
print(json.dumps({'archive':target.name,'files':len(rows),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
