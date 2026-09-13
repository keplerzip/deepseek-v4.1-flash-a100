#!/usr/bin/env python3
"""Check public source, documentation links, source hashes and fixed k5 configuration."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    roots=['deploy','build','docs','reports','licenses','examples']
    files=[p for name in roots for p in (ROOT/name).rglob('*') if p.is_file()
           and '__pycache__' not in p.parts and 'runtime' not in p.parts]
    files += [ROOT/name for name in ['README.md','CHANGELOG.md','release.json']]
    for path in files:
        if path.suffix=='.py': ast.parse(path.read_text(),filename=str(path))
        elif path.suffix=='.sh': subprocess.run(['bash','-n',str(path)],check=True)
        elif path.suffix=='.json': json.loads(path.read_text())
        if path.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if '://' in link or link.startswith('#'): continue
                target=link.split('#',1)[0]
                assert (path.parent/target).exists(),(str(path),link)
    cfg=json.loads((ROOT/'deploy/configs/runtime.json').read_text())
    assert cfg['speculative_config']['num_speculative_tokens']==5
    assert cfg['served_model_name']=='DeepSeek-V4.1-Flash' and cfg['max_model_len']==262144
    assert json.loads((ROOT/'release.json').read_text())['default_dspark_k']==5
    assert json.loads((ROOT/'release.json').read_text())['allowed_dspark_k']==[5]
    forbidden=['base64.sh','deploy/apply-hotfix.sh','deploy/scripts/hotfix_ops.py',
               'deploy/configs/dspark-k7.json','deploy/overrides/vllm_speculative.py']
    assert not any((ROOT/name).exists() for name in forbidden)
    readme=(ROOT/'README.md').read_text()
    assert re.search(r'^## (.+)$',readme,re.M).group(1).startswith('实测性能')
    for row in (ROOT/'deploy/manifests/deploy.sha256').read_text().splitlines():
        sha,name=row.split('  ',1)
        if name=='images/runtime-image.tar': continue  # intentionally excluded binary artifact
        assert hashlib.sha256((ROOT/'deploy'/name).read_bytes()).hexdigest()==sha,name
    print(json.dumps({'status':'PASS','checked_files':len(files),'default_k':5,
                      'readme_first_section':'performance','gpu_tested':False}))


if __name__=='__main__': main()
