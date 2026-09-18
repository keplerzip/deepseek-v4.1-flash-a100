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
    assert cfg['limit_mm_per_prompt']=={'image':999}
    assert cfg['served_model_name']=='DeepSeek-V4.1-Flash' and cfg['max_model_len']==262144
    assert cfg['tensor_parallel_size']==4 and cfg['data_parallel_size']==cfg['data_parallel_size_local']==2
    assert cfg['enable_expert_parallel'] and cfg['enable_eplb']
    assert cfg['eplb_config']['num_redundant_experts']==0
    assert not cfg.get('language_model_only',False)
    assert json.loads((ROOT/'release.json').read_text())['default_dspark_k']==5
    assert json.loads((ROOT/'release.json').read_text())['allowed_dspark_k']==[5]
    release=json.loads((ROOT/'release.json').read_text())
    update=json.loads((ROOT/'deploy/manifests/update.json').read_text())
    overlay=json.loads((ROOT/'deploy/overrides/performance/manifest.json').read_text())
    assert (ROOT/'VERSION').read_text().strip()==release['release']==update['release']==overlay['release']
    assert release['source_update']==update['id']==overlay['id']
    for name in ('deploy/configs/locked-scheme.json','deploy/manifests/baseline.json'):
        fixed=json.loads((ROOT/name).read_text())
        assert fixed['release']==release['release'] and fixed['tensor_parallel_size']==4 and fixed['data_parallel_size']==2
        assert fixed['concurrency_formula']==release['concurrency_formula']
    assert update['cumulative'] and not update['requires_prior_update']
    assert release['runtime_tag']==update['compatible_initial_image']
    assert release['performance_overlay_sha256']==hashlib.sha256((ROOT/'deploy/overrides/performance/manifest.json').read_bytes()).hexdigest()
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
