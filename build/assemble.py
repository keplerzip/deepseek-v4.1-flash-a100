#!/usr/bin/env python3
"""Assemble a new deploy directory from source plus a verified local image archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from verify_image import verify

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--inspect',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='new deploy directory')
    a=p.parse_args()
    if a.output.exists(): p.error('Output already exists; do not overwrite a deployed service')
    report=verify(a.archive,json.loads(a.inspect.read_text()))
    source=ROOT/'deploy'
    if a.output.resolve().is_relative_to(source.resolve()): p.error('Output must be outside the source deploy tree')
    shutil.copytree(source,a.output,ignore=shutil.ignore_patterns('runtime','images','__pycache__','*.pyc'))
    images=a.output/'images';images.mkdir()
    shutil.copyfile(a.archive,images/'runtime-image.tar')
    # Detect an archive changing during the copy, before freezing its checksum.
    if digest(images/'runtime-image.tar')!=report['archive_sha256']:
        raise RuntimeError('Copied image archive differs; incomplete output retained for inspection')
    manifest=a.output/'manifests'
    (manifest/'image-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    ids={report[k] for k in ('image_id','image_config_id','platform_manifest_id') if report.get(k)}
    (manifest/'allowed-image-ids.txt').write_text('\n'.join(sorted(ids))+'\n')
    baseline=json.loads((manifest/'baseline.json').read_text())
    baseline.update(runtime_archive_sha256=report['archive_sha256'],runtime_config_id=report['image_config_id'],
                    target_gpu_acceptance='NOT_RUN_FOR_THIS_ASSEMBLY')
    (manifest/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    lines=[]
    for path in sorted(a.output.rglob('*')):
        if path.is_symlink(): raise RuntimeError('Unexpected symlink in deploy source')
        if not path.is_file(): continue
        name=path.relative_to(a.output).as_posix()
        if name in ('deployment.env','manifests/deploy.sha256'): continue
        sha=report['archive_sha256'] if name=='images/runtime-image.tar' else digest(path)
        lines.append(f'{sha}  {name}\n')
    (manifest/'deploy.sha256').write_text(''.join(lines))
    print(json.dumps({'status':'ASSEMBLED','deploy':str(a.output),'default_k':5,
                      'model_snapshot':'place the fixed snapshot in the adjacent DeepSeek-V4.1-Flash directory',
                      'target_gpu_acceptance':'NOT_RUN_FOR_THIS_ASSEMBLY'}))


if __name__=='__main__': main()
