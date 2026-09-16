#!/usr/bin/env python3
"""Create a small update from deploy source, with no images, weights or state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    version = json.loads((ROOT / 'deploy/manifests/update.json').read_text())
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        stage = Path(tmp) / ('deepseek-v4.1-flash-a100-' + version['delivery_version'] + '-update')
        shutil.copytree(ROOT / 'deploy', stage / 'deploy',
                        ignore=shutil.ignore_patterns('runtime', 'images', '__pycache__', '*.pyc'))
        shutil.copyfile(ROOT / 'build/update-entry.sh', stage / 'install.sh')
        shutil.copyfile(ROOT / version['readme_source'], stage / 'README.zh-CN.md')
        for name in ('LICENSE', 'NOTICE'):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / 'licenses', stage / 'licenses')
        (stage / 'reports').mkdir()
        for name in ('dspark-k5-k7.user-reported.json', 'acceptance-20260913.user-reported.json'):
            shutil.copyfile(ROOT / 'reports' / name, stage / 'reports' / name)
        rows = []
        for f in sorted(stage.rglob('*')):
            if f.is_symlink():
                raise ValueError('Unexpected symlink in update source')
            if f.is_file():
                rows.append(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.relative_to(stage).as_posix() + '\n')
        (stage / 'SHA256SUMS').write_text(''.join(rows))
        with tarfile.open(a.output, 'w:gz', compresslevel=9) as archive:
            archive.add(stage, arcname=stage.name)
    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_name(a.output.name + '.sha256').write_text(digest + '  ' + a.output.name + '\n')
    print(f'{a.output} ({a.output.stat().st_size} bytes) sha256={digest}')


if __name__ == '__main__':
    main()
