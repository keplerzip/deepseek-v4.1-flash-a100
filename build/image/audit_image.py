"""Inventory an image and compare its Python implementation to the fixed source."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import sysconfig

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
args = parser.parse_args()
site = Path(sysconfig.get_paths()['purelib'])
packages = sorted([{'name': d.metadata['Name'], 'version': d.version}
                   for d in importlib.metadata.distributions()], key=lambda x: x['name'].lower())
same, missing, changed = [], [], []
if args.source:
    for path in sorted((args.source / 'vllm').rglob('*.py')):
        relative = path.relative_to(args.source).as_posix()
        installed = site / relative
        if not installed.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(installed.read_bytes()).digest():
            same.append(relative)
        else:
            changed.append({'path': relative, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'installed_sha256': hashlib.sha256(installed.read_bytes()).hexdigest()})
binaries = [{'path': str(p.relative_to(site)), 'bytes': p.stat().st_size}
            for p in (site / 'vllm').rglob('*.so')]
print(json.dumps({'scope': 'image metadata and source byte comparison; not GPU execution',
                  'python': sys.version, 'platform': platform.platform(), 'site_packages': str(site),
                  'packages': packages, 'tools': {n: shutil.which(n) for n in ['nvcc','ptxas','g++','gcc','ninja','cmake','cuobjdump','uv','ffmpeg']},
                  'vllm_python_source_identical': len(same), 'vllm_python_source_missing': missing,
                  'vllm_python_source_changed': changed, 'vllm_native_extensions': binaries}, indent=2))
