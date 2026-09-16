#!/usr/bin/env python3
"""Verify the pinned performance overlay and emit Docker bind arguments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import sysconfig


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(deploy, site, host_deploy, mhc, indexer=True, moe_align=True):
    root = deploy / 'overrides/performance'
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = []
    enabled = {'mhc': mhc, 'indexer': indexer, 'moe_align': moe_align, 'vision': True}
    for item in manifest['files']:
        name = item['path']
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not name.startswith('vllm/'):
            raise ValueError('Invalid performance overlay path')
        target = site / rel
        actual = digest(target) if target.exists() else None
        if actual not in {item['base_sha256'], item['patched_sha256']}:
            raise ValueError('Unsupported image source for performance overlay: ' + name)
        # Turning fusion off restores both layer ABIs, even in rebuilt images.
        variant = 'patched' if enabled[item['group']] else 'baseline'
        if variant == 'baseline' and item['base_sha256'] is None:
            continue
        source = root / variant / rel
        expected = item['base_sha256'] if variant == 'baseline' else item['patched_sha256']
        if digest(source) != expected:
            raise ValueError('Performance payload checksum mismatch: ' + str(source))
        host_source = host_deploy / source.relative_to(deploy)
        if any(c in str(host_source) for c in (':', '\n', '\x00')):
            raise ValueError('Unsupported delimiter in deployment path')
        rows.append({'source': str(host_source), 'target': str(target),
                     'path': name, 'sha256': expected, 'variant': variant})
    return {'id': manifest['id'], 'mhc_fusion': bool(mhc),
            'compact_indexer': bool(indexer), 'stable_moe_align': bool(moe_align), 'files': rows,
            'manifest_sha256': digest(root / 'manifest.json')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', type=Path, default=Path('/deploy'))
    parser.add_argument('--site', type=Path, default=Path(sysconfig.get_paths()['purelib']))
    parser.add_argument('--host-deploy', type=Path, required=True)
    parser.add_argument('--mhc', type=int, choices=(0, 1), default=1)
    parser.add_argument('--indexer', type=int, choices=(0, 1), default=1)
    parser.add_argument('--moe-align', type=int, choices=(0, 1), default=1)
    parser.add_argument('--nccl', choices=('auto', 'legacy'), default='auto')
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    result = plan(args.deploy, args.site, args.host_deploy, args.mhc, args.indexer, args.moe_align)
    result['nccl_selection'] = args.nccl
    if args.record:
        args.record.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['files']:
        sys.stdout.buffer.write(b'-v\x00')
        sys.stdout.buffer.write(f"{row['source']}:{row['target']}:ro".encode() + b'\x00')


if __name__ == '__main__':
    main()
