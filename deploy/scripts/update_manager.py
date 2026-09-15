#!/usr/bin/env python3
"""Apply or roll back a source-only update inside the project's Docker mount."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Invalid relative update path: ' + name)
    out = root / rel
    for path in (out, *out.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError('Update paths cannot contain symlinks: ' + name)
    return out


def atomic(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.dsv41-update-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def restore(target, backup, receipt, check_current=True):
    for row in receipt['files']:
        path = inside(target, row['path'])
        if check_current and (not path.is_file() or sha(path) != row['new_sha256']):
            raise ValueError('File changed after update; refusing to overwrite: ' + row['path'])
        if row['existed']:
            saved = inside(backup / 'files', row['path'])
            if sha(saved) != row['old_sha256']:
                raise ValueError('Rollback backup checksum mismatch: ' + row['path'])
    for row in reversed(receipt['files']):
        path = inside(target, row['path'])
        if row['existed']:
            saved = inside(backup / 'files', row['path'])
            atomic(path, saved.read_bytes(), row['old_mode'])
        else:
            path.unlink(missing_ok=True)
    receipt['status'] = 'ROLLED_BACK'
    atomic(backup / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())


def apply(payload, target):
    if not (target / 'deployment.env').is_file():
        raise ValueError('An existing deploy directory with deployment.env is required')
    changes = {}
    for line in (payload / 'manifests/deploy.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name == 'images/runtime-image.tar':
            continue  # Reuse the separately verified installed image.
        path = inside(payload, name)
        if sha(path) != expected:
            raise ValueError('Update payload checksum mismatch: ' + name)
        changes[name] = (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
    changes['manifests/deploy.sha256'] = ((payload / 'manifests/deploy.sha256').read_bytes(), 0o644)
    env_path = inside(target, 'deployment.env')
    lines = env_path.read_text().splitlines()
    lines = [s for s in lines if not s.startswith(('PERF_NCCL=', 'PERF_MHC='))]
    changes['deployment.env'] = (('\n'.join(lines + ['PERF_NCCL=auto', 'PERF_MHC=1']) + '\n').encode(), 0o600)
    for name in ('runtime/selected-config.json', 'runtime/tuning-active.json'):
        path = inside(target, name)
        if path.exists():
            config = json.loads(path.read_text())
            config['speculative_config']['num_speculative_tokens'] = 5
            config['limit_mm_per_prompt'] = {'image': 999}
            if config != json.loads(path.read_text()):
                changes[name] = ((json.dumps(config, indent=2) + '\n').encode(), 0o600)
    # Validate all paths and save every original before replacing any source.
    backup_rel = 'runtime/update-backups/' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    backup = inside(target, backup_rel)
    backup.mkdir(parents=True)
    receipt = {'status': 'PREPARED', 'update': '20260915-perf1', 'backup': backup_rel, 'files': []}
    for name, (data, mode) in sorted(changes.items()):
        dest = inside(target, name)
        existed = dest.exists()
        if existed and not dest.is_file():
            raise ValueError('A directory occupies update file: ' + name)
        row = {'path': name, 'existed': existed, 'new_sha256': hashlib.sha256(data).hexdigest()}
        if existed:
            row.update(old_sha256=sha(dest), old_mode=stat.S_IMODE(dest.stat().st_mode))
            saved = inside(backup / 'files', name)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest, saved)
        receipt['files'].append(row)
    atomic(backup / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    try:
        for name, (data, mode) in changes.items():
            atomic(inside(target, name), data, mode)
    except BaseException:
        restore(target, backup, receipt, check_current=False)
        raise
    receipt['status'] = 'APPLIED'
    encoded = (json.dumps(receipt, indent=2) + '\n').encode()
    atomic(backup / 'receipt.json', encoded)
    atomic(target / 'runtime/latest-update.json', encoded)
    print(json.dumps({'status': 'APPLIED', 'backup': backup_rel, 'files': len(changes)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', type=Path, default=Path('/target'))
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--apply', type=Path)
    action.add_argument('--rollback', action='store_true')
    a = p.parse_args()
    target = a.target.resolve(strict=True)
    state = inside(target, 'runtime')
    state.mkdir(exist_ok=True)
    with (state / 'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.apply:
            apply(a.apply, target)
        else:
            receipt = json.loads((state / 'latest-update.json').read_text())
            backup = inside(target, receipt['backup'])
            if receipt['status'] != 'APPLIED':
                raise ValueError('Latest update is not in APPLIED state')
            restore(target, backup, receipt)
            atomic(state / 'latest-update.json', (json.dumps(receipt, indent=2) + '\n').encode())
            print(json.dumps({'status': 'ROLLED_BACK', 'backup': receipt['backup']}))


if __name__ == '__main__':
    main()
