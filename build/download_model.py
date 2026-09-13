#!/usr/bin/env python3
"""Download the pinned ModelScope snapshot without a second cache or credentials."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / 'delivery/deepseek-v4.1-flash-a100/DeepSeek-V4.1-Flash'
CHUNK = 8 * 1024**2


class ContractError(RuntimeError):
    pass


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def response_end(status, headers, offset, expected):
    if headers.get('Content-Encoding', 'identity') not in ('identity', ''):
        raise ContractError('Encoded response is incompatible with byte offsets')
    if status == 200:
        if offset:
            raise ContractError('Server ignored Range; partial file preserved')
        return expected
    if status != 206:
        raise ContractError('Unexpected HTTP response status')
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('Content-Range', ''))
    if not match:
        raise ContractError('Invalid Content-Range')
    start, end, total = map(int, match.groups())
    if start != offset or total != expected or not start <= end < total:
        raise ContractError('Content-Range does not match the pinned file and offset')
    return end + 1


def safe_path(dest, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ContractError('Unsafe manifest path')
    target = dest.joinpath(*rel.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(dest.resolve()):
        raise ContractError('External or symlink model path')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reserve-gib', type=float, default=50)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'deploy/manifests/model-files.json')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be 1..16')
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        raise ContractError('Download destination must be the physical delivery directory')
    state_dir = ROOT / 'build-state'
    state_dir.mkdir(exist_ok=True)
    lock = (state_dir / 'model-download.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another model downloader holds the lock; do not restart it')
    manifest = json.loads(args.manifest.read_text())
    items = manifest['files']
    names = [x['path'] for x in items]
    if len(set(names)) != len(names):
        raise ContractError('Duplicate manifest paths')
    total = sum(x['size'] for x in items)
    partial_bytes = 0
    for item in items:
        final = safe_path(dest, item['path'])
        partial = final.with_name(final.name + '.partial')
        if partial.is_symlink():
            raise ContractError('Symlink partial file')
        if final.exists():
            partial_bytes += min(final.stat().st_size, item['size'])
        elif partial.exists():
            partial_bytes += min(partial.stat().st_size, item['size'])
    reserve = int(args.reserve_gib * 1024**3)
    if shutil.disk_usage(dest).free < total - partial_bytes + reserve:
        raise SystemExit('Insufficient space for remaining snapshot plus reserve; run the scoped storage preparation first')
    mutex = threading.Lock()
    progress = {}
    started = time.time()
    stop = threading.Event()

    def update(name, **fields):
        with mutex:
            progress.setdefault(name, {}).update(fields)

    def save():
        with mutex:
            files = {k: dict(v) for k, v in progress.items()}
        result = {'pid': os.getpid(), 'source': 'ModelScope', 'revision': manifest['modelscope_revision'],
                  'destination': str(dest), 'started_at_epoch': started, 'updated_at_epoch': time.time(),
                  'total_bytes': total, 'observed_bytes': sum(v.get('bytes', 0) for v in files.values()),
                  'verified_files': sum(v.get('status') == 'verified' for v in files.values()),
                  'file_count': len(items), 'files': files}
        atomic_json(state_dir / 'download-progress.json', result)
        return result

    def download(item):
        name, expected = item['path'], item['size']
        target = safe_path(dest, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        if target.exists():
            update(name, status='verifying_existing', bytes=target.stat().st_size)
            if target.stat().st_size != expected or sha256(target) != item['sha256']:
                raise ContractError('Existing final file differs from fixed snapshot: ' + name)
            update(name, status='verified', bytes=expected, sha256=item['sha256'])
            return
        for attempt in range(1, 11):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > expected:
                    raise ContractError('Partial file larger than pinned size: ' + name)
                while offset < expected:
                    if stop.is_set():
                        raise ContractError('Download stopped after another contract failure')
                    if shutil.disk_usage(dest).free < reserve:
                        raise ContractError('Space reserve reached; partial files preserved')
                    headers = {'User-Agent': 'DeepSeek-V4.1-Flash-offline-preparation/1', 'Accept-Encoding': 'identity'}
                    if offset:
                        headers['Range'] = 'bytes=' + str(offset) + '-'
                    request = urllib.request.Request(item['modelscope_url'], headers=headers)
                    with urllib.request.urlopen(request, timeout=90) as response:
                        end = response_end(response.status, response.headers, offset, expected)
                        update(name, status='downloading', bytes=offset, attempt=attempt)
                        last_space_check = time.monotonic()
                        with partial.open('ab' if offset else 'wb') as stream:
                            while True:
                                block = response.read(min(CHUNK, end - offset + 1))
                                if not block:
                                    break
                                if offset + len(block) > end:
                                    raise ContractError('Response exceeds declared range or pinned size')
                                stream.write(block)
                                offset += len(block)
                                update(name, bytes=offset)
                                if time.monotonic() - last_space_check > 10:
                                    if shutil.disk_usage(dest).free < reserve or stop.is_set():
                                        raise ContractError('Space reserve or stop condition reached')
                                    last_space_check = time.monotonic()
                            stream.flush()
                            os.fsync(stream.fileno())
                        if offset != end:
                            raise OSError('Incomplete HTTP body; retry with Range')
                update(name, status='verifying', bytes=expected)
                actual = sha256(partial)
                if actual != item['sha256']:
                    raise ContractError('Publisher SHA256 mismatch; partial file preserved: ' + name)
                partial.replace(target)
                update(name, status='verified', bytes=expected, sha256=actual,
                       verified_at=datetime.datetime.now().astimezone().isoformat())
                print('VERIFIED ' + name, flush=True)
                return
            except ContractError:
                stop.set()
                update(name, status='contract_failed')
                raise
            except Exception as exc:
                # Exception strings and redirect URLs may contain CDN signatures.
                update(name, status='retry', attempt=attempt, error_type=type(exc).__name__,
                       http_status=getattr(exc, 'code', None))
                print(f'RETRY {name} attempt={attempt} type={type(exc).__name__} HTTP={getattr(exc, "code", None)}', flush=True)
                if attempt == 10:
                    raise RuntimeError('Retries exhausted: ' + name) from None
                stop.wait(min(30, 3 * attempt))

    # Fetch auxiliary files first, then begin the largest shards early.
    ordered = sorted(items, key=lambda x: (x['path'].endswith('.safetensors'), -x['size']))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in ordered]
        while not all(f.done() for f in futures):
            result = save()
            print(f'PROGRESS {result["observed_bytes"] / 1024**3:.2f}/{total / 1024**3:.2f} GiB; {result["verified_files"]}/{len(items)} verified', flush=True)
            time.sleep(10)
        for future in futures:
            if future.exception():
                failures.append(str(future.exception()))
    result = save()
    result['status'] = 'FAIL' if failures else 'PASS'
    result['scope'] = 'downloaded file bytes and publisher SHA256; tensor headers and GPU inference are separate gates'
    result['failures'] = failures
    result['completed_at'] = datetime.datetime.now().astimezone().isoformat()
    atomic_json(state_dir / 'model-download.json', result)
    if failures:
        raise SystemExit('MODEL_DOWNLOAD=FAIL; see validation/model-download.json')
    print('MODEL_DOWNLOAD=PASS', flush=True)


if __name__ == '__main__':
    main()
