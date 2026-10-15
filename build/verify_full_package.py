#!/usr/bin/env python3
"""Stream-read a full offline archive and verify every file without extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from package_full import DEFAULT_ENV, configuration_metadata, excluded

BLOCK = 8 * 1024**2
CAPTURE = {'SHA256SUMS', 'PACKAGE.json', 'deploy/manifests/deploy.sha256',
           'deploy/manifests/image-integrity.json', 'deploy/configs/runtime.json',
           'deploy/configs/constraints.json',
           'deploy/deployment.env'}


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.sha.update(value)
        return value


def checked_path(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or any(char in name for char in ('\\', '\n', '\r'))):
        raise ValueError('Unsafe archive or checksum path: ' + repr(name))
    return path


def checksums(data):
    result = {}
    for line in data.decode('utf-8').splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        if not match:
            raise ValueError('Invalid checksum row')
        sha, name = match.groups()
        if checked_path(name).as_posix() != name or name in result:
            raise ValueError('Noncanonical or duplicate checksum path: ' + name)
        result[name] = sha
    return result


def verify(archive, expected_sha256=None):
    archive = Path(archive)
    sidecar = archive.with_name(archive.name + '.sha256')
    if expected_sha256 is None and sidecar.is_file():
        row = sidecar.read_text().strip().split('  ', 1)
        if len(row) != 2 or row[1] != archive.name:
            raise ValueError('Archive checksum sidecar names a different file')
        expected_sha256 = row[0]
    if expected_sha256 is not None and not re.fullmatch(r'[a-f0-9]{64}', expected_sha256):
        raise ValueError('Invalid expected archive SHA256')
    actual, sizes, captured, seen, package_root = {}, {}, {}, set(), None
    with archive.open('rb') as raw:
        reader = HashingReader(raw)
        with tarfile.open(fileobj=reader, mode='r|gz') as tar:
            for member in tar:
                path = checked_path(member.name)
                normalized = path.as_posix()
                if normalized in seen:
                    raise ValueError('Duplicate archive member: ' + normalized)
                seen.add(normalized)
                if package_root is None:
                    package_root = path.parts[0]
                if path.parts[0] != package_root:
                    raise ValueError('Archive must contain a single delivery root')
                if not (member.isdir() or member.isfile()) or member.issym() or member.islnk():
                    raise ValueError('Archive links and special files are forbidden: ' + normalized)
                if member.isdir():
                    if len(path.parts) > 1:
                        directory = PurePosixPath(*path.parts[1:])
                        if excluded(directory) and directory.as_posix() != 'deploy/images':
                            raise ValueError('Private or unwanted archive directory: ' + directory.as_posix())
                    continue
                if len(path.parts) < 2:
                    raise ValueError('File outside delivery directory')
                relative = PurePosixPath(*path.parts[1:])
                name = relative.as_posix()
                if excluded(relative) and name != 'deploy/images/runtime-image.tar':
                    raise ValueError('Private state, weights or unwanted artifact: ' + name)
                if name in CAPTURE and member.size > 16 * 1024**2:
                    raise ValueError('Unexpectedly large package metadata: ' + name)
                sha = hashlib.sha256()
                content = bytearray() if name in CAPTURE else None
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError('Missing regular member data: ' + name)
                length = 0
                for block in iter(lambda: stream.read(BLOCK), b''):
                    sha.update(block)
                    length += len(block)
                    if content is not None:
                        content.extend(block)
                if length != member.size:
                    raise ValueError('Truncated file: ' + name)
                actual[name], sizes[name] = sha.hexdigest(), length
                if content is not None:
                    captured[name] = bytes(content)
        # Include trailing gzip bytes in the outer archive checksum as well.
        while reader.read(BLOCK):
            pass
        archive_sha = reader.sha.hexdigest()
    if expected_sha256 is not None and archive_sha != expected_sha256:
        raise ValueError('Outer archive SHA256 mismatch')
    if not CAPTURE <= captured.keys():
        raise ValueError('Required package metadata is missing')
    expected = checksums(captured['SHA256SUMS'])
    measured = {k: v for k, v in actual.items() if k != 'SHA256SUMS'}
    if expected != measured:
        different = sorted(k for k in set(expected) | set(measured) if expected.get(k) != measured.get(k))
        raise ValueError('Package checksum coverage/mismatch: ' + ', '.join(different[:10]))
    deploy_expected = checksums(captured['deploy/manifests/deploy.sha256'])
    deploy_actual = {k.removeprefix('deploy/'): v for k, v in actual.items()
                     if k.startswith('deploy/') and k not in ('deploy/deployment.env', 'deploy/manifests/deploy.sha256')}
    if deploy_expected != deploy_actual:
        raise ValueError('deploy.sha256 coverage or checksum mismatch')
    if 'install.sh' not in actual or 'update-entry.sh' not in actual:
        raise ValueError('Required installer entry point is missing')
    package = json.loads(captured['PACKAGE.json'])
    if package_root != 'deepseek-v4.1-flash-a100-' + package['delivery_version'] + '-full':
        raise ValueError('Delivery directory and package version disagree')
    if (package.get('includes_weights') is not False
            or package.get('includes_container_image') is not True
            or package.get('includes_private_state') is not False):
        raise ValueError('Incorrect package contents declaration')
    image = 'deploy/images/runtime-image.tar'
    integrity = json.loads(captured['deploy/manifests/image-integrity.json'])
    if image not in actual or any(record.get('runtime_archive_sha256', record.get('archive_sha256')) != actual[image]
                                  or record.get('runtime_archive_bytes', record.get('archive_bytes')) != sizes[image]
                                  for record in (package, integrity)):
        raise ValueError('Bundled runtime image differs from the pinned metadata')
    runtime = json.loads(captured['deploy/configs/runtime.json'])
    constraints = json.loads(captured['deploy/configs/constraints.json'])
    configuration = configuration_metadata(runtime, constraints)
    if any(package.get(key) != value for key, value in configuration.items()):
        raise ValueError('Package configuration metadata differs from runtime')
    if captured['deploy/deployment.env'].decode('utf-8') != DEFAULT_ENV:
        raise ValueError('Packaged environment contains nondefault host settings')
    return dict(status='PASS', archive=str(archive), archive_sha256=archive_sha,
                external_checksum_checked=expected_sha256 is not None,
                release=package['release'], files=len(actual),
                uncompressed_regular_bytes=sum(sizes.values()), runtime_image_bytes=sizes[image],
                model='DeepSeek-Flash', context=512000, concurrency=64,
                max_num_batched_tokens=1024, dspark_k=5, per_dp_concurrency=32,
                vision_max_n_token=1020, requested_image_token_ceiling=1024, dspark_reserved_token_slots=4, disable_chunked_mm_input=True,
                enable_chunked_prefill=True,
                includes_weights=False, archive_links=False, extracted=False, gpu_executed=False,
                scope='Streaming outer/file/deploy checksums and fixed package contracts; no GPU validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.archive, args.expected_sha256)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
