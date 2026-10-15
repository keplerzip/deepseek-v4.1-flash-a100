#!/usr/bin/env python3
"""Package deploy + the pinned local runtime image, without weights or state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / 'delivery/deepseek-v4.1-flash-a100/deploy/images/runtime-image.tar'
DEFAULT_ENV = '''# Empty MODEL_DIR uses the adjacent DeepSeek-V4.1-Flash snapshot.
MODEL_DIR=
BIND_HOST=127.0.0.1
API_PORT=8005
START_TIMEOUT_SECONDS=7200
# Externally managed NewAPI; this package does not install or modify it.
NEWAPI_BASE_URL=
# Verified storage root from this isolated target's supplied inventory.
CONTAINERD_ROOT_DIR=/var/lib/containerd
PERF_NCCL=auto
PERF_MHC=1
PERF_INDEXER=1
PERF_MOE_ALIGN=1
PERF_DSPARK_KV=1
PERF_TOPK_REUSE=1
PERF_DECODE_METADATA=1
'''


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def excluded(relative):
    parts = relative.parts
    name = relative.name.lower()
    return (any(p in {'.git', '__pycache__', '.cache', '.pytest_cache', 'node_modules',
                      'runtime', 'weights', 'models'} for p in parts)
            or parts[:2] == ('deploy', 'images')
            or 'base64' in name or name.endswith(('.pyc', '.pyo', '.b64', '.bin', '.safetensors',
                                                 '.gguf', '.pt', '.pth', '.onnx', '.tar', '.gz'))
            or name == '.env' or name.startswith('.env.'))


def copy_source(source, target, relative):
    """Reject links before copying, so a link cannot import private files."""
    if any(char in relative.as_posix() for char in ('\\', '\n', '\r')):
        raise ValueError('Unsupported source filename for portable checksums')
    if excluded(relative):
        return
    if source.is_symlink():
        raise ValueError('Unexpected source symlink: ' + relative.as_posix())
    if source.is_dir():
        target.mkdir()
        for child in sorted(source.iterdir()):
            copy_source(child, target / child.name, relative / child.name)
    elif source.is_file():
        shutil.copy2(source, target)
    else:
        raise ValueError('Unexpected non-regular source: ' + relative.as_posix())


def checksum_tree(root, *, deploy=False, image_sha=None):
    rows = []
    omitted = {'deployment.env', 'manifests/deploy.sha256'} if deploy else {'SHA256SUMS'}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected packaged symlink')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name in omitted:
                continue
            sha = image_sha if name == ('images/runtime-image.tar' if deploy else 'deploy/images/runtime-image.tar') else digest(path)
            rows.append(f'{sha}  {name}\n')
    return ''.join(rows)


def normalize_tar_info(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mode = 0o755 if info.isdir() or info.name.endswith('.sh') else 0o600 if info.name.endswith('/deployment.env') else 0o644
    return info


def configuration_metadata(runtime, constraints):
    """Validate source templates, before resolve() adds the per-DP sequence cap."""
    fixed_runtime = dict(model='/models/DeepSeek-V4.1-Flash',
                         served_model_name='DeepSeek-Flash', max_model_len=512000,
                         max_num_batched_tokens=1024, tensor_parallel_size=4,
                         data_parallel_size=2, enable_chunked_prefill=True,
                         disable_chunked_mm_input=True)
    if any(runtime.get(key) != value for key, value in fixed_runtime.items()):
        raise ValueError('Fixed model/context/scheduling runtime contract differs')
    fixed_constraints = dict(served_model_name='DeepSeek-Flash', max_model_len=512000,
                             max_num_batched_tokens=1024, minimum_concurrency=64,
                             maximum_concurrency=64, concurrency_formula='fixed(64)',
                             tensor_parallel_size=4, data_parallel_size=2,
                             dspark_k=5, enable_chunked_prefill=True,
                             disable_chunked_mm_input=True, vision_max_n_token=1020,
                             requested_image_token_ceiling=1024, dspark_reserved_token_slots=4)
    if any(constraints.get(key) != value for key, value in fixed_constraints.items()):
        raise ValueError('Fixed concurrency or vision constraints differ')
    speculative = runtime.get('speculative_config', {})
    overrides = runtime.get('hf_overrides', {})
    if (not isinstance(speculative, dict) or not isinstance(overrides, dict)
            or speculative.get('method') != 'dspark' or speculative.get('num_speculative_tokens') != 5
            or overrides.get('vision_max_n_token') != 1020):
        raise ValueError('Underlying DSpark or vision token contract differs')
    global_concurrency = constraints['maximum_concurrency']
    per_dp = global_concurrency // runtime['data_parallel_size']
    if 'max_num_seqs' in runtime and runtime['max_num_seqs'] != per_dp:
        raise ValueError('Explicit per-DP max_num_seqs differs from fixed global concurrency')
    return dict(served_model_name=runtime['served_model_name'],
                weight_model='DeepSeek-V4.1-Flash', context=runtime['max_model_len'],
                max_num_batched_tokens=runtime['max_num_batched_tokens'],
                global_concurrency=global_concurrency, per_dp_concurrency=per_dp,
                vision_max_n_token=1020, requested_image_token_ceiling=1024,
                dspark_reserved_token_slots=4, disable_chunked_mm_input=True,
                enable_chunked_prefill=True)


def package(root, image, output, compresslevel=1):
    root, image, output = root.resolve(strict=True), image.resolve(strict=True), output.resolve()
    if output.exists():
        raise ValueError('Output already exists')
    if not image.is_file():
        raise ValueError('A local regular runtime image archive is required')
    configuration_metadata(json.loads((root / 'deploy/configs/runtime.json').read_text()),
                           json.loads((root / 'deploy/configs/constraints.json').read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    integrity = json.loads((root / 'deploy/manifests/image-integrity.json').read_text())
    if integrity.get('status') != 'PASS' or integrity.get('archive_bytes') != image.stat().st_size:
        raise ValueError('Local image does not match the verified archive size')
    image_sha = digest(image)
    if image_sha != integrity['archive_sha256']:
        raise ValueError('Local image does not match the pinned archive SHA256')
    image_stat = image.stat()
    version = json.loads((root / 'deploy/manifests/update.json').read_text())
    package_name = 'deepseek-v4.1-flash-a100-' + version['delivery_version'] + '-full'
    if Path(package_name).name != package_name or '\n' in package_name:
        raise ValueError('Invalid delivery version')

    with tempfile.TemporaryDirectory(prefix='.dsv41-full-', dir=output.parent) as temporary:
        stage = Path(temporary) / package_name
        stage.mkdir()
        for name in ('deploy', 'docs', 'reports', 'build', 'examples', 'licenses'):
            copy_source(root / name, stage / name, Path(name))
        for name in ('LICENSE', 'NOTICE', 'VERSION', 'release.json', 'CHANGELOG.md', 'README.md'):
            copy_source(root / name, stage / name, Path(name))
        # Use project defaults, never a previous delivery's target settings.
        (stage / 'deploy/deployment.env').write_text(DEFAULT_ENV)
        images = stage / 'deploy/images'
        images.mkdir()
        try:
            os.link(image, images / 'runtime-image.tar')
        except OSError:
            shutil.copyfile(image, images / 'runtime-image.tar')
            if digest(images / 'runtime-image.tar') != image_sha:
                raise ValueError('Runtime image copy changed or was incomplete')
        shutil.copyfile(root / 'build/full-install.sh', stage / 'install.sh')
        shutil.copyfile(root / 'build/update-entry.sh', stage / 'update-entry.sh')
        package_report = dict(schema=1, release=version['release'],
                              delivery_version=version['delivery_version'],
                              includes_weights=False, includes_container_image=True,
                              includes_private_state=False, target_host_modified=False,
                              runtime_archive_sha256=image_sha,
                              runtime_archive_bytes=image.stat().st_size,
                              target_gpu_acceptance='NOT_RUN_ON_BUILD_HOST')
        runtime = json.loads((stage / 'deploy/configs/runtime.json').read_text())
        constraints = json.loads((stage / 'deploy/configs/constraints.json').read_text())
        package_report.update(configuration_metadata(runtime, constraints))
        (stage / 'PACKAGE.json').write_text(json.dumps(package_report, indent=2) + '\n')
        (stage / 'deploy/manifests/deploy.sha256').write_text(
            checksum_tree(stage / 'deploy', deploy=True, image_sha=image_sha))
        (stage / 'SHA256SUMS').write_text(checksum_tree(stage, image_sha=image_sha))
        temporary_archive = Path(temporary) / 'package.tar.gz'
        with tarfile.open(temporary_archive, 'w:gz', compresslevel=compresslevel) as archive:
            archive.add(stage, arcname=package_name, filter=normalize_tar_info)
        after = image.stat()
        # Creating/removing the staging hardlink legitimately changes ctime.
        if (after.st_size, after.st_mtime_ns) != (image_stat.st_size, image_stat.st_mtime_ns):
            raise ValueError('Runtime image changed during packaging')
        archive_sha = digest(temporary_archive)
        os.link(temporary_archive, output)  # Atomic publish, refuses overwrite.
    output.with_name(output.name + '.sha256').write_text(archive_sha + '  ' + output.name + '\n')
    report = dict(status='PACKAGED', archive=str(output), archive_bytes=output.stat().st_size,
                  archive_sha256=archive_sha, **package_report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=DEFAULT_IMAGE,
                        help='Existing pinned runtime image tar; no download is performed')
    parser.add_argument('--compresslevel', type=int, choices=range(1, 10), default=1)
    args = parser.parse_args()
    package(ROOT, args.archive, args.output, args.compresslevel)


if __name__ == '__main__':
    main()
