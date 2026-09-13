"""Verify the physical snapshot, safetensors layouts and complete index mapping."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys

WIDTHS = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E4M3FN': 1,
          'F8_E5M2': 1, 'F8_E4M3FNUZ': 1, 'F8_E5M2FNUZ': 1, 'F8_E8M0': 1,
          'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4,
          'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_header(path, expected_size=None):
    size = expected_size if expected_size is not None else path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Short safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(64 * 1024**2, size - 8):
            raise ValueError('Invalid safetensors header length')
        data = stream.read(length)
        if len(data) != length:
            raise ValueError('Short safetensors header')
    header = json.loads(data, object_pairs_hook=no_duplicates)
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    ranges = []
    dtype_counts = {}
    for key, entry in tensors.items():
        a, b = entry['data_offsets']
        if type(a) is not int or type(b) is not int or not 0 <= a <= b <= size - 8 - length:
            raise ValueError('Out-of-bounds tensor: ' + key)
        shape = entry['shape']
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError('Invalid shape: ' + key)
        dtype = entry['dtype']
        if dtype not in WIDTHS:
            raise ValueError('Unreviewed safetensors dtype: ' + dtype)
        if b - a != math.prod(shape) * WIDTHS[dtype]:
            raise ValueError('Tensor shape/dtype/byte-count mismatch: ' + key)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        ranges.append((a, b))
    cursor = 0
    for a, b in sorted(ranges):
        if a != cursor:
            raise ValueError('Overlapping tensor ranges or holes in payload')
        cursor = b
    if cursor != size - 8 - length:
        raise ValueError('Trailing or missing tensor payload')
    return tensors, {'header_bytes': length, 'payload_bytes': cursor,
                     'tensor_count': len(tensors), 'dtype_counts': dtype_counts}


def verify(model, manifest_path, full_hash):
    errors, rows, mapping, dtype_counts = [], [], {}, {}
    manifest = json.loads(manifest_path.read_text(), object_pairs_hook=no_duplicates)
    expected_names = {item['path'] for item in manifest['files']}
    if len(expected_names) != len(manifest['files']):
        raise ValueError('Duplicate manifest files')
    links = [str(p.relative_to(model)) for p in model.rglob('*') if p.is_symlink()]
    if links or model.is_symlink():
        errors.append('Snapshot has symlinks: ' + ', '.join(links[:5]))
    actual_names = {p.relative_to(model).as_posix() for p in model.rglob('*') if p.is_file()}
    if actual_names != expected_names:
        errors.append('Snapshot file-set mismatch: ' + str(sorted(actual_names ^ expected_names)[:10]))
    payload_total = 0
    for item in manifest['files']:
        name = item['path']
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe manifest path')
        path = model / name
        if not path.is_file() or path.is_symlink():
            errors.append('Missing or non-regular file: ' + name)
            continue
        if not path.resolve().is_relative_to(model.resolve()):
            errors.append('External model path: ' + name)
            continue
        before = path.stat()
        if before.st_size != item['size']:
            errors.append('Size mismatch: ' + name)
            continue
        row = {'path': name, 'size': before.st_size, 'sha256_checked': False}
        if full_hash or not name.endswith('.safetensors'):
            actual = digest(path)
            row.update(sha256=actual, sha256_checked=True)
            if actual != item['sha256']:
                errors.append('SHA256 mismatch: ' + name)
        if name.endswith('.safetensors'):
            try:
                tensors, info = read_header(path)
                row.update(info)
                payload_total += info['payload_bytes']
                for dtype, count in info['dtype_counts'].items():
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
                for key in tensors:
                    if key in mapping:
                        errors.append('Duplicate tensor across shards: ' + key)
                    mapping[key] = name
            except (KeyError, TypeError, ValueError, struct.error) as exc:
                errors.append(name + ': ' + str(exc))
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            errors.append('File changed during verification: ' + name)
        rows.append(row)
        print('CHECKED ' + name, file=sys.stderr, flush=True)
    index = json.loads((model / 'model.safetensors.index.json').read_text(), object_pairs_hook=no_duplicates)
    config = json.loads((model / 'config.json').read_text())
    if mapping != index['weight_map']:
        errors.append('Actual tensor-to-shard mappings differ from model index')
    if len(mapping) != 96085:
        errors.append('Expected 96,085 tensors')
    if payload_total != index['metadata']['total_size']:
        errors.append('Header payload total differs from index metadata')
    if 'DeepseekV41ForCausalLM' not in config.get('architectures', []):
        errors.append('Unexpected architecture')
    quant = config.get('quantization_config', {})
    if quant.get('quant_method') != 'fp8' or quant.get('expert_dtype') != 'fp4':
        errors.append('Expected official FP8/FP4 mixed precision')
    text_config = config.get('text_config', {})
    if (text_config.get('model_type') != 'deepseek_v41_text'
            or text_config.get('num_nextn_predict_layers') != 3
            or text_config.get('dspark_block_size') != 5):
        errors.append('Unexpected DSpark checkpoint configuration')
    required_prefixes = ['vision.', 'aligner.', 'mtp.0.', 'mtp.1.', 'mtp.2.']
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in mapping):
            errors.append('Missing model component: ' + prefix)
    if 'mtp.2.confidence_head.proj.weight' not in mapping:
        errors.append('Missing DSpark confidence head')
    return {'status': 'FAIL' if errors else 'PASS', 'scope': 'snapshot bytes, tensor headers and index; no GPU inference',
            'full_hash': full_hash, 'modelscope_revision': manifest['modelscope_revision'],
            'completed_at': datetime.datetime.now().astimezone().isoformat(),
            'file_count': len(rows), 'tensor_count': len(mapping), 'tensor_payload_bytes': payload_total,
            'dtype_counts': dtype_counts, 'files': rows, 'errors': errors, 'gpu_inference_tested': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--full-hash', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.model, args.manifest, args.full_hash)
    except Exception as exc:
        result = {'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc), 'gpu_inference_tested': False}
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(args.output.name + '.tmp')
        tmp.write_text(text)
        tmp.replace(args.output)
    print(text)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
