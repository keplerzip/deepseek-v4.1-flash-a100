"""Port pinned vLLM #57885/#57860 onto R1.3; no downloads or image rebuild.

The R1.4-disabled variant of rocm.py retains R1.3's cross-layer reuse.
Only integer metadata is changed. Large batches keep the linear-time scan.
"""
import hashlib
import json
from pathlib import Path

from prepare_r13_overlays import replace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'source/vllm-backport-a350766628514d597670d2a5217777fdee2ba7f4'
OVERLAY = ROOT / 'deploy/overrides/performance'
UPDATE = '20260922-perf5'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fuse_ragged(text):
    start = text.index('@triton.jit\ndef _pack_global_topk_ragged_kernel(')
    end = text.index('\ndef compute_global_topk_ragged_indices_and_indptr(', start)
    fused = text[start:end].replace('_pack_global_topk_ragged_kernel(',
                                   '_pack_global_topk_ragged_fused_kernel(', 1)
    fused = replace(fused, '    topk_indptr_ptr,\n',
                    '    topk_indptr_ptr,\n    topk_lens_ptr,\n    num_tokens,\n')
    fused = replace(fused, '    BLOCK_SIZE: tl.constexpr,\n',
                    '    BLOCK_SIZE: tl.constexpr,\n    NTOK_PAD: tl.constexpr,\n')
    fused = replace(fused, '''    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
''', '''    # R1.4, adapted from vLLM #57860. Each program computes its prefix
    # directly from lengths; no inter-program dependency on indptr writes.
    rows = tl.arange(0, NTOK_PAD)
    lengths = tl.load(topk_lens_ptr + rows, mask=rows < num_tokens, other=0)
    out_start = tl.sum(tl.where(rows < token_idx, lengths, 0), axis=0)
    out_len = tl.load(topk_lens_ptr + token_idx)
    if block_idx == 0:
        if token_idx == 0:
            tl.store(topk_indptr_ptr, 0)
        tl.store(topk_indptr_ptr + token_idx + 1, out_start + out_len)
    # Write every indptr entry even for padded/empty rows, before returning.
''')
    text = text[:end] + '\n' + fused + text[end:]
    start = text.index('def compute_global_topk_ragged_indices_and_indptr(')
    end = text.index('\ndef _copy_ragged_to_graph_buffers(', start)
    before, after = text[:start], text[end:]
    text = text[start:end]
    text = replace(text, '    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()\n',
                    '    topk_indices = topk_indices.flatten(start_dim=1).contiguous()\n')
    anchor = '    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)\n'
    text = replace(text, anchor, anchor + '''    if num_tokens == 0:
        return (topk_lens, torch.zeros(1, dtype=torch.int32, device=topk_indices.device), topk_lens)
''')
    text = replace(text, '    topk_indptr = _build_indptr_from_lengths(topk_lens)\n', '''    # The fused prefix is quadratic in row count. Bound it to small decode
    # batches; larger batches and zero-width input retain the R1.3 scan.
    fuse_indptr = num_tokens <= 1024 and topk > 0
    topk_indptr = (
        torch.empty(num_tokens + 1, dtype=torch.int32, device=topk_indices.device)
        if fuse_indptr else _build_indptr_from_lengths(topk_lens)
    )
''')
    text = replace(text, '''        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
''', '''        if fuse_indptr:
            _pack_global_topk_ragged_fused_kernel[(num_tokens, triton.cdiv(topk, block))](
                global_topk_ragged, topk_indptr, topk_lens, num_tokens,
                topk_indices, topk_indices.stride(0), token_to_req_indices,
                block_table, block_table.stride(0), block_size, topk,
                BLOCK_SIZE=block, NTOK_PAD=triton.next_power_of_2(num_tokens),
            )
            return global_topk_ragged, topk_indptr, topk_lens
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
''')
    return before + text + after


def trim_indexer(text):
    text = replace(text, '        compressed_seq_lens = seq_lens\n', '')
    text = replace(text, '            compressed_seq_lens = seq_lens // self.compress_ratio\n', '')
    return replace(text, '''        prefill_metadata = None
        if num_prefills > 0:
''', '''        prefill_metadata = None
        if num_prefills > 0:
            # R1.4 / vLLM #57885: decode never consumes these lengths.
            compressed_seq_lens = (
                seq_lens // self.compress_ratio if self.compress_ratio > 1 else seq_lens
            )
''')


def main():
    path = OVERLAY / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['id'] == UPDATE:
        for row in manifest['files']:
            for variant in ('baseline', 'patched'):
                source = OVERLAY / variant / row['path']
                if source.exists():
                    assert sha(source.read_bytes()) == row.get(variant + '_sha256', row['base_sha256'])
        print('R1.4 overlay already exists and matches its manifest.')
        return
    if manifest['id'] != '20260920-perf4':
        raise ValueError('Build the pinned R1.3 overlay first')
    rows = {r['path']: r for r in manifest['files']}
    transforms = {
        'vllm/models/deepseek_v4_1/amd/rocm.py': fuse_ragged,
        'vllm/v1/attention/backends/mla/indexer.py': trim_indexer,
        'vllm/v1/attention/backends/mla/sparse_swa.py': lambda s: replace(
            s, '        is_valid_token.copy_(slot_mapping >= 0)\n',
            '        torch.ge(slot_mapping, 0, out=is_valid_token)\n'),
    }
    for name, transform in transforms.items():
        original = (BASE / name).read_bytes()
        # Preserve R1.3's complete implementation as the switch-off baseline.
        baseline = (OVERLAY / 'patched' / name).read_bytes() if name in rows else original
        if name in rows:
            assert sha(baseline) == rows[name]['patched_sha256']
        patched = transform(baseline.decode()).encode()
        for variant, data in [('baseline', baseline), ('patched', patched)]:
            dest = OVERLAY / variant / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        rows[name] = dict(path=name, group='decode_metadata', base_sha256=sha(original),
                         baseline_sha256=sha(baseline), patched_sha256=sha(patched))
    manifest.update(id=UPDATE, release='1.4.0', gpu_validated=False)
    manifest['files'] = list(rows.values())
    manifest['upstream_sources'] += [
        dict(url='https://github.com/vllm-project/vllm/pull/57885',
             commit='73ce1d9220507a4d06412983e6cb599ed163eb43', r14=True,
             component='persistent SWA validity output; prefill-only compressed lengths'),
        dict(url='https://github.com/vllm-project/vllm/pull/57860',
             commit='2031422bfd952cc65fc1a46a9b570f6f04dc318d', r14=True,
             component='fused ragged indptr/pack; local <=1024-row bound and empty-input handling'),
    ]
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(id=UPDATE, files=len(rows), gpu_validated=False)))


if __name__ == '__main__':
    main()
