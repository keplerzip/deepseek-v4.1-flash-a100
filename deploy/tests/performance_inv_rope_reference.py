"""Load actual inverse-RoPE code without importing vLLM or model weights."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import triton
import triton.language as tl

DEPLOY = Path(__file__).resolve().parents[1]
ROOT = DEPLOY / 'overrides/performance/patched'
OPS = ROOT / 'vllm/v1/attention/ops/rocm_aiter_mla_sparse.py'
ATTN = ROOT / 'vllm/models/deepseek_v4_1/amd/rocm.py'


def load_ops():
    names = {'_sparse_attn_decode_reduce_kernel', '_inverse_rope_gptj_kernel',
             '_fused_inverse_rope_gptj', 'rocm_inverse_rope_rows_', 'rocm_inv_rope_einsum'}
    nodes = [n for n in ast.parse(OPS.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ModuleType('r19_inverse_rope_ops')
    module.__file__ = str(OPS)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OPS), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), module.__dict__)
    return module


def inverse_reference(o, positions, cache, rope_dim=64):
    result = o.float().clone()
    nope = o.shape[-1] - rope_dim
    angles = cache[positions].float()
    cos, sin = angles[:, :rope_dim // 2, None], angles[:, rope_dim // 2:, None]
    cos, sin = cos.transpose(1, 2), sin.transpose(1, 2)
    even, odd = o[..., nope::2].float(), o[..., nope+1::2].float()
    result[..., nope::2] = even * cos + odd * sin
    result[..., nope+1::2] = odd * cos - even * sin
    return result.to(o.dtype)


def fixture(rows, heads, splits, device='cpu'):
    torch.manual_seed(73)
    shape = (rows, splits, heads)
    m = torch.randn(shape, device=device)
    l = torch.rand(shape, device=device) + .5
    acc = torch.randn((*shape, 512), device=device)
    positions = torch.arange(rows, device=device, dtype=torch.int64) * 3 + 1
    angle = torch.randn(max(512, rows * 3 + 3), 32, device=device)
    cache = torch.cat((angle.cos(), angle.sin()), dim=-1).contiguous()
    return m, l, acc, positions, cache


def reduce(module, m, l, acc, positions, cache, fused, adaptive=False, dtype=torch.float32):
    rows, splits, heads = m.shape
    out = torch.empty((rows, heads, 512), device=m.device, dtype=dtype)
    sink = torch.empty(1, device=m.device)
    module._sparse_attn_decode_reduce_kernel[(rows, heads)](
        m, l, acc, sink, out, positions if fused else None, cache if fused else None,
        out.stride(0), out.stride(1), m.stride(0), m.stride(1),
        acc.stride(0), acc.stride(1), acc.stride(2), cache.stride(0) if fused else 0,
        heads, HAS_ATTN_SINK=False, ADAPTIVE_SPLITS=adaptive, COMB_DIM=512,
        BLOCK_H=1, NUM_SPLITS=splits, SPLITS_PAD=triton.next_power_of_2(splits),
        FUSE_INV_ROPE=fused, NOPE=448, HALF=32, num_warps=4,
    )
    return out


def check_attention_dispatch():
    """Execute the actual eager forward_mqa on changing mixed-batch splits.

    Small CPU primitives stand in for attention kernels; the production row
    dispatch executes intact. This prevents double-rotation or stale Python
    split state across pure prefill, pure decode and mixed steps.
    """
    tree = ast.parse(ATTN.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'DeepseekV41ROCMAiterMLAAttention')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_mqa')
    context = SimpleNamespace(attn_metadata={})
    def rotate(o, positions, cache, rope_dim):
        o.copy_(inverse_reference(o, positions, cache, rope_dim))
    namespace = dict(torch=torch, cast=lambda typ, value: value,
                     get_forward_context=lambda: context, rocm_inverse_rope_rows_=rotate,
                     DeepseekV4FlashMLAMetadata=SimpleNamespace,
                     DeepseekV4ROCMAiterSparseSWAMetadata=SimpleNamespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ATTN), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), namespace)
    function = namespace['forward_mqa']
    _, _, _, _, cache = fixture(1, 2, 1)
    owner = SimpleNamespace(compressed_cache_prefix=None, compress_ratio=0,
                            swa_cache_layer=SimpleNamespace(prefix='swa', kv_cache=None),
                            n_local_heads=2, rotary_emb=SimpleNamespace(cos_sin_cache=cache),
                            rope_head_dim=64)
    checked = 0
    for decodes, prefills, fused in [(0, 3, False), (2, 0, True), (2, 3, True),
                                    (1, 4, False), (0, 1, False), (4, 1, True)]:
        rows = decodes + prefills
        context.attn_metadata = {'swa': SimpleNamespace(num_decodes=int(decodes > 0),
            num_prefills=int(prefills > 0), num_decode_tokens=decodes)}
        raw = torch.randn(rows, 2, 512).bfloat16()
        positions = torch.arange(rows, dtype=torch.int64) * 7 + 1
        def prefill(**kw):
            kw['output'].copy_(kw['q'])
        def decode(**kw):
            kw['output'].copy_(inverse_reference(kw['q'], kw['positions'], cache)
                               if fused else kw['q'])
            return decodes if fused else 0
        owner._forward_prefill, owner._forward_decode = prefill, decode
        output = torch.empty_like(raw)
        function(owner, raw, torch.empty(0), positions, output)
        torch.testing.assert_close(output, inverse_reference(raw, positions, cache), atol=0, rtol=0)
        checked += 1
    return checked


def check_decode_wrapper():
    tree = ast.parse(OPS.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'rocm_sparse_attn_decode')
    namespace = dict(torch=torch, _ON_GFX950=False,
                     _validate_dsv4_sparse_dims=lambda *a: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(OPS), 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), namespace)
    raw = torch.randn(3, 2, 512).bfloat16()
    positions = torch.tensor([1, 4, 7])
    _, _, _, _, cache = fixture(1, 2, 1)
    args = dict(q=raw, kv_cache=None, swa_k_cache=torch.empty(1, 1, 584, dtype=torch.uint8),
                swa_only=True, topk_indices=None, topk_lens=None,
                swa_indices=torch.zeros(3, 1, dtype=torch.int32),
                swa_lens=torch.ones(3, dtype=torch.int32), swa_ragged_indices=None,
                swa_ragged_indptr=None, topk_ragged_indices=None, topk_ragged_indptr=None,
                attn_sink=None, scale=1.0, head_dim=512, nope_head_dim=448, rope_head_dim=64,
                inv_rope_positions=positions, inv_rope_cos_sin_cache=cache)
    for split in (False, True):
        namespace['_use_split_k_decode'] = lambda: split
        def run(**kw):
            assert (kw['inv_rope_positions'] is not None) == split
            return inverse_reference(raw, positions, cache) if split else raw
        namespace['_rocm_sparse_attn_decode_triton'] = run
        output = torch.empty_like(raw)
        rotated = namespace['rocm_sparse_attn_decode'](**args, output=output)
        assert rotated == (3 if split else 0)
        if not split:
            output.copy_(inverse_reference(output, positions, cache))
        torch.testing.assert_close(output, inverse_reference(raw, positions, cache), atol=0, rtol=0)
    return 2
