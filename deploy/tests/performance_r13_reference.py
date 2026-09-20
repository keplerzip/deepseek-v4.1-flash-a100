"""Shared R1.3 checks: execute installed production methods, independent references."""
import __future__
import ast
from contextlib import ExitStack
from pathlib import Path
import random
import re
import sysconfig
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import torch

SITE = Path(sysconfig.get_paths()['purelib'])
DSPARK = 'vllm/models/deepseek_v4_1/nvidia/dspark.py'
ATTENTION = 'vllm/models/deepseek_v4_1/amd/rocm.py'


def production_method(path, cls, name, namespace):
    path = Path(path)
    if not path.is_absolute():
        path = SITE / path
    nodes = next(n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def draft_fixture(device='cpu', enabled=True, path=DSPARK, hidden=5120, head=512, qrank=1280):
    """Real native MXFP8 parameters and merged shard loaders, no full model needed.

    CPU only stubs backend selection; CUDA uses the installed kernel dispatcher.
    Simulated global TP rank 3/4 verifies disable_tp=True keeps complete Wkv.
    """
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod, QuantSpec, CkptCtx, kMxfp8Static, kMxfp8Dynamic,
    )
    quant = NS(online_quantization_config=None, weight_block_size=(32, 32))
    quant.get_quant_method = lambda *a, **kw: ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic), CkptCtx(scale_block_size=(32, 32)))
    owner = torch.nn.Module()
    owner.model = torch.nn.Module()
    owner.model.num_dspark_layers = 3
    owner.model.config = NS(head_dim=head)
    owner.model.confidence_head = None
    owner.model.layers = torch.nn.ModuleList()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with ExitStack() as stack:
            for name, value in [('rank', 3), ('world_size', 4)]:
                stack.enter_context(patch('vllm.model_executor.parameter.get_tensor_model_parallel_'+name,
                                          return_value=value))
            if device == 'cpu':
                stack.enter_context(patch('vllm.model_executor.layers.quantization.modelopt.select_linear_kernel',
                                          return_value=NS(input_quant_key=lambda: None)))
            stack.enter_context(torch.device(device))
            for _ in range(3):
                layer = torch.nn.Module()
                layer.ffn = NS(use_mega_moe=False)
                layer.attn = torch.nn.Module()
                layer.attn.q_lora_rank = qrank
                layer.attn.fused_wqa_wkv = MergedColumnParallelLinear(
                    hidden, [qrank, head], bias=False, quant_config=quant, disable_tp=True)
                layer.attn.kv_norm = torch.nn.RMSNorm(head, eps=1e-6, dtype=torch.bfloat16)
                owner.model.layers.append(layer)
            owner.model.context_wkv_proj = MergedColumnParallelLinear(
                hidden, [head]*3, bias=False, return_bias=False, quant_config=quant, disable_tp=True
            ) if enabled else None
    finally:
        torch.set_default_dtype(previous)
    owner.config = NS(dspark_n_routed_experts=128, n_routed_experts=384, num_attention_heads=64, expert_dtype='fp4')
    owner.linear_scale_name, owner.pad_shared_expert = 'weight_scale', False
    owner.process_weights_after_loading = lambda: None
    namespace = dict(torch=torch, re=re, logger=NS(info_once=lambda *a: None),
                     _EXPERT_SCALE_RE=re.compile(r'\.experts\.'),
                     get_tensor_model_parallel_world_size=lambda: 4,
                     get_tensor_model_parallel_rank=lambda: 3,
                     fused_moe_make_expert_params_mapping=lambda *a, **kw: [])
    cls = 'DSparkDeepseekV4ForCausalLM'
    owner._remap_dspark_name = MethodType(production_method(path, cls, '_remap_dspark_name', namespace), owner)
    owner.load_weights = MethodType(production_method(path, cls, 'load_weights', namespace), owner)
    # Use varying exponent bytes, rather than unit scales that could hide a
    # missing/reordered scale shard. Keep valid entries away from saturation.
    torch.manual_seed(413)
    weights, references = [], []
    for i in range(3):
        pair = []
        for label, width in [('wq_a', qrank), ('wkv', head)]:
            weight = (torch.randn(width, hidden, device=device, dtype=torch.float32)*.0625).to(torch.float8_e4m3fn)
            scale = torch.randint(125, 129, (width//32, hidden//32), dtype=torch.uint8, device=device)
            weights += [(f'mtp.{i}.attn.{label}.weight', weight),
                        (f'mtp.{i}.attn.{label}.scale', scale.view(torch.float8_e8m0fnu))]
            expanded = scale.repeat_interleave(32, dim=0)
            # Independent dequantization in FP32, then round as the BF16 path.
            dequant = (weight.float() * torch.exp2(expanded.float()-127).repeat_interleave(32, dim=1)).bfloat16()
            pair.append((weight, expanded, dequant))
        references.append(pair)
    random.Random(413).shuffle(weights)
    return owner, weights, references


def assert_loaded(owner, refs):
    for i, pair in enumerate(refs):
        layer = owner.model.layers[i].attn.fused_wqa_wkv
        assert layer.tp_rank == 0 and layer.tp_size == 1
        for attr, col in [('weight', 0), ('weight_scale', 1)]:
            actual = getattr(layer, attr).view(torch.uint8)
            expected = torch.cat([p[col].view(torch.uint8) for p in pair])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if owner.model.context_wkv_proj is not None:
        layer = owner.model.context_wkv_proj
        for attr, col in [('weight', 0), ('weight_scale', 1)]:
            torch.testing.assert_close(getattr(layer, attr).view(torch.uint8),
                                       torch.cat([p[1][col].view(torch.uint8) for p in refs]), rtol=0, atol=0)


def context_method(path=DSPARK):
    def insert(attn, kv, positions, slots):
        attn.recorded = (kv.clone(), positions.clone(), slots.clone())
    return production_method(path, 'DSparkDeepseekV4Model', 'precompute_and_store_context_kv',
                             dict(torch=torch, _insert_context_kv=insert))


def ragged_reference(indices, token_to_req, table, block_size, valid):
    """Independent CPU mapping of local logical tokens to physical KV slots."""
    indices, token_to_req, table, valid = [t.detach().cpu() for t in (indices, token_to_req, table, valid)]
    flat, ptr, lengths = [], [0], []
    for row, request, enabled in zip(indices.tolist(), token_to_req.tolist(), valid.tolist()):
        items = [v for v in row if v >= 0] if enabled else []
        flat.extend(int(table[request, v//block_size])*block_size + v % block_size for v in items)
        lengths.append(len(items))
        ptr.append(len(flat))
    return tuple(torch.tensor(v, dtype=torch.int32) for v in (flat, ptr, lengths))


def assert_ragged(actual, expected):
    flat, ptr, lengths = (v.detach().cpu() for v in actual)
    torch.testing.assert_close(ptr, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(lengths, expected[2], rtol=0, atol=0)
    # Capacity beyond the last indptr is intentionally uninitialized.
    torch.testing.assert_close(flat[:int(ptr[-1])], expected[0], rtol=0, atol=0)


def attention_fixture(device, pack, ratio=2, count=6):
    topk = torch.full((16, 512), -1, dtype=torch.int32, device=device)
    for row in range(16):
        length = (row+1)*11
        topk[row, :length] = torch.arange(length, dtype=torch.int32, device=device)
    # Noncompact row stride, contiguous columns as required by the real packer.
    table = torch.arange(96, dtype=torch.int32, device=device).view(3, 32)[:, :16]
    swa = NS(num_decode_tokens=count, num_decodes=3,
             token_to_req_indices=torch.arange(16, dtype=torch.int32, device=device) % 3,
             is_valid_token=torch.tensor([True, False, True, True]*4, device=device))
    meta = NS(block_size=64, block_table=table)
    ns = dict(torch=torch, compute_global_topk_ragged_indices_and_indptr=pack)
    fn = production_method(ATTENTION, 'DeepseekV41ROCMAiterMLAAttention', '_decode_topk_ragged', ns)
    source = NS(_reuse_topk_ragged=True, _topk_ragged_cache={}, _index_source_prefix='source',
                compress_ratio=ratio, topk_indices_buffer=topk)
    context = {'source': source}
    source._static_forward_context = context
    consumers = [NS(**vars(source)) for _ in range(4)]
    for owner in [source, *consumers]:
        owner.build = MethodType(fn, owner)
    return source, consumers, swa, meta
