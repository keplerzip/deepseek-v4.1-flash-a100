"""Execute production R1.2 host control paths in the unchanged offline image."""
import __future__
import ast
import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

SITE = Path(sysconfig.get_paths()['purelib'])


def method(path, cls, name, namespace):
    tree = ast.parse((SITE / path).read_text())
    nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
    fn = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SITE / path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def main():
    from vllm.config.compilation import CUDAGraphMode
    ns = dict(np=np, torch=torch, CUDAGraphMode=CUDAGraphMode, build_attn_metadata=lambda **kw: kw)
    fn = method('vllm/v1/worker/gpu/spec_decode/speculator.py', 'DraftModelSpeculator', '_build_attn_metadata', ns)
    spec = NS(max_model_len=262144, draft_max_seq_len=262144, attn_groups=[], kv_cache_config=None,
              block_tables=NS(input_block_tables=[torch.zeros(6, 8)], slot_mappings=torch.arange(96).view(1, -1), cp_size=1),
              input_buffers=NS(query_start_loc=torch.zeros(7, dtype=torch.int32), seq_lens=torch.ones(6, dtype=torch.int32), positions=torch.arange(96)))
    for mode, tokens in ((CUDAGraphMode.FULL, 30), (CUDAGraphMode.PIECEWISE, 10), (CUDAGraphMode.NONE, 10)):
        meta = fn(spec, 2, NS(cg_mode=mode, num_reqs=6, num_tokens=30), np.array([0, 5, 10], dtype=np.int32), torch.tensor([100, 101]), 1)
        assert meta['num_tokens'] == tokens and meta['positions'].numel() == tokens
        assert meta['slot_mappings'].shape == (1, tokens)
        assert meta['query_start_loc_cpu'].tolist() == [0, 5, 10, 10, 10, 10, 10]
        assert meta['seq_lens_cpu_upper_bound'].tolist() == [101, 102, 0, 0, 0, 0]

    apply_dense = method('vllm/model_executor/kernels/linear/mxfp8/marlin.py', 'MarlinMxfp8LinearKernel', 'apply_weights', dict(torch=torch))
    layer = NS(ampere_bf16_min_tokens=32, ampere_bf16_weight=torch.randn(8, 16).bfloat16(),
               ampere_bf16_bias=torch.arange(8).bfloat16(), weight=None, weight_scale=None, workspace=None,
               output_size_per_partition=8, input_size_per_partition=16)
    for rows in (1, 6, 31, 32, 168):
        x = torch.randn(rows, 16).bfloat16()
        # The Marlin bias has been permuted; BF16 must use the saved original.
        fallback = Mock(return_value='marlin')
        with patch.dict(sys.modules, {'vllm.model_executor.layers.quantization.utils.marlin_utils_fp8':NS(apply_mxfp8_marlin_linear=fallback)}):
            out = apply_dense(None, layer, x, layer.ampere_bf16_bias.flip(0))
            if rows < 32:
                assert out == 'marlin' and fallback.call_count == 1
            else:
                assert fallback.call_count == 0
                torch.testing.assert_close(out, torch.nn.functional.linear(x, layer.ampere_bf16_weight, layer.ampere_bf16_bias))

    @dataclass
    class EPLB:
        num_redundant_experts: int = 8
    @dataclass
    class Parallel:
        enable_eplb: bool
        eplb_config: EPLB
        tensor_parallel_size: int = 4
        pipeline_parallel_size: int = 1
        enable_elastic_ep: bool = False
        data_parallel_size: int = 2
        enable_expert_parallel: bool = True
    fn = method('vllm/v1/worker/gpu/spec_decode/dspark/utils.py', None, '_get_dspark_parallel_config',
                dict(replace=replace, logger=NS(warning_once=lambda *a: None)))
    target = Parallel(True, EPLB())
    draft = fn(target, 4)
    assert target.enable_eplb and target.eplb_config.num_redundant_experts == 8
    assert not draft.enable_eplb and draft.eplb_config.num_redundant_experts == 0
    assert draft.data_parallel_size == 2 and draft.enable_expert_parallel
    from vllm.model_executor.models.interfaces import _to_eplb_transport_view
    scales = torch.arange(32, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert _to_eplb_transport_view(scales).data_ptr() == scales.data_ptr()
    assert _to_eplb_transport_view(scales).dtype == torch.uint8

    from vllm.parser.deepseek_v41 import deepseek_v41_config, _PARAM_RE
    for thinking in (True, False):
        config = deepseek_v41_config(thinking)
        assert '<｜DSML｜calls>' in config.terminals['TOOL_START']
        assert '<｜DSML｜tool_calls>' in config.terminals['TOOL_START']
        assert '<｜DSML｜function_calls>' not in config.terminals['TOOL_START']
    for space in ('', ' '):
        text = '<｜DSML｜'+space+'parameter name="x" string="false">12</｜DSML｜'+space+'parameter>'
        assert _PARAM_RE.findall(text) == [('x', 'false', '12')]

    response_generate = method('vllm/entrypoints/openai/responses/serving.py', 'OpenAIServingResponses',
                               '_generate_with_builtin_tools', dict(cast=lambda _, value:value))
    async def check_responses_routing():
        calls = []
        async def generate(*args, **kw):
            calls.append(kw)
            yield 'first-result'
        owner = NS(model_config=NS(max_model_len=262144), engine_client=NS(generate=generate), _log_inputs=lambda *a,**k:None)
        context = NS(append_output=lambda _:None, need_builtin_tool_call=lambda:False)
        for rank in (0, 1):
            rows = [x async for x in response_generate(owner, 'test', {}, None, context, data_parallel_rank=rank)]
            assert rows == [context]
        assert [c['data_parallel_rank'] for c in calls] == [0, 1]
    asyncio.run(check_responses_routing())

    # Real CLI schema in the pinned image, without constructing a GPU engine.
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform
    vllm.platforms._current_platform = CpuPlatform()
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    sys.path.insert(0, '/deploy/scripts')
    from offline_ops import resolve
    config, argv = resolve(json.loads(Path('/deploy/configs/runtime.json').read_text()), 56)
    parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
    assert parsed.data_parallel_size == parsed.data_parallel_size_local == 2
    assert parsed.tensor_parallel_size == 4 and parsed.max_num_seqs == 28
    assert parsed.enable_expert_parallel and parsed.enable_eplb
    assert not parsed.language_model_only and parsed.max_model_len == 262144
    print(json.dumps(dict(status='PASS', gpu_executed=False,
        checks=['DP graph/eager padding control', 'dense dispatch including permuted bias',
                'EPLB draft isolation and E8M0 storage identity', 'DSML variants', 'Responses engine DP routing',
                'original image TP4 DP2 EP8 multimodal CLI schema'])))


if __name__ == '__main__': main()
