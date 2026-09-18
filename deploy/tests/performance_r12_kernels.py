"""R1.2 GPU references, executed only on the target SM80 device."""
import gc
import os
import time

import torch

from performance_r11_kernels import record


def check_dense(report):
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16
    kernel = object.__new__(MarlinMxfp8LinearKernel)
    os.environ['VLLM_AMPERE_DENSE_BF16_MIN_TOKENS'] = '32'
    torch.manual_seed(212)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        for n, k, bias in ((256, 512, False), (320, 512, True), (512, 5120, False)):
            layer = torch.nn.Module()
            layer.output_size_per_partition, layer.input_size_per_partition = n, k
            weight = (torch.randn(n, k, device='cuda').float()*.125).to(torch.float8_e4m3fn)
            scales = torch.full((n, k//32), 127, dtype=torch.uint8, device='cuda')
            ref_weight = dequant_mxfp8_to_bf16(weight, scales)
            ref_bias = torch.randn(n, device='cuda') if bias else None
            layer.register_parameter('weight', torch.nn.Parameter(weight, requires_grad=False))
            layer.register_parameter('weight_scale', torch.nn.Parameter(scales, requires_grad=False))
            layer.register_parameter('bias', torch.nn.Parameter(ref_bias, requires_grad=False) if bias else None)
            kernel.process_weights_after_loading(layer)
            for tokens in (1, 6, 31, 32, 168):
                started = time.monotonic()
                x = torch.randn(tokens, k, device='cuda')*.125
                expected = torch.nn.functional.linear(x, ref_weight, ref_bias)
                actual = kernel.apply_weights(layer, x, layer.bias)
                torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = kernel.apply_weights(layer, x, layer.bias)
                x.mul_(.5)
                graph.replay()
                torch.cuda.synchronize()
                expected = torch.nn.functional.linear(x, ref_weight, ref_bias)
                torch.testing.assert_close(replayed, expected, atol=3e-2, rtol=2e-2)
                del graph, replayed
                record(report, 'dense-bf16-or-marlin', started, tokens=tokens, n=n, k=k, bias=bias)
    finally:
        torch.set_default_dtype(previous_dtype)
    gc.collect()


def check_shared(report):
    from upstream.test_shared_experts_multistream import (
        test_decode_width_runs_on_aux_stream_and_matches_serial,
        test_above_threshold_stays_serial_on_main_stream,
    )
    started = time.monotonic()
    test_decode_width_runs_on_aux_stream_and_matches_serial()
    test_above_threshold_stays_serial_on_main_stream()
    record(report, 'shared-expert-stream-vs-serial', started)


def check_sparse(report):
    from upstream.test_rocm_triton_attn_dsv4 import (
        _pack_fp8_ds_mla_cache, _ragged_from_rows, _ref_sparse_decode_ragged,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _rocm_sparse_attn_decode_ragged_triton as decode
    torch.manual_seed(312)
    block = 64
    main_kv = torch.randn(128, 512, device='cuda', dtype=torch.bfloat16)*.125
    extra_kv = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)*.125
    main = _pack_fp8_ds_mla_cache(main_kv, block, False)
    extra = _pack_fp8_ds_mla_cache(extra_kv, block, False)
    for count in (5, 6, 24):
        started = time.monotonic()
        q = torch.randn(count, 16, 512, device='cuda', dtype=torch.bfloat16)*.125
        main_rows = [list(range(128)) if i%2 else [0, 7, 127] for i in range(count)]
        extra_rows = [list(range(512)) if i%2 else [] for i in range(count)]
        mi, mp = _ragged_from_rows(main_rows, q.device)
        ei, ep = _ragged_from_rows(extra_rows, q.device)
        sink = torch.randn(16, device='cuda')
        kwargs = dict(q=q, main_cache=main, main_indices=mi, main_indptr=mp,
                      scale=512**-.5, attn_sink=sink, nope_head_dim=448, rope_head_dim=64,
                      extra_cache=extra, extra_indices=ei, extra_indptr=ep)
        expected = _ref_sparse_decode_ragged(q, main, main_rows, 512**-.5, sink, block,
                                            extra, extra_rows, False)
        actual = decode(**kwargs)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): replayed = decode(**kwargs)
        q.mul_(.5)
        graph.replay()
        torch.cuda.synchronize()
        expected = _ref_sparse_decode_ragged(q, main, main_rows, 512**-.5, sink, block,
                                            extra, extra_rows, False)
        torch.testing.assert_close(replayed, expected, atol=2e-2, rtol=2e-2)
        del graph, replayed
        record(report, 'sparse-decode-8-warps-reference', started, queries=count, heads=16, graph_mutation=True)
