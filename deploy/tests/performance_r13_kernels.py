"""Target-only R1.3 tests of real MXFP8 GEMM and ragged pack CUDA graphs."""
import gc
import time

import torch
from performance_r11_kernels import record
from performance_r13_reference import (
    draft_fixture, assert_loaded, context_method, attention_fixture,
    ragged_reference, assert_ragged,
)


@torch.inference_mode()
def check_dspark_kv(report):
    # Real model dimensions and scale geometry from the fixed model snapshot.
    owner, weights, refs = draft_fixture(device='cuda')
    owner.load_weights(iter(weights))
    assert_loaded(owner, refs)
    linears = [l.attn.fused_wqa_wkv for l in owner.model.layers] + [owner.model.context_wkv_proj]
    for layer in linears:
        layer.quant_method.process_weights_after_loading(layer)
    precompute = context_method()
    graph = None
    try:
        for rows in (1, 5, 6, 31, 32, 168, 512):
            started = time.monotonic()
            x = torch.randn(rows, 5120, dtype=torch.bfloat16, device='cuda')*.0625
            positions = torch.arange(rows, device='cuda')
            slots = [torch.arange(rows, device='cuda') + i*1024 for i in range(3)]
            precompute(owner.model, x, positions, slots)
            def compare():
                for i, layer in enumerate(owner.model.layers):
                    expected = layer.attn.kv_norm(torch.nn.functional.linear(x, refs[i][1][2]))
                    actual, pos, slot = layer.attn.recorded
                    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)
                    torch.testing.assert_close(pos, positions, atol=0, rtol=0)
                    torch.testing.assert_close(slot, slots[i], atol=0, rtol=0)
                    original, _ = layer.attn.fused_wqa_wkv(x)
                    original = layer.attn.kv_norm(original[:, 1280:])
                    torch.testing.assert_close(actual, original, atol=3e-2, rtol=2e-2)
            compare()
            for _ in range(3):
                precompute(owner.model, x, positions, slots)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                precompute(owner.model, x, positions, slots)
            for _ in range(2):
                x.mul_(.75)
                graph.replay()
                torch.cuda.synchronize()
                compare()
            graph = None
            record(report, 'dspark-stacked-kv-native-mxfp8', started, rows=rows,
                   hidden=5120, qrank=1280, kv=512, draft_layers=3, mutated_graph_replays=2)
    finally:
        graph = None
        del owner, linears
        gc.collect()


@torch.inference_mode()
def check_topk_reuse(report):
    from vllm.models.deepseek_v4_1.amd.rocm import compute_global_topk_ragged_indices_and_indptr as pack
    for ratio in (1, 2):
        started = time.monotonic()
        source, consumers, swa, meta = attention_fixture('cuda', pack, ratio=ratio)
        def build():
            out = source.build(swa, meta)
            for consumer in consumers:
                assert consumer.build(swa, meta) is out
            return out
        def compare(out, count):
            expected = ragged_reference(source.topk_indices_buffer[:count], swa.token_to_req_indices[:count],
                                        meta.block_table, meta.block_size//ratio, swa.is_valid_token[:count])
            assert_ragged(out, expected)
        graph = None
        try:
            for _ in range(3):
                compare(build(), 6)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = build()
            for step in range(3):
                source.topk_indices_buffer[:, 0].add_(1)
                meta.block_table.add_(1)
                swa.is_valid_token.logical_not_()
                graph.replay()
                torch.cuda.synchronize()
                compare(captured, 6)
                # Eager batches can evict Python references without breaking
                # storage owned by an earlier captured graph.
                for count in (2, 12):
                    swa.num_decode_tokens = count
                    compare(build(), count)
                swa.num_decode_tokens = 6
            graph.replay()
            torch.cuda.synchronize()
            compare(captured, 6)
            for consumer in consumers:
                consumer._reuse_topk_ragged = False
                compare(consumer.build(swa, meta), 6)
            record(report, 'ragged-topk-reuse-reference', started, compress_ratio=ratio,
                   consumers=4, tokens=[6, 2, 12], graph_replays=4,
                   mutated_inputs=['indices', 'block table', 'validity mask'])
        finally:
            graph = None
            gc.collect()
