"""Target SM80 inverse-RoPE fusion and mutable CUDA graph regression."""
import gc
import time

import torch
from performance_r11_kernels import record
from performance_inv_rope_reference import load_ops, fixture, reduce, inverse_reference


@torch.inference_mode()
def check_inverse_rope(report):
    module = load_ops()
    for rows, heads, splits in [(1, 16, 4), (6, 16, 4), (32, 16, 3), (192, 16, 1)]:
        started = time.monotonic()
        m, l, acc, positions, cache = fixture(rows, heads, splits, 'cuda')
        def check(fused):
            plain = reduce(module, m, l, acc, positions, cache, False, dtype=torch.bfloat16)
            expected = inverse_reference(plain, positions, cache)
            torch.testing.assert_close(fused, expected, atol=2e-2, rtol=2e-2)
            assert torch.equal(fused[..., :448], plain[..., :448])
            assert not torch.equal(fused[..., 448:], plain[..., 448:])
        for _ in range(3):
            check(reduce(module, m, l, acc, positions, cache, True, dtype=torch.bfloat16))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured = reduce(module, m, l, acc, positions, cache, True, dtype=torch.bfloat16)
            for _ in range(2):
                positions.add_(1)
                cache.neg_()
                graph.replay()
                torch.cuda.synchronize()
                check(captured)
            # In-place fallback must preserve strides, not allocate output.
            raw = torch.randn(rows, heads * 2, 512, device='cuda', dtype=torch.bfloat16)[:, ::2]
            expected = inverse_reference(raw, positions, cache)
            address = raw.data_ptr()
            module.rocm_inverse_rope_rows_(raw, positions, cache, 64)
            assert raw.data_ptr() == address
            torch.testing.assert_close(raw, expected, atol=2e-2, rtol=2e-2)
            record(report, 'inverse-rope-reduce-epilogue', started, rows=rows,
                   heads=heads, splits=splits, unchanged_nope=True,
                   mutated_graph_replays=2, strided_inplace_fallback=True)
        finally:
            graph = None
            gc.collect()
