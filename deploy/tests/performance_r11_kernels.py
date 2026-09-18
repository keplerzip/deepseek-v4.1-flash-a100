"""SM80 numerical checks for R1.1; run only inside the pinned image on GPU.

References use full dense scoring and original stable alignment, not another
copy of the compact/fused implementation. No throughput result is inferred.
"""
import importlib.util
import os
from pathlib import Path
import time

import torch


def record(report, name, started, **details):
    import json
    row = dict(name=name, status='PASS', seconds=time.monotonic()-started, **details)
    report['cases'].append(row)
    print(json.dumps(row), flush=True)


def check_moe(report):
    from vllm.model_executor.layers.fused_moe.stable_moe_align import stable_moe_align
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import _moe_align_block_size_deterministic as dispatch
    path = Path('/deploy/overrides/performance/baseline/vllm/model_executor/layers/fused_moe/moe_align_block_size.py')
    spec = importlib.util.spec_from_file_location('r11_moe_reference', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    reference = baseline._moe_align_block_size_deterministic
    torch.manual_seed(101)
    for n, experts, block, mapped in ((0, 256, 16, False), (6, 256, 16, False),
            (192, 256, 32, False), (336, 256, 64, False), (4096, 256, 16, False),
            (65, 7, 16, True), (257, 32, 32, True),
            (168, 384, 16, True), (140, 128, 16, True)):
        started = time.monotonic()
        ids = torch.randint(-2, experts+3, (n, 8), device='cuda', dtype=torch.int32)
        mapping = None
        if mapped:
            mapping = torch.arange(experts, device='cuda', dtype=torch.int32).flip(0)
            mapping[::3] = -1
        size = ids.numel() + experts * (block-1)
        size = ((size+block-1)//block)*block
        def buffers():
            return (torch.empty(size, device='cuda', dtype=torch.int32),
                    torch.empty(size//block, device='cuda', dtype=torch.int32),
                    torch.empty(1, device='cuda', dtype=torch.int32))
        expected, actual, routed = buffers(), buffers(), buffers()
        reference(ids, experts, block, *expected, mapping)
        stable_moe_align(ids, experts, block, *actual, mapping)
        os.environ['VLLM_FUSED_STABLE_MOE_ALIGN'] = '1'
        dispatch(ids, experts, block, *routed, mapping)
        for result in (actual, routed):
            for got, ref in zip(result, expected):
                torch.testing.assert_close(got, ref, atol=0, rtol=0)
        # Capture and then mutate routing: replay must re-read inputs and not
        # accidentally depend on warmup counts or old graph outputs.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stable_moe_align(ids, experts, block, *actual, mapping)
        ids.fill_(-1)
        if ids.numel():
            ids.view(-1)[::3] = 0
        graph.replay()
        torch.cuda.synchronize()
        reference(ids, experts, block, *expected, mapping)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
        del graph
        record(report, 'stable-moe-align', started, tokens=n, experts=experts,
               block_size=block, expert_map=mapped, graph_mutation=True)


def assert_selection(scores, output, starts=None, sorted_output=False):
    """Validate membership, uniqueness and top-k threshold, allowing score ties."""
    for i in range(scores.shape[0]):
        ids = output[i].long()
        valid = ids >= 0
        selected = ids[valid]
        start = int(starts[i]) if starts is not None else 0
        selected = selected + start
        finite = torch.isfinite(scores[i])
        expected_count = min(output.shape[1], int(finite.sum()))
        assert selected.numel() == expected_count
        assert selected.unique().numel() == expected_count
        if expected_count:
            assert bool(((selected >= 0) & (selected < scores.shape[1])).all())
            chosen = scores[i, selected]
            assert bool(torch.isfinite(chosen).all())
            threshold = scores[i].topk(expected_count).values[-1]
            torch.testing.assert_close(chosen.min(), threshold, atol=2e-4, rtol=2e-5)
        if sorted_output:
            assert bool((ids[:expected_count].diff() >= 0).all())
            assert bool((ids[expected_count:] == -1).all())


def check_indexer(report):
    from vllm.v1.attention.ops import mqa_logits_candidates_triton as compact
    from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton, fp8_paged_mqa_logits_triton
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import apply_candidate_mask
    torch.manual_seed(103)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype, device, d, topk, n = torch.float8_e4m3fn, 'cuda', 128, 512, 2048
    # Candidate slots are deliberately permuted and -1 padded. Includes
    # request offsets, causal tails, empty rows, noncontiguous TP row slices.
    for heads, n, slots in ((8, 2048, 80), (16, 32768, 2048), (64, 2048, 80)):
        started = time.monotonic()
        rows = 6
        q = torch.randn(rows, heads, d, device=device).to(dtype)
        k = torch.randn(n, d, device=device).to(dtype)
        scales = torch.rand(n, device=device)*.1 + .01
        w = torch.rand(rows, heads, device=device)*.1
        domain = (n-1024)//8
        cand = torch.stack([torch.randperm(domain, device=device)[:slots] for _ in range(rows)]).int()
        cand[:, -3:] = -1
        starts = torch.tensor([0, 512, 0, 256, 1024, 128], device=device, dtype=torch.int32)
        ends = starts + torch.tensor([n-1024, n-1100, 0, 7, n-1024, n-1025], device=device, dtype=torch.int32)
        dense = fp8_mqa_logits_triton(q, (k, scales), w, starts, ends, clean_logits=True)
        apply_candidate_mask(dense, starts, ends, cand, 8)
        actual = torch.full((rows, topk), -1, device=device, dtype=torch.int32)
        compact.candidate_topk_prefill(q, (k, scales), w, starts, ends, cand, 8, actual, topk)
        assert_selection(dense, actual, starts, sorted_output=True)
        # Compare scores for ALL candidate slots to the established full path.
        logits = compact._launch(q.bfloat16(), k.view(torch.uint8), scales, w, cand,
                                 starts, ends, None, 1, 8, 1, False)
        positions = starts[:, None] + (cand[:, :, None]*8 + torch.arange(8, device=device)).flatten(1)
        valid = (cand.repeat_interleave(8, dim=1) >= 0) & (positions < ends[:, None])
        expected = dense.gather(1, positions.clamp(0, n-1).long()).masked_fill(~valid, -torch.inf)
        torch.testing.assert_close(logits, expected, atol=2e-4, rtol=2e-5)
        shard = torch.full_like(actual[1:4], -1)
        compact.candidate_topk_prefill(q[1:4], (k, scales), w[1:4], starts[1:4], ends[1:4], cand[1:4], 8, shard, topk)
        torch.testing.assert_close(shard, actual[1:4], atol=0, rtol=0)
        compact.candidate_topk_prefill(q[:0], (k, scales), w[:0], starts[:0], ends[:0], cand[:0], 8, actual[:0], topk)
        record(report, 'compact-prefill', started, heads=heads, candidate_slots=slots, packed_offsets=True, shard_slice=True)

    for block, next_n, native in ((64, 1, True), (128, 6, True), (256, 6, False)):
        started = time.monotonic()
        batch, heads = 3, 16
        n, slots = (32768, 2048) if native and next_n == 6 else (2048, 80)
        rows = batch*next_n
        blocks = n//block
        # Each request has a separate, shuffled page table.
        table = torch.randperm(batch*blocks, device=device).int().reshape(batch, blocks)
        cache = torch.empty((batch*blocks, block, 1, d+4), dtype=torch.uint8, device=device)
        flat = cache.view(batch*blocks, -1)
        k_bytes = torch.randn(batch*blocks, block, d, device=device).to(dtype).view(torch.uint8)
        flat[:, :block*d] = k_bytes.reshape(batch*blocks, -1)
        kv_scales = torch.rand(batch*blocks, block, device=device)*.1 + .01
        flat[:, block*d:] = kv_scales.view(torch.uint8)
        q = torch.randn(batch, next_n, heads, d, device=device).to(dtype)
        w = torch.rand(rows, heads, device=device)*.1
        ends = torch.tensor([n-3, n//2-21, 0], device=device, dtype=torch.int32).view(batch, 1)
        if native:
            ends = (ends - next_n + 1 + torch.arange(next_n, device=device)).clamp(min=0).int()
        cand = torch.stack([torch.randperm(n//8, device=device)[:slots] for _ in range(rows)]).int()
        cand[:, -5:] = -1
        def dense_scores():
            # The pinned low-level paged kernel accepts one bound per group.
            # Express native per-row bounds as separate one-query groups so
            # the reference reads exactly the intended causal context.
            vis = ends.flatten() if native else (ends-next_n+1+torch.arange(next_n, device=device)).clamp(min=0).int().flatten()
            scores = fp8_paged_mqa_logits_triton(q.reshape(rows, 1, heads, d), cache, w, vis,
                table.repeat_interleave(next_n, dim=0), max_model_len=n, clean_logits=True)
            apply_candidate_mask(scores, None, vis, cand, 8, 1)
            return scores
        reference = dense_scores()
        output = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        compact.candidate_topk_decode(q, cache, w, ends, table, cand, 8, output, topk)
        assert_selection(reference, output)
        # Test the exact group/row slicing used by TP decode query sharding.
        shard = torch.empty_like(output[next_n:2*next_n])
        compact.candidate_topk_decode(q[1:2], cache, w[next_n:2*next_n], ends[1:2], table[1:2],
                                     cand[next_n:2*next_n], 8, shard, topk)
        torch.testing.assert_close(shard, output[next_n:2*next_n], atol=0, rtol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compact.candidate_topk_decode(q, cache, w, ends, table, cand, 8, output, topk)
        cand.copy_(cand.roll(7, dims=1))
        graph.replay()
        torch.cuda.synchronize()
        assert_selection(dense_scores(), output)
        del graph
        record(report, 'compact-paged-decode', started, kv_block=block, next_n=next_n,
               native_bounds=native, graph_mutation=True, shard_slice=True)
