"""Target-only tests: integer equality and CUDA graph updates, not speed claims."""
import gc
from pathlib import Path
import time
from types import SimpleNamespace as NS

import torch
from performance_r11_kernels import record
from performance_r13_reference import ragged_reference, assert_ragged
from performance_r14_reference import load_pack, ragged_inputs, mutate_inputs, ROCM, SITE
from performance_r14_cpu import prefix_function


@torch.inference_mode()
def check_decode_metadata(report):
    from vllm.models.deepseek_v4_1.amd.rocm import compute_global_topk_ragged_indices_and_indptr as pack
    baseline_path = Path('/deploy/overrides/performance/baseline')/ROCM
    old = load_pack(baseline_path, 'r14_r13_pack').compute_global_topk_ragged_indices_and_indptr
    for rows, width, padded in [(0, 512, False), (3, 0, False), (1, 1, False),
                                 (2, 17, False), (6, 512, False), (6, 512, True),
                                 (168, 512, False), (336, 512, False),
                                 (1024, 512, False), (1025, 512, False)]:
        started = time.monotonic()
        args = ragged_inputs(rows, width, device='cuda', all_padding=padded)
        graph = None
        try:
            for _ in range(3):
                assert_ragged(pack(*args), ragged_reference(*args))
            # The R1.3 empty-input reshape is undefined; R1.4 handles it.
            if rows:
                assert_ragged(old(*args), ragged_reference(*args))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = pack(*args)
            addresses = [t.data_ptr() for t in captured]
            for _ in range(2):
                mutate_inputs(args)
                graph.replay()
                torch.cuda.synchronize()
                assert_ragged(captured, ragged_reference(*args))
                assert addresses == [t.data_ptr() for t in captured]
                if rows:
                    assert_ragged(old(*args), ragged_reference(*args))
            record(report, 'r14-decode-metadata', started, rows=rows, width=width,
                   initially_all_padding=padded, exact_integer_reference=True,
                   compared_r13=bool(rows), mutated_graph_replays=2)
        finally:
            graph = None
            gc.collect()
    # Exercise the changed persistent-buffer write from the real SWA builder
    # on device, then replay with a different sign pattern and stable addresses.
    for count in (6, 168, 1024):
        started = time.monotonic()
        common = NS(seq_lens=torch.tensor([4096], device='cuda'),
                    seq_lens_cpu_upper_bound=torch.tensor([4096]),
                    query_start_loc=torch.tensor([0, count], device='cuda'),
                    query_start_loc_cpu=torch.tensor([0, count]),
                    block_table_tensor=torch.zeros((1, 32), dtype=torch.int32, device='cuda'),
                    slot_mapping=torch.arange(count, device='cuda') - count // 2,
                    token_to_req_indices=lambda buf: buf[:count])
        owner = NS(decode_threshold=6, token_to_req_indices=torch.zeros(count, dtype=torch.int32, device='cuda'),
                   is_valid_token=torch.ones(count+4, dtype=torch.bool, device='cuda'))
        fn = prefix_function(SITE/'vllm/v1/attention/backends/mla/sparse_swa.py',
                             'DeepseekSparseSWAMetadataBuilder', 'swa', (1, 0, count, 0))
        for _ in range(3):
            fn(owner, 0, common)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                out = fn(owner, 0, common)['is_valid_token']
            for _ in range(2):
                common.slot_mapping.neg_().sub_(1)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, common.slot_mapping >= 0, rtol=0, atol=0)
                assert owner.is_valid_token[count:].all()
                assert out.data_ptr() == owner.is_valid_token.data_ptr()
            record(report, 'r14-swa-validity-graph', started, rows=count, mutated_graph_replays=2)
        finally:
            graph = None
            gc.collect()
