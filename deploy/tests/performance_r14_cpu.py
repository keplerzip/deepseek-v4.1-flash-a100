"""Verify actual changed builder prefixes on CPU, including ATen operation counts."""
import __future__
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from performance_r14_reference import SITE, metadata_prefix


class Operations(TorchDispatchMode):
    def __init__(self):
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


def prefix_function(path, cls, kind, split):
    namespace = dict(torch=torch, split_decodes_and_prefills=lambda *a, **k: split,
                     get_compressed_slot_mapping=lambda *a, out: out[:a[0]])
    exec(compile(metadata_prefix(path, cls, kind), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace['build']


def main():
    cases = []
    name = 'vllm/v1/attention/backends/mla/indexer.py'
    for ratio in (1, 4, 8):
        for prefill in (False, True):
            split = (1, 1, 1, 4) if prefill else (2, 0, 2, 0)
            common = NS(num_reqs=2, num_actual_tokens=sum(split[2:]),
                        query_start_loc=torch.tensor([0, 1, 5] if prefill else [0, 1, 2]),
                        query_start_loc_cpu=torch.tensor([0, 1, 5] if prefill else [0, 1, 2]),
                        seq_lens=torch.tensor([1025, 2051], dtype=torch.int32),
                        slot_mapping=torch.arange(sum(split[2:])),
                        block_table_tensor=torch.arange(32).view(2, 16), dcp_local_seq_lens=None)
            owner = NS(decode_threshold=6, use_flattening=True, supports_varlen=True,
                       use_pcp=False, compress_ratio=ratio, kernel_block_size=None,
                       pcp_world_size=1, kv_cache_spec=NS(num_states=16),
                       compressed_slot_mapping_buffer=torch.zeros(16, dtype=torch.int64))
            out, ops = [], []
            for root in (Path('/deploy/overrides/performance/baseline'), SITE):
                fn = prefix_function(root/name, 'DeepseekV32IndexerMetadataBuilder', 'indexer', split)
                with Operations() as trace:
                    result = fn(owner, 0, common)
                out.append(result)
                ops.append(trace.counts)
            assert ops[0]['aten::floor_divide'] == int(ratio > 1)
            assert ops[1]['aten::floor_divide'] == int(ratio > 1 and prefill)
            if prefill:
                torch.testing.assert_close(out[1]['compressed_seq_lens'], out[0]['compressed_seq_lens'], rtol=0, atol=0)
            else:
                assert 'compressed_seq_lens' not in out[1]
            for key in ('compressed_slot_mapping', 'indexer_block_table'):
                torch.testing.assert_close(out[1][key], out[0][key], rtol=0, atol=0)
            cases.append(dict(kind='indexer-builder-prefix', ratio=ratio, prefill=prefill,
                              old_divisions=ops[0]['aten::floor_divide'],
                              new_divisions=ops[1]['aten::floor_divide'], status='PASS'))

    name = 'vllm/v1/attention/backends/mla/sparse_swa.py'
    for count in (0, 1, 6, 33):
        common = NS(seq_lens=torch.tensor([1025]), seq_lens_cpu_upper_bound=torch.tensor([1025]),
                    query_start_loc=torch.tensor([0, count]), query_start_loc_cpu=torch.tensor([0, count]),
                    block_table_tensor=torch.zeros((1, 32), dtype=torch.int32),
                    slot_mapping=torch.arange(count) - count // 2,
                    token_to_req_indices=lambda buf: buf[:count])
        counts, outputs = [], []
        for root in (Path('/deploy/overrides/performance/baseline'), SITE):
            owner = NS(decode_threshold=6, token_to_req_indices=torch.zeros(count+4, dtype=torch.int32),
                       is_valid_token=torch.ones(count+4, dtype=torch.bool))
            fn = prefix_function(root/name, 'DeepseekSparseSWAMetadataBuilder', 'swa', (1, 0, count, 0))
            with Operations() as trace:
                result = fn(owner, 0, common)
            assert result['is_valid_token'].data_ptr() == owner.is_valid_token[:count].data_ptr()
            assert owner.is_valid_token[count:].all()
            outputs.append(result['is_valid_token'])
            counts.append(trace.counts)
        torch.testing.assert_close(outputs[1], common.slot_mapping >= 0, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        assert counts[0]['aten::copy_'] == 1 and counts[1]['aten::copy_'] == 0
        cases.append(dict(kind='swa-builder-prefix', tokens=count, persistent_output=True,
                          old_copies=1, new_copies=0, status='PASS'))
    print(json.dumps(dict(status='PASS', gpu_executed=False, whole_builder_executed=False, cases=cases)))


if __name__ == '__main__':
    main()
