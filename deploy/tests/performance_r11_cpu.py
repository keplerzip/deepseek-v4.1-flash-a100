"""Execute production indexer dispatch on CPU with recording kernel stubs.

These tests verify row slicing and fallback control flow, not GPU arithmetic.
"""
import ast
import __future__
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sysconfig
import unittest

import torch

# Import only the production host dispatch functions. vLLM intentionally
# disables its Triton facade on a CPU-only machine; do not fake a CUDA driver.
candidate_file = Path(sysconfig.get_paths()['purelib'])/'vllm/v1/attention/ops/mqa_logits_candidates_triton.py'
host_nodes = [n for n in ast.parse(candidate_file.read_text()).body if isinstance(n, ast.FunctionDef)
              and n.name in {'candidate_logits_enabled', 'use_candidate_logits'}]
exec(compile(ast.Module(body=host_nodes, type_ignores=[]), str(candidate_file), 'exec'), globals())


class DispatchChecks(unittest.TestCase):
    def test_candidate_source_short_context_and_switch_keep_original_path(self):
        cand = torch.empty(4, 2048, dtype=torch.int32)
        for value, expected in (('0', False), ('1', True)):
            os.environ['VLLM_DSV41_CAND_LOGITS'] = value
            self.assertEqual(use_candidate_logits(cand, False, 8, 32768), expected)
            self.assertFalse(use_candidate_logits(cand, True, 8, 32768))
            self.assertFalse(use_candidate_logits(cand, False, 8, 16384))
            self.assertFalse(use_candidate_logits(None, False, 8, 32768))
            self.assertFalse(use_candidate_logits(cand, False, 4, 32768))
        os.environ['VLLM_DSV41_CAND_LOGITS'] = '1'

    def fixture(self, metadata):
        site = Path(sysconfig.get_paths()['purelib'])
        path = site/'vllm/model_executor/layers/sparse_attn_indexer.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sparse_attn_indexer')
        fn.decorator_list = []
        self.calls = []
        def compact(q, kv, w, starts_or_ends, ends_or_table, candidates, block, out, k):
            self.calls.append((q.clone(), w.clone(), candidates.clone()))
            out.copy_(candidates[:, :k])
        def unexpected(*args, **kw):
            raise AssertionError('Dense path must not run after compact top-k')
        platform = NS(fp8_dtype=lambda: torch.bfloat16, is_cuda=lambda: True,
            is_xpu=lambda: False, is_device_capability=lambda _: True)
        workspace = NS(get_simultaneous=lambda *specs: [torch.zeros(*spec[0], dtype=spec[1]) for spec in specs])
        namespace = dict(torch=torch, current_platform=platform,
            get_forward_context=lambda: NS(attn_metadata={'cache':metadata}, cudagraph_runtime_mode='FULL'),
            _resolve_layer_name=lambda s:s, DeepseekV32IndexerMetadata=NS,
            CUDAGraphMode=NS(FULL='FULL'), is_deep_gemm_supported=lambda:False,
            current_workspace_manager=lambda:workspace,
            _gather_workspace_shapes=lambda n,d,dtype,fp4: (((n,d),dtype),((n,1),torch.float32)),
            use_candidate_logits=use_candidate_logits, candidate_topk_prefill=compact, candidate_topk_decode=compact,
            envs=NS(VLLM_DSV4_LOGITS_ROW_CHUNK=1024),
            ops=NS(top_k_per_row_prefill=unexpected, top_k_per_row_decode=unexpected),
            _apply_prefill_candidates=unexpected, _apply_candidate_mask=unexpected,
            _select_candidate_blocks=unexpected, _merge_dcp_topk_global=lambda *a,**k:None,
            kv_cache_as_quant_view=lambda cache,*a:cache, logger=NS(info_once=lambda *a:None),
            indexer_decode_shard_rows=lambda bounds,batch,next_n: tuple(v*next_n for v in (bounds or (0,batch))),
            _all_reduce_decode_topk=lambda buf,*a:buf)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec', flags=__future__.annotations.compiler_flag),namespace)
        return namespace['sparse_attn_indexer']

    def run_case(self, metadata, rows):
        function = self.fixture(metadata)
        ids = torch.arange(rows).view(-1,1).expand(rows,2048).contiguous().int()
        w = torch.arange(rows).view(-1,1).expand(rows,16).contiguous().float()
        q = torch.zeros(rows,16,128)
        out = torch.empty(rows,512,dtype=torch.int32)
        function(torch.zeros(rows,5120), 'cache', torch.zeros(1,128,1,132), q, None, None, w,
                 128, 'ue8m0',512,128,262144,32768,out,True,False,'', candidate_blocks=ids,
                 candidate_block_size=8, candidate_write=False)
        return out

    def test_prefill_chunk_does_not_reenter_dense_topk(self):
        chunk=NS(cu_seqlen_ks=torch.zeros(3,dtype=torch.int32),cu_seqlen_ke=torch.ones(3,dtype=torch.int32),
                 local_cu_seq_lens=torch.tensor([0,32768]),max_local_total_seq_lens=32768,
                 local_total_seq_lens=32768,skip_kv_gather=True, token_start=2,token_end=5,shard_row_counts=None)
        meta=NS(slot_mapping=torch.arange(6),num_decodes=0,num_prefills=1,num_decode_tokens=0,prefill=NS(chunks=[chunk]))
        out=self.run_case(meta,6)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][1][:,0].tolist(),[2,3,4])
        self.assertEqual(out[:,0].tolist(),[-1,-1,2,3,4,-1])

    def test_decode_preserves_tp_group_and_candidate_row_slices(self):
        dec=NS(decode_lens=torch.tensor([6]*4),requires_padding=False,shard_bounds=(1,3),
               seq_lens=torch.full((4,6),32768),block_table=torch.zeros(4,256),
               max_seq_len=32768,global_seq_lens=None)
        meta=NS(slot_mapping=torch.arange(24),num_decodes=4,num_prefills=0,num_decode_tokens=24,decode=dec)
        out=self.run_case(meta,24)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(list(self.calls[0][0].shape),[2,6,16,128])
        self.assertEqual(self.calls[0][1][:,0].tolist(),list(range(6,18)))
        self.assertEqual(out[:,0].tolist(),[-1]*6+list(range(6,18))+[-1]*6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
