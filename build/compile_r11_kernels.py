"""Compile R1.1 Triton kernels to SM80 cubin without executing a GPU kernel."""
import argparse
import ast
import json
from pathlib import Path
import sys
import sysconfig
from types import ModuleType

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def load_kernels(name, path, names, **dependencies):
    # Use production function bodies verbatim, with real Triton. vLLM's
    # platform detection disables its facade without a GPU; no fake driver
    # or execution is needed to compile an explicit GPUTarget.
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(triton=triton, tl=tl, **dependencies)
    sys.modules[name] = module
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    site = Path(sysconfig.get_paths()['purelib'])/'vllm'
    decode = load_kernels('r11_fp8_decode', site/'v1/attention/ops/mqa_logits_triton.py', ['_decode_e4m3fn_bf16_lut'])
    idx = load_kernels('r11_indexer', site/'v1/attention/ops/mqa_logits_candidates_triton.py',
                       ['_candidate_logits_kernel', '_map_topk_kernel'],
                       _decode_e4m3fn_bf16_lut=decode._decode_e4m3fn_bf16_lut)
    moe = load_kernels('r11_moe', site/'model_executor/layers/fused_moe/stable_moe_align.py',
                       ['_maximum', '_sort_count', '_scan_counts', '_bases_initialize', '_scatter_stable'])
    rows = []
    def compile_one(fn, pointers, const, **options):
        signature = {name: ('constexpr' if name in const else pointers.get(name, 'i32')) for name in fn.arg_names}
        kernel = triton.compile(ASTSource(fn, signature, constexprs=const),
                                target=GPUTarget('cuda', 80, 32), options={'num_warps': 4, **options})
        assert kernel.asm['cubin'] and '.target sm_80' in kernel.asm['ptx']
        row = dict(kernel=fn.__name__, constants=const, status='COMPILED_SM80', cubin_bytes=len(kernel.asm['cubin']))
        rows.append(row)
        print(json.dumps(row), flush=True)
    pointers = dict(q_ptr='*bf16', k_ptr='*u8', k_scale_ptr='*fp32', weights_ptr='*fp32', fp8_lut_ptr='*bf16',
                    cand_ptr='*i32', start_ptr='*i32', end_ptr='*i32', block_tables_ptr='*i32', logits_ptr='*fp32')
    for heads in (8, 16, 64):
        for block, next_n, start in ((1<<30, 1, True), (64, 1, False), (128, 6, False), (256, 6, False)):
            compile_one(idx._candidate_logits_kernel, pointers,
                dict(num_heads=heads, head_dim=128, CAND_BLOCK=8, KV_BLOCK=block, NEXT_N=next_n,
                     HAS_START=start, BLOCK_H=max(16, heads), BLOCK_D=128, TILE=16, BLOCK_N=128), num_stages=2)
    compile_one(idx._map_topk_kernel,
                dict(idx_ptr='*i32', logits_ptr='*fp32', cand_ptr='*i32', out_ptr='*i32'),
                dict(CAND_BLOCK=8, TOPK=512, BLOCK_K=512))
    for experts, mapped in ((256, 0), (32, 32), (7, 7)):
        compile_one(moe._sort_count, dict(ids='*i32', expert_map='*i32', ordered='*i64', counts='*i32'),
                    dict(E=experts, MAP_SIZE=mapped, TILE=256, BINS=triton.next_power_of_2(experts+1)))
        compile_one(moe._scan_counts, dict(counts='*i32', offsets='*i32', totals='*i32'), dict(E=experts, BLOCK=128))
        compile_one(moe._bases_initialize, dict(totals='*i32', bases='*i32', sorted_ids='*i32', expert_ids='*i32', padded_total='*i32'),
                    dict(E=experts, PAD=16, EXPERT_BLOCK=triton.next_power_of_2(experts), BLOCK=1024))
        compile_one(moe._scatter_stable, dict(ordered='*i64', offsets='*i32', bases='*i32', sorted_ids='*i32', expert_ids='*i32'),
                    dict(E=experts, PAD=16, TILE=256))
    args.output.write_text(json.dumps(dict(status='PASS', triton=triton.__version__, target='sm_80',
        gpu_executed=False, cases=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
