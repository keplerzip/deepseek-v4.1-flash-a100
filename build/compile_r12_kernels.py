"""AOT-compile the actual 8-warp sparse decode kernel for SM80, no execution."""
import argparse
import json
from pathlib import Path
import sysconfig

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from compile_r11_kernels import load_kernels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    site = Path(sysconfig.get_paths()['purelib']) / 'vllm/v1/attention/ops'
    fp8 = load_kernels('r12_lut', site/'fp8_sm80.py', ['_decode_fp8_lut'], _NATIVE_FP8_CAST=tl.constexpr(False))
    module = load_kernels('r12_sparse', site/'rocm_aiter_mla_sparse.py',
                          ['_sparse_attn_decode_partial_kernel'], _decode_fp8_lut=fp8._decode_fp8_lut)
    fn = module._sparse_attn_decode_partial_kernel
    pointers = dict(q_ptr='*bf16', main_cache_ptr='*u8', main_indices_ptr='*i32', main_indptr_ptr='*i32',
                    extra_cache_ptr='*u8', extra_indices_ptr='*i32', extra_indptr_ptr='*i32', part_m_ptr='*fp32',
                    part_l_ptr='*fp32', part_acc_ptr='*fp32', fp8_lut_ptr='*bf16', scale='fp32')
    rows = []
    for extra in (True, False):
        for splits in (1, 8, 16):
            constants = dict(HAS_EXTRA=extra, NOPE_DIM=448, NOPE_BLOCK=512, ROPE_DIM=64,
                             IS_FNUZ_MAIN=False, IS_FNUZ_EXTRA=False, BLOCK_H=16, BLOCK_K=32,
                             NUM_SPLITS=splits, NUM_STAGES=1)
            signature = {k:'constexpr' if k in constants else pointers.get(k, 'i32') for k in fn.arg_names}
            kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                    target=GPUTarget('cuda', 80, 32), options=dict(num_warps=8, num_stages=1))
            assert kernel.asm['cubin'] and '.target sm_80' in kernel.asm['ptx']
            row = dict(extra_cache=extra, splits=splits, num_warps=8, cubin_bytes=len(kernel.asm['cubin']), status='COMPILED_SM80')
            rows.append(row)
            print(json.dumps(row), flush=True)
    a.output.write_text(json.dumps(dict(status='PASS', target='sm_80', gpu_executed=False, cases=rows), indent=2)+'\n')


if __name__ == '__main__': main()
