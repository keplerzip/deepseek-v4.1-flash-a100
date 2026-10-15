"""Compile installed R1.9 inverse-RoPE kernels for SM80 without a GPU.

Run inside the original runtime image with the selected performance overlay
mounted by build/test_runtime.py. This checks generated cubin/PTX, not GPU
execution, numerical correctness, CUDA Graph behavior or performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sysconfig

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from compile_r11_kernels import load_kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = (Path(sysconfig.get_paths()['purelib'])
              / 'vllm/v1/attention/ops/rocm_aiter_mla_sparse.py')
    names = ['_sparse_attn_decode_reduce_kernel', '_inverse_rope_gptj_kernel']
    module = load_kernels('r19_inverse_rope_compile', source, names)
    report = dict(status='RUNNING', gpu_executed=False, target='sm_80',
                  triton=triton.__version__, source=str(source),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=[],
                  scope='offline SM80 compilation only; GPU accuracy, graph replay and speed require target tests')

    def compile_one(function, pointers, constants):
        signature = {name: 'constexpr' if name in constants else pointers.get(name, 'i32')
                     for name in function.arg_names}
        kernel = triton.compile(ASTSource(function, signature, constexprs=constants),
                                target=GPUTarget('cuda', 80, 32), options={'num_warps': 4})
        assert kernel.asm['cubin'] and '.target sm_80' in kernel.asm['ptx']
        row = dict(kernel=function.__name__, constants=constants, status='COMPILED_SM80',
                   pointer_types=pointers, cubin_bytes=len(kernel.asm['cubin']),
                   ptx_bytes=len(kernel.asm['ptx']), shared_bytes=kernel.metadata.shared)
        report['cases'].append(row)
        print(json.dumps(row), flush=True)

    try:
        pointers = dict(part_m_ptr='*fp32', part_l_ptr='*fp32', part_acc_ptr='*fp32',
                        attn_sink_ptr='*fp32', out_ptr='*bf16', pos_ptr='*i64', cos_sin_ptr='*fp32')
        for splits in (1, 3, 4, 16):
            for fused in (False, True):
                for adaptive in (False, True):
                    for sink in (False, True):
                        compile_one(module._sparse_attn_decode_reduce_kernel, pointers,
                                    dict(HAS_ATTN_SINK=sink, ADAPTIVE_SPLITS=adaptive,
                                         COMB_DIM=512, BLOCK_H=1, NUM_SPLITS=splits,
                                         SPLITS_PAD=triton.next_power_of_2(splits),
                                         FUSE_INV_ROPE=fused, NOPE=448, HALF=32))
        for input_dtype in ('*bf16', '*fp32'):
            compile_one(module._inverse_rope_gptj_kernel,
                        dict(o_ptr=input_dtype, out_ptr='*bf16', pos_ptr='*i64', cos_sin_ptr='*fp32'),
                        dict(NOPE=448, HALF=32, BLOCK_NOPE=512, BLOCK_HALF=32))
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=str(exc))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], target=report['target'],
                          compiled_cases=len(report['cases']), gpu_executed=False)), flush=True)


if __name__ == '__main__':
    main()
