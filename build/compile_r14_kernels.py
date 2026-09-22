"""Compile the production R1.4 fused pack for SM80 without GPU execution."""
import argparse
import json
from pathlib import Path
import sysconfig

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from compile_r11_kernels import load_kernels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    path = Path(sysconfig.get_paths()['purelib'])/'vllm/models/deepseek_v4_1/amd/rocm.py'
    name = '_pack_global_topk_ragged_fused_kernel'
    fn = getattr(load_kernels('r14_ragged', path, [name]), name)
    rows = []
    for pad in (1, 2, 8, 32, 256, 512, 1024):
        constants = dict(BLOCK_SIZE=128, NTOK_PAD=pad)
        signature = {n: 'constexpr' if n in constants else '*i32' if n.endswith('_ptr') else 'i32'
                     for n in fn.arg_names}
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                target=GPUTarget('cuda', 80, 32), options={'num_warps': 4})
        assert kernel.asm['cubin'] and '.target sm_80' in kernel.asm['ptx']
        rows.append(dict(kernel=name, constants=constants, status='COMPILED_SM80',
                         cubin_bytes=len(kernel.asm['cubin'])))
    report = dict(status='PASS', gpu_executed=False, target='sm_80', triton=triton.__version__, cases=rows)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
