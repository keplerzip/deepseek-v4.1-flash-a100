"""Compile the ragged packing kernels used by R1.3 to SM80, no GPU execution."""
import argparse
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
    path = Path(sysconfig.get_paths()['purelib'])/'vllm/models/deepseek_v4_1/amd/rocm.py'
    module = load_kernels('r13_ragged', path,
                          ['_compute_topk_lens_kernel', '_pack_global_topk_ragged_kernel'])
    rows = []
    for name, key in [('_compute_topk_lens_kernel', 'TRITON_BLOCK_SIZE'),
                      ('_pack_global_topk_ragged_kernel', 'BLOCK_SIZE')]:
        fn = getattr(module, name)
        for tile in ((256, 512, 1024) if key == 'TRITON_BLOCK_SIZE' else (128, 256, 512)):
            sig = {n: 'constexpr' if n == key else '*i1' if n == 'is_valid_token_ptr'
                   else '*i32' if n.endswith('_ptr') else 'i32' for n in fn.arg_names}
            kernel = triton.compile(ASTSource(fn, sig, constexprs={key: tile}),
                                    target=GPUTarget('cuda', 80, 32), options={'num_warps': 4})
            assert kernel.asm['cubin'] and '.target sm_80' in kernel.asm['ptx']
            rows.append(dict(kernel=name, tile=tile, status='COMPILED_SM80', cubin_bytes=len(kernel.asm['cubin'])))
    report = dict(status='PASS', gpu_executed=False, target='sm_80', triton=triton.__version__, cases=rows)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
