"""Run real Triton pack code via its CPU interpreter; not a GPU result."""
import os
os.environ['TRITON_INTERPRET'] = '1'

import json
from performance_r14_reference import load_pack, ragged_inputs, mutate_inputs
from performance_r13_reference import ragged_reference, assert_ragged


def main():
    module = load_pack()
    cases = []
    for rows, width, padded in [(0, 512, False), (3, 0, False), (1, 1, False),
                                 (2, 17, False), (6, 512, False), (6, 512, True),
                                 (168, 512, False), (336, 129, False),
                                 (1024, 17, False), (1025, 17, False)]:
        args = ragged_inputs(rows, width, all_padding=padded)
        for _ in range(2):
            actual = module.compute_global_topk_ragged_indices_and_indptr(*args)
            assert_ragged(actual, ragged_reference(*args))
            mutate_inputs(args)
        row = dict(rows=rows, width=width, initially_all_padding=padded,
                   fused=0 < rows <= 1024 and width > 0, changed_inputs_checked=True, status='PASS')
        cases.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(status='PASS', scope='production Triton CPU interpreter vs independent slot mapping',
                          gpu_executed=False, cases=cases)))


if __name__ == '__main__':
    main()
