"""Real Triton reduce via CPU interpreter; no GPU correctness or speed claim."""
import os
os.environ['TRITON_INTERPRET'] = '1'

import json
import torch
from performance_inv_rope_reference import (
    load_ops, fixture, reduce, inverse_reference,
    check_attention_dispatch, check_decode_wrapper,
)


def main():
    module = load_ops()
    cases = []
    for rows, heads, splits in [(1, 1, 1), (5, 4, 3), (6, 2, 4)]:
        for adaptive in (False, True):
            m, l, acc, positions, cache = fixture(rows, heads, splits)
            if adaptive and splits > 1:
                # Stale NaNs in an inactive split must remain masked.
                m[:, -1].fill_(-torch.finfo(torch.float32).max)
                l[:, -1].zero_()
                acc[:, -1].fill_(float('nan'))
            for mutation in range(2):
                plain = reduce(module, m, l, acc, positions, cache, False, adaptive)
                fused = reduce(module, m, l, acc, positions, cache, True, adaptive)
                expected = inverse_reference(plain, positions, cache)
                torch.testing.assert_close(fused, expected, atol=1e-5, rtol=1e-5)
                assert torch.equal(fused[..., :448], plain[..., :448])
                assert not torch.equal(fused[..., 448:], plain[..., 448:])
                positions.add_(1)
                cache.neg_()
            cases.append(dict(rows=rows, heads=heads, splits=splits,
                              adaptive=adaptive, changed_inputs_checked=True, status='PASS'))
    print(json.dumps(dict(status='PASS', gpu_executed=False, fp32_interpreter_only=True,
                         mixed_dispatch_cases=check_attention_dispatch(),
                         split_and_fallback_wrapper_cases=check_decode_wrapper(), cases=cases)))


if __name__ == '__main__':
    main()
