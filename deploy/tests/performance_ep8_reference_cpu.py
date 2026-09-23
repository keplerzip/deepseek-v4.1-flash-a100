"""Validate the eight-rank EP8 test's tensor contracts without a GPU."""
import json

import torch

from performance_ep8 import nonuniform_case, uniform_case


def check_uniform(rows, world_size):
    cases = [uniform_case(rank, rows, world_size, 'cpu')
             for rank in range(world_size)]
    gathered = torch.cat([case[0] for case in cases])
    reduced = torch.stack([case[1] for case in cases]).sum(dim=0)
    for rank, (x, y, expected_ag, expected_rs) in enumerate(cases):
        assert x.dtype == y.dtype == expected_ag.dtype == expected_rs.dtype == torch.bfloat16
        assert x.shape == expected_rs.shape == (rows, 5120)
        assert y.shape == expected_ag.shape == (rows * world_size, 5120)
        torch.testing.assert_close(gathered, expected_ag, atol=0, rtol=0)
        torch.testing.assert_close(reduced.narrow(0, rank * rows, rows),
                                   expected_rs, atol=0, rtol=0)
    for x, y, _, _ in cases:
        x.add_(1)
        y.add_(1)
    gathered = torch.cat([case[0] for case in cases])
    reduced = torch.stack([case[1] for case in cases]).sum(dim=0)
    for rank, (_, _, expected_ag, expected_rs) in enumerate(cases):
        torch.testing.assert_close(gathered, expected_ag + 1, atol=0, rtol=0)
        torch.testing.assert_close(reduced.narrow(0, rank * rows, rows),
                                   expected_rs + world_size, atol=0, rtol=0)


def check_nonuniform(world_size):
    sizes = list(range(1, world_size + 1))
    cases = [nonuniform_case(rank, sizes, 'cpu') for rank in range(world_size)]
    gathered = torch.cat([case[0] for case in cases])
    reduced = torch.stack([case[1] for case in cases]).sum(dim=0)
    slices = reduced.split(sizes, dim=0)
    for rank, (x, y, expected_ag, expected_rs) in enumerate(cases):
        assert x.dtype == y.dtype == expected_ag.dtype == expected_rs.dtype == torch.float32
        assert x.shape == expected_rs.shape == (sizes[rank], 32)
        assert y.shape == expected_ag.shape == (sum(sizes), 32)
        torch.testing.assert_close(gathered, expected_ag, atol=0, rtol=0)
        torch.testing.assert_close(slices[rank], expected_rs, atol=0, rtol=0)


def main():
    world_size = 8
    rows = (1, 6, 24, 32)
    for n in rows:
        check_uniform(n, world_size)
    check_nonuniform(world_size)
    print(json.dumps({'status': 'PASS', 'gpu_executed': False,
                      'world_size': world_size, 'uniform_rows': rows,
                      'nonuniform_sizes': list(range(1, world_size + 1)),
                      'dtype_contracts': ['BF16 custom', 'FP32 NCCL fallback']}))


if __name__ == '__main__':
    main()
