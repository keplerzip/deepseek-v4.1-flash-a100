"""Eight-rank custom EP collectives vs mathematical references, with replay."""
import datetime
import gc
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
    assert dist.get_world_size() == 8
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    comm = CudaCommunicator(dist.group.WORLD, torch.device('cuda', rank), unique_name='ep:r12-test')
    assert comm.dp_ag_rs_comm is not None and not comm.dp_ag_rs_comm.disabled
    called = dict(ag=0, rs=0)
    for method, key in [('custom_all_gather', 'ag'), ('custom_reduce_scatter', 'rs')]:
        original = getattr(comm.dp_ag_rs_comm, method)
        def counted(*args, _fn=original, _key=key, **kwargs):
            out = _fn(*args, **kwargs)
            if out is not None:
                called[_key] += 1
            return out
        setattr(comm.dp_ag_rs_comm, method, counted)
    for rows in (1, 6, 24, 32):
        print(json.dumps(dict(rank=rank, phase='EP8 eager and graph', rows=rows)), flush=True)
        x = torch.full((rows, 5120), rank + 1, dtype=torch.bfloat16, device='cuda')
        y = torch.full((rows * 8, 5120), rank + 1, dtype=torch.bfloat16, device='cuda')
        expected_ag = torch.cat([torch.full_like(x, i + 1) for i in range(8)])
        expected_rs = torch.full_like(x, 36)
        for _ in range(2):
            ag = comm.all_gatherv(x)
            rs = comm.reduce_scatterv(y, dim=0)
        torch.cuda.synchronize()
        torch.testing.assert_close(ag, expected_ag, atol=0, rtol=0)
        torch.testing.assert_close(rs, expected_rs, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            ag = comm.all_gatherv(x)
            rs = comm.reduce_scatterv(y, dim=0)
        x.add_(1)
        y.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(ag, expected_ag + 1, atol=0, rtol=0)
        torch.testing.assert_close(rs, expected_rs + 8, atol=0, rtol=0)
        del graph, ag, rs
        gc.collect()  # Release graph references BEFORE communicator teardown.
    assert called['ag'] > 0 and called['rs'] > 0, called
    # Nonuniform DP work and non-BF16 router tensors must retain the NCCL path.
    sizes = list(range(1, 9))
    x = torch.full((sizes[rank], 32), rank + 1, dtype=torch.float32, device='cuda')
    ag = comm.all_gatherv(x, sizes=sizes)
    expected = torch.cat([torch.full((n, 32), i + 1, device='cuda') for i, n in enumerate(sizes)])
    torch.testing.assert_close(ag, expected, atol=0, rtol=0)
    y = torch.full((sum(sizes), 32), rank + 1, device='cuda')
    rs = comm.reduce_scatterv(y, dim=0, sizes=sizes)
    torch.testing.assert_close(rs, torch.full_like(x, 36), atol=0, rtol=0)
    torch.cuda.synchronize()
    dist.barrier()
    comm.destroy()
    dist.destroy_process_group()
    row = dict(status='PASS', rank=rank, custom_calls=called, graph_input_mutation=True,
               nonuniform_fp32_fallback=True, scope='small EP8 operators; not full-model performance')
    (Path(sys.argv[1]) / f'rank-{rank}.json').write_text(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
