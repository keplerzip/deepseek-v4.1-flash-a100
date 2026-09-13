"""Eight-rank SM80/UVA/NCCL/graph probe with explicit device binding and progress."""
import datetime
import faulthandler
import json
import os
from pathlib import Path
import time
import traceback

rank = int(os.environ['LOCAL_RANK'])
output = Path(os.environ['PROBE_OUTPUT'])
output.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
last_phase = 'python_start'
stack = (output/f'stack-rank-{rank}.log').open('a', buffering=1)
faulthandler.enable(file=stack, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True, file=stack)


def phase(name, status='RUNNING', **extra):
    global last_phase
    last_phase = name
    row = {'rank': rank, 'pid': os.getpid(), 'phase': name, 'status': status,
           'elapsed_seconds': round(time.monotonic()-started, 2), **extra}
    path = output/f'progress-rank-{rank}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row)+'\n')
    temporary.replace(path)
    print('[GPU rank] '+json.dumps(row), flush=True)


def main():
    phase('import_torch')
    import torch
    import torch.distributed as dist
    phase('bind_device')
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    assert torch.cuda.current_device() == rank
    assert torch.cuda.device_count() == 8
    assert torch.cuda.get_device_capability(device) == (8, 0)
    phase('import_uva_helpers')
    from vllm.utils.platform_utils import is_uva_available
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    assert is_uva_available(), 'Engram CPU offload requires UVA'
    phase('gloo_init')
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=120))
    assert dist.get_rank() == rank and dist.get_world_size() == 8

    def barrier(name):
        phase(name)
        dist.monitored_barrier(timeout=datetime.timedelta(seconds=120), wait_all_ranks=True)

    barrier('all_ranks_ready')
    phase('uva_allocate_and_map')
    cpu = torch.arange(128, dtype=torch.int32, device='cpu', pin_memory=True)
    view = get_accelerator_view_from_cpu_tensor(cpu)
    assert view.device == device, (view.device, device)
    phase('uva_write')
    view.add_(3)
    phase('uva_synchronize')
    torch.cuda.synchronize(device)
    torch.testing.assert_close(cpu, torch.arange(128, dtype=torch.int32)+3)
    phase('bf16_matmul')
    x = torch.ones((64, 64), device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    torch.mm(x, x, out=y)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(y, torch.full_like(y, 64), rtol=0, atol=0)
    barrier('before_nccl_init')
    phase('nccl_init_explicit_device')
    nccl = dist.new_group(backend='nccl', timeout=datetime.timedelta(seconds=120), device_id=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for index in range(3):
            phase(f'nccl_eager_{index}_submit')
            torch.mm(x, x, out=y)
            dist.all_reduce(y, group=nccl)
            phase(f'nccl_eager_{index}_synchronize')
            torch.cuda.synchronize(device)
            torch.testing.assert_close(y, torch.full_like(y, 512), rtol=0, atol=0)
    barrier('before_graph_capture')
    phase('graph_capture')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.mm(x, x, out=y)
        dist.all_reduce(y, group=nccl)
    barrier('before_graph_replay')
    # Poison the output so an absent/no-op replay cannot reuse a warmup PASS.
    phase('graph_output_reset')
    y.zero_()
    torch.cuda.synchronize(device)
    phase('graph_replay')
    graph.replay()
    phase('graph_synchronize')
    torch.cuda.synchronize(device)
    torch.testing.assert_close(y, torch.full_like(y, 512), rtol=0, atol=0)
    barrier('all_ranks_validated')
    # NCCL finalization waits for persistent references held by captured graphs.
    # Release the executable and graph before destroying the communicator.
    phase('graph_release')
    graph.reset()
    del graph
    phase('graph_release_synchronize')
    torch.cuda.synchronize(device)
    barrier('all_graphs_released')
    phase('nccl_destroy')
    dist.destroy_process_group(nccl)
    phase('gloo_destroy')
    dist.destroy_process_group()
    row = {'status': 'PASS', 'rank': rank, 'world_size': 8, 'device_index': rank,
           'gpu': torch.cuda.get_device_name(device), 'sm': [8, 0],
           'torch': torch.__version__, 'cuda': torch.version.cuda, 'nccl': torch.cuda.nccl.version(),
           'uva_cpu_mutation': True, 'bf16_matmul': True, 'nccl_eager_all_reduce': True,
           'nccl_graph_all_reduce': True, 'graph_output_reset_before_replay': True,
           'graph_released_before_nccl_destroy': True,
           'scope': 'small GPU operators only; full model and DSpark require separate acceptance'}
    (output/f'rank-{rank}.json').write_text(json.dumps(row, indent=2)+'\n')
    phase('complete', status='PASS')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    phase('python_start')
    try:
        main()
    except BaseException as exc:
        failed_phase = last_phase
        phase(failed_phase, status='FAIL', error=repr(exc))
        traceback.print_exc()
        # An unhealthy communicator can also hang during graceful teardown.
        # Leave cleanup to the torchrun parent and the independent supervisor.
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        stack.close()
