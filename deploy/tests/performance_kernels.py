"""Actual CUDA regression for the SM80 mHC port, without model weights."""
import argparse
import json
from pathlib import Path
import time

import torch


def check_mhc(report):
    from vllm.model_executor.kernels.mhc import dsv41_sm80 as mhc
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(7)
    device = 'cuda'
    h, hc = 5120, 4
    fn = torch.randn(24, h * hc, device=device) * .02
    scale = torch.tensor([.5, .25, 1.], device=device)
    base = torch.randn(24, device=device)
    weight = torch.empty(h, dtype=torch.bfloat16, device=device).uniform_(.5, 1.5)
    mix_args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2., 20)
    for n in (0, 1, 5, 6, 8, 16, 17, 192, 336):
        for carried in (False, True):
            started = time.monotonic()
            x = torch.randn(n, h, dtype=torch.bfloat16, device=device)
            r = torch.randn(n, hc, h, dtype=torch.bfloat16, device=device)
            post = torch.rand(n, hc, 1, device=device)
            comb = torch.rand(n, hc, hc, device=device)
            pre = torch.rand(n, hc, device=device) if carried else None
            args = (x, r, post, comb, *mix_args, pre, weight, 1e-6)
            actual = mhc.mhc_fused_post_pre_delayed_tilelang(*args, True)
            assert actual[5].shape == (n, h)
            if n:
                original_post = mhc_post_tilelang(x, r, post, comb)
                original = mhc_pre_delayed_tilelang(original_post, *mix_args,
                    pre_mix=pre, norm_weight=weight, norm_eps=1e-6)
                torch.testing.assert_close(actual[0], original_post, atol=0, rtol=0)
                torch.testing.assert_close(actual[3], original[2], atol=0, rtol=0)
                torch.testing.assert_close(actual[5], original_post.mean(1), atol=0, rtol=0)
                if mhc.mhc_fused_post_pre_split_config(n, h, hc) is not None:
                    fp32_post = torch.einsum('tij,tih->tjh', comb, r.float()) + post * x.float().unsqueeze(1)
                    reference = mhc_pre_delayed_torch(fp32_post.bfloat16(), *mix_args,
                        pre_mix=pre, x=fp32_post.flatten(1))
                    for got, ref in zip((actual[1], actual[2], actual[4]),
                                        (reference[0], reference[1], reference[3])):
                        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
                else:
                    for got, ref in zip((actual[1], actual[2], actual[4]),
                                        (original[0], original[1], original[3])):
                        torch.testing.assert_close(got, ref, atol=0, rtol=0)
                without_aux = mhc.mhc_fused_post_pre_delayed_tilelang(*args, False)
                for got, ref in zip(actual[:5], without_aux[:5]):
                    torch.testing.assert_close(got, ref, atol=0, rtol=0)
                bare = mhc.mhc_fused_post_pre_delayed_tilelang(
                    x, r, post, comb, *mix_args, pre, None, 1e-6, True)
                torch.testing.assert_close(bare[5], original_post.mean(1), atol=0, rtol=0)
                # Warm before capture; release the graph before process exit.
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = mhc.mhc_fused_post_pre_delayed_tilelang(*args, True)
                graph.replay()
                torch.cuda.synchronize()
                for got, ref in zip(replay, actual):
                    torch.testing.assert_close(got, ref, atol=0, rtol=0)
                del graph, replay
            row = {'tokens': n, 'carried_pre_mix': carried, 'status': 'PASS',
                   'seconds': time.monotonic() - started}
            report['cases'].append(row)
            print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mhc', type=int, choices=(0, 1), default=1)
    p.add_argument('--indexer', type=int, choices=(0, 1), default=1)
    p.add_argument('--moe-align', type=int, choices=(0, 1), default=1)
    a = p.parse_args()
    report = {'status': 'RUNNING', 'cases': [], 'full_model_tested': False,
              'groups': {'mhc': a.mhc, 'indexer': a.indexer, 'moe_align': a.moe_align}}
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
            raise RuntimeError('This test must execute on an SM80 GPU; CPU is not a passing result')
        from performance_r11_kernels import check_moe, check_indexer
        from performance_r12_kernels import check_dense, check_shared, check_sparse
        for enabled, name, check in ((a.indexer, 'compact-indexer', check_indexer),
                                    (a.moe_align, 'stable-moe-align', check_moe),
                                    (a.mhc, 'mhc', check_mhc),
                                    (True, 'dense-bf16', check_dense),
                                    (True, 'shared-expert-overlap', check_shared),
                                    (True, 'sparse-decode-8-warps', check_sparse)):
            if enabled:
                print('[R1.2 kernels] Starting '+name, flush=True)
                check(report)
        report.update(status='PASS', gpu=torch.cuda.get_device_name(),
                      scope='Enabled SM80 kernel reference checks and graph replay; no model speed result')
    except Exception as exc:
        report.update(status='FAIL', error=str(exc))
        raise
    finally:
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
