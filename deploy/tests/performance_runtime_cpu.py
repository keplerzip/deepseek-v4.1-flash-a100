"""Installed overlay checks on a CPU build host; GPU math is tested separately."""
import ast
import importlib.util
import json
from pathlib import Path
import sysconfig
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.model_executor.kernels.mhc import dsv41_sm80 as mhc


def check_decoder_flow(site):
    """Execute old/new production forward bodies with CPU reference operators."""
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    def post(x, r, p, c):
        return (torch.einsum('tij,tih->tjh', c, r.float()) + p * x.float().unsqueeze(1)).bfloat16()

    def pre(*args, norm_weight, norm_eps, **kw):
        p, c, x, carried = mhc_pre_delayed_torch(*args, **kw)
        f = x.float()
        x = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + norm_eps) * norm_weight).bfloat16()
        return p, c, x, carried

    def fused(x, r, p, c, *args, capture_aux=False, **kw):
        r = post(x, r, p, c)
        return r, *pre(r, *args, **kw), r.mean(1) if capture_aux else r.new_empty((0, r.shape[-1]))

    def load(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepseekV4DecoderLayer')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        namespace = {'torch': torch, 'mhc_post_tilelang': post, 'mhc_pre_delayed_tilelang': pre,
                     'mhc_fused_post_pre_delayed_tilelang': fused}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace['forward']

    name = 'vllm/models/deepseek_v4_1/nvidia/model.py'
    original = load(Path('/deploy/overrides/performance/baseline') / name)
    updated = load(site / name)
    torch.manual_seed(11)
    h = 128
    norm = SimpleNamespace(weight=torch.ones(h, dtype=torch.bfloat16), variance_epsilon=1e-6)
    decoder = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        rms_norm_eps=1e-6, hc_post_alpha=2., use_sequence_parallel=False, engram=None,
        attn_norm=norm, ffn_norm=norm, attn=lambda positions, x, _: x * .5,
        ffn=lambda x, ids: x * .25, hc_attn_fn=torch.randn(24, 4*h)*.02,
        hc_ffn_fn=torch.randn(24, 4*h)*.02, hc_attn_scale=torch.ones(3),
        hc_ffn_scale=torch.ones(3), hc_attn_base=torch.randn(24), hc_ffn_base=torch.randn(24))
    decoder.hc_attn_fn_broadcast = decoder.hc_attn_fn.view(24, 4, h).sum(1).contiguous()
    positions = torch.arange(3)
    for entry in ('broadcast', 'pipeline', 'residual', 'engram'):
        x = torch.randn(3, h, dtype=torch.bfloat16)
        kw = {}
        if entry == 'pipeline':
            x = torch.randn(3, 4, h, dtype=torch.bfloat16)
            kw['pre_mix'] = torch.rand(3, 4)
        elif entry in ('residual', 'engram'):
            kw = dict(pre_mix=torch.rand(3, 4), residual=torch.randn(3, 4, h, dtype=torch.bfloat16),
                      post_mix=torch.rand(3, 4, 1), res_mix=torch.rand(3, 4, 4))
        if entry == 'engram':
            class Engram:
                layer_hash_index = 0
                def __call__(self, residual, hashes, mask):
                    return residual + .125
            decoder.engram = Engram()
            kw['engram_hashes'] = torch.zeros(3, 1, 1, dtype=torch.int32)
        old = original(decoder, x, positions, None, **kw)
        new = updated(decoder, x, positions, None, **kw)
        for left, right in zip(old, new[:5]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert new[5] is None
        if entry in ('residual', 'engram'):
            captured = updated(decoder, x, positions, None, **kw, capture_previous_aux=True)
            expected = post(x, kw['residual'], kw['post_mix'], kw['res_mix']).mean(1)
            torch.testing.assert_close(captured[5], expected, atol=0, rtol=0)


def main():
    checks = []
    for n in (1, 5, 6, 8, 12, 16):
        tile, splits, threads = mhc.mhc_fused_post_pre_split_config(n, 5120, 4)
        assert 24 % tile == 0 and 5120 % (splits * threads) == 0
    assert mhc.mhc_fused_post_pre_split_config(17, 5120, 4) is None
    assert mhc.mhc_fused_post_pre_split_config(6, 5137, 4) is None
    assert mhc.mhc_fused_post_pre_splits(5120, 4) == (4,)
    fields = dict(hidden_size=5120, hc_mult=4, rms_numel=20480, rms_eps=1e-6,
                  hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=2.,
                  sinkhorn_repeat=20, norm_eps=1e-6, use_pre_mix_in=True, write_aux=True)
    keys = mhc.DSV41_MHC_PRE_NORM_KERNEL.get_warmup_keys(max_tokens=4096, extra_splits=(4,), **fields)
    assert {k.n_splits for k in keys} == {1, 4}
    checks.append('SM80 complete tiles and split-4 aux warmup')
    for n in (0, 6, 192, 336):
        for aux in (False, True):
            with FakeTensorMode():
                x = torch.empty(n, 5120, dtype=torch.bfloat16)
                residual = torch.empty(n, 4, 5120, dtype=torch.bfloat16)
                args = (x, residual, torch.empty(n, 4, 1), torch.empty(n, 4, 4),
                        torch.empty(24, 20480), torch.ones(3), torch.zeros(24),
                        1e-6, 1e-6, 1e-6, 2., 20, torch.empty(n, 4), None, 1e-6, aux)
                result = torch.ops.vllm.mhc_fused_post_pre_delayed_tilelang(*args)
                assert result[0].shape == residual.shape
                assert result[3].shape == (n, 5120)
                assert result[5].shape == (n if aux else 0, 5120)
    checks.append('registered fake tensor ABI for empty/C1/C32/C56 k5 shapes')
    # Load just the processor module: importing the whole model package on a
    # GPU-less host triggers unrelated SM80 Triton driver discovery.
    site = Path(sysconfig.get_paths()['purelib'])
    check_decoder_flow(site)
    checks.append('production decoder control flow vs old source using CPU reference ops, including pre-Engram aux')
    mm_path = site / 'vllm/models/deepseek_v4_1/common/mm_preprocess.py'
    spec = importlib.util.spec_from_file_location('dsv41_mm_cpu', mm_path)
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    assert mm.image_sentinel_mask(torch.tensor([129264, 129265, 1])).tolist() == [True, False, False]
    assert '_apply_token_matches_with_placeholders' not in mm.DeepseekV4VLMultiModalProcessor.__dict__
    assert not hasattr(mm, 'IMAGE_PAD_ID')
    checks.append('image padding removed; default placeholder splicing and exact sentinel mask')
    # Syntax check each installed model file without claiming GPU model import.
    for name in ('model.py', 'dspark.py', 'vl_model.py'):
        ast.parse((site / 'vllm/models/deepseek_v4_1/nvidia' / name).read_text())
    print(json.dumps({'status': 'PASS', 'checks': checks, 'gpu_math_tested': False,
                      'full_model_import_tested': False}))


if __name__ == '__main__':
    main()
