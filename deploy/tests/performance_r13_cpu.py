"""R1.3 actual checkpoint loading and per-step cache lifecycle without GPUs."""
import json
from pathlib import Path
from unittest.mock import Mock

import torch
# Restore the installed Triton language objects for import-time annotations.
# No driver is fabricated, and no Triton kernel is launched in this CPU test.
import triton
import triton.language as tl
import vllm.triton_utils as facade
facade.triton, facade.tl = triton, tl

from performance_r13_reference import (
    DSPARK, draft_fixture, assert_loaded, context_method, attention_fixture,
    ragged_reference, assert_ragged,
)


def check_loader():
    cases = 0
    for path in (DSPARK, Path('/deploy/overrides/performance/baseline') / DSPARK):
        for enabled in (False, True):
            owner, weights, refs = draft_fixture(enabled=enabled, path=path, hidden=128, head=64, qrank=96)
            loaded = owner.load_weights(iter(weights))
            assert_loaded(owner, refs)
            assert ('model.context_wkv_proj.weight_scale' in loaded) == enabled
            if enabled:
                try:
                    owner.load_weights((n, w) for n, w in weights if n != 'mtp.1.attn.wkv.scale')
                except ValueError as e:
                    assert 'Incomplete DSpark context KV weights' in str(e)
                else:
                    raise AssertionError('missing native scale shard was not rejected')
            # Run the real context loop with an independent BF16 math backend.
            for layer, pair in zip(owner.model.layers, refs):
                del layer.attn.fused_wqa_wkv
                layer.attn.fused_wqa_wkv = Mock(side_effect=lambda x, w=torch.cat([p[2] for p in pair]):
                                               (torch.nn.functional.linear(x, w), None))
            if enabled:
                del owner.model.context_wkv_proj
                owner.model.context_wkv_proj = Mock(side_effect=lambda x:
                    torch.nn.functional.linear(x, torch.cat([p[1][2] for p in refs])))
            fn = context_method(path)
            x = torch.randn(7, 128).bfloat16()
            positions = torch.arange(7)
            slots = [torch.arange(7)+i*100 for i in range(3)]
            fn(owner.model, x, positions, [slots[0], None, slots[2]])
            for i in (0, 2):
                actual, pos, slot = owner.model.layers[i].attn.recorded
                expected = owner.model.layers[i].attn.kv_norm(torch.nn.functional.linear(x, refs[i][1][2]))
                torch.testing.assert_close(actual, expected)
                assert torch.equal(pos, positions) and torch.equal(slot, slots[i])
            assert not hasattr(owner.model.layers[1].attn, 'recorded')
            fn(owner.model, x, positions, None)  # memory profiling: project, no writes
            cases += 1
    return cases


def check_cache():
    builds = []
    def pack(*args):
        builds.append(1)
        return ragged_reference(*args)
    source, consumers, swa, meta = attention_fixture('cpu', pack)
    def expected(owner=source, metadata=meta):
        n = swa.num_decode_tokens
        return ragged_reference(owner.topk_indices_buffer[:n], swa.token_to_req_indices[:n],
                                metadata.block_table[:swa.num_decodes], metadata.block_size//owner.compress_ratio,
                                swa.is_valid_token[:n])
    for count in (6, 2, 12, 6):
        swa.num_decode_tokens = count
        source.topk_indices_buffer[:, 0].add_(1)
        out = source.build(swa, meta)
        assert_ragged(out, expected())
        before = len(builds)
        for consumer in consumers:
            cached = consumer.build(swa, meta)
            assert cached is out
        assert len(builds) == before  # one build, not five identical packs
    # Distinct block table storage must not reuse physical addresses.
    alt = type(meta)(block_size=64, block_table=meta.block_table.clone()+10)
    assert_ragged(consumers[0].build(swa, alt), expected(consumers[0], alt))
    assert len(builds) == 5
    consumers[1].compress_ratio = 1
    assert_ragged(consumers[1].build(swa, meta), expected(consumers[1]))
    assert len(builds) == 6
    # Same pointer, different contents is safe only after the source rebuilds.
    swa.is_valid_token.fill_(False)
    source.build(swa, meta)
    assert_ragged(consumers[0].build(swa, meta), expected())
    assert len(builds) == 7
    # A separate DP pool owns independent state even at the same layer prefix.
    other, _, s2, m2 = attention_fixture('cpu', pack)
    other.build(s2, m2)
    assert other._topk_ragged_cache is not source._topk_ragged_cache
    for owner in [source, *consumers]:
        owner._reuse_topk_ragged = False
        assert_ragged(owner.build(swa, meta), expected(owner))
    assert len(builds) == 13
    return len(builds)


if __name__ == '__main__':
    print(json.dumps(dict(status='PASS', gpu_executed=False, loader_cases=check_loader(),
        cache_builds_checked=check_cache(), scope='real merged MXFP8 loader; independent context/metadata references')))
