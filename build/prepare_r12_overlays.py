"""Reproduce the R1.2 additions from pinned, locally archived upstream sources.

Run after the R1.1 overlay exists. No download, image modification or GPU access.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'source/vllm-backport-a350766628514d597670d2a5217777fdee2ba7f4'
TURBO = ROOT / 'research/sources/speed-review-20260916/turbo-source/patches'
RESEARCH = ROOT / 'research/sources/speed-review-20260918'
OVERLAY = ROOT / 'deploy/overrides/performance'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def patch(text, diff):
    """Apply GitHub's context hunks, rejecting ambiguous or mismatched sources."""
    chunks = re.split(r'^@@[^\n]*\n', diff, flags=re.M)[1:]
    for chunk in chunks:
        lines = chunk.splitlines(keepends=True)
        old = ''.join(s[1:] for s in lines if s[:1] in (' ', '-'))
        new = ''.join(s[1:] for s in lines if s[:1] in (' ', '+'))
        if old and not old.endswith('\n'):
            old += '\n'
        if new and not new.endswith('\n'):
            new += '\n'
        if not old or text.count(old) != 1:
            raise ValueError('Upstream patch context does not match uniquely')
        text = text.replace(old, new, 1)
    return text


def main():
    manifest = json.loads((OVERLAY / 'manifest.json').read_text())
    rows = {r['path']: r for r in manifest['files']}
    turbo_rows = {r['target']: r for r in json.loads((TURBO / 'manifest.json').read_text())['files']}

    def put(name, data, group='r12', baseline=None):
        original = (BASE / name).read_bytes() if name.startswith('vllm/') else (ROOT / 'build/image' / name).read_bytes()
        baseline = original if baseline is None else baseline
        for variant, content in [('baseline', baseline), ('patched', data)]:
            path = OVERLAY / variant / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        rows[name] = dict(path=name, group=group, base_sha256=sha(original),
                          baseline_sha256=sha(baseline), patched_sha256=sha(data))

    selected = [
        'vllm/model_executor/kernels/linear/mxfp8/marlin.py',
        'vllm/distributed/device_communicators/cuda_communicator.py',
        'vllm/model_executor/models/interfaces.py',
        'vllm/v1/worker/gpu/spec_decode/dspark/utils.py',
        'vllm/v1/worker/gpu/spec_decode/speculator.py',
        'vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py',
        'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py',
        'vllm/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py',
    ]
    for name in selected:
        data = (TURBO / 'files' / name).read_bytes()
        assert sha((BASE / name).read_bytes()) == turbo_rows[name]['before'], name
        assert sha(data) == turbo_rows[name]['after'], name
        put(name, data)

    # EPLB metadata must work with both mHC switch positions. Image provenance
    # and the disabled-fusion payload have separate checksums for this reason.
    name = 'vllm/models/deepseek_v4_1/nvidia/model.py'
    old = '        self.num_redundant_experts = example_moe.n_redundant_experts\n'
    addition = '''        if not example_moe.use_mega_moe:
            expert_map = example_moe.experts.expert_map_manager
            self.num_physical_experts = expert_map.global_num_experts
            self.num_local_physical_experts = expert_map.local_num_experts
            assert self.num_physical_experts == self.num_logical_experts + self.num_redundant_experts
'''
    fused = (OVERLAY / 'patched' / name).read_text()
    if addition not in fused:
        assert fused.count(old) == 1
        fused = fused.replace(old, old + addition)
    baseline = (BASE / name).read_text().replace(old, old + addition)
    put(name, fused.encode(), 'mhc', baseline.encode())

    for archive in ('backport-moe-commit.json', 'backport-latest-compare.json'):
        for item in json.loads((RESEARCH / archive).read_text())['files']:
            name = item['filename']
            if archive.endswith('compare.json') and name != 'vllm/parser/deepseek_v41.py':
                continue
            if not name.startswith('vllm/'):
                # The upstream CUDA numerical tests are carried into our suite.
                if name.endswith('test_shared_experts_multistream.py'):
                    data = ''.join(s[1:] for s in item['patch'].splitlines(True) if s.startswith('+'))
                    (ROOT / 'deploy/tests/upstream/test_shared_experts_multistream.py').write_text(data + '\n')
                continue
            put(name, patch((BASE / name).read_text(), item['patch']).encode())

    # Take only the independently justified 8-warp launch change. Keep the
    # original split heuristic; the newer env-default change is not required.
    name = 'vllm/v1/attention/ops/rocm_aiter_mla_sparse.py'
    text = (BASE / name).read_text()
    old = '            NUM_SPLITS=num_splits,\n            NUM_STAGES=1,\n            num_warps=4,\n        )'
    assert text.count(old) == 1
    text = text.replace(old, '            NUM_SPLITS=num_splits,\n            NUM_STAGES=1,\n            num_warps=8 if current_platform.is_cuda() else 4,\n        )')
    put(name, text.encode())

    # Explicitly isolate the 128-expert draft from the 384-expert target EPLB
    # controller as well as from its ParallelConfig.
    name = 'vllm/v1/worker/gpu/eplb_utils.py'
    text = (BASE / name).read_text()
    old = '        draft_model = speculator.model\n'
    assert text.count(old) == 1
    text = text.replace(old, '''        if getattr(speculative_config, "method", None) == "dspark":
            return False
        draft_model = speculator.model
''')
    put(name, text.encode())

    # Unlike Chat/Messages, the pinned Responses handler did not forward the
    # router's DP header to engine.generate, including built-in tool turns.
    name = 'vllm/entrypoints/openai/responses/serving.py'
    text = (BASE / name).read_text()
    old = '                priority=self._get_priority(request, raw_request),\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '                data_parallel_rank=self._get_data_parallel_rank(raw_request),\n')
    old = '        reasoning_parser_kwargs: dict[str, Any] | None = None,\n'
    assert text.count(old) == 1
    text = text.replace(old, old + '        data_parallel_rank: int | None = None,\n')
    old = '                session_id=session_id,\n                reasoning_parser_kwargs=reasoning_parser_kwargs,\n'
    # The public handler call and the nested engine call share this pair.
    assert text.count(old) == 2
    prefix, generator = text.split('    async def _generate_with_builtin_tools(', 1)
    generator = generator.replace(old, old + '                data_parallel_rank=data_parallel_rank,\n')
    put(name, (prefix + '    async def _generate_with_builtin_tools(' + generator).encode())

    # The four original callback sites stay in the image; only their recorder
    # gains explicit DP coordinates, so no image rebuild is necessary.
    name = 'dsv41_capacity.py'
    text = (ROOT / 'build/image' / name).read_text()
    text = text.replace("_SEEN_REPLAYS = set()", '''_SEEN_REPLAYS = set()


def _parallel(config):
    pc = config.parallel_config
    return {'data_parallel_rank': int(pc.data_parallel_rank),
            'data_parallel_size': int(pc.data_parallel_size),
            'tensor_parallel_size': int(pc.tensor_parallel_size),
            'expert_parallel_enabled': bool(pc.enable_expert_parallel)}''')
    text = text.replace("{'max_model_len': length,", "{**_parallel(config), 'max_model_len': length,")
    text = text.replace("{'speculator': type(speculator).__name__,", "{**_parallel(speculator.vllm_config), 'speculator': type(speculator).__name__,")
    text = text.replace("{'manager': type(manager).__name__,", "{**_parallel(manager.vllm_config), 'manager': type(manager).__name__,")
    put(name, text.encode())

    manifest.update(id='20260918-perf3', release='1.2.0', gpu_validated=False)
    manifest['files'] = list(rows.values())
    manifest['upstream_sources'] = [r for r in manifest['upstream_sources'] if not r.get('r12')]
    manifest['upstream_sources'] += [
        dict(url='https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo',
             commit='fae324ae62ac5cef31b7d38f5d369618e1cae1fa', r12=True,
             component='dense BF16, EP8 collectives, DP draft metadata, EPLB draft isolation'),
        dict(url='https://github.com/wtdcode/vllm-backport', r12=True,
             commit='a5bb095f0b97d513ac0c657c2c481334e9302ad8', component='8-warp sparse decode'),
        dict(url='https://github.com/wtdcode/vllm-backport', r12=True,
             commit='9925c45ab4c0740dfc8e77b4715f665e8b205447', component='CUDA shared-expert overlap'),
        dict(url='https://github.com/wtdcode/vllm-backport/pull/89', r12=True,
             commit='19a4e635aa46e4e58342b836833c163c60af8552',
             component='V4.1 DSML spelling variants; pinned comparison archived locally')]
    (OVERLAY / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(rows), 'id': manifest['id'], 'gpu_validated': False}))


if __name__ == '__main__':
    main()
