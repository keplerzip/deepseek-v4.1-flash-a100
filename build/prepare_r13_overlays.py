"""Adapt pinned vLLM #57627/#57434 to the R1.2 offline SM80 overlay.

Run after prepare_r12_overlays.py. Uses only local source; no image rebuild.
Already-produced R1.3 payloads are checked rather than patched a second time.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'source/vllm-backport-a350766628514d597670d2a5217777fdee2ba7f4'
OVERLAY = ROOT / 'deploy/overrides/performance'
UPDATE = '20260920-perf4'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source context is not unique: ' + old[:120])
    return text.replace(old, new, 1)


def stack_projection(text):
    text = replace(text, 'from collections.abc import Iterable\n',
                   'from collections.abc import Iterable\nimport os\n')
    text = replace(text, 'from vllm.model_executor.layers.linear import ReplicatedLinear\n',
                   'from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear\n')
    anchor = '        # Heads: final norm, and the Markov + confidence heads.\n'
    text = replace(text, anchor, '''        # R1.3, adapted from vLLM #57627. The context needs only Wkv;
        # retain each layer's original fused projection for its attention.
        self.context_wkv_proj = None
        if os.environ.get("VLLM_DSV41_STACKED_DRAFT_KV", "1") == "1":
            self.context_wkv_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [config.head_dim] * self.num_dspark_layers,
                bias=False,
                return_bias=False,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "context_wkv_proj"),
                disable_tp=True,
            )
        logger.info_once("R1.3 DSpark stacked context KV: %s",
                         self.context_wkv_proj is not None)

''' + anchor)
    text = replace(text, '        for i, layer in enumerate(self.layers):\n', '''        all_kv = None
        if self.context_wkv_proj is not None:
            all_kv = self.context_wkv_proj(main_x).view(
                main_x.shape[0], self.num_dspark_layers, self.config.head_dim
            )
        for i, layer in enumerate(self.layers):
''')
    text = replace(text, '''            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = qr_kv[..., attn.q_lora_rank :]
''', '''            if all_kv is None:
                qr_kv, _ = attn.fused_wqa_wkv(main_x)
                kv = qr_kv[..., attn.q_lora_rank :]
            else:
                kv = all_kv[:, i, :]
''')
    text = replace(text, '        loaded_confidence_head = False\n',
                   '        loaded_confidence_head = False\n        loaded_context_shards: set[tuple[str, int]] = set()\n')
    anchor = '            # Stacked rules only apply to decoder-layer weights. Head-stack params\n'
    text = replace(text, anchor, '''            # Duplicate the original checkpoint KV shards, including their
            # native 32x32 scales, through the real merged-linear loader.
            # Do not concatenate Marlin-repacked tensors after loading.
            match = re.fullmatch(r"model\\.layers\\.(\\d+)\\.attn\\.wkv\\.(.+)", name)
            if match is not None and self.model.context_wkv_proj is not None:
                shard = int(match.group(1))
                if not 0 <= shard < self.model.num_dspark_layers:
                    raise ValueError("DSpark context KV shard outside draft layers")
                context_name = f"model.context_wkv_proj.{match.group(2)}"
                param = params_dict[context_name]
                param.weight_loader(param, loaded_weight, shard)
                loaded_params.add(context_name)
                loaded_context_shards.add((context_name, shard))

''' + anchor)
    anchor = '        if self.model.confidence_head is not None and not loaded_confidence_head:\n'
    text = replace(text, anchor, '''        if self.model.context_wkv_proj is not None:
            expected_context_shards = {
                (name, shard)
                for name in params_dict
                if name.startswith("model.context_wkv_proj.")
                for shard in range(self.model.num_dspark_layers)
            }
            missing_context = expected_context_shards - loaded_context_shards
            if missing_context:
                raise ValueError(f"Incomplete DSpark context KV weights: {sorted(missing_context)}")

''' + anchor)
    return text


def reuse_topk(text):
    text = replace(text, 'from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention\n',
                   'from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention, _replace_layer_index\n')
    # The file already imports os for its existing ROCm control paths.
    if '\nimport os\n' not in text:
        text = replace(text, 'import torch\n', 'import os\n\nimport torch\n')
    anchor = '        self._fused_compressor_split_sizes: tuple[int, int] | None = None\n'
    text = replace(text, anchor, anchor + '''        # R1.3: Ampere inherits this same Triton path. The index-producing
        # layer owns the cache and invalidates it on every eager/captured step.
        self._reuse_topk_ragged = os.environ.get("VLLM_DSV41_REUSE_TOPK_RAGGED", "1") == "1"
        self._topk_ragged_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._index_source_prefix: str | None = None
        if self._reuse_topk_ragged and self.compress_ratio > 0:
            assert self.index_source_layer_id is not None
            self._index_source_prefix = _replace_layer_index(self.prefix, self.index_source_layer_id)
            if self._index_source_prefix not in self._static_forward_context:
                raise NotImplementedError("Index source is outside this rank; cannot reuse ragged metadata")
        logger.info_once("R1.3 Ampere/ROCm ragged top-k reuse: %s", self._reuse_topk_ragged)
''')
    anchor = '    def _forward_decode(\n'
    text = replace(text, anchor, '''    def _decode_topk_ragged(
        self,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse this step's pack only for identical input buffers and geometry.

        The owner runs before every consumer of its indices. Capture records
        the build once and all consumers read those graph-owned output buffers.
        Replays execute the recorded build; no Python cache crosses replay steps.
        """
        assert swa_metadata.is_valid_token is not None
        assert self.topk_indices_buffer is not None
        tokens, requests = swa_metadata.num_decode_tokens, swa_metadata.num_decodes
        source = None
        if self._reuse_topk_ragged:
            assert self._index_source_prefix is not None
            source = self._static_forward_context[self._index_source_prefix]
            if source is self:
                source._topk_ragged_cache.clear()
            # Unlike a ratio-only key, this also isolates distinct KV groups,
            # validity masks or block-table views without reading GPU values.
            key = (
                self.compress_ratio, attn_metadata.block_size, tokens, requests,
                self.topk_indices_buffer.data_ptr(), self.topk_indices_buffer.stride(),
                attn_metadata.block_table.data_ptr(), attn_metadata.block_table.stride(),
                tuple(attn_metadata.block_table.shape),
                swa_metadata.token_to_req_indices.data_ptr(),
                swa_metadata.token_to_req_indices.stride(),
                swa_metadata.is_valid_token.data_ptr(), swa_metadata.is_valid_token.stride(),
            )
            cached = source._topk_ragged_cache.get(key)
            if cached is not None:
                return cached
        built = compute_global_topk_ragged_indices_and_indptr(
            self.topk_indices_buffer[:tokens],
            swa_metadata.token_to_req_indices,
            attn_metadata.block_table[:requests],
            attn_metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[:tokens],
        )
        if source is not None:
            source._topk_ragged_cache[key] = built
        return built

''' + anchor)
    old = '''            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            (
                topk_ragged_indices,
                topk_ragged_indptr,
                topk_lens,
            ) = compute_global_topk_ragged_indices_and_indptr(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
'''
    new = '''            assert attn_metadata is not None
            topk_ragged_indices, topk_ragged_indptr, topk_lens = self._decode_topk_ragged(
                swa_metadata, attn_metadata
            )
'''
    return replace(text, old, new)


def main():
    manifest = json.loads((OVERLAY / 'manifest.json').read_text())
    if manifest['id'] == UPDATE:
        for row in manifest['files']:
            assert sha((OVERLAY / 'patched' / row['path']).read_bytes()) == row['patched_sha256']
            if (OVERLAY / 'baseline' / row['path']).is_file():
                assert sha((OVERLAY / 'baseline' / row['path']).read_bytes()) == row.get('baseline_sha256', row['base_sha256'])
        print('R1.3 overlay already exists and matches its manifest.')
        return
    if manifest['id'] != '20260918-perf3':
        raise ValueError('Build the pinned R1.2 overlay first')
    rows = {r['path']: r for r in manifest['files']}
    name = 'vllm/models/deepseek_v4_1/nvidia/dspark.py'
    for variant in ('baseline', 'patched'):
        path = OVERLAY / variant / name
        before = path.read_bytes()
        expected = rows[name]['base_sha256'] if variant == 'baseline' else rows[name]['patched_sha256']
        assert sha(before) == expected, (variant, name)
        after = stack_projection(before.decode()).encode()
        path.write_bytes(after)
        rows[name][variant + '_sha256'] = sha(after)
    name = 'vllm/models/deepseek_v4_1/amd/rocm.py'
    original = (BASE / name).read_bytes()
    patched = reuse_topk(original.decode()).encode()
    for variant, data in [('baseline', original), ('patched', patched)]:
        path = OVERLAY / variant / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    rows[name] = dict(path=name, group='r13', base_sha256=sha(original),
                      baseline_sha256=sha(original), patched_sha256=sha(patched))
    manifest.update(id=UPDATE, release='1.3.0', gpu_validated=False)
    manifest['files'] = list(rows.values())
    manifest['upstream_sources'] += [
        dict(url='https://github.com/vllm-project/vllm/pull/57627',
             commit='f0defc90f6645242ee61bccecd6146f385cff4b7', r13=True,
             component='stacked DSpark KV projection; SM80 merged loader and independent fallback'),
        dict(url='https://github.com/vllm-project/vllm/pull/57434',
             commit='bea2fcdfcf2ef385db7b93352d98c7ee35739126', r13=True,
             component='Ampere inherited ragged top-k reuse; additional buffer/geometry guards'),
    ]
    (OVERLAY / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(id=UPDATE, files=len(rows), gpu_validated=False)))


if __name__ == '__main__':
    main()
