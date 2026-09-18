"""Count the unchanged V4.1 snapshot using safetensors headers only."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct


def audit(root):
    cfg = json.loads((root / 'config.json').read_text())
    assert cfg['architectures'] == ['DeepseekV41ForCausalLM']
    total, by_dtype, headers = Counter(), Counter(), []
    dense_bf16 = 0
    for path in sorted(root.glob('*.safetensors')):
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            if not 0 < length < 64 * 1024**2:
                raise ValueError('Unexpected tensor header size')
            raw = f.read(length)
        headers.append(dict(file=path.name, header_sha256=hashlib.sha256(raw).hexdigest()))
        for name, item in json.loads(raw).items():
            if name == '__metadata__':
                continue
            if '.experts.' in name:
                group = 'routed_experts_including_draft'
            elif '.engram.embed.' in name:
                group = 'engram_cpu_tables'
            else:
                group = 'non_expert_including_vision_and_engram_projection'
            size = item['data_offsets'][1] - item['data_offsets'][0]
            total[group] += size
            by_dtype[group + ':' + item['dtype']] += size
            if group.startswith('non_expert') and item['dtype'] == 'F8_E4M3':
                dense_bf16 += 2 * math.prod(item['shape'])
    assert len(headers) == 48
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    assert sum(total.values()) == index['metadata']['total_size']
    expert = total['routed_experts_including_draft']
    nonexpert = total['non_expert_including_vision_and_engram_projection']
    # Expert scale bytes are expanded from E8M0 to BF16 by SM80 Marlin.
    scale_expansion = by_dtype['routed_experts_including_draft:F8_E8M0']
    # Deliberately pessimistic replication for all non-expert tensors: eight
    # copies, even though TP4 partitions many of them. This is a weight-only
    # scenario, NOT a measured peak and NOT a bound on runtime allocations.
    weight_scenario = (expert + scale_expansion) / 8 + nonexpert + dense_bf16
    return dict(model='DeepSeek-V4.1-Flash', architecture=cfg['architectures'][0],
        source='local original 48-shard snapshot; headers only, payload not rehashed',
        bytes_by_group=dict(total), bytes_by_group_and_dtype=dict(by_dtype),
        dense_bf16_extra_bytes_if_all_eligible_nonexpert_weights_materialized=dense_bf16,
        expert_bf16_scale_expansion_bytes=scale_expansion,
        engram_host_table_budget_bytes_dp2=2 * total['engram_cpu_tables'],
        engram_replication_note='Tables are TP-sharded inside each DP group; DP2 needs two table replicas in host RAM, excluding loading peaks',
        expert_weight_bytes_per_gpu_ep8=(expert + scale_expansion) / 8,
        pessimistic_nonexpert_replication_weight_gib_per_gpu=weight_scenario / 2**30,
        gpu_memory_utilization=0.90, configured_budget_gib_per_gpu=72,
        excluded=['KV cache', 'CUDA graphs', 'activation/workspace', 'packing padding',
                  'loading temporary copies', 'EPLB staging and communication buffers'],
        decision='weight accounting supports a TP4 DP2 EP8 trial; actual fit and 256K capacity require target calibration',
        gpu_inference_tested=False, shard_headers=headers)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = audit(a.model)
    a.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'shard_headers'}, indent=2))
