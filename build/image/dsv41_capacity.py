"""Write factual capacity/DSpark records for the local offline installer."""
import json
import os
from pathlib import Path
import time

_SEEN_REPLAYS = set()


def _write(kind, data):
    directory = os.environ.get('DSV41_OBSERVATION_DIR')
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = {'kind': kind, 'pid': os.getpid(), 'time_ns': time.time_ns(), **data}
    path = root / f'{kind}-{record["pid"]}-{record["time_ns"]}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(path)


def record_capacity(config, cache, reported_tokens, reported_concurrency):
    groups = []
    for i, group in enumerate(cache.kv_cache_groups):
        spec = group.kv_cache_spec
        page = int(spec.page_size_bytes)
        maximum = int(spec.max_memory_usage_bytes(config))
        groups.append({'group': i, 'spec_type': type(spec).__name__,
                       'layer_names': list(group.layer_names),
                       'page_size_bytes': page, 'max_request_bytes': maximum,
                       'full_window_blocks': (maximum + page - 1) // page,
                       'block_size_tokens': int(spec.block_size)})
    blocks_per_request = sum(g['full_window_blocks'] for g in groups)
    if blocks_per_request <= 0:
        raise ValueError('No KV blocks for full-window capacity accounting')
    pool = int(cache.num_blocks)
    length = int(config.model_config.max_model_len)
    full_windows = pool // blocks_per_request
    # Integer arithmetic prevents boundary errors from rounded log values.
    effective_tokens = pool * length // blocks_per_request
    _write('capacity', {'max_model_len': length,
                        'configured_max_num_seqs': int(config.scheduler_config.max_num_seqs),
                        'pool_blocks': pool, 'blocks_per_full_window': blocks_per_request,
                        'full_windows': full_windows, 'required_concurrency': max(32, 2 * full_windows),
                        'effective_kv_tokens': effective_tokens,
                        'upstream_reported_tokens': int(reported_tokens),
                        'upstream_reported_concurrency': float(reported_concurrency),
                        'groups': groups})


def record_dspark(speculator):
    _write('dspark', {'speculator': type(speculator).__name__,
                      'num_speculative_tokens': int(speculator.num_speculative_steps),
                      'draft_query_per_request': int(speculator.num_query_per_req),
                      'sample_from_anchor': bool(speculator.sample_from_anchor),
                      'local_argmax': bool(speculator.use_local_argmax_reduction),
                      'fused_markov_built': speculator._fused_markov is not None,
                      'adaptive_verification': bool(speculator.enable_adaptive_verification),
                      'device': str(speculator.device),
                      'scope': 'draft loaded; acceptance counters and generated outputs still require requests'})


def _descriptor(desc):
    return {'mode': desc.cg_mode.name, 'num_tokens': int(desc.num_tokens),
            'num_reqs': desc.num_reqs, 'uniform_token_count': desc.uniform_token_count}


def record_graph_capture(manager, description):
    if not os.environ.get('DSV41_OBSERVATION_DIR'):
        return
    _write('graph-capture', {'manager': type(manager).__name__, 'description': description,
            'manager_id': id(manager), 'device': str(manager.device),
            'mode': manager.cudagraph_mode.name,
            'memory_estimation_probe': manager._capture_mem_samples is not None,
            'decode_query_len': int(manager.decode_query_len),
            'max_num_reqs': int(manager.max_num_reqs),
            'breakable_piecewise': bool(manager.use_breakable_cg),
            'captured_full_graphs': [_descriptor(d) for d in manager.graphs]})


def record_graph_replay(manager, desc):
    if not os.environ.get('DSV41_OBSERVATION_DIR'):
        return
    key = (id(manager), desc)
    if key in _SEEN_REPLAYS:
        return
    _SEEN_REPLAYS.add(key)
    _write('graph-replay', {'manager': type(manager).__name__, 'manager_id': id(manager),
            'device': str(manager.device), 'decode_query_len': int(manager.decode_query_len),
            'descriptor': _descriptor(desc),
            'scope': 'first replay issued for this shape; request success and CUDA errors checked separately'})
