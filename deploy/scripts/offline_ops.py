"""Dependency-free deployment calculations, executed using the packaged image."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

MODEL = 'DeepSeek-V4.1-Flash'
LENGTH = 262144


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def resolve(config, concurrency, fixed_blocks=None):
    expected = {'model':'/models/'+MODEL,'served_model_name':MODEL,'tensor_parallel_size':4,
                'data_parallel_size':2,'data_parallel_size_local':2,'enable_expert_parallel':True,
                'all2all_backend':'allgather_reducescatter','enable_eplb':True,
                'eplb_config':{'window_size':1000,'step_interval':3000,'num_redundant_experts':0,
                               'use_async':True,'log_balancedness':True,'log_balancedness_interval':500},
                'pipeline_parallel_size':1,'max_model_len':LENGTH,'tokenizer_mode':'deepseek_v41',
                'kv_cache_dtype':'fp8_ds_mla','engram_config':{'cpu_offload':True},
                'tool_call_parser':'deepseek_v41','reasoning_parser':'deepseek_v41',
                'enable_prefix_caching':True,'enable_prompt_tokens_details':True,'enable_auto_tool_choice':True,
                'limit_mm_per_prompt':{'image':999},'generation_config':'vllm','disable_custom_all_reduce':True}
    allowed=set(expected)|{'speculative_config','max_num_batched_tokens','gpu_memory_utilization'}
    if set(config)!=allowed: raise ValueError('Unexpected or missing runtime keys')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Protected runtime constraint changed: '+key)
    spec = config['speculative_config']
    k = spec.get('num_speculative_tokens') if isinstance(spec, dict) else None
    if type(k) is not int or k != 5:
        raise ValueError('This deployment is fixed to native DSpark k=5')
    if spec != {'method':'dspark','num_speculative_tokens':k,'draft_sample_method':'greedy',
                'rejection_sample_method':'standard','use_local_argmax_reduction':True,'enable_adaptive_verification':False}:
        raise ValueError('DSpark contract changed')
    if not (32 <= concurrency and concurrency % 2 == 0):
        raise ValueError('Invalid concurrency; no silent clamping is performed')
    if config['gpu_memory_utilization'] not in [0.90,0.92,0.94]:
        raise ValueError('Memory utilization is outside the bounded tuning matrix')
    if config['max_num_batched_tokens'] not in [4096,8192,16384]:
        raise ValueError('Batch-token budget is outside the bounded tuning matrix')
    # The user-facing C is global; vLLM's scheduler limit is per DP engine.
    local_concurrency = concurrency // expected['data_parallel_size']
    result = dict(config, max_num_seqs=local_concurrency)
    if fixed_blocks is not None:
        if fixed_blocks<=0:raise ValueError('Fixed KV pool must be positive')
        result['num_gpu_blocks_override']=fixed_blocks
    result['max_num_batched_tokens'] = max(result['max_num_batched_tokens'], (((k+1)*local_concurrency+127)//128)*128)
    batches = {1,2,4,8,16,32,local_concurrency}
    rung = 64
    while rung < local_concurrency:
        batches.add(rung)
        rung *= 2
    batches = sorted(b for b in batches if b <= local_concurrency)
    shapes = sorted({size*b for size in [k,k+1] for b in batches})
    result['compilation_config'] = {'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes':shapes,'max_cudagraph_capture_size':max(shapes)}
    args = [result['model'],'--host','0.0.0.0','--port','8000','--middleware','dsv41_guard.OfflineGuard']
    for key,value in result.items():
        if key == 'model':
            continue
        flag = '--'+key.replace('_','-')
        if isinstance(value,bool):
            args.append(flag if value else '--no-'+key.replace('_','-'))
        else:
            args += [flag,json.dumps(value,separators=(',',':')) if isinstance(value,(dict,list)) else str(value)]
    return result,args


def capacity(directory, configured):
    rows = [json.loads(p.read_text()) for p in directory.glob('capacity-*.json')]
    if not rows:
        raise ValueError('No capacity records from this launch')
    # The final initialization record from each process is authoritative.
    latest = {}
    for row in rows:
        if row['time_ns'] > latest.get(row['pid'], {}).get('time_ns', -1):
            latest[row['pid']] = row
    rows = list(latest.values())
    for row in rows:
        if row['max_model_len'] != LENGTH or row['configured_max_num_seqs'] != configured//2:
            raise ValueError('Stale or mismatching capacity record')
        validate_topology(row)
        blocks = sum(g['full_window_blocks'] for g in row['groups'])
        if blocks != row['blocks_per_full_window'] or blocks <= 0:
            raise ValueError('KV group accounting mismatch')
        if row['pool_blocks']//blocks != row['full_windows']:
            raise ValueError('Full-window arithmetic mismatch')
        if row['pool_blocks']*LENGTH//blocks != row['effective_kv_tokens']:
            raise ValueError('Effective KV token arithmetic mismatch')
    # Each DP engine has a distinct KV pool. TP ranks are shards, never extra
    # pools. Floors must be taken per DP: fractional windows cannot be joined.
    pools = []
    for rank in (0, 1):
        local = [r for r in rows if r['data_parallel_rank'] == rank]
        if not local:
            raise ValueError(f'Missing capacity record for DP rank {rank}')
        effective = min(r['effective_kv_tokens'] for r in local)
        pools.append({'data_parallel_rank':rank, 'effective_kv_tokens':effective,
                      'full_windows':effective//LENGTH})
    if any(p['full_windows'] < 1 for p in pools):
        raise ValueError('Each DP pool must fit at least one full 256K window')
    windows = sum(p['full_windows'] for p in pools)
    effective = sum(p['effective_kv_tokens'] for p in pools)
    required = max(32,2*windows)
    if windows < 1:
        raise ValueError('Cannot fit even one full 256K window')
    return {'status':'CALIBRATED' if required == configured else 'RESTART_REQUIRED',
            'max_model_len':LENGTH,'full_windows':windows,'effective_kv_tokens':effective,
            'configured_concurrency':configured,'required_concurrency':required,
            'minimum_pool_blocks':min(r['pool_blocks'] for r in rows),
            'formula':'max(32,2*sum(dp_effective_kv_tokens//262144))','tp_capacity_multiplier':1,
            'topology':'TP4-DP2-EP8','data_parallel_size':2,'tensor_parallel_size':4,
            'per_dp_max_num_seqs':configured//2,'pools':pools,
            'records':rows,'scope':'actual initialized KV groups; full-context correctness and stress are separate gates'}


def validate_topology(row):
    if (row.get('data_parallel_size') != 2 or row.get('tensor_parallel_size') != 4
            or row.get('expert_parallel_enabled') is not True or row.get('data_parallel_rank') not in (0, 1)):
        raise ValueError('Missing or mismatching TP4 DP2 EP8 telemetry')


def eight_workers(rows):
    for row in rows:
        validate_topology(row)
    return all(len({r['pid'] for r in rows if r['data_parallel_rank'] == dp}) == 4 for dp in (0, 1)) and len({r['pid'] for r in rows}) == 8


def graphs(directory, concurrency, k=5):
    if type(k) is not int or k != 5: raise ValueError("Graph validation requires fixed k=5")
    rows = [json.loads(p.read_text()) for p in directory.glob('*.json')]
    drafts = [r for r in rows if r.get('kind') == 'dspark']
    if not eight_workers(drafts):
        raise ValueError('Need draft-load records from all eight workers')
    if any(r['num_speculative_tokens'] != k or r['draft_query_per_request'] != k or not r['fused_markov_built'] or not r['local_argmax'] for r in drafts):
        raise ValueError('DSpark path does not match the fixed route')
    captures = [r for r in rows if r.get('kind') == 'graph-capture' and not r['memory_estimation_probe']]
    for query in [k,k+1]:
        eligible = [r for r in captures if r['decode_query_len'] == query]
        matching = [r for r in eligible if any(d['num_tokens'] == query*(concurrency//2) and d['mode'] == 'FULL' for d in r['captured_full_graphs'])]
        if not eight_workers(matching):
            raise ValueError(f'Missing full CUDA graph at query={query}, C={concurrency} on eight workers')
    return {'status':'PASS','scope':'draft load and completed graph capture; replay and accepted drafts need successful requests',
            'concurrency':concurrency,'per_dp_concurrency':concurrency//2,'draft_workers':8,'num_speculative_tokens':k,'target_shape':(k+1)*(concurrency//2),'draft_shape':k*(concurrency//2)}


def replays(directory, concurrency, k=5):
    if type(k) is not int or k != 5: raise ValueError("Replay validation requires fixed k=5")
    rows = [json.loads(p.read_text()) for p in directory.glob('*.json')]
    real = {(r['pid'],r['manager_id']) for r in rows
            if r.get('kind')=='graph-capture' and not r['memory_estimation_probe']}
    counts = {}
    for query in [k,k+1]:
        valid = [r for r in rows if r.get('kind')=='graph-replay'
                 and (r['pid'],r['manager_id']) in real and r['decode_query_len']==query
                 and r['descriptor']['mode']=='FULL']
        if not eight_workers(valid): raise ValueError(f'Missing actual FULL graph replay on both DP groups: query={query}')
        counts[str(query)] = sorted({r['descriptor']['num_tokens'] for r in valid})
    return {'status':'PASS','scope':'FULL target and draft graph replay on eight workers; pair with successful requests',
            'concurrency':concurrency,'num_speculative_tokens':k,'replayed_token_shapes':counts}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command',required=True)
    q = sub.add_parser('resolve'); q.add_argument('--config',type=Path,required=True); q.add_argument('--concurrency',type=int,required=True); q.add_argument('--output',type=Path,required=True)
    q.add_argument('--fixed-blocks',type=int)
    q = sub.add_parser('argv'); q.add_argument('path',type=Path)
    for name in ['capacity','graphs','replays']:
        q = sub.add_parser(name); q.add_argument('--directory',type=Path,required=True); q.add_argument('--concurrency',type=int,required=True); q.add_argument('--output',type=Path,required=True)
        if name != 'capacity': q.add_argument('--resolved',type=Path,required=True)
    q = sub.add_parser('get'); q.add_argument('path',type=Path); q.add_argument('key')
    q = sub.add_parser('fingerprint'); q.add_argument('--model',type=Path,required=True); q.add_argument('--manifest',type=Path,required=True)
    a = p.parse_args()
    if a.command == 'resolve':
        config,args = resolve(json.loads(a.config.read_text()),a.concurrency,a.fixed_blocks)
        write(a.output, {'config':config,'argv':args,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
    elif a.command == 'argv':
        for arg in json.loads(a.path.read_text())['argv']:
            sys.stdout.buffer.write(arg.encode()+b'\0')
    elif a.command == 'capacity':
        write(a.output,capacity(a.directory,a.concurrency))
    elif a.command == 'graphs':
        write(a.output,graphs(a.directory,a.concurrency,json.loads(a.resolved.read_text())['config']['speculative_config']['num_speculative_tokens']))
    elif a.command == 'replays':
        write(a.output,replays(a.directory,a.concurrency,json.loads(a.resolved.read_text())['config']['speculative_config']['num_speculative_tokens']))
    elif a.command == 'get':
        value = json.loads(a.path.read_text())[a.key]
        print(value if isinstance(value,(str,int,float)) else json.dumps(value))
    elif a.command == 'fingerprint':
        expected={r['path']:r['size'] for r in json.loads(a.manifest.read_text())['files']}
        files={p.relative_to(a.model).as_posix():p for p in a.model.rglob('*') if p.is_file()}
        if any(p.is_symlink() for p in a.model.rglob('*')) or set(files)!=set(expected):
            raise ValueError('Model file set or symlink policy changed')
        rows=[]
        for name,path in sorted(files.items()):
            st=path.stat()
            if st.st_size!=expected[name]:raise ValueError('Model size changed: '+name)
            rows.append([name,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino])
        print(hashlib.sha256(json.dumps(rows).encode()).hexdigest())


if __name__ == '__main__':
    main()
