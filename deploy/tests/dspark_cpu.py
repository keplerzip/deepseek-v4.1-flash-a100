"""CPU-only contract tests. They do not validate GPU kernels or model output."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY / 'scripts'))
sys.path.insert(0, str(DEPLOY / 'tests'))
import offline_ops as ops
import dspark_benchmark as bench


class ContractTests(unittest.TestCase):
    def test_graph_sizes_and_constraints(self):
        base = json.loads((DEPLOY / 'configs/runtime.json').read_text())
        for k, shapes in [(5, (160, 192))]:
            candidate = copy.deepcopy(base)
            candidate['speculative_config']['num_speculative_tokens'] = k
            result, argv = ops.resolve(candidate, 64)
            self.assertEqual(result['max_model_len'], 512000)
            self.assertEqual(result['max_num_seqs'], 32)
            self.assertEqual(result['served_model_name'], 'DeepSeek-Flash')
            self.assertEqual(result['tensor_parallel_size'] * result['data_parallel_size'], 8)
            self.assertEqual(result['engram_config'], {'cpu_offload': True})
            self.assertEqual(result['gpu_memory_utilization'], 0.92)
            self.assertEqual(result['limit_mm_per_prompt'], {'image': 999})
            self.assertEqual(json.loads(argv[argv.index('--limit-mm-per-prompt')+1]), {'image': 999})
            self.assertTrue(set(shapes) <= set(result['compilation_config']['cudagraph_capture_sizes']))
            self.assertEqual(result['compilation_config']['max_cudagraph_capture_size'], 192)
            self.assertEqual(result['max_num_batched_tokens'], 1024)
            self.assertEqual(argv[argv.index('--max-num-batched-tokens') + 1], '1024')
            self.assertIs(result['enable_chunked_prefill'], True)
            self.assertIn('--enable-chunked-prefill', argv)
            self.assertIs(result['disable_chunked_mm_input'], True)
            self.assertIn('--disable-chunked-mm-input', argv)
            self.assertEqual(result['hf_overrides'], {'vision_max_n_token': 1020})
            self.assertEqual(json.loads(argv[argv.index('--hf-overrides') + 1]), {'vision_max_n_token': 1020})
            self.assertEqual(result['speculative_config']['num_speculative_tokens'], k)
        for invalid in (True, 4, 6, 7, 8, 10):
            candidate = copy.deepcopy(base)
            candidate['speculative_config']['num_speculative_tokens'] = invalid
            with self.assertRaises(ValueError): ops.resolve(candidate, 64)
        for key, value in [('max_model_len', 32768), ('max_model_len', 1048576),
                           ('served_model_name', 'alias'), ('served_model_name', 'DeepSeek-V4-Flash'),
                           ('served_model_name', 'DeepSeek-V4.1-Flash'),
                           ('max_num_batched_tokens', 512), ('max_num_batched_tokens', 4096), ('max_num_batched_tokens', 8192),
                           ('max_num_batched_tokens', True), ('enable_chunked_prefill', False),
                           ('disable_chunked_mm_input', False), ('hf_overrides', {'vision_max_n_token': 2048}), ('hf_overrides', {'vision_max_n_token': 1024}),
                           ('hf_overrides', {'is_mm_prefix_lm': False}),
                           ('engram_config', {'cpu_offload': False}), ('limit_mm_per_prompt', {'image': 4}),
                           ('gpu_memory_utilization', 0.90), ('gpu_memory_utilization', 0.94)]:
            candidate = copy.deepcopy(base)
            candidate[key] = value
            with self.assertRaises(ValueError): ops.resolve(candidate, 64)
        for invalid_concurrency in (0, 31, 32, 33, 56, 63, 65, True):
            with self.assertRaises(ValueError): ops.resolve(base, invalid_concurrency)

    def test_eight_rank_graph_and_replay_checks_not_bypassed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for pid in range(8):
                topology = dict(data_parallel_rank=pid//4, data_parallel_size=2, tensor_parallel_size=4, expert_parallel_enabled=True)
                (directory / f'draft-{pid}.json').write_text(json.dumps({'kind': 'dspark', 'pid': pid,
                    **topology,
                    'num_speculative_tokens': 5, 'draft_query_per_request': 5, 'fused_markov_built': True, 'local_argmax': True}))
                for query in (5, 6):
                    (directory / f'capture-{pid}-{query}.json').write_text(json.dumps({'kind': 'graph-capture',
                        **topology,
                        'pid': pid, 'manager_id': query, 'memory_estimation_probe': False, 'decode_query_len': query,
                        'draft_context_kv_captured': query == 5,
                        'captured_full_graphs': [{'num_tokens': query*32, 'mode': 'FULL'}]}))
                    (directory / f'replay-{pid}-{query}.json').write_text(json.dumps({'kind': 'graph-replay',
                        **topology,
                        'pid': pid, 'manager_id': query, 'decode_query_len': query,
                        'draft_context_kv_captured': query == 5,
                        'descriptor': {'mode': 'FULL', 'num_tokens': query}}))
            graphs = ops.graphs(directory, 64, 5)
            self.assertEqual(graphs['status'], 'PASS')
            self.assertEqual(graphs['concurrency'], 64)
            self.assertEqual(graphs['per_dp_concurrency'], 32)
            self.assertEqual(graphs['target_shape'], 192)
            self.assertEqual(graphs['draft_shape'], 160)
            self.assertEqual(ops.replays(directory, 64, 5)['status'], 'PASS')
            with self.assertRaises(ValueError): ops.graphs(directory, 64, 7)
            with self.assertRaises(ValueError): ops.replays(directory, 64, 7)
            for record_type, checker in (('capture', ops.graphs), ('replay', ops.replays)):
                path = directory / f'{record_type}-7-5.json'
                original = json.loads(path.read_text())
                for invalid in (False, None):
                    bad = dict(original)
                    if invalid is None:
                        bad.pop('draft_context_kv_captured')
                    else:
                        bad['draft_context_kv_captured'] = invalid
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): checker(directory, 64, 5)
                path.write_text(json.dumps(original))
            (directory / 'replay-7-6.json').unlink()
            with self.assertRaises(ValueError): ops.replays(directory, 64, 5)
            (directory / 'capture-7-6.json').unlink()
            with self.assertRaises(ValueError): ops.graphs(directory, 64, 5)

    def test_requests_and_per_position_metrics(self):
        first = bench.workload('fixed', 'high', 'single', 0)
        self.assertEqual(first, bench.workload('fixed', 'high', 'single', 0))
        self.assertNotEqual(first, bench.workload('fixed', 'high', 'load', 0))
        self.assertEqual(first['max_tokens'], 1024)
        rows = bench.parse_metrics('vllm:spec_decode_num_drafts_total{model_name="DeepSeek-Flash"} 100\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 700\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 200\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="6"} 4\n')
        delta = bench.metric_delta({'totals': {}, 'positions': {}}, rows)
        self.assertEqual(delta['mean_draft_length'], 7)
        self.assertEqual(delta['accepted_fraction_of_rounds_by_position']['6'], .04)
        self.assertEqual(delta['accepted_per_draft'], 2)

    def test_private_history_benchmark_preserves_images_without_mutating_input(self):
        body = {'model': 'DeepSeek-Flash', 'messages': [
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]},
            {'role': 'assistant', 'content': 'Previous analysis'},
            {'role': 'user', 'content': 'Continue the explanation'}], 'max_tokens': 256}
        original = copy.deepcopy(body)
        result = bench.template_workload(body, 'high')
        self.assertEqual(body, original)
        self.assertEqual(result['messages'], original['messages'])
        self.assertEqual(result['max_tokens'], 1024)
        self.assertTrue(result['chat_template_kwargs']['thinking'])
        with self.assertRaises(ValueError):
            bench.template_workload({**body, 'model': 'alias'}, 'high')

    def test_capacity_takes_full_500k_windows_per_pool_before_summing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = {'pid': 1, 'time_ns': 1, 'max_model_len': 512000, 'configured_max_num_seqs': 32,
                   'data_parallel_rank':0, 'data_parallel_size':2, 'tensor_parallel_size':4, 'expert_parallel_enabled':True,
                   'groups': [{'full_window_blocks': 10}], 'blocks_per_full_window': 10,
                   'pool_blocks': 149, 'full_windows': 14, 'effective_kv_tokens': 149*512000//10}
            (directory / 'capacity-1.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'Missing capacity'):
                ops.capacity(directory, 64)
            (directory / 'capacity-2.json').write_text(json.dumps(dict(row, pid=2, data_parallel_rank=1)))
            self.assertEqual(ops.capacity(directory, 64)['required_concurrency'], 64)
            self.assertEqual(ops.capacity(directory, 64)['tp_capacity_multiplier'], 1)
            # 14.9 + 14.9 windows is 28 resident full windows, NOT 29.
            self.assertEqual(ops.capacity(directory, 64)['full_windows'], 28)
            (directory / 'capacity-3.json').write_text(json.dumps(dict(row, pid=3)))
            self.assertEqual(ops.capacity(directory, 64)['required_concurrency'], 64)

    def test_500k_capacity_floor_minimum_and_stale_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = {'pid': 1, 'time_ns': 1, 'max_model_len': 512000, 'configured_max_num_seqs': 32,
                   'data_parallel_rank': 0, 'data_parallel_size': 2, 'tensor_parallel_size': 4,
                   'expert_parallel_enabled': True, 'groups': [{'full_window_blocks': 10}],
                   'blocks_per_full_window': 10}
            def pool(rank, blocks, **changed):
                value = dict(row, pid=rank+1, data_parallel_rank=rank, pool_blocks=blocks,
                             full_windows=blocks//10, effective_kv_tokens=blocks*512000//10)
                value.update(changed)
                (directory / f'capacity-{rank}.json').write_text(json.dumps(value))
            for counts, windows in [((19, 19), 2), ((89, 89), 16),
                                    ((99, 89), 17)]:
                for rank, blocks in enumerate(counts): pool(rank, blocks)
                result = ops.capacity(directory, 64)
                self.assertEqual(result['full_windows'], windows)
                self.assertEqual(result['required_concurrency'], 64)
                self.assertEqual(result['formula'], 'fixed(64)')
            pool(0, 9)
            with self.assertRaisesRegex(ValueError, 'Each DP pool'):
                ops.capacity(directory, 64)
            pool(0, 19, max_model_len=262144)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                ops.capacity(directory, 64)

    def test_start_seed_does_not_reuse_256k_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'capacity.json'
            self.assertEqual(ops.initial_concurrency(path), 64)
            valid = dict(status='CALIBRATED', topology='TP4-DP2-EP8', max_model_len=512000,
                         formula=ops.CAPACITY_FORMULA, required_concurrency=56)
            path.write_text(json.dumps(valid))
            self.assertEqual(ops.initial_concurrency(path), 64)
            for changed in ({'max_model_len':262144}, {'topology':'TP8'},
                            {'formula':'max(32,2*sum(dp_effective_kv_tokens//262144))'},
                            {'required_concurrency':True}, {'required_concurrency':31},
                            {'required_concurrency':33}, {'required_concurrency':64},
                            {'status':'RESTART_REQUIRED'}):
                path.write_text(json.dumps(dict(valid, **changed)))
                self.assertEqual(ops.initial_concurrency(path), 64)
            for text in ('{broken', 'null', '[]'):
                path.write_text(text)
                self.assertEqual(ops.initial_concurrency(path), 64)

    def test_long_acceptance_reserves_output_inside_fixed_window(self):
        import acceptance
        self.assertEqual(acceptance.LENGTH, ops.LENGTH)
        self.assertEqual(acceptance.LONG_INPUT_BUDGETS[-1], 507904)
        self.assertNotIn(524288, acceptance.LONG_INPUT_BUDGETS)
        self.assertEqual(acceptance.LONG_INPUT_BUDGETS[-1] + 4096, ops.LENGTH)


if __name__ == '__main__':
    unittest.main(verbosity=2)
