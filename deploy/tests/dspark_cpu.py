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
        for k, shapes in [(5, (280, 336))]:
            candidate = copy.deepcopy(base)
            candidate['speculative_config']['num_speculative_tokens'] = k
            result, argv = ops.resolve(candidate, 56)
            self.assertEqual(result['max_model_len'], 262144)
            self.assertEqual(result['engram_config'], {'cpu_offload': True})
            self.assertTrue(set(shapes) <= set(result['compilation_config']['cudagraph_capture_sizes']))
            self.assertEqual(result['speculative_config']['num_speculative_tokens'], k)
        for invalid in (True, 4, 6, 7, 8, 10):
            candidate = copy.deepcopy(base)
            candidate['speculative_config']['num_speculative_tokens'] = invalid
            with self.assertRaises(ValueError): ops.resolve(candidate, 56)
        for key, value in [('max_model_len', 32768), ('served_model_name', 'alias'), ('engram_config', {'cpu_offload': False})]:
            candidate = copy.deepcopy(base)
            candidate[key] = value
            with self.assertRaises(ValueError): ops.resolve(candidate, 56)

    def test_eight_rank_graph_and_replay_checks_not_bypassed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for pid in range(8):
                (directory / f'draft-{pid}.json').write_text(json.dumps({'kind': 'dspark', 'pid': pid,
                    'num_speculative_tokens': 5, 'draft_query_per_request': 5, 'fused_markov_built': True, 'local_argmax': True}))
                for query in (5, 6):
                    (directory / f'capture-{pid}-{query}.json').write_text(json.dumps({'kind': 'graph-capture',
                        'pid': pid, 'manager_id': query, 'memory_estimation_probe': False, 'decode_query_len': query,
                        'captured_full_graphs': [{'num_tokens': query*56, 'mode': 'FULL'}]}))
                    (directory / f'replay-{pid}-{query}.json').write_text(json.dumps({'kind': 'graph-replay',
                        'pid': pid, 'manager_id': query, 'decode_query_len': query, 'descriptor': {'mode': 'FULL', 'num_tokens': query}}))
            self.assertEqual(ops.graphs(directory, 56, 5)['status'], 'PASS')
            self.assertEqual(ops.replays(directory, 56, 5)['status'], 'PASS')
            with self.assertRaises(ValueError): ops.graphs(directory, 56, 7)
            with self.assertRaises(ValueError): ops.replays(directory, 56, 7)
            (directory / 'replay-7-6.json').unlink()
            with self.assertRaises(ValueError): ops.replays(directory, 56, 5)
            (directory / 'capture-7-6.json').unlink()
            with self.assertRaises(ValueError): ops.graphs(directory, 56, 5)

    def test_requests_and_per_position_metrics(self):
        first = bench.workload('fixed', 'high', 'single', 0)
        self.assertEqual(first, bench.workload('fixed', 'high', 'single', 0))
        self.assertNotEqual(first, bench.workload('fixed', 'high', 'load', 0))
        self.assertEqual(first['max_tokens'], 1024)
        rows = bench.parse_metrics('vllm:spec_decode_num_drafts_total{model_name="DeepSeek-V4.1-Flash"} 100\n'
            'vllm:spec_decode_num_draft_tokens_total{engine="0"} 700\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 200\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="6"} 4\n')
        delta = bench.metric_delta({'totals': {}, 'positions': {}}, rows)
        self.assertEqual(delta['mean_draft_length'], 7)
        self.assertEqual(delta['accepted_fraction_of_rounds_by_position']['6'], .04)
        self.assertEqual(delta['accepted_per_draft'], 2)

    def test_capacity_formula_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            row = {'pid': 1, 'time_ns': 1, 'max_model_len': 262144, 'configured_max_num_seqs': 56,
                   'groups': [{'full_window_blocks': 10}], 'blocks_per_full_window': 10,
                   'pool_blocks': 280, 'full_windows': 28, 'effective_kv_tokens': 28*262144}
            (directory / 'capacity-1.json').write_text(json.dumps(row))
            self.assertEqual(ops.capacity(directory, 56)['required_concurrency'], 56)
            self.assertEqual(ops.capacity(directory, 56)['tp_capacity_multiplier'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
