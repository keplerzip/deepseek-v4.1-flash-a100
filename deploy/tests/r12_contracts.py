"""CPU-only routing, topology, telemetry and source-version contracts."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

DEPLOY = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load('r12_guard', DEPLOY / 'overrides/dsv41_guard.py')
recorder = load('r12_recorder', DEPLOY / 'overrides/performance/patched/dsv41_capacity.py')


class Checks(unittest.TestCase):
    def test_history_sticks_to_same_pool_and_independent_conversations_spread(self):
        for key in ('messages', 'input'):
            original = {key: [{'role': 'user', 'content': 'initial question'}]}
            following = {key: original[key] + [{'role': 'assistant', 'content': 'response'}, {'role': 'user', 'content': 'continue'}]}
            self.assertEqual(guard.dp_affinity(original, [], 2), guard.dp_affinity(following, [], 2))
        ranks = {guard.dp_affinity({'input': 'independent request ' + str(i)}, [], 2) for i in range(64)}
        self.assertEqual(ranks, {0, 1})
        self.assertIsNone(guard.dp_affinity({'previous_response_id': 'resp_1'}, [], 2))
        for rank in (0, 1):
            self.assertEqual(guard.dp_affinity({}, [(b'x-data-parallel-rank', str(rank).encode())], 2), rank)
        with self.assertRaises(ValueError): guard.dp_affinity({}, [(b'x-data-parallel-rank', b'2')], 2)

    def test_middleware_injects_rank_and_preserves_exact_body_for_each_protocol(self):
        async def run(path, payload):
            body = json.dumps(payload).encode()
            sent = []
            async def app(scope, receive, send):
                self.assertEqual(dict(scope['headers'])[b'x-data-parallel-rank'], b'1')
                self.assertEqual((await receive())['body'], body)
                await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            async def receive(): return {'type': 'http.request', 'body': body}
            async def send(event): sent.append(event)
            with patch.dict(os.environ, DSV41_DP_SIZE='2', VLLM_API_KEY=''):
                await guard.OfflineGuard(app)({'type':'http','method':'POST','path':path,
                    'headers':[(b'x-data-parallel-rank', b'1')]}, receive, send)
            self.assertEqual(dict(sent[0]['headers'])[b'x-dsv41-dp-rank'], b'1')
        for path in ('chat/completions', 'responses', 'messages'):
            asyncio.run(run('/v1/'+path, {'model': guard.MODEL, 'input': 'test'}))

    def test_capacity_and_graph_callbacks_record_actual_dp_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, DSV41_OBSERVATION_DIR=tmp):
            config = NS(parallel_config=NS(data_parallel_rank=1, data_parallel_size=2, tensor_parallel_size=4, enable_expert_parallel=True),
                        model_config=NS(max_model_len=262144), scheduler_config=NS(max_num_seqs=28))
            spec = NS(page_size_bytes=128, max_memory_usage_bytes=lambda _:1280, block_size=32)
            recorder.record_capacity(config, NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=['x'])], num_blocks=149), 0, 0)
            row = json.loads(next(Path(tmp).glob('capacity-*.json')).read_text())
            self.assertEqual((row['full_windows'], row['data_parallel_rank'], row['configured_max_num_seqs']), (14, 1, 28))


if __name__ == '__main__': unittest.main(verbosity=2)
