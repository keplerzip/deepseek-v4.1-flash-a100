"""Run benchmark accounting and frontend lifecycle contracts without a GPU."""
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor as RealExecutor
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import dspark_benchmark as bench


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.completed = 0
        self.lock = threading.Lock()

    def request(self, path, body=None, **kwargs):
        if path == '/v1/models':
            return {'data': [{'id': bench.MODEL}]}
        assert path == '/v1/chat/completions'
        with self.lock:
            self.completed += 1
            rank = self.completed % 2
        return {'elapsed': 8., '_dp_rank': str(rank)}


def fake_stream(stream):
    return 'Valid measured fixture output', {
        'prompt_tokens': 31, 'completion_tokens': 1024,
        'prompt_tokens_details': {'cached_tokens': 0}}, [.1, 4.1]


class BenchmarkContracts(unittest.TestCase):
    def run_benchmark(self, *, contaminated=False, extra=()):
        clients, executor_sizes = [], []
        noise = [0]

        def client(*args, **kwargs):
            value = FakeClient(*args, **kwargs)
            clients.append(value)
            return value

        def metrics(value):
            noise[0] += 17 if contaminated else 0
            n = value.completed
            return {'settled': True, 'positions': {}, 'totals': {
                'vllm:generation_tokens_total': n*1024+noise[0],
                'vllm:spec_decode_num_drafts_total': n*300,
                'vllm:spec_decode_num_draft_tokens_total': n*1500,
                'vllm:spec_decode_num_accepted_tokens_total': n*800}}

        def executor(*, max_workers):
            executor_sizes.append(max_workers)
            return RealExecutor(max_workers=max_workers)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'report.json'
            args = ['dspark_benchmark.py', '--output', str(output), '--seed', 'cpu-contract',
                    '--mode', 'off', *extra]
            with patch.object(sys, 'argv', args), patch.object(bench, 'Client', client), \
                 patch.object(bench, 'idle_metrics', metrics), patch.object(bench, 'sse_chat', fake_stream), \
                 patch.object(bench, 'ThreadPoolExecutor', executor), redirect_stdout(io.StringIO()):
                result = bench.main()
            report = json.loads(output.read_text())
            rows = [json.loads(line) for line in output.with_suffix('.jsonl').read_text().splitlines()]
        return result, report, rows, clients[0].completed, executor_sizes

    def test_default_c64_clean(self):
        result, report, rows, requests, executor_sizes = self.run_benchmark()
        self.assertFalse(result)
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['signoff_eligible'])
        self.assertEqual(report['concurrency'], 64)
        self.assertEqual(executor_sizes, [64])
        self.assertEqual((len(rows), requests), (207, 207))
        self.assertEqual(report['load_completed_requests'], 200)
        self.assertEqual(report['load_output_tokens'], 204800)
        self.assertAlmostEqual(report['single_decode_median'], 255.75)
        self.assertEqual(report['load_output_tokens_per_second'],
                         204800/report['phases']['load']['seconds'])

    def test_ambient_metrics_keep_all_phases_and_c32_samples(self):
        result, report, rows, requests, executor_sizes = self.run_benchmark(
            contaminated=True, extra=('--concurrency', '32', '--clients-isolated'))
        self.assertTrue(result)
        self.assertEqual(report['status'], 'CONTAMINATED')
        self.assertFalse(report['signoff_eligible'])
        self.assertEqual(report['concurrency'], 32)
        self.assertEqual(executor_sizes, [32])
        self.assertEqual((len(rows), requests), (207, 207))
        self.assertEqual(report['load_output_tokens'], 204800)
        self.assertGreater(report['load_output_tokens_per_second'], 0)
        self.assertEqual(len(report['contamination']), 3)
        self.assertTrue(report['frontend_paused'])
        for phase in ('warmup', 'single', 'load'):
            self.assertFalse(report['phases'][phase]['global_counters_attributable'])

    def test_private_request_history_and_images_preserved(self):
        original = {'model': bench.MODEL, 'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'private history'},
             {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJD'}}]}],
             'temperature': .5}
        baseline = json.loads(json.dumps(original))
        body = bench.template_workload(original, 'high')
        self.assertEqual(original, baseline)
        self.assertEqual(body['messages'], original['messages'])
        self.assertEqual(body['model'], 'DeepSeek-Flash')
        self.assertEqual(body['max_tokens'], 1024)
        self.assertTrue(body['chat_template_kwargs']['thinking'])
        self.assertEqual(len(bench.vision_request(8)['messages'][0]['content']), 9)

    def test_unsettled_metrics_do_not_abort_measurement(self):
        snapshots = iter({'totals': {'vllm:num_requests_running': 1,
                                    'vllm:generation_tokens_total': i}, 'positions': {}}
                         for i in range(15))
        with patch.object(bench, 'snapshot', side_effect=lambda _: next(snapshots)), \
             patch.object(bench.time, 'sleep'):
            value = bench.idle_metrics(None)
        self.assertFalse(value['settled'])

    def test_counter_reset_and_missing_dspark_are_not_signoff(self):
        before = {'totals': {'vllm:generation_tokens_total': 2048}, 'positions': {}, 'settled': True}
        after = {'totals': {'vllm:generation_tokens_total': 0}, 'positions': {}, 'settled': True}
        issues = bench.accounting_issues(before, after, bench.metric_delta(before, after), 1024, 5)
        self.assertGreaterEqual(len(issues), 3)


FAKE_COMMON = r'''
DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
STATE_DIR="$DEPLOY_DIR/runtime"
ENGINE_NAME=deepseek-v4.1-flash-engine
FRONTEND_NAME=deepseek-v4.1-flash-api
RUNTIME_IMAGE=sha256:cpu-test
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_operator() { :; }
resolve_image() { :; }
engine_health() { :; }
own_container() {
  [[ "$1" == "$ENGINE_NAME" || "${BENCH_CASE:-clean}" != foreign-owner ]]
}
docker() {
  printf '%s\n' "$*" >> "$STATE_DIR/calls.log"
  if [[ "$1" == inspect && "${2:-}" == --format ]]; then
    case "$3" in
      *frontend.py*)
        [[ "${BENCH_CASE:-clean}" != foreign-mount ]] || { printf '/other/frontend.py\n'; return; }
        printf '%s/scripts/local_access_frontend.py\n' "$DEPLOY_DIR" ;;
      *access.json*) printf '%s/local-access.json\n' "$STATE_DIR" ;;
      *State.Running*) printf 'true\n' ;;
    esac
  elif [[ "$1" == exec ]]; then
    [[ "${BENCH_CASE:-clean}" != signal ]] || kill -TERM $$
    if [[ "${BENCH_CASE:-clean}" == benchmark-fail && "$*" == *'--mode off'* ]]; then
      return 1
    fi
  fi
}
'''


class FrontendLifecycleContracts(unittest.TestCase):
    def run_shell(self, case='clean', extra=(), lock=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('tests', 'scripts', 'runtime'):
                (root/name).mkdir()
            script = Path(__file__).with_name('performance-benchmark.sh').read_text()
            (root/'tests/performance-benchmark.sh').write_text(script)
            (root/'scripts/common.sh').write_text(FAKE_COMMON)
            (root/'scripts/local-access.sh').write_text(
                '#!/bin/bash\nprintf "restored\\n" >> "$(dirname "$0")/../runtime/calls.log"\n')
            (root/'runtime/current-run.txt').write_text('current\n')
            for name in ('runtime.resolved.json', 'capacity.json'):
                (root/'runtime'/name).write_text('{}\n')
            with (root/'runtime/operation.lock').open('w') as handle:
                if lock:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(['bash', str(root/'tests/performance-benchmark.sh'),
                                         '--isolate-clients', *extra],
                                        env={**os.environ, 'BENCH_CASE': case},
                                        text=True, capture_output=True, timeout=20)
            calls_path = root/'runtime/calls.log'
            calls = calls_path.read_text().splitlines() if calls_path.exists() else []
        return result, calls

    def assert_only_frontend_stopped(self, calls):
        stops = [line for line in calls if line.startswith('stop ')]
        self.assertEqual(stops, ['stop -t 90 deepseek-v4.1-flash-api'])
        self.assertFalse(any('new-api' in line or 'newapi' in line for line in calls))
        self.assertFalse(any(line.startswith(('run ', 'restart ')) for line in calls))

    def test_isolation_stops_only_frontend_and_restores_after_both_modes(self):
        result, calls = self.run_shell(extra=('--images', '8', '--concurrency', '32'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_only_frontend_stopped(calls)
        self.assertEqual(calls[-1], 'restored')
        runs = [line for line in calls if line.startswith('exec ')]
        self.assertEqual(len(runs), 2)
        self.assertTrue(all('--images 8 --concurrency 32 --clients-isolated' in line for line in runs))

    def test_contaminated_or_failed_off_still_runs_high_and_restores(self):
        result, calls = self.run_shell('benchmark-fail')
        self.assertEqual(result.returncode, 1)
        self.assert_only_frontend_stopped(calls)
        self.assertTrue(any('--mode high' in line for line in calls))
        self.assertEqual(calls[-1], 'restored')

    def test_term_restores_frontend(self):
        result, calls = self.run_shell('signal')
        self.assertEqual(result.returncode, 143)
        self.assert_only_frontend_stopped(calls)
        self.assertEqual(calls[-1], 'restored')

    def test_foreign_owner_or_mount_is_never_stopped(self):
        for case in ('foreign-owner', 'foreign-mount'):
            result, calls = self.run_shell(case)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(any(line.startswith('stop ') for line in calls))
            self.assertNotIn('restored', calls)

    def test_operation_lock_prevents_overlap(self):
        result, calls = self.run_shell(lock=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Another deployment operation is active', result.stderr)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
