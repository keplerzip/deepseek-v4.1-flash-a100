"""Paired fixed-request benchmark. GPU results are created only by live HTTP."""
import argparse
from collections import Counter
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import threading
import time

from acceptance import Client, MODEL, LENGTH, sse_chat, vision_request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values)-1, int((len(values)-1)*fraction)))]


def workload(seed, mode, phase, index):
    # Same seed/body for both k values; distinct early prefixes per request.
    tag = hashlib.sha256(f'{seed}/{mode}/{phase}/{index}'.encode()).hexdigest()[:24]
    return {'model': MODEL, 'messages': [{'role': 'user', 'content':
        f'Run {tag}. Explain how to implement a reliable LRU cache in Python. '
        'Include code, invariants, edge cases, tests and performance analysis. '
        'Continue with concrete examples and discuss concurrent callers.'}],
        'temperature': 0, 'seed': 42, 'max_tokens': 1024, 'ignore_eos': True,
        'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'thinking': mode == 'high', 'reasoning_effort': 'high'}}


def template_workload(template, mode):
    """Keep the same conversation/images while controlling benchmark sampling."""
    if template.get('model', MODEL) != MODEL or not isinstance(template.get('messages'), list):
        raise ValueError('Request file must be a Chat Completions body for DeepSeek-Flash')
    body = copy.deepcopy(template)
    for key in ('max_completion_tokens', 'stop', 'reasoning_effort', 'thinking'):
        body.pop(key, None)
    body.update(model=MODEL, temperature=0, seed=42, max_tokens=1024, ignore_eos=True,
                n=1, stream=True, stream_options={'include_usage': True})
    body['chat_template_kwargs'] = {**body.get('chat_template_kwargs', {}),
        'thinking': mode == 'high', 'reasoning_effort': 'high'}
    return body


def parse_metrics(text):
    totals, positions = {}, {}
    for line in text.splitlines():
        m = re.fullmatch(r'(vllm:[a-zA-Z0-9_:]+)(?:\{(.*)\})?\s+([-+0-9.eE]+)', line)
        if not m:
            continue
        name, labels, value = m.groups()
        totals[name] = totals.get(name, 0) + float(value)
        if name == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            pos = re.search(r'(?:^|,)position="(\d+)"', labels or '')
            if pos:
                positions[pos[1]] = positions.get(pos[1], 0) + float(value)
    return {'totals': totals, 'positions': positions}


def snapshot(client):
    return parse_metrics(client.request('/metrics', raw=True))


def idle_metrics(client):
    previous = None
    for _ in range(15):
        current = snapshot(client)
        totals = current['totals']
        busy = sum(totals.get(n, 0) for n in ['vllm:num_requests_running', 'vllm:num_requests_waiting'])
        counters = {k: v for k, v in totals.items() if k.endswith('_total')}
        if busy == 0 and previous == counters:
            current['settled'] = True
            return current
        previous = counters
        time.sleep(1)
    # Request timings and usage remain useful under contention. Preserve them,
    # then label the report non-signoff instead of silently skipping load.
    current['settled'] = False
    return current


def metric_delta(before, after):
    totals = {k: v-before['totals'].get(k, 0) for k, v in after['totals'].items() if k.endswith('_total')}
    drafts = totals.get('vllm:spec_decode_num_drafts_total', 0)
    proposed = totals.get('vllm:spec_decode_num_draft_tokens_total', 0)
    accepted = totals.get('vllm:spec_decode_num_accepted_tokens_total', 0)
    positions = {k: v-before['positions'].get(k, 0) for k, v in after['positions'].items()}
    return {'counters': totals, 'drafts': drafts, 'draft_tokens': proposed, 'accepted_tokens': accepted,
            'acceptance_rate': accepted/proposed if proposed else None,
            'accepted_per_draft': accepted/drafts if drafts else None,
            'mean_draft_length': proposed/drafts if drafts else None,
            'accepted_tokens_by_position_zero_based': positions,
            'accepted_fraction_of_rounds_by_position': {k: v/drafts for k, v in positions.items()} if drafts else {}}


def accounting_issues(before, after, delta, expected_tokens, k):
    """Global metrics can be attributed only after exact request accounting."""
    issues = []
    if not before.get('settled', True) or not after.get('settled', True):
        issues.append('Engine counters did not settle; other clients may be active')
    observed = delta['counters'].get('vllm:generation_tokens_total')
    if observed is None or observed != expected_tokens:
        issues.append(f'Generation accounting mismatch: metrics={observed}, benchmark={expected_tokens}; possible other traffic or counter reset')
    if any(v < 0 for v in delta['counters'].values()):
        issues.append('Global metric counters decreased during the phase')
    if not (delta['drafts'] > 0 and 0 < delta['accepted_tokens'] <= delta['draft_tokens']):
        issues.append('DSpark draft/accept counters are missing or inconsistent')
    if delta['mean_draft_length'] is None or not 0 < delta['mean_draft_length'] <= k + 1e-6:
        issues.append('DSpark mean draft length is outside the configured bound')
    return issues


def sample_summary(rows, phases):
    """Compute client measurements independently of shared service counters."""
    single = [r for r in rows if r['phase'] == 'single' and r['status'] == 'PASS']
    load = [r for r in rows if r['phase'] == 'load' and r['status'] == 'PASS']
    load_seconds = phases.get('load', {}).get('seconds', 0)
    return dict(
        completed_requests=sum(r['status'] == 'PASS' for r in rows),
        load_completed_requests=len(load),
        load_output_tokens=sum(r['output_tokens'] for r in load),
        load_requests_by_dp=dict(Counter(str(r.get('data_parallel_rank')) for r in load)),
        single_decode_median=statistics.median([r['decode_tps'] for r in single]) if single else None,
        single_decode_min=min([r['decode_tps'] for r in single], default=None),
        single_decode_max=max([r['decode_tps'] for r in single], default=None),
        single_end_to_end_tps_median=statistics.median([r['end_to_end_tps'] for r in single]) if single else None,
        single_ttft_median=statistics.median([r['ttft_seconds'] for r in single]) if single else None,
        load_output_tokens_per_second=sum(r['output_tokens'] for r in load)/load_seconds if load and load_seconds > 0 else None,
        load_latency_p50=percentile([r['seconds'] for r in load], .5),
        load_latency_p95=percentile([r['seconds'] for r in load], .95),
        load_ttft_p95=percentile([r['ttft_seconds'] for r in load], .95))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', required=True)
    p.add_argument('--mode', choices=['off', 'high'], required=True)
    p.add_argument('--k', type=int, choices=[5], default=5)
    p.add_argument('--concurrency', type=int, choices=(32, 64), default=64,
                   help='Load concurrency; use 32 for the historical comparison')
    p.add_argument('--clients-isolated', action='store_true',
                   help='Shell wrapper paused this deployment frontend; direct internal clients can still contend')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--request-file', type=Path, help='Private Chat body; preserved images/history, controlled 1024-token sampling')
    source.add_argument('--images', type=int, choices=(1, 5, 8), help='Small bundled OCR fixtures, not a real screenshot-history benchmark')
    a = p.parse_args()
    template = json.loads(a.request_file.read_text()) if a.request_file else None
    if a.images:
        template = vision_request(a.images, 'chat')
        template['messages'][0]['content'][0]['text'] = (
            'Explain a robust screenshot OCR and document processing pipeline in Python. '
            'Use the attached images as examples. Include code, edge cases and tests. Continue in detail.')

    def make_workload(phase, index):
        return template_workload(template, a.mode) if template is not None else workload(a.seed, a.mode, phase, index)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    client = Client(a.base_url, '', timeout=600)
    rows, errors, contamination, phases = [], [], [], {}
    trace = a.output.with_suffix('.jsonl')
    trace.write_text('')
    report = {'status': 'RUNNING', 'model': MODEL, 'k': a.k, 'thinking': a.mode == 'high',
        'reasoning_effort': 'high' if a.mode == 'high' else None, 'context': LENGTH,
        'concurrency': a.concurrency, 'output_tokens_per_request': 1024, 'warmups': 2, 'single_samples': 5,
        'load_requests': 200, 'seed': a.seed, 'sampling_seed': 42, 'temperature': 0,
        'started_at': datetime.datetime.now().astimezone().isoformat(),
        'timing_note': 'completion tokens include reasoning; SSE chunks are not tokens. Decode estimate=(usage completion_tokens-1)/(last-first token-bearing event). Also report completion_tokens/total HTTP duration.',
        'engram_cpu_offload': True, 'trace_file': trace.name,
        'frontend_paused': a.clients_isolated,
        'aggregate_timing_note': 'load usage completion tokens / entire client load-phase wall time, excluding metric settlement; includes queueing and prefill',
        'signoff_note': 'CONTAMINATED retains real client timings but is not an isolated performance signoff; global DSpark counters cannot be attributed to these requests'}
    if template is None:
        bodies = [make_workload(phase, i) for phase, n in [('warmup', 2), ('single', 5), ('load', 200)] for i in range(n)]
        report['request_set_sha256'] = digest(bodies)
    else:
        report['request_set_sha256'] = digest({'repeated_body': make_workload('single', 0), 'layout': [2, 5, 200]})
    report['workload_source'] = 'private_request_file' if a.request_file else 'small_image_fixtures' if a.images else 'synthetic_text'
    report['fixture_images'] = a.images
    report['cache_policy'] = 'two warmups; repeated conversation shares prefix cache' if template is not None else 'distinct request prefixes'

    def save():
        temp = a.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(a.output)

    def one(phase, index):
        body = make_workload(phase, index)
        start = time.monotonic()
        try:
            stream = client.request('/v1/chat/completions', body)
            text, usage, arrivals = sse_chat(stream)
            assert usage['completion_tokens'] == 1024, usage
            assert stream['elapsed'] > 0 and len(arrivals) > 1 and arrivals[-1] > arrivals[0]
            assert '\ufffd' not in text, 'Replacement characters in output'
            assert not re.search(r'<[｜|]?DSML[｜|]?', text), 'Unparsed DSML in output'
            details = usage.get('completion_tokens_details') or {}
            return {'phase': phase, 'index': index, 'status': 'PASS', 'request_sha256': digest(body),
                'data_parallel_rank': stream.get('_dp_rank'),
                'input_tokens': usage['prompt_tokens'], 'output_tokens': usage['completion_tokens'],
                'reasoning_tokens': details.get('reasoning_tokens'),
                'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0),
                'ttft_seconds': arrivals[0], 'seconds': stream['elapsed'],
                'decode_tps': 1023/(arrivals[-1]-arrivals[0]), 'end_to_end_tps': 1024/stream['elapsed'],
                'token_bearing_events': len(arrivals), 'visible_text_sha256': hashlib.sha256(text.encode()).hexdigest()}
        except Exception as exc:
            return {'phase': phase, 'index': index, 'status': 'FAIL', 'request_sha256': digest(body),
                    'seconds': time.monotonic()-start, 'error': str(exc)[:2000]}

    def record(row):
        rows.append(row)
        with trace.open('a') as f:
            f.write(json.dumps(row) + '\n')
        if row['status'] != 'PASS':
            raise RuntimeError(f"{row['phase']} request {row['index']}: {row['error']}")

    phase_state = ['initializing']
    stop_heartbeat = threading.Event()
    def heartbeat():
        while not stop_heartbeat.wait(15):
            print(f'[k={a.k} thinking={a.mode}] {phase_state[0]}: {len(rows)} requests recorded', flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    save()
    try:
        assert [m['id'] for m in client.request('/v1/models')['data']] == [MODEL]
        for phase, count, concurrency in [('warmup', 2, 1), ('single', 5, 1), ('load', 200, a.concurrency)]:
            phase_state[0] = phase
            before = idle_metrics(client)
            start = time.monotonic()
            # Keep a partial phase if a real request fails, so successful
            # client samples do not disappear from the final report.
            phases[phase] = {'seconds': None, 'requests': count, 'concurrency': concurrency,
                             'completed': False}
            if concurrency == 1:
                for i in range(count):
                    row = one(phase, i)
                    record(row)
                    print(f"[k={a.k} thinking={a.mode}] {phase} {i+1}/{count}: {row.get('decode_tps', 0):.2f} tok/s", flush=True)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    for future in as_completed([pool.submit(one, phase, i) for i in range(count)]):
                        record(future.result())
            elapsed = time.monotonic()-start  # Excludes metrics-settlement polling.
            phases[phase].update(seconds=elapsed, completed=True)
            after = idle_metrics(client)
            delta = metric_delta(before, after)
            # Near output limits a round may be shorter than k. Do not invent
            # exact-k proof from this mean; startup observations prove actual k.
            issues = accounting_issues(before, after, delta, count*1024, a.k)
            contamination.extend(f'{phase}: {issue}' for issue in issues)
            phases[phase].update(metrics=delta, accounting_status='CONTAMINATED' if issues else 'PASS',
                                  accounting_issues=issues, global_counters_attributable=not issues)
            if issues:
                print(f'[k={a.k} thinking={a.mode}] {phase}: CONTAMINATED; preserving client samples and continuing load', flush=True)
            report['phases'] = phases
            save()
    except Exception as exc:
        errors.append(str(exc)[:4000])
    finally:
        stop_heartbeat.set()
        if phase_state[0] in phases and phases[phase_state[0]]['seconds'] is None:
            phases[phase_state[0]]['seconds'] = time.monotonic()-start
        status = 'FAIL' if errors or len(rows) != 207 else 'CONTAMINATED' if contamination else 'PASS'
        report.update(status=status, signoff_eligible=status == 'PASS', errors=errors,
                      contamination=contamination, **sample_summary(rows, phases),
                      phases=phases, finished_at=datetime.datetime.now().astimezone().isoformat())
        save()
    print(json.dumps(report), flush=True)
    return report['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
