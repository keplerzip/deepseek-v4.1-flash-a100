"""Compare one unchanged prefix after an idle interval; never resets cache."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

from acceptance import Client, chat_text, metrics


def now():
    return datetime.datetime.now().astimezone().isoformat()


def identity():
    p = Path('/state/runtime.resolved.json')
    return {'container_hostname': socket.gethostname(),
            'pid1_start_ticks': Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()[19],
            'resolved_config_sha256': hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}


def observe(client, body):
    before = metrics(client)
    response = client.request('/v1/chat/completions', body)
    assert chat_text(response) == 'CACHE_READY', 'Unexpected model response to fixed-prefix probe'
    usage = response['usage']
    cached = usage['prompt_tokens_details']['cached_tokens']
    assert 0 <= cached <= usage['prompt_tokens']
    return {'time': now(), 'unix_seconds': time.time(), 'identity': identity(),
            'request_body_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
            'usage': usage, 'cached_tokens': cached, 'request_seconds': response['_elapsed'],
            'metrics_before': {k: v for k, v in before.items() if 'cache' in k or 'generation_tokens' in k or 'num_requests' in k}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['seed', 'check'])
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    client = Client('http://127.0.0.1:8000', '', timeout=180)
    directory = a.directory
    if a.action == 'seed':
        directory.mkdir(parents=True, exist_ok=False)
        salt = uuid.uuid4().hex
        text = 'Probe '+salt+'. Reference records follow.\n' + '\n'.join(
            f'Row {n}: this is a stable local record with unchanged text.' for n in range(1024)) + '\nReply exactly CACHE_READY.'
        body = {'model':'DeepSeek-V4.1-Flash', 'messages':[{'role':'user', 'content':text}],
                'max_tokens':32, 'temperature':0, 'chat_template_kwargs':{'thinking':False}}
        (directory/'request.json').write_text(json.dumps(body, indent=2)+'\n')
        cold = observe(client, body)
        hot = observe(client, body)
        assert hot['cached_tokens'] > cold['cached_tokens'], (cold, hot)
        result = {'status':'SEEDED', 'cold':cold, 'hot':hot,
                  'instructions':'Keep the service running and avoid all other requests. Run check once after the chosen interval. Each check itself refreshes this prefix.'}
        (directory/'seed.json').write_text(json.dumps(result, indent=2)+'\n')
    else:
        body = json.loads((directory/'request.json').read_text())
        seeded = json.loads((directory/'seed.json').read_text())['hot']
        current = observe(client, body)
        same_engine = seeded['identity'] == current['identity']
        assert seeded['request_body_sha256'] == current['request_body_sha256']
        result = {'status':'OBSERVED', 'same_request_body':True, 'same_engine_and_config':same_engine,
                  'elapsed_since_seed_seconds':current['unix_seconds']-seeded['unix_seconds'],
                  'seed_cached_tokens':seeded['cached_tokens'], 'current':current,
                  'interpretation':('ENGINE_OR_CONFIG_CHANGED' if not same_engine else
                     'PREFIX_STILL_REUSABLE' if current['cached_tokens'] >= seeded['cached_tokens'] else
                     'PREFIX_HIT_DECREASED_REQUIRES_DIAGNOSIS'),
                  'scope':'One controlled prefix, direct model route; this probe cannot certify no other traffic occurred or establish a cache TTL.'}
        (directory/f'check-{time.time_ns()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
