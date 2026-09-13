"""HF005: JSON Schema type fields must survive the offline media guard."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request

MODEL = 'DeepSeek-V4.1-Flash'
SCHEMA = {'type': 'object', 'properties': {
    'type': {'type': 'string', 'description': 'The record category.'},
    'optional': {'type': ['string', 'null']}}, 'required': ['type'], 'additionalProperties': False}


def request_body(protocol, stream=False):
    prompt = 'Call record_probe with type SCHEMA_READY. Omit optional. Do not answer in prose.'
    body = {'model': MODEL, 'stream': stream, 'temperature': 0,
            'chat_template_kwargs': {'thinking': False}}
    function = {'name': 'record_probe', 'description': 'Records a category.', 'parameters': copy.deepcopy(SCHEMA)}
    if protocol == 'responses':
        body.update(input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}],
                    tools=[{'type': 'function', **function}], store=False, max_output_tokens=256)
    elif protocol == 'chat':
        body.update(messages=[{'role': 'user', 'content': prompt}], max_tokens=256,
                    tools=[{'type': 'function', 'function': function}])
        if stream:
            body['stream_options'] = {'include_usage': True}
    else:
        body.update(messages=[{'role': 'user', 'content': prompt}], max_tokens=256,
                    thinking={'type': 'disabled'}, tools=[{'name': function['name'],
                    'description': function['description'], 'input_schema': function['parameters']}])
    return body


def check_cpu(module, captured=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post('/v1/{tail:path}')
    async def echo(request: Request):
        return Response(content=await request.body(), media_type='application/json')

    app.add_middleware(module.OfflineGuard)
    tests = []
    with TestClient(app) as client:
        for protocol in ['chat', 'responses', 'messages']:
            path = '/v1/' + ('chat/completions' if protocol == 'chat' else protocol)
            for stream in [False, True]:
                body = request_body(protocol, stream)
                raw = json.dumps(body, indent=2).encode()
                response = client.post(path, content=raw, headers={'Content-Type': 'application/json'})
                assert response.status_code == 200, response.text
                assert response.content == raw, 'The guard rewrote tool definitions'
                tests.append({'case': protocol + ' stream=' + str(stream), 'status': 'PASS', 'body_bytes_preserved': True})
        body = request_body('responses')
        body['tools'] = [{'type': 'namespace', 'name': 'records', 'description': 'Records', 'tools': body['tools']}]
        body['text'] = {'format': {'type': 'json_schema', 'name': 'record', 'schema': copy.deepcopy(SCHEMA)}}
        response = client.post('/v1/responses', json=body)
        assert response.status_code == 200 and response.json() == body, response.text
        tests.append({'case': 'namespace and structured-output schema', 'status': 'PASS'})
        if captured:
            rows = [json.loads(line) for line in captured.read_text().splitlines() if line.strip()]
            count = 0
            for row in rows:
                if row.get('path') != '/v1/responses':
                    continue
                raw = json.dumps(row['body']).encode()
                response = client.post('/v1/responses', content=raw, headers={'Content-Type': 'application/json'})
                assert response.status_code == 200 and response.content == raw, response.text[:500]
                count += 1
            assert count
            tests.append({'case': 'captured actual Codex requests pass guard unchanged', 'status': 'PASS', 'requests': count})
        for protocol, content in [
            ('chat', {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}),
            ('responses', {'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA'}),
            ('messages', {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAAA'}}),
        ]:
            path = '/v1/' + ('chat/completions' if protocol == 'chat' else protocol)
            body = {'model': MODEL, 'input' if protocol == 'responses' else 'messages': [{'role': 'user', 'content': [content]}]}
            assert client.post(path, json=body).status_code == 200
            bad = copy.deepcopy(content)
            if protocol == 'chat':
                bad['image_url']['url'] = 'https://example.invalid/image.png'
            elif protocol == 'responses':
                bad['image_url'] = 'https://example.invalid/image.png'
            else:
                bad['source'] = {'type': 'url', 'url': 'https://example.invalid/image.png'}
            body['input' if protocol == 'responses' else 'messages'][0]['content'] = [bad]
            assert client.post(path, json=body).status_code == 400
            tests.append({'case': protocol + ' inline image allowed / remote image rejected', 'status': 'PASS'})
        bad_source = {'model': MODEL, 'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': []}]}]}
        assert client.post('/v1/messages', json=bad_source).status_code == 400
        assert client.post('/v1/responses', json={**request_body('responses'), 'model': 'wrong'}).status_code == 400
        assert client.get('/download-model').status_code == 404
        tests.append({'case': 'malformed source / wrong model / unsupported route rejected', 'status': 'PASS'})
    return tests


def check_live():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tests = []
    for streaming in [False, True]:
        request = urllib.request.Request('http://127.0.0.1:8000/v1/responses',
            data=json.dumps(request_body('responses', streaming)).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=180) as response:
            if streaming:
                completed = None
                for line in response:
                    if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                        continue
                    event = json.loads(line[5:])
                    assert not event.get('error') and event.get('type') != 'error', event
                    if event.get('type') == 'response.completed':
                        completed = event['response']
                assert completed, 'Missing final response event'
                result = completed
            else:
                result = json.load(response)
        assert result['model'] == MODEL and not result.get('error'), result
        call = next((item for item in result['output'] if item['type'] == 'function_call'), None)
        assert call and call['name'] == 'record_probe', result
        assert json.loads(call['arguments'])['type'] == 'SCHEMA_READY', call
        tests.append({'case': 'live Responses schema tool stream=' + str(streaming), 'status': 'PASS',
                      'authorization_header_sent': False, 'usage': result.get('usage')})
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guard-path', type=Path)
    parser.add_argument('--captured-requests', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.guard_path:
        spec = importlib.util.spec_from_file_location('guard_under_test', args.guard_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        import dsv41_guard as module
    report = {'status': 'RUNNING', 'module_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              'scope': 'actual running engine guard and live Responses tool requests' if args.live else 'CPU-only ASGI guard; no model inference',
              'tests': []}
    try:
        assert not os.environ.get('VLLM_API_KEY'), 'Expected the frozen keyless API mode'
        report['tests'] += check_cpu(module, args.captured_requests)
        if args.live:
            report['tests'] += check_live()
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=str(exc)[:2000])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report['status'] != 'PASS'


if __name__ == '__main__':
    raise SystemExit(main())
