"""Real endpoint acceptance. No canned response is accepted as GPU validation."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid

MODEL = 'DeepSeek-V4.1-Flash'
LENGTH = 262144


class Client:
    def __init__(self, base, key, timeout=3600, dp_rank=None):
        self.base = base.rstrip('/')
        self.key = key.strip()
        self.timeout = timeout
        self.dp_rank = dp_rank
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, body=None, raw=False):
        headers = {'Content-Type':'application/json','Authorization':'Bearer '+self.key,'x-api-key':self.key,
                   'anthropic-version':'2023-06-01'}
        if self.dp_rank is not None:
            headers['X-data-parallel-rank'] = str(self.dp_rank)
        req = urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
        started = time.monotonic()
        with self.opener.open(req,timeout=self.timeout) as response:
            if raw:
                return response.read().decode()
            if body and body.get('stream'):
                events, arrivals = [], []
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data and data != b'[DONE]':
                        events.append(json.loads(data))
                        arrivals.append(time.monotonic()-started)
                return {'events':events,'event_times':arrivals,'elapsed':time.monotonic()-started,
                        '_dp_rank':response.headers.get('X-DSV41-DP-Rank')}
            result = json.load(response)
            result['_elapsed'] = time.monotonic()-started
            result['_gateway_request_id'] = response.headers.get('X-Request-Id')
            result['_dp_rank'] = response.headers.get('X-DSV41-DP-Rank')
            return result

    def chat(self, text, **kwargs):
        return self.request('/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':text}],
                    'max_tokens':256,'temperature':0,'chat_template_kwargs':{'thinking':False},**kwargs})


def metrics(client):
    text = client.request('/metrics',raw=True)
    result = {}
    for line in text.splitlines():
        m = re.match(r'(vllm:[a-zA-Z0-9_:]+)(?:\{.*\})?\s+([-+0-9.eE]+)$',line)
        if m:
            result[m[1]] = result.get(m[1],0)+float(m[2])
    return result


def chat_text(response):
    assert response['model'] == MODEL, response.get('model')
    assert not response.get('error'), response.get('error')
    text = response['choices'][0]['message'].get('content') or ''
    assert '\ufffd' not in text, 'Replacement characters in visible output'
    assert not re.search(r'<[｜|]?DSML[｜|]?',text), 'Unparsed DSML in visible output'
    return text.strip()


def sse_chat(stream):
    texts, token_arrivals, usage = [], [], None
    for event, arrival in zip(stream['events'],stream['event_times']):
        assert not event.get('error'), event
        assert event.get('model',MODEL) == MODEL
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices',[]):
            delta = choice.get('delta',{})
            if delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content'):
                token_arrivals.append(arrival)
            texts.append(delta.get('content') or '')
    assert usage is not None and token_arrivals, 'Missing stream token usage or token events'
    return ''.join(texts),usage,token_arrivals


def vision_request(count, protocol='chat'):
    """Build the exact requests shared by schema checks and live OCR acceptance."""
    if count < 1 or protocol not in ('chat', 'responses', 'messages'):
        raise ValueError('Invalid vision acceptance case')
    images = [base64.b64encode((Path(__file__).parent/'fixtures'/f'code-{(n-1)%4+1}.png').read_bytes()).decode()
              for n in range(1, count+1)]
    prompt = 'Read the code printed on each image, in image order. Return only the codes separated by commas.'
    body = {'model': MODEL, 'temperature': 0, 'stream': False, 'chat_template_kwargs': {'thinking': False}}
    if protocol == 'responses':
        parts = [{'type': 'input_text', 'text': prompt}] + [
            {'type': 'input_image', 'detail': 'auto', 'image_url': 'data:image/png;base64,'+x} for x in images]
        body.update(input=[{'role': 'user', 'content': parts}], store=False, max_output_tokens=128)
    elif protocol == 'messages':
        parts = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': x}} for x in images]
        body.update(messages=[{'role': 'user', 'content': parts}], max_tokens=128, thinking={'type': 'disabled'})
    else:
        parts = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+x}} for x in images]
        body.update(messages=[{'role': 'user', 'content': parts}], max_tokens=128)
    return body


def run(args):
    client = Client(args.base_url,args.key_file.read_text(), dp_rank=getattr(args, 'dp_rank', None))
    report = {'model':MODEL,'max_model_len':LENGTH,'suite':args.suite,'started_at':datetime.datetime.now().astimezone().isoformat(),
              'base_url':args.base_url,'scope':'real HTTP requests; GPU path proven separately by runtime observations',
              'tests':[],'status':'RUNNING'}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        tmp.replace(args.output)

    def test(name, fn):
        started = time.monotonic()
        try:
            details = fn()
            row = {'name':name,'status':'PASS','details':details}
        except Exception as exc:
            message = exc.read().decode(errors='replace') if isinstance(exc,urllib.error.HTTPError) else str(exc)
            row = {'name':name,'status':'FAIL','error':message[:4000].replace(client.key,'[REDACTED]')}
        row['seconds'] = round(time.monotonic()-started,3)
        report['tests'].append(row)
        save()
        print(json.dumps(row,ensure_ascii=False),flush=True)

    def identity():
        response = client.request('/v1/models')
        ids = [x['id'] for x in response['data']]
        assert ids == [MODEL], ids
        return {'ids':ids}

    def text_and_stream():
        prompt = 'Reply with the exact ASCII string OFFLINE_READY and nothing else.'
        reply = client.chat(prompt)
        assert chat_text(reply) == 'OFFLINE_READY', reply
        stream = client.chat(prompt,stream=True,stream_options={'include_usage':True})
        text,usage,arrivals = sse_chat(stream)
        assert text.strip() == 'OFFLINE_READY', text
        return {'text':text.strip(),'usage':usage,'first_token_seconds':arrivals[0]}

    def protocol_cache(protocol):
        text = 'Cache probe '+uuid.uuid4().hex+'\n' + '\n'.join(
            f'Record {i}: stable offline prefix data for protocol cache verification.' for i in range(1024))
        text += '\nReply only CACHE_READY.'
        body = {'model':MODEL, 'temperature':0, 'chat_template_kwargs':{'thinking':False}}
        if protocol == 'responses':
            path = '/v1/responses'
            body.update(input=[{'role':'user','content':text}], max_output_tokens=32, store=False)
        elif protocol == 'messages':
            path = '/v1/messages'
            body.update(messages=[{'role':'user','content':text}], max_tokens=32, thinking={'type':'disabled'})
        else:
            path = '/v1/chat/completions'
            body.update(messages=[{'role':'user','content':text}], max_tokens=32)
        def usage(reply, streaming):
            if not streaming:
                assert not reply.get('error'), reply
                return reply['usage']
            merged = {}
            for event in reply['events']:
                assert event.get('type') != 'error' and not event.get('error'), event
                if protocol == 'responses' and event.get('type') == 'response.completed':
                    merged.update(event['response']['usage'])
                elif protocol == 'messages':
                    merged.update(event.get('message', {}).get('usage', {}))
                    merged.update(event.get('usage', {}))
                elif protocol == 'chat':
                    merged.update(event.get('usage') or {})
            assert merged, 'Missing SSE usage'
            return merged
        def cached(value):
            if protocol == 'responses': return value['input_tokens_details']['cached_tokens']
            if protocol == 'messages': return value['cache_read_input_tokens']
            return value['prompt_tokens_details']['cached_tokens']
        cold = usage(client.request(path, body), False)
        hot = usage(client.request(path, body), False)
        stream_body = dict(body, stream=True)
        if protocol == 'chat': stream_body['stream_options'] = {'include_usage':True}
        streamed = usage(client.request(path, stream_body), True)
        assert cached(hot) > cached(cold) and cached(streamed) > cached(cold), (cold, hot, streamed)
        return {'protocol':protocol, 'dp_rank':client.dp_rank, 'cold_cached_tokens':cached(cold),
                'hot_cached_tokens':cached(hot), 'sse_cached_tokens':cached(streamed)}

    def cache():
        salt = uuid.uuid4().hex
        prefix = 'Run '+salt+'. Reference records follow.\n'+('\n'.join(f'Row {n}: this is a stable local record with unchanged text.' for n in range(1024)))
        prompt = prefix+'\nReply exactly CACHE_READY.'
        cold = client.chat(prompt,max_tokens=128)
        hot = client.chat(prompt,max_tokens=128)
        assert chat_text(cold) == chat_text(hot) == 'CACHE_READY'
        first = cold['usage']['prompt_tokens_details']['cached_tokens']
        second = hot['usage']['prompt_tokens_details']['cached_tokens']
        total = hot['usage']['prompt_tokens']
        assert 0 <= first < second <= total, (first,second,total)
        return {'cold_cached_tokens':first,'hot_cached_tokens':second,'prompt_tokens':total,
                'token_hit_ratio':second/total,'cold_seconds':cold['_elapsed'],'hot_seconds':hot['_elapsed'],
                'hot_gateway_request_id':hot.get('_gateway_request_id')}

    def dspark():
        before = metrics(client)
        reply = client.chat('Write a detailed numbered list explaining 30 properties of a binary search tree. Use at least 500 words.',
                            max_tokens=1024,stream=True,stream_options={'include_usage':True})
        text,usage,arrivals = sse_chat(reply)
        assert len(text) > 100 and usage['completion_tokens'] >= 128
        time.sleep(2)
        after = metrics(client)
        names = ['vllm:spec_decode_num_drafts_total','vllm:spec_decode_num_draft_tokens_total','vllm:spec_decode_num_accepted_tokens_total']
        delta = {n:after.get(n,0)-before.get(n,0) for n in names}
        assert all(v > 0 for v in delta.values()), delta
        assert delta[names[2]] <= delta[names[1]], delta
        return {'counter_deltas':delta,'acceptance_rate':delta[names[2]]/delta[names[1]],
                'completion_tokens':usage['completion_tokens'],'sse_events':len(arrivals)}

    def rejects():
        invalid = [('alias',{'model':'invalid-alias','messages':[{'role':'user','content':'test'}]}),
                   ('remote-image',{'model':MODEL,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://example.invalid/image.png'}}]}]})]
        for name,body in invalid:
            try:
                client.request('/v1/chat/completions',body)
            except urllib.error.HTTPError as e:
                assert e.code in [400,404], (name,e.code)
            else:
                raise AssertionError('Did not reject '+name)
        return {'invalid_model':'rejected','remote_media':'rejected before fetching'}

    def tools(api, streaming):
        value = 'RECEIPT_'+uuid.uuid4().hex[:12]
        schema = {'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}
        prompt = 'Call record_probe with value CHECK. After receiving its tool result, repeat its receipt exactly and nothing else.'
        body = {'model':MODEL,'stream':streaming,'temperature':0,'chat_template_kwargs':{'thinking':False}}
        if api == 'responses':
            body.update(input=[{'role':'user','content':[{'type':'input_text','text':prompt}]}],store=False,max_output_tokens=512,
                        tools=[{'type':'function','name':'record_probe','description':'Returns a receipt.','parameters':schema}],tool_choice='auto')
            first = client.request('/v1/responses',body)
            result = next(e['response'] for e in first['events'] if e['type']=='response.completed') if streaming else first
            assert result['model'] == MODEL
            call = next(x for x in result['output'] if x['type']=='function_call')
            assert call['name']=='record_probe' and json.loads(call['arguments'])=={'value':'CHECK'} and call['call_id']
            body['input'] += result['output']+[{'type':'function_call_output','call_id':call['call_id'],'output':json.dumps({'receipt':value})}]
            second = client.request('/v1/responses',body)
            final = next(e['response'] for e in second['events'] if e['type']=='response.completed') if streaming else second
            text = ''.join(c['text'] for x in final['output'] if x['type']=='message' for c in x['content'] if c['type']=='output_text')
        elif api == 'messages':
            body.update(messages=[{'role':'user','content':prompt}],max_tokens=512,thinking={'type':'disabled'},
                        tools=[{'name':'record_probe','description':'Returns a receipt.','input_schema':schema}])
            first = client.request('/v1/messages',body)

            def contents(result):
                if not streaming:
                    return result['content']
                blocks, parts = {}, {}
                for e in result['events']:
                    assert e['type'] != 'error',e
                    if e['type']=='content_block_start': blocks[e['index']]=dict(e['content_block'])
                    elif e['type']=='content_block_delta':
                        d=e['delta']; i=e['index']
                        if d['type']=='text_delta': blocks[i]['text']=blocks[i].get('text','')+d['text']
                        elif d['type']=='input_json_delta': parts[i]=parts.get(i,'')+d['partial_json']
                        elif d['type']=='thinking_delta': blocks[i]['thinking']=blocks[i].get('thinking','')+d['thinking']
                        elif d['type']=='signature_delta': blocks[i]['signature']=blocks[i].get('signature','')+d['signature']
                for i,part in parts.items(): blocks[i]['input']=json.loads(part)
                return [blocks[i] for i in sorted(blocks)]
            content = contents(first)
            call = next(x for x in content if x['type']=='tool_use')
            assert call['name']=='record_probe' and call['input']=={'value':'CHECK'} and call['id']
            body['messages'] += [{'role':'assistant','content':content},{'role':'user','content':[{'type':'tool_result','tool_use_id':call['id'],'content':json.dumps({'receipt':value})}]}]
            final = contents(client.request('/v1/messages',body))
            text = ''.join(c['text'] for c in final if c['type']=='text')
        else:
            body.update(messages=[{'role':'user','content':prompt}],max_tokens=512,
                        tools=[{'type':'function','function':{'name':'record_probe','description':'Returns a receipt.','parameters':schema}}])
            first = client.request('/v1/chat/completions',body)
            if streaming:
                merged = {}
                for event in first['events']:
                    for choice in event.get('choices',[]):
                        for c in choice.get('delta',{}).get('tool_calls',[]):
                            item=merged.setdefault(c['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                            if c.get('id'): item['id']=c['id']
                            for key in ['name','arguments']: item['function'][key]+=c.get('function',{}).get(key) or ''
                calls=[merged[i] for i in sorted(merged)]
            else: calls=first['choices'][0]['message']['tool_calls']
            assert len(calls)==1 and calls[0]['function']['name']=='record_probe'
            assert json.loads(calls[0]['function']['arguments'])=={'value':'CHECK'}
            body['messages'] += [{'role':'assistant','content':None,'tool_calls':calls},{'role':'tool','tool_call_id':calls[0]['id'],'content':json.dumps({'receipt':value})}]
            if streaming: body['stream_options']={'include_usage':True}
            second=client.request('/v1/chat/completions',body)
            text=sse_chat(second)[0] if streaming else chat_text(second)
        assert text.strip()==value,(text,value)
        return {'protocol':api,'stream':streaming,'receipt_matched':True,'tool_call_id_matched':True}

    def vision(count, protocol='chat'):
        reply=client.request('/v1/'+('chat/completions' if protocol=='chat' else protocol), vision_request(count, protocol))
        assert reply['model']==MODEL
        if protocol=='responses':
            text=''.join(c['text'] for item in reply['output'] if item['type']=='message'
                         for c in item['content'] if c['type']=='output_text')
        elif protocol=='messages':
            text=''.join(c['text'] for c in reply['content'] if c['type']=='text')
        else:
            text=chat_text(reply)
        text=re.sub(r'\s+','',text)
        expected=','.join(f'LOCAL{(n-1)%4+1}A100' for n in range(1,count+1))
        assert text==expected,(text,expected)
        return {'protocol':protocol,'images':count,'ocr':text,'usage':reply['usage']}

    def parallel():
        n=args.concurrency
        def one(i):
            tag=f'PARALLEL_{i:04d}'
            r=client.chat('Reply exactly '+tag+'.',max_tokens=128)
            assert chat_text(r)==tag,(i,r)
            return r['_elapsed']
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=n) as pool: times=list(pool.map(one,range(n)))
        return {'submitted_concurrency':n,'successful':len(times),'elapsed':time.monotonic()-start,
                'p50_seconds':statistics.median(times),'p95_seconds':sorted(times)[max(0,int(len(times)*.95)-1)],
                'scope':'independent short requests; not simultaneous full-window residency'}

    def long_context(budget):
        # Ask the serving tokenizer to count the actual template, not text bytes.
        nonce=uuid.uuid4().hex[:12]
        records=[f'POSITION_{i}=VALUE_{nonce}_{i}' for i in range(4)]
        filler='This is an irrelevant archive record. Ignore it when retrieving the POSITION values.\n'
        count=max(1,budget//16)
        prompt=''
        actual=0
        for _ in range(15):
            quarter=(filler*(count//4))
            prompt='Retrieve all four POSITION values in numeric order.\n'+quarter.join(records)+'\nReturn only the four values separated by commas.'
            counted=client.request('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':prompt}],
                    'chat_template_kwargs':{'thinking':False},'add_generation_prompt':True})
            actual=counted['count']
            if budget-2048 <= actual <= budget: break
            count=max(1,int(count*(budget-512)/max(actual,1)))
        assert budget-2048 <= actual <= budget,(budget,actual)
        reply=client.chat(prompt,max_tokens=4096)
        expected=','.join(f'VALUE_{nonce}_{i}' for i in range(4))
        assert chat_text(reply).replace(' ','')==expected,(actual,reply)
        assert reply['usage']['prompt_tokens'] <= LENGTH-4096
        return {'requested_input_budget':budget,'tokenizer_count':actual,'actual_prompt_tokens':reply['usage']['prompt_tokens'],
                'output_budget':4096,'retrieval_positions':4,'seconds':reply['_elapsed']}

    test('identity',identity)
    if args.suite in ['smoke','full','gateway']:
        test('text-and-SSE',text_and_stream)
        test('cache-cold-and-hot',cache)
        if args.suite != 'gateway': test('DSpark-draft-and-accept-counters',dspark)
        test('offline-input-rejections',rejects)
    if args.suite in ['full','gateway']:
        for api in ['chat','responses','messages']:
            test(f'{api}-cache-JSON-and-SSE',lambda a=api:protocol_cache(a))
        for api in ['chat','responses','messages']:
            for streaming in [False,True]: test(f'{api}-tools-stream={streaming}',lambda a=api,s=streaming:tools(a,s))
        for count in [1,2,4,5,8]: test(f'vision-{count}',lambda n=count:vision(n))
        for api in ['responses','messages']: test(f'vision-5-{api}',lambda a=api:vision(5,a))
        test('concurrent-requests',parallel)
    if args.suite == 'long':
        for budget in [32768,65536,131072,258048]: test('long-'+str(budget),lambda b=budget:long_context(b))
    if args.suite == 'parallel': test('concurrent-requests',parallel)
    report['status']='PASS' if all(r['status']=='PASS' for r in report['tests']) else 'FAIL'
    report['finished_at']=datetime.datetime.now().astimezone().isoformat()
    save()
    return report['status']!='PASS'


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',required=True)
    p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=['smoke','full','long','parallel','gateway'],default='full')
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--dp-rank',type=int,choices=(0,1))
    raise SystemExit(run(p.parse_args()))
