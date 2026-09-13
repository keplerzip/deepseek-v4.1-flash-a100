"""Repeatable real HTTP/SSE timing; counts come from server usage, not SSE chunks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid
from acceptance import Client, MODEL, sse_chat


def percentile(values,fraction):
    values=sorted(values)
    return values[max(0,min(len(values)-1,int((len(values)-1)*fraction)))]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',required=True); p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--requests',type=int,default=200); p.add_argument('--output-tokens',type=int,default=1024)
    p.add_argument('--thinking',action='store_true'); p.add_argument('--duration-hours',type=float,default=0)
    a=p.parse_args()
    client=Client(a.base_url,a.key_file.read_text())
    rows=[]
    report={'model':MODEL,'context':262144,'protocol':'Chat Completions HTTP/SSE','thinking':a.thinking,
            'reasoning_effort':'high','temperature':0,'concurrency':a.concurrency,'warmups':2,'single_stream_samples':5,
            'requested_output_tokens':a.output_tokens,'ignore_eos':True,'status':'RUNNING',
            'started_at':datetime.datetime.now().astimezone().isoformat(),
            'timing_note':'decode=(usage completion_tokens-1)/(last token-bearing event-first); DSpark may emit several tokens per event; inter-event latency is not per-token latency'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trace=a.output.with_suffix('.jsonl')
    trace.write_text('')

    def one(index):
        salt=uuid.uuid4().hex
        prompt=f'Independent benchmark {salt}. Explain how to implement a reliable LRU cache in Python, including code, invariants, edge cases, and performance analysis. Continue with concrete examples.'
        started=time.monotonic()
        try:
            stream=client.chat(prompt,max_tokens=a.output_tokens,ignore_eos=True,stream=True,
                               stream_options={'include_usage':True},
                               chat_template_kwargs={'thinking':a.thinking,'reasoning_effort':'high'})
            text,usage,arrivals=sse_chat(stream)
            assert usage['completion_tokens']==a.output_tokens,usage
            duration=arrivals[-1]-arrivals[0]
            assert duration>0
            return {'index':index,'status':'PASS','input_tokens':usage['prompt_tokens'],
                    'output_tokens':usage['completion_tokens'],'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens',0),
                    'ttft_seconds':arrivals[0],'end_to_end_seconds':stream['elapsed'],
                    'decode_seconds':duration,'decode_tokens_per_second':(usage['completion_tokens']-1)/duration,
                    'inter_event_p95_seconds':percentile([b-c for c,b in zip(arrivals,arrivals[1:])],.95) if len(arrivals)>1 else None,
                    'token_bearing_events':len(arrivals),'visible_text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        except Exception as exc:
            return {'index':index,'status':'FAIL','error':str(exc)[:1000].replace(client.key,'[REDACTED]'),'elapsed':time.monotonic()-started}

    def record(row):
        rows.append(row)
        with trace.open('a') as f:f.write(json.dumps(row)+'\n')

    for i in range(2):
        row=one(-i-1); row['phase']='warmup'; record(row)
    for i in range(5):
        row=one(i); row['phase']='single'; record(row)
    load_start=time.monotonic(); rounds=0
    while True:
        count=max(a.requests,2*a.concurrency,200)
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            for row in pool.map(one,range(rounds*count,(rounds+1)*count)):
                row['phase']='load'; record(row)
        rounds+=1
        print(json.dumps({'completed_rounds':rounds,'requests':count*rounds,'elapsed_seconds':time.monotonic()-load_start}),flush=True)
        if time.monotonic()-load_start>=a.duration_hours*3600:break
        if any(r['status']=='FAIL' for r in rows[-count:]):break
    load_seconds=time.monotonic()-load_start
    single=[r for r in rows if r['phase']=='single' and r['status']=='PASS']
    load=[r for r in rows if r['phase']=='load' and r['status']=='PASS']
    failed=[r for r in rows if r['status']=='FAIL']
    report.update(status='FAIL' if failed else 'PASS',failed_requests=len(failed),load_completed_requests=len(load),
                  load_seconds=load_seconds,requested_duration_hours=a.duration_hours,
                  duration_satisfied=load_seconds>=a.duration_hours*3600,
                  single_decode_median=statistics.median([r['decode_tokens_per_second'] for r in single]) if single else None,
                  load_output_tokens_per_second=sum(r['output_tokens'] for r in load)/load_seconds,
                  load_latency_p50=percentile([r['end_to_end_seconds'] for r in load],.5) if load else None,
                  load_latency_p95=percentile([r['end_to_end_seconds'] for r in load],.95) if load else None,
                  load_ttft_p95=percentile([r['ttft_seconds'] for r in load],.95) if load else None,
                  finished_at=datetime.datetime.now().astimezone().isoformat(),trace_file=trace.name)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return bool(failed)


if __name__=='__main__':raise SystemExit(main())
