"""Real full-window admission, cancellation, reuse and repeated mixed workloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid
from acceptance import Client, MODEL, LENGTH, chat_text, metrics


def full_window(client,index):
    nonce=uuid.uuid4().hex
    target=LENGTH-512
    count=20000
    for attempt in range(20):
        prompt='Independent record '+nonce+'\n'+('Local archive filler record.\n'*count)+'\nReply exactly WINDOW_'+nonce
        counted=client.request('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':prompt}],
                    'chat_template_kwargs':{'thinking':False},'add_generation_prompt':True})['count']
        if target-128<=counted<=target:break
        count=max(1,int(count*(target-64)/counted))
    assert target-128<=counted<=target,(target,counted)
    result=client.chat(prompt,max_tokens=512)
    assert chat_text(result)=='WINDOW_'+nonce,chat_text(result)
    actual=result['usage']['prompt_tokens']
    assert target-256<=actual<=target,(actual,target)
    return {'index':index,'status':'PASS','actual_prompt_tokens':actual,'output_budget':512,
            'seconds':result['_elapsed'],'usage':result['usage']}


def reuse(client):
    nonce=uuid.uuid4().hex
    prefix=nonce+'\n'+('Reusable record with stable prefix.\n'*4096)
    cold=client.chat(prefix+'Reply exactly CACHE_A.',max_tokens=128)
    hot=client.chat(prefix+'Reply exactly CACHE_A.',max_tokens=128)
    partial=client.chat(prefix+'Reply exactly CACHE_B.',max_tokens=128)
    assert [chat_text(x) for x in [cold,hot,partial]]==['CACHE_A','CACHE_A','CACHE_B']
    counts=[x['usage']['prompt_tokens_details']['cached_tokens'] for x in [cold,hot,partial]]
    assert counts[1]>counts[0] and 0<counts[2]<=partial['usage']['prompt_tokens'],counts
    body={'model':MODEL,'messages':[{'role':'user','content':'Write a very long detailed account of balanced trees.'}],
          'stream':True,'max_tokens':8192,'chat_template_kwargs':{'thinking':False}}
    req=urllib.request.Request(client.base+'/v1/chat/completions',data=json.dumps(body).encode(),
          headers={'Content-Type':'application/json','Authorization':'Bearer '+client.key})
    before=metrics(client)
    with client.opener.open(req,timeout=client.timeout) as response:
        seen=False
        for line in response:
            if b'"content"' in line and line.startswith(b'data:'):seen=True;break
        assert seen,'No stream content to cancel'
    # The socket is now closed while the long request is unfinished.
    time.sleep(3)
    reply=client.chat('Reply exactly AFTER_CANCEL.',max_tokens=128)
    assert chat_text(reply)=='AFTER_CANCEL'
    after=metrics(client)
    return {'status':'PASS','cold_hot_partial_cached_tokens':counts,'cancel_and_followup':'PASS',
            'preemption_delta':after.get('vllm:num_preemptions_total',0)-before.get('vllm:num_preemptions_total',0)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',required=True);p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--capacity',type=Path,required=True)
    p.add_argument('--mode',choices=['full-window','reuse','stability'],required=True)
    p.add_argument('--hours',type=float,default=2)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    client=Client(a.base_url,a.key_file.read_text(),timeout=10800)
    cap=json.loads(a.capacity.read_text());concurrency=cap['required_concurrency'];windows=cap['full_windows']
    assert concurrency==max(32,2*windows) and cap['max_model_len']==LENGTH
    report={'model':MODEL,'context':LENGTH,'mode':a.mode,'status':'RUNNING','tests':[],
            'full_windows':windows,'concurrency':concurrency,'started_at':datetime.datetime.now().astimezone().isoformat()}
    start=time.monotonic()
    def save():
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    def record(name,fn):
        try: row={'name':name,'status':'PASS','details':fn()}
        except Exception as exc: row={'name':name,'status':'FAIL','error':str(exc)[:2000].replace(client.key,'[REDACTED]')}
        report['tests'].append(row);save();print(json.dumps(row),flush=True)
        if row['status']!='PASS':raise RuntimeError('Failed '+name)
    def wave(count):
        before=metrics(client);started=time.monotonic()
        with ThreadPoolExecutor(max_workers=count) as pool: rows=list(pool.map(lambda i:full_window(client,i),range(count)))
        after=metrics(client)
        return {'submitted':count,'completed':len(rows),'seconds':time.monotonic()-started,'requests':rows,
                'preemption_delta':after.get('vllm:num_preemptions_total',0)-before.get('vllm:num_preemptions_total',0),
                'scope':'independent near-full 256K requests; admission/completion proof, C is not simultaneous KV residency'}
    def suite(name,round_no):
        target=a.output.parent/f'round-{round_no}-{name}.json'
        proc=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('acceptance.py')),
             '--base-url',a.base_url,'--key-file',str(a.key_file),'--suite',name,
             '--concurrency',str(concurrency),'--output',str(target)],timeout=43200)
        assert proc.returncode==0,(name,proc.returncode)
        return {'report':target.name,'status':json.loads(target.read_text())['status']}
    try:
        if a.mode=='full-window':
            for count in dict.fromkeys([windows,32,concurrency]):record('full-window-C'+str(count),lambda c=count:wave(c))
        elif a.mode=='reuse': record('cancel-and-prefix-reuse',lambda:reuse(client))
        else:
            assert a.hours>=2,'Stability duration must be at least 2 hours'
            round_no=0
            while time.monotonic()-start<a.hours*3600 or round_no==0:
                record('full-'+str(round_no),lambda:suite('full',round_no))
                record('long-'+str(round_no),lambda:suite('long',round_no))
                record('reuse-'+str(round_no),lambda:reuse(client))
                report.setdefault('resource_samples',[]).append({'elapsed':time.monotonic()-start,'metrics':metrics(client)})
                round_no+=1;save()
        report['status']='PASS'
    except Exception as exc:report.update(status='FAIL',error=str(exc))
    report.update(elapsed_seconds=time.monotonic()-start,requested_hours=a.hours if a.mode=='stability' else None,
                  finished_at=datetime.datetime.now().astimezone().isoformat())
    save();return report['status']!='PASS'


if __name__=='__main__':raise SystemExit(main())
