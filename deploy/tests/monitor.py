"""Record actual GPU memory and per-process RSS/PSS/locked-memory observations."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-file',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
started=time.monotonic();next_sample=0;count=0;peak_rss=peak_pss=0;first=None;last=None
while not a.stop_file.exists() and time.monotonic()-started<172800:
    now=time.monotonic()-started
    if now<next_sample:time.sleep(1);continue
    processes=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            fields={}
            for line in (proc/'status').read_text().splitlines():
                key,_,value=line.partition(':')
                if key in ['Name','VmRSS','VmLck','VmSwap']:fields[key]=value.strip()
            pss=0
            for line in (proc/'smaps_rollup').read_text().splitlines():
                if line.startswith('Pss:'):pss=int(line.split()[1])
            fields.update(pid=int(proc.name),pss_kib=pss,rss_kib=int(fields.get('VmRSS','0 kB').split()[0]));processes.append(fields)
        except (OSError,ValueError):continue
    rss=sum(x['rss_kib'] for x in processes);pss=sum(x['pss_kib'] for x in processes)
    peak_rss=max(peak_rss,rss);peak_pss=max(peak_pss,pss)
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        gpu=result.stdout.strip() if result.returncode==0 else result.stderr.strip()
    except (OSError,subprocess.TimeoutExpired) as exc:gpu=str(exc)
    sample={'elapsed_seconds':now,'gpu_csv':gpu,'processes':processes,'rss_sum_kib':rss,'pss_sum_kib':pss,
            'meminfo':Path('/proc/meminfo').read_text(),'scope':'container processes; summed RSS double-counts shared pages, PSS is proportional'}
    if first is None:first=sample
    last=sample
    with a.output.open('a') as out:out.write(json.dumps(sample)+'\n')
    count+=1;next_sample=now+10
summary={'status':'RECORDED','samples':count,'peak_rss_sum_kib':peak_rss,'peak_pss_sum_kib':peak_pss,
         'first_pss_kib':first['pss_sum_kib'] if first else None,'last_pss_kib':last['pss_sum_kib'] if last else None,
         'scope':'measurements only; memory growth and steady-state behavior require workload-aware review'}
a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
