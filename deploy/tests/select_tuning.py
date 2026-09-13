"""Choose a measured candidate, retaining the baseline if evidence is incomplete."""
import json
from pathlib import Path
import sys

directory=Path(sys.argv[1]);rows=json.loads((directory/'matrix.json').read_text())
base=rows[0]
assert base['status']=='PASS','A passing baseline is required'
def qualify(row):
    if row['status']!='PASS':return False
    a,b=row['benchmark'],base['benchmark']
    return a['single_decode_median']>=b['single_decode_median']*.95 and a['load_latency_p50']<=b['load_latency_p50']*1.05
eligible=[r for r in rows if qualify(r)]
winner=min(eligible,key=lambda r:r['benchmark']['load_latency_p50'])
# A 3% margin is required before claiming a winner; noise keeps the baseline.
if winner['benchmark']['load_latency_p50']>base['benchmark']['load_latency_p50']*.97:winner=base
report={'status':'SELECTED_PENDING_FULL_ACCEPTANCE','winner':winner['name'],'config':winner['config'],
        'rule':'C32 median completion time, >=3% improvement, C1 decode regression <=5%; fixed k5 and exact C formula',
        'scope':'bounded batch/memory matrix; not a claim of global optimality; full-context, final-C, quality and stability gates remain',
        'candidates':rows}
(directory/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
(directory/'selected-config.json').write_text(json.dumps(winner['config'],indent=2)+'\n')
print(winner['name'])
