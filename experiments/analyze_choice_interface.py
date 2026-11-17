"""Paired latency and within/between-interface label stability; no accuracy claim."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def run():
    roots=[Path('artifacts/choice-interface-comparison-v1'),Path('artifacts/choice-interface-comparison-repeat-v1')]
    collections=[];inputs={}
    for root in roots:
        protocol=json.loads((root/'protocol.json').read_text());rows={}
        for qid in protocol['query_ids']:
            path=root/(qid+'.json');row=json.loads(path.read_text());rows[qid]=row;inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        collections.append(rows)
    if set(collections[0])!=set(collections[1]):raise ValueError('Different cohorts')
    if any(collections[0][q]['input_sha256']!=collections[1][q]['input_sha256'] for q in collections[0]):raise ValueError('Changed repeated inputs')
    comparisons={}
    for name,left,right in [('native-repeat',(0,'native'),(1,'native')),('chat-repeat',(0,'chat'),(1,'chat')),('between-first',(0,'native'),(0,'chat')),('between-repeat',(1,'native'),(1,'chat'))]:
        confusion=Counter();paired=0
        for q in collections[0]:
            a=collections[left[0]][q]['arms'][left[1]];b=collections[right[0]][q]['arms'][right[1]]
            if a['status']!='valid' or b['status']!='valid':continue
            x=a['result']['labels'];y=b['result']['labels'];paired+=1
            if set(x)!=set(y):raise ValueError('Different candidates')
            confusion.update((x[k],y[k]) for k in x)
        comparisons[name]=dict(paired_queries=paired,labels=sum(confusion.values()),agreements=sum(v for (a,b),v in confusion.items() if a==b),confusion=[dict(left=a,right=b,count=v) for (a,b),v in sorted(confusion.items())])
    # Cluster bootstrap by query, averaging the two repeats within each query.
    differences=[]
    for q in collections[0]:differences.append(sum(c[q]['arms']['chat']['seconds']-c[q]['arms']['native']['seconds'] for c in collections)/2)
    values=np.asarray(differences);rng=np.random.default_rng(20261117);boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    timing=dict(mean_chat_minus_native_seconds=float(values.mean()),query_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),scope='All attempts including errors; query-cluster bootstrap over fixed20 development queries and2repeats, not broad-population guarantee.')
    report=dict(queries=len(values),comparisons=comparisons,timing=timing,inputs_sha256=inputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Same-model short-choice latency and reproducibility; neither API is semantic gold, and no expert-search accuracy, witness correctness or E2E advantage is established.')
    out=Path('artifacts/choice-interface-analysis-v1');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['inputs_sha256','comparisons']},indent=2));print({k:{p:v for p,v in x.items() if p!='confusion'} for k,x in comparisons.items()})


if __name__=='__main__':run()
