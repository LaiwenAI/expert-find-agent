"""Within-reviewer lexical baseline on a fixed upstream CMU profile realization."""
import json
import math
from collections import Counter
from pathlib import Path
from .cmu_gold import load_annotations,pairwise_loss
from .retrieval import tokens


def text(row):
    c=row.get('content',row)
    return str(c.get('title') or '')+'\n'+str(c.get('abstract') or '')


def run(root='artifacts/public/cmu-gold-upstream',output='artifacts/public/cmu-results'):
    root=Path(root);dataset=root/'evaluation_datasets/d_20_1'
    annotations=load_annotations(root/'data/evaluations.csv')
    submissions=json.loads((dataset/'submissions.json').read_text())
    if isinstance(submissions,list):submissions={r['id']:r for r in submissions}
    profiles={};documents=[]
    for path in sorted((dataset/'archives').glob('*.jsonl')):
        rid=path.stem.lstrip('~');profile=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        profiles[rid]=profile;documents.extend(profile)
    df=Counter(t for row in documents for t in set(tokens(text(row))))
    n=len(documents)
    def vector(row):
        counts=Counter(tokens(text(row)))
        return {t:(1+math.log(count))*math.log(1+(n+.5)/(df.get(t,0)+.5)) for t,count in counts.items()}
    def cosine(a,b):
        norm=math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
        return sum(v*b.get(t,0) for t,v in a.items())/norm if norm else 0
    predictions={};omissions=[];leaks=[]
    for rid,grades in annotations.items():
        if rid not in profiles:raise ValueError('Missing reviewer profile')
        predictions[rid]={}
        for pid in grades:
            if pid not in submissions:raise ValueError('Missing submission')
            q=vector(submissions[pid]);scores=[]
            for row in profiles[rid]:
                if row['id']==pid:leaks.append((rid,pid));continue
                scores.append(cosine(q,vector(row)))
            if not scores:omissions.append((rid,pid))
            predictions[rid][pid]=max(scores,default=0)
    result=dict(method='max publication log-TF IDF cosine',profile_realization='d_20_1',
        **pairwise_loss(predictions,annotations),self_paper_matches_excluded=len(leaks),empty_profiles=len(omissions),
        caveat='Single profile realization, not ten-run reproduction; within-reviewer ordering only. No model training.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'lexical-predictions.json').write_text(json.dumps(predictions,indent=2))
    (out/'lexical-summary.json').write_text(json.dumps(result,indent=2));return result
