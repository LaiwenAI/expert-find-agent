"""Author-connected bootstrap for a local temporal proxy comparison."""
from collections import defaultdict
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

PARENT=Path('artifacts/local-temporal-author-proxy-v1')
COMPARISON=Path('artifacts/local-temporal-candidate-comparison-v1')


def analyze(replicates=10000, seed=20261117):
    parent_summary=json.loads((PARENT/'summary.json').read_text());payload=(PARENT/'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=parent_summary['results_sha256']:raise ValueError('Changed query labels')
    source=[json.loads(x) for x in payload.splitlines()]; comparison_path=COMPARISON/'results.json'
    comparison=json.loads(comparison_path.read_text()); summary=json.loads((COMPARISON/'summary.json').read_text())
    if summary['parent_results_sha256']!=parent_summary['results_sha256'] or len(source)!=200 or len(comparison)!=200:
        raise ValueError('Changed comparison cohort')
    if [r['id'] for r in source]!=[r['id'] for r in comparison] or len({r['id'] for r in source})!=200:
        raise ValueError('Changed query IDs')
    for reference, result in zip(source, comparison):
        truth=set(reference['known_author_ids'])
        if not truth:raise ValueError('Empty known-author set')
        for method in ('candidate-lm','compiled-bm25-max'):
            row=result['methods'][method]; ranking=row['ranking']
            if len(ranking)>10 or len(set(ranking))!=len(ranking) or not math.isfinite(row['seconds']) or row['seconds']<0:
                raise ValueError('Invalid ranking or time')
            positions=[i for i,eid in enumerate(ranking,1) if eid in truth]
            expected=(int(bool(positions)),1/min(positions) if positions else 0,len(set(ranking)&truth)/len(truth))
            if any(not math.isclose(row[key],value,abs_tol=1e-12) for key,value in zip(('hit_at_10','mrr_at_10','known_author_recall_at_10'),expected)):
                raise ValueError('Metric inconsistent with ranking')
    links=list(range(len(source)))
    def find(x):
        while x!=links[x]:links[x]=links[links[x]];x=links[x]
        return x
    owner={}
    for i,row in enumerate(source):
        for author in row['known_author_ids']:
            if author in owner:
                a,b=find(i),find(owner[author]);links[a]=b
            else:owner[author]=i
    groups=defaultdict(list)
    for i in range(len(source)):groups[find(i)].append(i)
    groups=list(groups.values())
    if replicates<1000:raise ValueError('Too few bootstrap replicates')
    metrics=('hit_at_10','mrr_at_10','known_author_recall_at_10','seconds');differences={}
    for metric in metrics:
        differences[metric]=[r['methods']['candidate-lm'][metric]-r['methods']['compiled-bm25-max'][metric] for r in comparison]
    rng=random.Random(seed); samples={m:[] for m in metrics}
    for _ in range(replicates):
        selected=[rng.choice(groups) for _ in groups];indices=[i for group in selected for i in group]
        for metric in metrics:samples[metric].append(sum(differences[metric][i] for i in indices)/len(indices))
    report={}
    for metric in metrics:
        values=sorted(samples[metric]);report[metric]=dict(mean=statistics.mean(differences[metric]),
            percentile_interval_95=[values[int(.025*replicates)],values[int(.975*replicates)-1]],
            improved=sum(v>0 for v in differences[metric]),worse=sum(v<0 for v in differences[metric]),tied=sum(v==0 for v in differences[metric]))
    return dict(queries=len(source),author_connected_groups=len(groups),largest_group=max(map(len,groups)),
                seed=seed,replicates=replicates,comparison_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),metrics=report,
                scope='Post-hoc paired bootstrap on selected temporal author-recovery proxy. Groups connect queries sharing known authors; other reviewer, coauthor, topic, near-duplicate and time dependencies remain unmodeled. Known author labels are unverified and incomplete. One timing per method/query, not an end-to-end latency distribution. Confidence intervals are not equivalence guarantees or industrial relevance evidence.')


if __name__=='__main__':print(json.dumps(analyze(),indent=2))
