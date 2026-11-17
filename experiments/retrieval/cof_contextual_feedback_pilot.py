"""Development-only contextual feedback replay with disjoint evaluation folds."""
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

from expert_search.cof_benchmark import load


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=Path('artifacts/cof-single-query-v1/rows.json')
    summary_path=path.parent/'summary.json'
    summary=json.loads(summary_path.read_text())
    if sha(path)!=summary['rows_sha256']:
        raise ValueError('Recorded rows changed')
    root,papers,_,labels=load('SIGIR')
    rows=json.loads(path.read_text())
    if len(rows)!=73 or len({r['query_id'] for r in rows})!=73 or {r['query_id'] for r in rows}!=set(labels['soft']):
        raise ValueError('Unexpected cohort')
    # Group identical normalized query content before assigning folds.
    groups={}
    for row in rows:
        paper=papers[row['query_id']]
        text=' '.join((paper.get('title','')+' '+paper.get('abstract','')).casefold().split())
        if not text:raise ValueError('Empty query')
        groups.setdefault(hashlib.sha256(text.encode()).hexdigest(),[]).append(row)
    rng=random.Random(20261123)
    group_keys=sorted(groups);rng.shuffle(group_keys)
    if len(group_keys)<60:raise ValueError('Insufficient distinct groups')
    old_keys,update_keys,new_keys=group_keys[:20],group_keys[20:40],group_keys[40:]
    old=[r for key in old_keys for r in groups[key]]
    update=[r for key in update_keys for r in groups[key]]
    new=[r for key in new_keys for r in groups[key]]
    if {r['query_id'] for r in old}&{r['query_id'] for r in update+new} or {r['query_id'] for r in update}&{r['query_id'] for r in new}:
        raise ValueError('Split overlap')
    arms=('lexical','dense')
    for row in rows:
        for arm in arms:
            if not math.isfinite(row['timings'][arm]) or row['timings'][arm]<0 or not 0<=row['metrics']['soft'][arm]['p@5']<=1:
                raise ValueError('Invalid result')
    penalty=5.0;epsilon=.2;threshold=200
    arms=('lexical','dense')
    totals={bucket:{a:0.0 for a in arms} for bucket in ('short','long')}
    counts={bucket:{a:0 for a in arms} for bucket in ('short','long')}
    def context(row): return 'short' if row['token_length'] <= threshold else 'long'
    def quality(row,arm): return row['metrics']['soft'][arm]['p@5']
    def utility(row,arm): return quality(row,arm)-penalty*row['timings'][arm]
    def select(bucket):
        if not any(counts[bucket].values()):return 'lexical'
        if any(counts[bucket][a]==0 for a in arms):return max(arms,key=lambda a: counts[bucket][a])
        return max(arms,key=lambda a:(totals[bucket][a]/counts[bucket][a],a=='lexical'))
    def metrics(rows,policy):
        def arm(row): return policy[context(row)]
        return {'quality':statistics.mean(quality(r,arm(r)) for r in rows),
                'component_seconds':statistics.mean(r['timings'][arm(r)] for r in rows),
                'utility':statistics.mean(utility(r,arm(r)) for r in rows),
                'dense_fraction':statistics.mean(arm(r)=='dense' for r in rows)}
    lexical={'short':'lexical','long':'lexical'}
    dense={'short':'dense','long':'dense'}
    checkpoints=[]
    for i,row in enumerate(update):
        bucket=context(row)
        unobserved=[a for a in arms if counts[bucket][a]==0]
        if unobserved: arm=unobserved[0]
        elif rng.random()<epsilon:arm=rng.choice(arms)
        else:arm=select(bucket)
        totals[bucket][arm]+=utility(row,arm);counts[bucket][arm]+=1
        if i+1 in (5,10,len(update)):
            policy={bucket:select(bucket) for bucket in counts}
            checkpoints.append({'updates':i+1,'policy':policy,
                                'feedback_counts':{k:dict(v) for k,v in counts.items()},
                                'old':metrics(old,policy),'new':metrics(new,policy)})
    report={'queries':len(rows),'query_groups':len(groups),'old_queries':len(old),'update_queries':len(update),'new_queries':len(new),
            'seed':20261123,'initial_policy':'lexical','epsilon':epsilon,'cost_penalty_per_second':penalty,'token_length_threshold':threshold,
            'old_initial':metrics(old,lexical),'new_initial':metrics(new,lexical),
            'old_fixed_dense':metrics(old,dense),'new_fixed_dense':metrics(new,dense),
            'checkpoints':checkpoints,
            'input_sha256':{'rows':sha(path),'summary':sha(summary_path),'papers':sha(root/'SIGIR_papers_test.json'),
                            'soft_labels':sha(root/'SIGIR_queries_test_soft.json'),'script':sha(Path(__file__))},
            'scope':'Single post-hoc development split of 73 repeatedly examined public reviewer queries. Query-text grouping only; does not remove all topic, reviewer, source or semantic dependencies. Soft reviewer self-ratings and recorded component time simulate chosen-arm feedback; updates do not receive old or new labels. Old/new metrics are descriptive oracle evaluation, not real feedback, independent industrial gold, or untouched held-out results. The pre-existing data and choices were inspected; this is not preregistration, a chronological task stream, evidence of generalizable transfer, or full-agent latency/cost. Arbitrary cost exchange rate and different CPU/GPU hardware. The policy uses two length buckets and two arms; the 200-token cutoff was chosen after observing the data. Length is not a validated proxy for task difficulty. No text corrections, source-level attribution, or model training.'}
    output=Path('artifacts/cof-contextual-feedback-pilot-v1');output.mkdir(exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
