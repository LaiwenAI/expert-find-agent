"""Exploratory cost-sensitive feedback replay on recorded single-query components."""
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('artifacts/cof-single-query-v1')
    summary = json.loads((root/'summary.json').read_text())
    rows_path = root/'rows.json'
    if sha(rows_path) != summary['rows_sha256'] or sha(root/'protocol.json') != summary['protocol_sha256']:
        raise ValueError('Changed recorded inputs')
    rows = json.loads(rows_path.read_text())
    if len(rows) != 73 or len({r['query_id'] for r in rows}) != 73:
        raise ValueError('Incomplete cohort')
    arms = ('lexical', 'dense')
    for row in rows:
        for arm in arms:
            t = row['timings'][arm]
            q = row['metrics']['soft'][arm]['p@5']
            if not math.isfinite(t) or t < 0 or not math.isfinite(q) or not 0 <= q <= 1:
                raise ValueError('Invalid feedback or timing')
    # A hypothetical exchange rate, not dollars or a calibrated user utility.
    penalty_per_second = 5.0
    streams=[]
    for seed in range(30):
        rng=random.Random(20261122+seed)
        sequence=list(rows);rng.shuffle(sequence)
        totals={a:0.0 for a in arms};counts={a:0 for a in arms};chosen=[]
        for i,row in enumerate(sequence):
            if i < 2: arm=arms[i]
            elif rng.random() < .2: arm=rng.choice(arms)
            else: arm=max(arms,key=lambda a:(totals[a]/counts[a],a=='lexical'))
            quality=row['metrics']['soft'][arm]['p@5']
            seconds=row['timings'][arm]
            totals[arm] += quality-penalty_per_second*seconds
            counts[arm] += 1
            chosen.append((row,arm))
        evaluation=chosen[20:]
        outcome={}
        for policy in ('route',)+arms:
            def actual(row,choice): return choice if policy=='route' else policy
            outcome[policy]={'quality':statistics.mean(r['metrics']['soft'][actual(r,a)]['p@5'] for r,a in evaluation),
                             'seconds':statistics.mean(r['timings'][actual(r,a)] for r,a in evaluation),
                             'utility':statistics.mean(r['metrics']['soft'][actual(r,a)]['p@5']-penalty_per_second*r['timings'][actual(r,a)] for r,a in evaluation)}
        outcome['dense_fraction']=statistics.mean(a=='dense' for _,a in evaluation)
        streams.append(outcome)
    report={'queries':len(rows),'streams':len(streams),'warmup':20,'evaluation_per_stream':53,'epsilon':.2,
            'seed_start':20261122,'penalty_per_second':penalty_per_second,
            'across_stream_means':{policy:{k:statistics.mean(s[policy][k] for s in streams) for k in ('quality','seconds','utility')} for policy in ('route',)+arms},
            'mean_dense_fraction':statistics.mean(s['dense_fraction'] for s in streams),
            'input_sha256':{'rows':sha(rows_path),'protocol':sha(root/'protocol.json'),'summary':sha(root/'summary.json'),'script':sha(Path(__file__))},
            'scope':'Post-hoc simulation on 73 reused CoF reviewer queries and one recorded timing per arm. Shuffled streams overlap and are not independent or temporal; reviewer self-ratings simulate feedback. The selected arm receives soft P@5 minus an arbitrary hypothetical seconds penalty, not user utility or monetary cost. GPU dense and CPU lexical use different hardware; measured times include query encoding and scoring but exclude initialization, model reasoning, feedback collection, training, concurrency, and serving. Timing values are replayed, not newly measured; no enterprise fit, held-out continual learning, forgetting, causal quality preservation, or end-to-end cost claim.'}
    out=Path('artifacts/cof-cost-feedback-replay-v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'streams.json').write_text(json.dumps(streams,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
