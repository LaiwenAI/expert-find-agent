"""Test logged-feedback estimators against known frozen reviewer outcomes."""
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from expert_search.offpolicy import evaluate


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('artifacts/cof-single-query-v1');path=root/'rows.json'
    summary=json.loads((root/'summary.json').read_text())
    if sha(path)!=summary['rows_sha256']:raise ValueError('Changed frozen rows')
    rows=json.loads(path.read_text())
    if len(rows)!=73 or len({r['query_id'] for r in rows})!=73:raise ValueError('Changed cohort')
    truth=statistics.mean(r['metrics']['soft']['dense']['p@5'] for r in rows)
    target={'all':{'lexical':0.,'dense':1.}}
    arms=('lexical','dense');results={}
    for dense_probability in (.5,.1):
        ips=[];snips=[];esses=[];zero=0
        for rep in range(1000):
            rng=random.Random(20261124+rep)
            events=[]
            for i,row in enumerate(rows):
                arm='dense' if rng.random()<dense_probability else 'lexical'
                events.append({'event_id':str(i),'context':'all','arm':arm,
                    'outcome':row['metrics']['soft'][arm]['p@5'],
                    'logging_probabilities':{'dense':dense_probability,'lexical':1-dense_probability}})
            estimate=evaluate(events,target)
            ips.append(estimate['ips']);esses.append(estimate['effective_sample_size'])
            if estimate['snips'] is None:zero+=1
            else:snips.append(estimate['snips'])
        results[str(dense_probability)]={'mean_ips':statistics.mean(ips),
             'ips_rmse':math.sqrt(statistics.mean((x-truth)**2 for x in ips)),
             'mean_snips':statistics.mean(snips) if snips else None,
             'snips_rmse_when_defined':math.sqrt(statistics.mean((x-truth)**2 for x in snips)) if snips else None,
             'mean_effective_sample_size':statistics.mean(esses),'zero_target_actions':zero,
             'max_weight':1/dense_probability}
    report={'queries':len(rows),'repetitions':1000,'seed_start':20261124,'fixed_dense_oracle':truth,
            'logging_dense_probabilities':results,
            'input_sha256':{'rows':sha(path),'summary':sha(root/'summary.json'),
                            'evaluator':sha(Path('src/expert_search/offpolicy.py')),'script':sha(Path(__file__))},
            'scope':'Algorithm sanity check on a fixed, previously explored CoF reviewer cohort with simulated randomized actions and fully observed counterfactual labels. The 1000 random logs are repeated draws over the same queries, not independent datasets; the oracle is the same public reviewer self-rating, not enterprise fit or real user utility. Constant action probabilities are known by construction; no missing outcomes, time ordering, true feedback behavior, inference latency, confounding, or uncertainty from model outputs is modeled. This does not validate off-policy evaluation on real logs.'}
    out=Path('artifacts/cof-offpolicy-simulation-v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
