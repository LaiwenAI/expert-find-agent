"""Prequential simulated-feedback routing replay on frozen reviewer rankings."""
import hashlib
import json
import random
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('artifacts/cof-gpu-quality-v1')
    rows_path = root / 'rows.json'
    summary = json.loads((root/'summary.json').read_text())
    if sha(rows_path) != summary['rows_sha256']:
        raise ValueError('Frozen rankings changed')
    rows = json.loads(rows_path.read_text())
    if len(rows) != 73 or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid cohort')
    # Soft/hard reviewer P@5 are feedback signals, not expert-match truth.
    for r in rows:
        for method in ('lexical', 'gpu'):
            if len(r['rankings'][method][:5]) != 5 or not 0 <= r['metrics']['soft'][method]['p@5'] <= 1:
                raise ValueError('Invalid frozen result')
    summaries=[]
    for seed in range(30):
        rng = random.Random(20261122+seed)
        stream = list(rows); rng.shuffle(stream)
        # Fixed initial alternating exploration prevents a single zero-reward
        # observation from locking the decision. No future feedback is used.
        totals = {'lexical': 0., 'gpu': 0.}; counts = {'lexical': 0, 'gpu': 0}
        chosen=[]
        for i,r in enumerate(stream):
            if i < 2: arm = ('lexical','gpu')[i]
            elif rng.random() < .2: arm = rng.choice(('lexical','gpu'))
            else: arm = max(('lexical','gpu'),key=lambda m:(totals[m]/counts[m],m=='lexical'))
            reward = r['metrics']['soft'][arm]['p@5']
            totals[arm] += reward; counts[arm] += 1
            chosen.append({'arm':arm,'soft':reward,'hard':r['metrics']['hard'][arm]['p@5'],
                           'lexical_soft':r['metrics']['soft']['lexical']['p@5'],
                           'gpu_soft':r['metrics']['soft']['gpu']['p@5']})
        later=chosen[20:]
        summaries.append({'route_soft':statistics.mean(r['soft'] for r in later),
                          'route_hard':statistics.mean(r['hard'] for r in later),
                          'lexical_soft':statistics.mean(r['lexical_soft'] for r in later),
                          'gpu_soft':statistics.mean(r['gpu_soft'] for r in later),
                          'gpu_fraction':statistics.mean(r['arm']=='gpu' for r in later)})
    report={'queries':len(rows),'stream_count':len(summaries),'feedback':'simulated soft reviewer P@5 for chosen frozen ranking',
            'initial_alternating':2,'warmup_queries':20,'evaluation_queries_per_stream':53,
            'exploration_probability':.2,'seed_start':20261122,
            'across_stream_means':{k:statistics.mean(x[k] for x in summaries) for k in summaries[0]},
            'routing_minus_lexical_range':[min(x['route_soft']-x['lexical_soft'] for x in summaries),max(x['route_soft']-x['lexical_soft'] for x in summaries)],
            'routing_minus_gpu_range':[min(x['route_soft']-x['gpu_soft'] for x in summaries),max(x['route_soft']-x['gpu_soft'] for x in summaries)],
            'input_sha256':{'rows':sha(rows_path),'summary':sha(root/'summary.json'),'script':sha(Path(__file__))},
            'scope':'Exploratory replay, not a new test set: all rankings and labels were previously examined. Shuffled streams reuse 73 queries, contain no temporal chronology, and are not independent datasets. Each update sees only its chosen arm simulated soft P@5, derived from complete reviewer self-ratings rather than real user feedback or enterprise fit. Fixed epsilon and two arms; no context, feedback text, cost, model training, actual retrieval or latency measurement. Ranges over overlapping streams are not confidence intervals. No held-out transfer, forgetting, non-inferiority or continual-learning claim.'}
    out=Path('artifacts/cof-feedback-routing-simulation-v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'streams.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
