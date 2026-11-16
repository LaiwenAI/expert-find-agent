"""Analyze complete paired cache trials with query-level timing aggregation."""
import json
import statistics
from pathlib import Path


def analyze(fp32=False):
    name = 'qwen-prefix-fp32-pilot' if fp32 else 'qwen-prefix-pilot'
    report=json.loads((Path('docs/exploration')/(name+'.json')).read_text())
    rows=report['rows']
    expected=report['protocol']['queries_per_dataset']*report['protocol']['repetitions']*2
    if len(rows)!=expected:
        raise ValueError('Incomplete paired pilot')
    keys=[(r['dataset'],r['query_id'],r['repetition']) for r in rows]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate trial')
    summaries=[]
    for dataset in ('NIPS','SIGIR'):
        selected=[r for r in rows if r['dataset']==dataset]
        queries=sorted({r['query_id'] for r in selected})
        if len(queries)!=report['protocol']['queries_per_dataset']:
            raise ValueError('Unexpected query cohort')
        for qid in queries:
            if sorted(r['repetition'] for r in selected if r['query_id']==qid)!=[0,1]:
                raise ValueError('Missing paired repetition')
        full=statistics.mean(r['seconds']['full'] for r in selected)
        cached=statistics.mean(r['seconds']['cached'] for r in selected)
        summaries.append(dict(dataset=dataset,queries=len(queries),trials=len(selected),
            mean_full_seconds=full,mean_cached_seconds=cached,speed_ratio=full/cached,
            identical_top5_trials=sum(r['top5_identical'] for r in selected),
            maximum_probability_difference=max(r['maximum_probability_difference'] for r in selected),
            metric_changes={task:{metric:statistics.mean(r['metrics']['cached'][task][metric]-r['metrics']['full'][task][metric] for r in selected)
                for metric in ('p@5','ndcg@10')} for task in ('soft','hard')}))
    result=dict(summaries=summaries,limitations='Twelve fixed protocol queries, two timing repetitions each. Arithmetic is recorded in the pilot protocol, and segmented-forward score differences are reported. Query cohort is reused, and this pilot does not establish quality noninferiority or complete-system latency.')
    (Path('docs/exploration')/(name+'-analysis.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32',action='store_true')
    analyze(parser.parse_args().fp32)
