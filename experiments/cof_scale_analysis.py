"""Recompute scale-check metrics with reported exact-copy query sensitivity."""
import json
import statistics
from pathlib import Path
from urllib.parse import quote
from expert_search.cof_benchmark import load,precision_metrics


def analyze(dataset):
    root,_,_,labels=load(dataset)
    audit=json.loads(Path(f'docs/exploration/cof-{dataset.lower()}-audit.json').read_text())
    copied=set(audit['exact_query_texts_in_profiles'])
    rows=[]
    for qid in sorted(labels['soft']):
        path=root/'scale-check-v1'/f"{quote(qid, safe='')}.json"
        if not path.exists():raise ValueError('Incomplete scale check')
        row=json.loads(path.read_text())
        if row['query_id']!=qid or len(set(row['ranking']))!=len(row['ranking']):
            raise ValueError('Invalid query/ranking')
        if set(row['ranking'])!=set(labels['soft'][qid]):
            raise ValueError('Incomplete judged candidate ranking')
        for task,truth in labels.items():
            if row['metrics'][task]!=precision_metrics(row['ranking'],truth[qid]):
                raise ValueError('Changed source metrics')
        rows.append(row)
    cohorts={'original':rows,'exclude_exact_copy_queries':[r for r in rows if r['query_id'] not in copied]}
    result=dict(dataset=dataset,exact_copy_queries=sorted(copied),
        cohorts={name:dict(queries=len(selected),metrics={task:{metric:statistics.mean(r['metrics'][task][metric] for r in selected)
            for metric in ('p@5','ndcg@10')} for task in labels}) for name,selected in cohorts.items()},
        scope='Sensitivity analysis excludes whole exact-copy queries identified in the prior corpus audit. Original protocol results retained. Near duplicates and broader contamination remain unmeasured.')
    Path(f'docs/exploration/cof-scale-{dataset.lower()}-analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='KDD',choices=['KDD','SciRepEval'])
    analyze(parser.parse_args().dataset)
