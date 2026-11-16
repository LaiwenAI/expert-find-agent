"""Recompute paired semantic-transport quality against the frozen Model1 control."""
import hashlib
import json
from pathlib import Path
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.packing_analysis import paired_interval


def analyze(dataset):
    root,_,_,labels=load(dataset)
    qids=sorted(labels['soft'])
    directory=root/'semantic-transport-v1'
    if {p.stem.removeprefix('query-') for p in directory.glob('query-*.json')}!=set(qids):
        raise ValueError('Incomplete or unexpected semantic-transport query set')
    rows=[];hashes={}
    for qid in qids:
        paths=[directory/f'query-{qid}.json',root/'language-models-v2'/f'query-{qid}.json']
        semantic,baseline=[json.loads(p.read_text()) for p in paths]
        if semantic['query_id']!=qid or baseline['query_id']!=qid:raise ValueError('Wrong query ID')
        rankings={'semantic':semantic['ranking'],'model1':baseline['rankings']['model1-document']}
        for ranking in rankings.values():
            if len(set(ranking))!=len(ranking) or set(ranking)!=set(labels['soft'][qid]):
                raise ValueError('Methods do not score the same judged candidates')
        metrics={method:{task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()}
                 for method,ranking in rankings.items()}
        if metrics['semantic']!=semantic['metrics']:raise ValueError('Changed semantic metrics')
        if any(metrics['model1'][task]!=baseline['metrics'][task]['model1-document'] for task in labels):
            raise ValueError('Changed reference metrics')
        rows.append(dict(query_id=qid,metrics=metrics,
            top5_changed=rankings['semantic'][:5]!=rankings['model1'][:5]))
        hashes[qid]=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    return dict(dataset=dataset,queries=len(rows),changed_top5_queries=sum(r['top5_changed'] for r in rows),
        contrasts={task:{metric:paired_interval([r['metrics']['semantic'][task][metric]-r['metrics']['model1'][task][metric]
            for r in rows]) for metric in ('p@5','p@10','ndcg@10')} for task in labels},
        paired_input_hashes=hashes,
        warning='Exploratory query bootstrap, no multiplicity correction; shared profiles/topics may violate query independence. Intervals crossing zero do not prove equivalence. No label tuning, novel-formula or industrial-quality claim.')


if __name__=='__main__':
    for dataset in ('SIGIR','NIPS'):
        report=analyze(dataset)
        Path(f'docs/exploration/semantic-transport-{dataset.lower()}-analysis.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='paired_input_hashes'},indent=2))
