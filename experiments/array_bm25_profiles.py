"""Complete expert aggregation/ranking check against frozen lexical baselines."""
import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote
from expert_search.array_bm25 import ArrayBM25
from expert_search.cof_benchmark import load,rank_judged,precision_metrics


def run(dataset):
    root,papers,reviewers,labels=load(dataset)
    engine=ArrayBM25(root/'unique-profile-corpus/fast.sqlite')
    profiles={eid:r['papers'] for eid,r in reviewers.items()};rows=[]
    for qid in sorted(labels['soft']):
        paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        reference=json.loads((root/'baseline-lexical'/(quote(qid,safe='')+'.json')).read_text())
        start=time.perf_counter();scores=engine.profile_scores(query,profiles)
        results={task:{method:rank_judged({eid:s[method] for eid,s in scores.items()},truth[qid]) for method in ('bm25-max','bm25-top3')} for task,truth in labels.items()}
        seconds=time.perf_counter()-start
        for task,methods in results.items():
            for method,ranking in methods.items():
                expected=reference['results'][task][method]
                if ranking[:10]!=expected['ranking'] or precision_metrics(ranking,labels[task][qid])!=expected['metrics']:
                    raise AssertionError('Changed ranking or metrics')
        rows.append(dict(query_id=qid,seconds=seconds,exact_top10=True))
    report=dict(dataset=dataset,queries=len(rows),mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
        array_bytes=engine.array_bytes,rows=rows,scope='Resident BM25 document scoring, unique-profile max/top3 aggregation, and judged-candidate sorting for both tasks. Excludes initialization and input loading; all top10 rankings and metrics equal frozen SQLite outputs.')
    path=Path('artifacts')/('array-bm25-profiles-'+dataset.lower()+'.json');path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='NIPS');run(parser.parse_args().dataset)
