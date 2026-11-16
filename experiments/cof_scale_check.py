"""Fixed classical-LM scalability check on additional original CoF datasets."""
import json
import statistics
import time
from pathlib import Path
from urllib.parse import quote
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM


def run(dataset):
    root,papers,reviewers,labels=load(dataset)
    documents={pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
               for pid in sorted({p for r in reviewers.values() for p in r['papers']})}
    start=time.perf_counter()
    reference=ExpertLanguageModels(documents,{eid:r['papers'] for eid,r in reviewers.items()})
    build_seconds=time.perf_counter()-start
    compiled=CompiledCandidateLM(reference)
    available = sorted(compiled.available)
    judged = set().union(*(set(row) for row in labels['soft'].values()))
    if judged - compiled.available:
        raise ValueError('Judged profiles have no usable evidence')
    rows=[]
    output=root/'scale-check-v1';output.mkdir(exist_ok=True)
    for qid in sorted(labels['soft']):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        start=time.perf_counter();scores=compiled.score(query)
        ranking=sorted(scores,key=lambda eid:(-scores[eid],eid))
        seconds=time.perf_counter()-start
        if scores!=reference.score(query,eligible=available):raise ValueError('Compiled scores changed')
        judged=[e for e in ranking if e in labels['soft'][qid]]
        row=dict(query_id=qid,ranking=judged,seconds=seconds,
                 metrics={task:precision_metrics(judged,truth[qid]) for task,truth in labels.items()})
        (output/f"{quote(qid, safe='')}.json").write_text(json.dumps(row,indent=2));rows.append(row)
        print(dataset,qid,'complete',flush=True)
    report=dict(dataset=dataset,queries=len(rows),profiles=len(compiled.available),documents=len(documents),
        reference_build_seconds=build_seconds,compiled_array_bytes=compiled.array_bytes,
        mean_seconds=statistics.mean(r['seconds'] for r in rows),exact_reference_scores=True,
        metrics={task:{key:statistics.mean(r['metrics'][task][key] for r in rows)
                       for key in ('p@5','ndcg@10')} for task in labels},
        scope='Fixed JM=.5 classical control. Original released candidate judgments. Full-profile score and sort timing, single pass. Source/query duplication requires separate leakage analysis before quality interpretation.')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    Path(f'docs/exploration/cof-scale-{dataset.lower()}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['KDD','SciRepEval'],default='KDD')
    run(parser.parse_args().dataset)
