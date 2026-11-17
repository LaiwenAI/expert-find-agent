"""Analyze bilingual ranking stability across all frozen construction intents."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_interval(rows, key, seed=20261118, repeats=2000):
    grouped=defaultdict(list)
    for row in rows:grouped[row['group_id']].append(row[key])
    groups=sorted(grouped);sums=np.asarray([sum(grouped[g]) for g in groups]);counts=np.asarray([len(grouped[g]) for g in groups])
    rng=np.random.default_rng(seed);draws=rng.integers(len(groups),size=(repeats,len(groups)))
    boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return dict(queries=len(rows),company_groups=len(groups),mean=float(sums.sum()/counts.sum()),percentile_95=list(map(float,np.quantile(boot,[.025,.975]))),seed=seed,repeats=repeats)


def main(output):
    root=Path('artifacts/bilingual-intent-retrieval-results-v1');summary=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if sha(root/'protocol.json')!=summary['protocol_sha256'] or sha(root/'private-results.json')!=summary['private_results_sha256']:raise ValueError('Changed full-intent results')
    selection=Path('artifacts/bilingual-intent-retrieval-v1');spec=json.loads((selection/'selection.json').read_text())
    if sha(selection/'selection.json')!=protocol['selection_sha256'] or sha(selection/'queries.jsonl')!=spec['query_sha256'] or spec['query_sha256']!=protocol['query_sha256']:raise ValueError('Changed query selection')
    query_rows=[json.loads(s) for s in (selection/'queries.jsonl').read_text().splitlines()];results=json.loads((root/'private-results.json').read_text())
    if len(query_rows)!=5000 or len(results)!=5000 or [q['id'] for q in query_rows]!=[r['id'] for r in results]:raise ValueError('Changed query order')
    evaluated=[]
    for q,r in zip(query_rows,results):
        if any(q[k]!=r[k] for k in ('id','parent_id','group_id','intent_type','status')):raise ValueError('Changed query metadata')
        if any(len(r['ranks'][lang])!=10 or len(set(r['ranks'][lang]))!=10 for lang in ('zh','en')):raise ValueError('Invalid result ranking')
        x,y=r['ranks']['zh'],r['ranks']['en'];evaluated.append(dict(id=r['id'],group_id=r['group_id'],intent_type=r['intent_type'],status=r['status'],overlap=len(set(x)&set(y))/10,ordered_equal=x==y))
    earlier=Path('artifacts/bilingual-retrieval-v1');old_summary=json.loads((earlier/'summary.json').read_text());old_selection=json.loads((earlier/'selection.json').read_text())
    if sha(earlier/'evaluation.jsonl')!=old_summary['evaluation_sha256'] or sha(earlier/'protocol.json')!=old_summary['protocol_sha256'] or sha(earlier/'queries.jsonl')!=old_selection['queries_sha256']:
        raise ValueError('Changed earlier comparison cohort')
    old=[json.loads(x) for x in (earlier/'evaluation.jsonl').read_text().splitlines()];by_id={r['id']:r for r in results}
    if len(old)!=1000 or len({r['id'] for r in old})!=1000 or any(r['id'] not in by_id for r in old):raise ValueError('Changed earlier query subset')
    drift={}
    for lang in ('zh','en'):
        equal=0;overlaps=[]
        for oldrow in old:
            new=by_id[oldrow['id']]
            if any(new[k]!=oldrow[k] for k in ('parent_id','group_id','intent_type','status')):raise ValueError('Changed earlier metadata')
            previous=oldrow['languages'][lang]['exact']['ranking'];current=new['ranks'][lang]
            equal+=previous==current;overlaps.append(len(set(previous)&set(current))/10)
        drift[lang]=dict(queries=len(old),ordered_top10_equal=equal,mean_top10_set_overlap=statistics.mean(overlaps))
    def aggregate(items):
        return dict(stability=group_interval(items,'overlap'),disjoint=sum(r['overlap']==0 for r in items),identical_sets=sum(r['overlap']==1 for r in items),identical_order=sum(r['ordered_equal'] for r in items))
    report=dict(queries=len(evaluated),parents=len({q['parent_id'] for q in query_rows}),overall=aggregate(evaluated),
        by_intent={k:aggregate([r for r in evaluated if r['intent_type']==k]) for k in sorted({r['intent_type'] for r in evaluated})},
        by_screen_status={k:aggregate([r for r in evaluated if r['status']==k]) for k in sorted({r['status'] for r in evaluated})},prior_1000_replay=drift,
        input_sha256={str(p):sha(p) for p in (root/'summary.json',root/'protocol.json',root/'private-results.json',selection/'selection.json',selection/'queries.jsonl',earlier/'summary.json',earlier/'evaluation.jsonl',earlier/'queries.jsonl')},script_sha256=sha(__file__),
        scope='Observed cross-language expert ranking overlap, not semantic equivalence, expert relevance, or held-out accuracy. Bootstrap resamples company groups and weights all views; repeated parents within groups remain dependent. Type and status estimates are descriptive and unadjusted. Earlier rankings used separately encoded queries and CPU FP32 on another node; differences cannot be attributed to one cause. No source text or identifiers in this report.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
