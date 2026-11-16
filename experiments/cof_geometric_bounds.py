"""Does existing source-ball pruning actually save work on frozen embeddings?"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.bounded_profile import BoundedProfile
from expert_search.cof_benchmark import load, rank_judged, precision_metrics


def run(dataset,stable_dots=False):
    root,papers,reviewers,labels=load(dataset)
    source=root/'qwen-embedding-torch-v1'/'vectors.npy'
    matrix=np.load(source,allow_pickle=False).astype(np.float64)
    pids=sorted({pid for r in reviewers.values() for pid in r['papers']}); qids=sorted(labels['soft'])
    if matrix.shape!=(len(pids)+len(qids),1024) or not np.isfinite(matrix).all():raise ValueError('Invalid source vectors')
    position={pid:i for i,pid in enumerate(pids)}
    owners={eid:np.array([position[p] for p in sorted(set(r['papers']))],dtype=np.int64) for eid,r in reviewers.items()}
    start=time.perf_counter();indexes={eid:BoundedProfile(matrix[rows],groups=4) for eid,rows in owners.items() if len(rows)};build=time.perf_counter()-start
    rows=[]
    for i,qid in enumerate(qids):
        q=matrix[len(pids)+i];timings=dict(full=[],bounded=[]);full=None;bounded=None
        for repeat in range(3):
            for mode in (('full','bounded') if repeat%2==0 else ('bounded','full')):
                start=time.perf_counter()
                if mode=='full':
                    similarities=np.einsum('ij,j->i',matrix[:len(pids)],q,optimize=False) if stable_dots else matrix[:len(pids)]@q
                    full={eid:float(np.max(similarities[ids])) if len(ids) else 0. for eid,ids in owners.items()}
                else:
                    results={eid:index.maximum(q,stable_dots=stable_dots) for eid,index in indexes.items()}
                    bounded={eid:results[eid]['score'] if eid in results else 0. for eid in owners}
                # Both timings include complete deterministic expert sorting.
                ranking=sorted(owners,key=lambda eid:(-float((full if mode=='full' else bounded)[eid]),eid))
                timings[mode].append(time.perf_counter()-start)
        full_order=sorted(full,key=lambda eid:(-full[eid],eid));bounded_order=sorted(bounded,key=lambda eid:(-bounded[eid],eid))
        rows.append(dict(query_id=qid,ordered_top5_equal=full_order[:5]==bounded_order[:5],max_score_difference=max(abs(full[e]-bounded[e]) for e in full),
                         full_median_seconds=float(np.median(timings['full'])),bounded_median_seconds=float(np.median(timings['bounded'])),
                         owned_source_occurrences=sum(len(ids) for ids in owners.values()),unique_source_dots=len(pids),
                         evaluated_member_dots=sum(r['member_dot_products'] for r in results.values()),bound_dots=sum(r['bound_dot_products'] for r in results.values()),
                         skipped_member_dots=sum(r['skipped_members'] for r in results.values()),
                         metrics={task:precision_metrics(rank_judged(full,truth[qid]),truth[qid]) for task,truth in labels.items()}))
        print(dataset,qid,rows[-1]['skipped_member_dots'],flush=True)
    report=dict(dataset=dataset,queries=len(rows),experts=len(owners),unique_sources=len(pids),groups=4,repeats=3,stable_dots=stable_dots,build_seconds=build,
                vectors_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ordered_top5_equal=sum(r['ordered_top5_equal'] for r in rows),
                mean_full_seconds=float(np.mean([r['full_median_seconds'] for r in rows])),mean_bounded_seconds=float(np.mean([r['bounded_median_seconds'] for r in rows])),
                mean_skipped_member_fraction=float(np.mean([r['skipped_member_dots']/max(1,r['owned_source_occurrences']) for r in rows])),rows=rows,
                warning='Mature ball-bound baseline, not new theory. Empirical float64 agreement with 1e-12 pruning slack, not formal real-arithmetic certification. Full path shares a source dot across owners; bounded path duplicates owned members. Cached queries, no online encoder/agent time. Member counts alone do not establish speed.')
    out=Path('artifacts')/('cof-geometric-bounds-'+dataset.lower()+('-stable-v2' if stable_dots else '-v1'));out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='SIGIR',choices=['SIGIR','NIPS']);parser.add_argument('--stable-dots',action='store_true');args=parser.parse_args();run(args.dataset,args.stable_dots)
