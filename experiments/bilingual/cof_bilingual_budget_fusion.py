"""Compare paired-language candidate fusion on existing CoF reviewer labels."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from expert_search.channel_fusion import fuse
from expert_search.cof_benchmark import load,precision_metrics,rank_judged
from expert_search.statistics import paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    root,papers,reviewers,labels=load('SIGIR');folder=Path('artifacts/cof-sigir-bilingual-translation-v1');previous=Path('artifacts/cof-sigir-bilingual-evaluation-v5')
    summary=json.loads((previous/'summary.json').read_text());selection=json.loads((folder/'encode-manifest.json').read_text());encoding=json.loads((folder/'encoding-summary.json').read_text())
    if sha(previous/'results.json')!=summary['results_sha256'] or sha(folder/'encoding-summary.json')!=summary['encoding_summary_sha256'] or sha(folder/'query-vectors.npz')!=encoding['vectors_sha256'] or sha(folder/'encode-manifest.json')!=encoding['selection_sha256'] or sha(folder/'encode-input.jsonl')!=selection['queries_sha256']:
        raise ValueError('Changed translation or baseline input')
    source=root/'qwen-embedding-torch-v1';spec=json.loads((source/'protocol.json').read_text())
    if sha(source/'vectors.npy')!=summary['source_vectors_sha256'] or sha(source/'protocol.json')!=summary['source_protocol_sha256']:raise ValueError('Changed source model or vectors')
    qrows=[json.loads(x) for x in (folder/'encode-input.jsonl').read_text().splitlines()];old=json.loads((previous/'results.json').read_text())
    pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    def text(pid):return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    items=[dict(id='doc:'+p,text=text(p),query=False) for p in pids]+[dict(id='query:'+q,text=text(q),query=True) for q in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=spec['input_sha256']:raise ValueError('Changed source order')
    if any(set(labels[t][qid])!=set(reviewers) for t in labels for qid in qids):raise ValueError('Changed judged reviewer universe')
    matrix=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/'query-vectors.npz',allow_pickle=False) as data:
        vectors=data['vectors'].copy();ids=data['ids'].tolist();eligible=data['eligible_indices'].tolist()
    if ids!=[qrows[i]['id'] for i in eligible] or ids!=[r['query_id'] for r in old] or len(ids)!=19 or vectors.shape!=(38,1024) or matrix.shape!=(len(items),1024):raise ValueError('Changed query vectors or selection')
    positions={p:i for i,p in enumerate(pids)}
    def rank(vector,truth):
        similarity=matrix[:len(pids)]@vector
        scores={eid:max(float(similarity[positions[p]]) for p in set(r['papers'])) for eid,r in reviewers.items()}
        return rank_judged(scores,truth)
    protocol=dict(queries=len(ids),planned=len(selection['all_query_ids']),source_vectors_sha256=sha(source/'vectors.npy'),source_protocol_sha256=sha(source/'protocol.json'),translation_vectors_sha256=sha(folder/'query-vectors.npz'),selection_sha256=sha(folder/'encode-manifest.json'),previous_results_sha256=sha(previous/'results.json'),script_sha256=sha(__file__),fusion_sha256=sha('src/expert_search/channel_fusion.py'),
        methods='English and Chinese full source score max; RRF with constant 60 and per-language depths 20 or 50. Same judged reviewers and unique source corpus. Candidate union sizes count deduplicated reviewers.',
        scope='Post-hoc reused public labels on 19 successfully translated views; one selected query lacks translation. Same-model screening is not human translation verification. Reviewer labels do not establish enterprise fit, source entailment or individual contribution. Cached scoring excludes query encoding, generation, initialization, reasoning, network and concurrent load.')
    output.mkdir(parents=True,exist_ok=False);(output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[]
    for i,qid in enumerate(ids):
        start=time.perf_counter();rankings={'english':rank(vectors[2*i],labels['soft'][qid]),'chinese':rank(vectors[2*i+1],labels['soft'][qid])};seconds=time.perf_counter()-start
        if any(rankings[lang][:10]!=old[i][lang] for lang in rankings):raise ValueError('Changed prior ranking')
        methods={lang:rankings[lang][:10] for lang in rankings};unions={};fusion_times={}
        for depth in (20,50):
            start=time.perf_counter();methods[f'rrf{depth}']=[r['id'] for r in fuse(rankings,k=10,depth=depth,method='rrf',constant=60)];fusion_times[str(depth)]=time.perf_counter()-start
            unions[str(depth)]=len(set(rankings['english'][:depth])|set(rankings['chinese'][:depth]))
        metrics={task:{m:precision_metrics(ranking,labels[task][qid]) for m,ranking in methods.items()} for task in labels}
        rows.append(dict(query_id=qid,status=old[i]['status'],rankings=methods,metrics=metrics,union_sizes=unions,cached_both_scoring_seconds=seconds,fusion_seconds=fusion_times))
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(queries=len(rows),matched_baseline_rankings=2*len(rows),planned=protocol['planned'],
        mean_union_size={str(d):statistics.mean(r['union_sizes'][str(d)] for r in rows) for d in (20,50)},
        unions_exceed_50={str(d):sum(r['union_sizes'][str(d)]>50 for r in rows) for d in (20,50)},
        metrics={task:{m:{k:statistics.mean(r['metrics'][task][m][k] for r in rows) for k in ('p@5','ndcg@10')} for m in ('english','chinese','rrf20','rrf50')} for task in labels},
        fused_minus_english={task:{m:paired_interval([r['metrics'][task][m]['p@5']-r['metrics'][task]['english']['p@5'] for r in rows]) for m in ('rrf20','rrf50')} for task in labels},
        mean_cached_both_scoring_seconds=statistics.mean(r['cached_both_scoring_seconds'] for r in rows),mean_fusion_seconds={str(d):statistics.mean(r['fusion_seconds'][str(d)] for r in rows) for d in (20,50)},
        protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'results.json'),scope=protocol['scope']+' One observation per query. Intervals resample queries with independence assumed, are unadjusted for multiple comparisons, and exclude the missing translation; no verified intent or causal effect.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
