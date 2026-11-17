"""Evaluate fixed bilingual fusion on the remaining public reviewer queries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from expert_search.channel_fusion import fuse
from expert_search.cof_benchmark import load,precision_metrics,rank_judged
from expert_search.statistics import paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    root,papers,reviewers,labels=load('SIGIR');folder=Path('artifacts/cof-sigir-bilingual-remaining-v1');prior=Path('artifacts/cof-sigir-bilingual-remaining-eval-v1')
    summary=json.loads((prior/'summary.json').read_text());selection=json.loads((folder/'encode-manifest.json').read_text());encoding=json.loads((folder/'encoding-summary.json').read_text());translation=json.loads((folder/'protocol.json').read_text())
    if sha(prior/'results.json')!=summary['results_sha256'] or sha(folder/'encoding-summary.json')!=summary['encoding_summary_sha256'] or sha(folder/'encode-manifest.json')!=encoding['selection_sha256'] or sha(folder/'query-vectors.npz')!=encoding['vectors_sha256'] or any(sha(root/n)!=h for n,h in translation['source_sha256'].items()):raise ValueError('Changed baseline, translations or labels')
    previous=json.loads((prior/'results.json').read_text());source=root/'qwen-embedding-torch-v1';spec=json.loads((source/'protocol.json').read_text())
    if sha(source/'vectors.npy')!=summary['source_vectors_sha256'] or sha(source/'protocol.json')!=summary['source_protocol_sha256']:raise ValueError('Changed source vectors')
    pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    def text(pid):return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    items=[dict(id='doc:'+p,text=text(p),query=False) for p in pids]+[dict(id='query:'+q,text=text(q),query=True) for q in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=spec['input_sha256'] or any(set(labels[t][q])!=set(reviewers) for t in labels for q in qids):raise ValueError('Changed source order or labels')
    matrix=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/'query-vectors.npz',allow_pickle=False) as data:vectors=data['vectors'].copy();ids=data['ids'].tolist()
    if ids!=[r['query_id'] for r in previous] or len(ids)!=summary['encoded_pairs'] or not ids or vectors.shape!=(2*len(ids),1024) or matrix.shape!=(len(items),1024):raise ValueError('Changed query cohort')
    positions={p:i for i,p in enumerate(pids)};rows=[]
    for i,q in enumerate(ids):
        rankings={}
        for j,lang in enumerate(('english','chinese')):
            similarity=matrix[:len(pids)]@vectors[2*i+j];scores={e:max(float(similarity[positions[p]]) for p in set(r['papers'])) for e,r in reviewers.items()}
            rankings[lang]=rank_judged(scores,labels['soft'][q])
            if rankings[lang][:10]!=previous[i][lang]:raise ValueError('Changed baseline ranking')
        fused=[x['id'] for x in fuse(rankings,k=50,depth=50,method='rrf',constant=60)]
        if len(fused)!=50:raise ValueError('Incomplete fused ranking')
        metrics={}
        for task in labels:
            positives={e for e,v in labels[task][q].items() if v>0}
            if not positives:raise ValueError('Undefined candidate recall')
            metrics[task]={}
            for method,ranking in [('english',rankings['english']),('chinese',rankings['chinese']),('rrf',fused)]:
                selected=ranking[:50];score=precision_metrics(selected,labels[task][q]);metrics[task][method]=dict(candidate_recall=len(set(selected)&positives)/len(positives),p5=score['p@5'],ndcg10=score['ndcg@10'])
        rows.append(dict(query_id=q,status=previous[i]['status'],metrics=metrics))
    def compare(task,metric):
        values=[r['metrics'][task]['rrf'][metric]-r['metrics'][task]['english'][metric] for r in rows]
        return dict(interval=paired_interval(values),wins=sum(x>1e-12 for x in values),ties=sum(abs(x)<=1e-12 for x in values),losses=sum(x< -1e-12 for x in values))
    planned=len(selection['all_query_ids']);missing=planned-len(rows)
    if planned!=53 or missing<0:raise ValueError('Changed planned cohort')
    metrics={task:{method:{metric:statistics.mean(r['metrics'][task][method][metric] for r in rows) for metric in ('candidate_recall','p5','ndcg10')} for method in ('english','chinese','rrf')} for task in labels}
    bounds={task:{method:{metric:[sum(r['metrics'][task][method][metric] for r in rows)/planned,(sum(r['metrics'][task][method][metric] for r in rows)+missing)/planned] for metric in ('candidate_recall','p5','ndcg10')} for method in ('english','chinese','rrf')} for task in labels}
    result=dict(planned=planned,queries=len(rows),missing=missing,status_counts=dict(Counter(r['status'] for r in rows)),budget=50,input_depth_per_language=50,rrf_constant=60,metrics=metrics,planned_cohort_box_bounds=bounds,
        fused_minus_english={task:{metric:compare(task,metric) for metric in ('candidate_recall','p5','ndcg10')} for task in labels},
        evaluation_summary_sha256=sha(prior/'summary.json'),evaluation_results_sha256=sha(prior/'results.json'),encoding_summary_sha256=sha(folder/'encoding-summary.json'),translation_protocol_sha256=sha(folder/'protocol.json'),source_vectors_sha256=sha(source/'vectors.npy'),script_sha256=sha(__file__),fusion_sha256=sha('src/expert_search/channel_fusion.py'),
        scope='All remaining 53 CoF SIGIR queries after earlier 20; dataset globally explored, not a sealed test. Fixed RRF 60/top50 and English baseline carried from earlier exploratory run. Original reviewer score >0 defines positives; translations lack independent equivalence review. Missing translations remain unknown; box bounds allow each missing metric [0,1]. Equal candidate count but unequal translation and retrieval costs. Cached vectors exclude encoding, agent reasoning and service time. Query bootstrap assumes independent queries and is unadjusted.')
    output.mkdir(parents=True,exist_ok=False);(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');result['results_sha256']=sha(output/'results.json');(output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
