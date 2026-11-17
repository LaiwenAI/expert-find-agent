"""Compare fused and monolingual candidates at identical output budgets."""
import argparse
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
    root,papers,reviewers,labels=load('SIGIR');folder=Path('artifacts/cof-sigir-bilingual-translation-v1');prior=Path('artifacts/cof-bilingual-budget-fusion-v1')
    report=json.loads((prior/'summary.json').read_text());protocol=json.loads((prior/'protocol.json').read_text())
    if sha(prior/'protocol.json')!=report['protocol_sha256'] or sha(prior/'results.json')!=report['results_sha256'] or sha(folder/'query-vectors.npz')!=protocol['translation_vectors_sha256'] or sha(root/'qwen-embedding-torch-v1/vectors.npy')!=protocol['source_vectors_sha256'] or sha(root/'qwen-embedding-torch-v1/protocol.json')!=protocol['source_protocol_sha256'] or sha('src/expert_search/channel_fusion.py')!=protocol['fusion_sha256']:
        raise ValueError('Changed frozen inputs or fusion')
    previous=json.loads((prior/'results.json').read_text());source=root/'qwen-embedding-torch-v1';spec=json.loads((source/'protocol.json').read_text())
    pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    def text(pid):return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    items=[dict(id='doc:'+p,text=text(p),query=False) for p in pids]+[dict(id='query:'+q,text=text(q),query=True) for q in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=spec['input_sha256']:raise ValueError('Changed paper or query vector order')
    if any(set(labels[t][q])!=set(reviewers) for t in labels for q in qids):raise ValueError('Incomplete reviewer labels')
    matrix=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/'query-vectors.npz',allow_pickle=False) as data:vectors=data['vectors'].copy();ids=data['ids'].tolist()
    if len(ids)!=19 or ids!=[r['query_id'] for r in previous] or vectors.shape!=(38,1024) or matrix.shape!=(len(items),1024):raise ValueError('Changed query cohort')
    positions={p:i for i,p in enumerate(pids)};rows=[]
    for i,q in enumerate(ids):
        rankings={}
        for j,lang in enumerate(('english','chinese')):
            similarity=matrix[:len(pids)]@vectors[2*i+j]
            scores={e:max(float(similarity[positions[p]]) for p in set(r['papers'])) for e,r in reviewers.items()}
            rankings[lang]=rank_judged(scores,labels['soft'][q])
            if rankings[lang][:10]!=previous[i]['rankings'][lang]:raise ValueError('Changed monolingual ranking')
        fused=[x['id'] for x in fuse(rankings,k=50,depth=50,method='rrf',constant=60)]
        if len(fused)!=50 or fused[:10]!=previous[i]['rankings']['rrf50']:raise ValueError('Changed prior fusion ranking')
        metrics={}
        for task in labels:
            positives={e for e,v in labels[task][q].items() if v>0}
            if not positives:raise ValueError('Undefined candidate recall')
            metrics[task]={}
            for budget in (20,50):
                metrics[task][str(budget)]={}
                for method,ranking in [('english',rankings['english']),('chinese',rankings['chinese']),('rrf',fused)]:
                    selected=ranking[:budget];score=precision_metrics(selected,labels[task][q])
                    metrics[task][str(budget)][method]=dict(candidate_recall=len(set(selected)&positives)/len(positives),p5=score['p@5'],ndcg10=score['ndcg@10'])
        rows.append(dict(query_id=q,status=previous[i]['status'],metrics=metrics))
    def compare(task,budget,metric):
        differences=[r['metrics'][task][str(budget)]['rrf'][metric]-r['metrics'][task][str(budget)]['english'][metric] for r in rows]
        return dict(interval=paired_interval(differences),wins=sum(x>1e-12 for x in differences),ties=sum(abs(x)<=1e-12 for x in differences),losses=sum(x< -1e-12 for x in differences))
    metrics={task:{str(budget):{method:{metric:statistics.mean(r['metrics'][task][str(budget)][method][metric] for r in rows) for metric in ('candidate_recall','p5','ndcg10')} for method in ('english','chinese','rrf')} for budget in (20,50)} for task in labels}
    results=dict(queries=len(rows),planned=protocol['planned'],budgets=[20,50],input_depth_per_language=50,rrf_constant=60,metrics=metrics,
        fused_minus_english={task:{str(b):{m:compare(task,b,m) for m in ('candidate_recall','p5','ndcg10')} for b in (20,50)} for task in labels},
        prior_protocol_sha256=sha(prior/'protocol.json'),prior_results_sha256=sha(prior/'results.json'),translation_vectors_sha256=sha(folder/'query-vectors.npz'),source_vectors_sha256=sha(source/'vectors.npy'),script_sha256=sha(__file__),fusion_sha256=sha('src/expert_search/channel_fusion.py'),
        scope='Post-hoc, previously examined 19/20 public reviewer queries. Reviewer positive is original score >0 and recall denominator is all judged positives; translation not independently verified. Equal output candidate count, not equal retrieval/translation cost. Both language vectors and rankings available in advance; no agent reasoning or end-to-end timing. Query bootstrap assumes independence and is unadjusted; labels do not establish industrial expert fit or source entailment.')
    output.mkdir(parents=True,exist_ok=False);(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');results['results_sha256']=sha(output/'results.json');(output/'summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
