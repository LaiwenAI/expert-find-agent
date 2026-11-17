"""Compare judged-reviewer coverage of bilingual and single-language pools."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from expert_search.cof_benchmark import load,rank_judged
from expert_search.statistics import paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    root,papers,reviewers,labels=load('SIGIR');folder=Path('artifacts/cof-sigir-bilingual-translation-v1');prior=Path('artifacts/cof-bilingual-budget-fusion-v1')
    summary=json.loads((prior/'summary.json').read_text());protocol=json.loads((prior/'protocol.json').read_text())
    if sha(prior/'protocol.json')!=summary['protocol_sha256'] or sha(prior/'results.json')!=summary['results_sha256'] or sha(folder/'query-vectors.npz')!=protocol['translation_vectors_sha256'] or sha(root/'qwen-embedding-torch-v1/vectors.npy')!=protocol['source_vectors_sha256'] or sha(root/'qwen-embedding-torch-v1/protocol.json')!=protocol['source_protocol_sha256']:
        raise ValueError('Changed paired experiment inputs')
    previous=json.loads((prior/'results.json').read_text());source=root/'qwen-embedding-torch-v1';spec=json.loads((source/'protocol.json').read_text())
    pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    def text(pid):return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    items=[dict(id='doc:'+p,text=text(p),query=False) for p in pids]+[dict(id='query:'+q,text=text(q),query=True) for q in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=spec['input_sha256']:raise ValueError('Changed vector order')
    if any(set(labels[t][q])!=set(reviewers) for t in labels for q in qids):raise ValueError('Incomplete original label universe')
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
            if rankings[lang][:10]!=previous[i]['rankings'][lang]:raise ValueError('Changed baseline rank')
        pools={'english20':set(rankings['english'][:20]),'english50':set(rankings['english'][:50]),'chinese20':set(rankings['chinese'][:20]),'union20':set(rankings['english'][:20])|set(rankings['chinese'][:20])}
        if len(pools['union20'])!=previous[i]['union_sizes']['20'] or len(pools['union20'])>40:raise ValueError('Changed bilingual pool')
        metrics={}
        for task in labels:
            positives={e for e,v in labels[task][q].items() if v>0}
            if not positives:raise ValueError('Cannot compute recall without positive labels')
            metrics[task]={m:len(pool&positives)/len(positives) for m,pool in pools.items()}
        rows.append(dict(query_id=q,status=previous[i]['status'],pool_sizes={m:len(p) for m,p in pools.items()},positive_counts={t:sum(v>0 for v in labels[t][q].values()) for t in labels},recall=metrics))
    report=dict(queries=len(rows),planned=protocol['planned'],mean_pool_sizes={m:statistics.mean(r['pool_sizes'][m] for r in rows) for m in ('english20','english50','chinese20','union20')},
        mean_recall={task:{m:statistics.mean(r['recall'][task][m] for r in rows) for m in ('english20','english50','chinese20','union20')} for task in labels},
        union_minus_english50={task:paired_interval([r['recall'][task]['union20']-r['recall'][task]['english50'] for r in rows]) for task in labels},
        union_minus_english20={task:paired_interval([r['recall'][task]['union20']-r['recall'][task]['english20'] for r in rows]) for task in labels},
        wins_ties_losses_vs_english50={task:{'wins':sum(r['recall'][task]['union20']>r['recall'][task]['english50'] for r in rows),'ties':sum(r['recall'][task]['union20']==r['recall'][task]['english50'] for r in rows),'losses':sum(r['recall'][task]['union20']<r['recall'][task]['english50'] for r in rows)} for task in labels},
        input_sha256={str(p):sha(p) for p in (prior/'protocol.json',prior/'summary.json',prior/'results.json',folder/'query-vectors.npz',source/'vectors.npy',source/'protocol.json')},script_sha256=sha(__file__),
        scope='Original CoF reviewer labels, all judged candidate reviewers, positive defined as score > 0; macro query-level candidate recall. 19 previously inspected pairs from 20 selected queries; one translation missing. Target labels concern reviewer matching, not industrial expertise or source entailment. No translated-query human equivalence review. Same-query pairing, unadjusted query bootstrap; cached query encodings and upstream translation costs excluded. Candidate coverage does not demonstrate top-k quality or end-to-end agent latency.')
    output.mkdir(parents=True,exist_ok=False);(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');report['results_sha256']=sha(output/'results.json');(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
