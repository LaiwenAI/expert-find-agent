"""Evaluate frozen Chinese, backtranslated, and fixed RRF reviewer rankings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

from expert_search.cof_benchmark import load, precision_metrics, rank_judged
from expert_search.channel_fusion import fuse
from expert_search.statistics import paired_interval
from experiments.bilingual.evaluate_cof_bilingual import sha


def main(output):
    root,papers,reviewers,labels=load('SIGIR')
    original=Path('artifacts/cof-sigir-bilingual-translation-v1')
    folder=Path('artifacts/cof-sigir-backtranslation-v1')
    selection=json.loads((folder/'encode-manifest.json').read_text())
    encoding=json.loads((folder/'encoding-summary.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    old_protocol=json.loads((original/'protocol.json').read_text())
    checks=[(folder/'encode-input.jsonl',selection['queries_sha256']),
            (folder/'encode-manifest.json',encoding['selection_sha256']),
            (folder/'query-vectors.npz',encoding['vectors_sha256']),
            (folder/'protocol.json',selection['backtranslation_protocol_sha256']),
            (folder/'summary.json',selection['backtranslation_summary_sha256']),
            (original/'encode-manifest.json',selection['original_selection_sha256']),
            (original/'encode-input.jsonl',protocol['input_sha256']),
            (Path('experiments/bilingual/cof_query_encode.py'),selection['encoder_script_sha256']),
            (Path('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json'),selection['checkpoint_provenance_sha256'])]
    if any(sha(p)!=expected for p,expected in checks) or encoding['checkpoint_provenance_sha256']!=selection['checkpoint_provenance_sha256'] or summary['protocol_sha256']!=sha(folder/'protocol.json'):
        raise ValueError('Changed frozen input or checkpoint')
    if any(sha(root/name)!=value for name,value in old_protocol['source_sha256'].items()):raise ValueError('Changed public data')
    source=root/'qwen-embedding-torch-v1'; source_protocol=json.loads((source/'protocol.json').read_text()); source_summary=json.loads((source/'summary.json').read_text())
    if sha(source/'vectors.npy')!=source_summary['vectors_sha256'] or source_protocol['checkpoint']!=json.loads(Path('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json').read_text()) or source_protocol['checkpoint']['revision']!=encoding['checkpoint_revision']:
        raise ValueError('Changed source vectors or checkpoint')
    parents={r['id']:r for r in map(json.loads,(original/'encode-input.jsonl').read_text().splitlines())}
    rows=list(map(json.loads,(folder/'encode-input.jsonl').read_text().splitlines()))
    if [r['id'] for r in rows]!=selection['available_query_ids'] or set(selection['available_query_ids'])|set(selection['missing_query_ids'])!=set(selection['all_query_ids']) or set(selection['available_query_ids'])&set(selection['missing_query_ids']):raise ValueError('Changed coverage')
    for row in rows:
        qid=row['id'];name=hashlib.sha256(qid.encode()).hexdigest()+'.json';path=folder/'queries'/name
        if sha(path)!=summary['record_sha256'][name]:raise ValueError('Changed generation record')
        record=json.loads(path.read_text());parent=parents[qid]
        if row['chinese']!=parent['chinese'] or row['parent_status']!=parent['status'] or row['english']!=record['backtranslation'] or row['status']!=record['status'] or record['chinese_sha256']!=hashlib.sha256(row['chinese'].encode()).hexdigest():raise ValueError('Changed paired text')
    pids=sorted({pid for r in reviewers.values() for pid in r['papers']});qids=sorted(labels['soft'])
    items=[dict(id='doc:'+pid,text=(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip(),query=False) for pid in pids]
    items += [dict(id='query:'+qid,text=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip(),query=True) for qid in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=source_protocol['input_sha256']:raise ValueError('Changed vector ordering')
    matrix=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/'query-vectors.npz',allow_pickle=False) as d:
        vectors=d['vectors'].copy();ids=d['ids'].tolist();eligible=d['eligible_indices'].tolist();lengths=d['token_lengths'].tolist()
    if ids!=[rows[i]['id'] for i in eligible] or len(ids)!=encoding['encoded_pairs'] or vectors.shape!=(2*len(ids),1024) or lengths!=encoding['token_lengths'] or any(lengths[2*i]>512 or lengths[2*i+1]>512 or not rows[i]['contains_han'] for i in eligible) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):raise ValueError('Invalid query vectors')
    with np.load(original/'query-vectors.npz',allow_pickle=False) as d:
        oldids=d['ids'].tolist();oldvectors=d['vectors'].copy()
    oldsummary=json.loads((original/'encoding-summary.json').read_text())
    if sha(original/'query-vectors.npz')!=oldsummary['vectors_sha256']:raise ValueError('Changed original vectors')
    pos={pid:i for i,pid in enumerate(pids)}
    def score(qid,v):
        similarity=matrix[:len(pids)]@v
        scores={eid:max((float(similarity[pos[pid]]) for pid in set(reviewers[eid]['papers'])),default=0.) for eid in labels['soft'][qid]}
        return rank_judged(scores,labels['soft'][qid])
    results=[]
    for index,qid in enumerate(ids):
        row=rows[eligible[index]]
        if set(labels['soft'][qid])!=set(labels['hard'][qid]):raise ValueError('Changed judged candidates')
        back=score(qid,vectors[2*index]);zh=score(qid,vectors[2*index+1]);old=score(qid,oldvectors[2*oldids.index(qid)+1])
        fused=[x['id'] for x in fuse({'chinese':zh,'backtranslation':back},k=len(zh),depth=len(zh),constant=60)]
        arms={'chinese':zh,'backtranslation':back,'fusion':fused}
        results.append({'query_id':qid,'status':row['status'],'parent_status':row['parent_status'],
            'top10':{k:v[:10] for k,v in arms.items()},'old_chinese_top10_overlap':len(set(zh[:10])&set(old[:10]))/10,
            'old_chinese_cosine':float(vectors[2*index+1]@oldvectors[2*oldids.index(qid)+1]),
            'metrics':{t:{a:precision_metrics(rank,labels[t][qid]) for a,rank in arms.items()} for t in labels}})
    def aggregate(group):
        if not group:return {'queries':0}
        return {'queries':len(group),'metrics':{t:{a:{m:statistics.mean(r['metrics'][t][a][m] for r in group) for m in ('p@5','ndcg@10')} for a in ('chinese','backtranslation','fusion')} for t in labels},
                'paired_p5':{t:{a:paired_interval([r['metrics'][t][a]['p@5']-r['metrics'][t]['chinese']['p@5'] for r in group]) if len(group)>=10 else None for a in ('backtranslation','fusion')} for t in labels}}
    output.mkdir(parents=True,exist_ok=False)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    missing=len(selection['all_query_ids'])-len(results);n=len(selection['all_query_ids'])
    report={'planned':n,'encoded':len(results),'missing':missing,'all_encoded':aggregate(results),
      'both_model_screened':aggregate([r for r in results if r['status']=='model_screened' and r['parent_status']=='model_screened']),
      'full_cohort_p5_bounds':{t:{a:[sum(r['metrics'][t][a]['p@5'] for r in results)/n,(sum(r['metrics'][t][a]['p@5'] for r in results)+missing)/n] for a in ('chinese','backtranslation','fusion')} for t in labels},
      'mean_old_chinese_cosine':statistics.mean(r['old_chinese_cosine'] for r in results),'mean_old_chinese_top10_overlap':statistics.mean(r['old_chinese_top10_overlap'] for r in results),
      'mean_backtranslation_seconds':summary['mean_translation_seconds_available'],'batch_encoding_seconds':encoding['batch_encoding_seconds'],
      'source_vectors_sha256':sha(source/'vectors.npy'),'encoding_summary_sha256':sha(folder/'encoding-summary.json'),'results_sha256':sha(output/'results.json'),'code_sha256':sha(__file__),
      'scope':'Previously explored public reviewer queries; labels transfer only if both translations preserve meaning, which is not independently established. Screens use the generating model. Missing queries remain unknown. Translation timing excludes auditing; GPU batch timing is not single-query latency. No industrial relevance claim. Bootstrap is unadjusted.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output)
