"""Evaluate remaining frozen bilingual queries on original CoF reviewer labels."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
import numpy as np

from expert_search.cof_benchmark import load,precision_metrics,rank_judged
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    root,papers,reviewers,labels=load('SIGIR');folder=Path('artifacts/cof-sigir-bilingual-remaining-v1')
    protocol=json.loads((folder/'protocol.json').read_text());selection=json.loads((folder/'encode-manifest.json').read_text());encoding=json.loads((folder/'encoding-summary.json').read_text())
    if sha(folder/'encode-input.jsonl')!=selection['queries_sha256'] or sha(folder/'encode-manifest.json')!=encoding['selection_sha256'] or sha(folder/'query-vectors.npz')!=encoding['vectors_sha256']:
        raise ValueError('Changed query encoding')
    if any(sha(root/n)!=h for n,h in protocol['source_sha256'].items()):raise ValueError('Changed public benchmark')
    if sha(folder/'protocol.json')!=selection['translation_protocol_sha256'] or sha(folder/'summary.json')!=selection['translation_summary_sha256']:
        raise ValueError('Changed translation selection')
    qrows=[json.loads(line) for line in (folder/'encode-input.jsonl').read_text().splitlines()];by_id={r['id']:r for r in qrows}
    if [r['id'] for r in qrows]!=selection['available_query_ids'] or len(by_id)!=len(qrows):raise ValueError('Changed query IDs')
    translation_summary=json.loads((folder/'summary.json').read_text())
    if set(selection['available_query_ids'])&set(selection['missing_query_ids']) or set(selection['available_query_ids'])|set(selection['missing_query_ids'])!=set(selection['all_query_ids']):raise ValueError('Changed selection coverage')
    for qid in selection['all_query_ids']:
        name=hashlib.sha256(qid.encode()).hexdigest()+'.json';path=folder/'queries'/name
        if sha(path)!=translation_summary['record_sha256'][name]:raise ValueError('Changed translation record')
        source_record=json.loads(path.read_text());paper=papers[qid];english=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
        if source_record['query_id']!=qid or hashlib.sha256(english.encode()).hexdigest()!=source_record['english_sha256']:raise ValueError('Changed original query')
        if qid in by_id:
            row=by_id[qid];han=any('\u4e00'<=c<='\u9fff' for c in row['chinese'])
            if row['english']!=english or row['chinese']!=source_record['chinese'] or row['status']!=source_record['status'] or row['contains_han']!=han:raise ValueError('Changed paired view')
        elif source_record['chinese'] is not None:raise ValueError('Unexpected missing translation')
    source=root/'qwen-embedding-torch-v1';source_protocol=json.loads((source/'protocol.json').read_text());source_summary=json.loads((source/'summary.json').read_text())
    if sha(source/'vectors.npy')!=source_summary['vectors_sha256'] or hashlib.sha256(inspect.getsource(QwenTorchEmbedding).encode()).hexdigest()!=source_protocol['encoder_sha256']:
        raise ValueError('Changed source vector model or array')
    if source_protocol['checkpoint']['revision']!=encoding['checkpoint_revision']:raise ValueError('Checkpoint revisions differ')
    if source_protocol['checkpoint']!=json.loads(Path('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json').read_text()):raise ValueError('Changed source checkpoint provenance')
    scripts=[Path('experiments/bilingual/cof_query_encode.py'),Path('experiments/bilingual/cof_query_encode_v1.py')]
    if not any(path.is_file() and sha(path)==selection['encoder_script_sha256'] for path in scripts):raise ValueError('Missing frozen encoder script')
    if 'checkpoint_provenance_sha256' in encoding and encoding['checkpoint_provenance_sha256']!=sha('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json'):raise ValueError('Checkpoint provenance differs')
    pids=sorted({pid for r in reviewers.values() for pid in r['papers']});qids=sorted(labels['soft'])
    items=[dict(id='doc:'+pid,text=(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip(),query=False) for pid in pids]
    items += [dict(id='query:'+qid,text=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip(),query=True) for qid in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()!=source_protocol['input_sha256']:raise ValueError('Changed source vector ordering')
    matrix=np.load(source/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(items),1024):raise ValueError('Invalid source vectors')
    with np.load(folder/'query-vectors.npz',allow_pickle=False) as data:
        vectors=data['vectors'].copy();ids=data['ids'].tolist();eligible=data['eligible_indices'].tolist();lengths=data['token_lengths'].tolist()
    if sorted(set(eligible))!=eligible or any(type(i)!=int or not 0<=i<len(qrows) for i in eligible):raise ValueError('Invalid encoded positions')
    if ids!=[qrows[i]['id'] for i in eligible] or len(lengths)!=2*len(qrows) or lengths!=encoding['token_lengths'] or len(ids)!=encoding['encoded_pairs'] or vectors.shape!=(2*len(ids),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid query vectors')
    if any(lengths[2*i]>512 or lengths[2*i+1]>512 or not qrows[i]['contains_han'] for i in eligible):raise ValueError('Truncated or non-Chinese-marked query')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):raise ValueError('Unnormalized queries')
    doc_position={pid:i for i,pid in enumerate(pids)};query_position={qid:len(pids)+i for i,qid in enumerate(qids)}
    def score(qid,vector):
        start=time.perf_counter();similarity=matrix[:len(pids)]@vector
        scores={eid:max((float(similarity[doc_position[pid]]) for pid in set(reviewers[eid]['papers'])),default=0.) for eid in labels['soft'][qid]}
        ranking=rank_judged(scores,labels['soft'][qid]);return ranking,time.perf_counter()-start
    rows=[]
    for i,qid in enumerate(ids):
        if set(labels['soft'][qid])!=set(labels['hard'][qid]):raise ValueError('Changed judged candidates')
        arms={};times={}
        for language in (('english','chinese') if i%2==0 else ('chinese','english')):
            ranking,elapsed=score(qid,vectors[2*i+(language=='chinese')]);arms[language]=ranking;times[language]=elapsed
        old,_=score(qid,np.asarray(matrix[query_position[qid]]));cosine=float(vectors[2*i]@matrix[query_position[qid]])
        rows.append({'query_id':qid,'status':by_id[qid]['status'],'english':arms['english'][:10],'chinese':arms['chinese'][:10],
                     'english_cached_ordered_top10_equal':arms['english'][:10]==old[:10],
                     'english_cached_overlap_at_10':len(set(arms['english'][:10])&set(old[:10]))/10,'english_cached_cosine':cosine,
                     'cached_english_metrics':{task:precision_metrics(old,labels[task][qid]) for task in labels},
                     'overlap_at_10':len(set(arms['english'][:10])&set(arms['chinese'][:10]))/10,'seconds':times,
                     'metrics':{task:{language:precision_metrics(arms[language],labels[task][qid]) for language in arms} for task in labels}})
    output.mkdir(parents=True,exist_ok=False);(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    missing=set(selection['all_query_ids'])-set(ids)
    def aggregate(subset):
        if not subset:return {'queries':0}
        return {'queries':len(subset),'mean_overlap_at_10':statistics.mean(r['overlap_at_10'] for r in subset),
                'metrics':{t:{a:{m:statistics.mean(r['metrics'][t][a][m] for r in subset) for m in ('p@5','ndcg@10')} for a in ('english','chinese')} for t in labels},
                'paired_p_at_5':{t:paired_interval([r['metrics'][t]['chinese']['p@5']-r['metrics'][t]['english']['p@5'] for r in subset]) if len(subset)>=10 else None for t in labels}}
    report={'planned':len(selection['all_query_ids']),'encoded_pairs':len(rows),'missing_pairs':len(missing),'status_counts':dict(Counter(r['status'] for r in rows)),
            'all_encoded':aggregate(rows),'model_screened':aggregate([r for r in rows if r['status']=='model_screened']),
            'english_cached_top10_equal':sum(r['english_cached_ordered_top10_equal'] for r in rows),
            'mean_english_cached_overlap_at_10':statistics.mean(r['english_cached_overlap_at_10'] for r in rows) if rows else None,
            'fresh_minus_cached_english_p5':{t:paired_interval([r['metrics'][t]['english']['p@5']-r['cached_english_metrics'][t]['p@5'] for r in rows]) if len(rows)>=10 else None for t in labels},
            'mean_english_cached_cosine':statistics.mean(r['english_cached_cosine'] for r in rows) if rows else None,
            'mean_cached_scoring_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in ('english','chinese')} if rows else {},
            'chinese_full_cohort_p5_bounds':{t:[sum(r['metrics'][t]['chinese']['p@5'] for r in rows)/len(selection['all_query_ids']),
                                               (sum(r['metrics'][t]['chinese']['p@5'] for r in rows)+len(missing))/len(selection['all_query_ids'])] for t in labels},
            'code_sha256':sha(Path(__file__)),'source_protocol_sha256':sha(source/'protocol.json'),'selection_sha256':sha(folder/'encode-manifest.json'),
            'checkpoint_provenance_pinned_at_encoding':'checkpoint_provenance_sha256' in encoding,
            'source_vectors_sha256':sha(source/'vectors.npy'),'encoding_summary_sha256':sha(folder/'encoding-summary.json'),'results_sha256':sha(output/'results.json'),
            'scope':'Post-hoc reused public reviewer test queries; Chinese translations and same-model screens lack human verification. Reviewer-level labels do not establish industrial fit or source entailment. Failed translations/encoding remain unknown in full-cohort bounds. Cached scoring excludes query encoding, initialization and reasoning. Bootstrap assumes independent queries and is unadjusted.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
