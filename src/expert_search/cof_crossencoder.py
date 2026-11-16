"""Frozen cross-encoder control using the exact packed candidate source inputs."""
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path
from .cof_benchmark import load, precision_metrics
from .shared_evidence import expand_candidates


def checked_ranking(results, aliases):
    ids=[r.get('id') for r in results]
    if len(ids)!=len(aliases) or any(type(i)!=int for i in ids) or sorted(ids)!=list(range(len(aliases))):
        raise ValueError('Missing or duplicated cross-encoder candidate')
    if any(type(r.get('score')) not in (float,int) or not math.isfinite(r['score']) for r in results):
        raise ValueError('Invalid cross-encoder score')
    return [aliases[r['id']] for r in sorted(results,key=lambda r:(-r['score'],aliases[r['id']]))]


def source_max_scores(results,payload):
    expected={s['source_id'] for s in payload['source_windows']}
    ids=[r.get('id') for r in results]
    if len(ids)!=len(expected) or any(type(i)!=int for i in ids) or set(ids)!=expected:
        raise ValueError('Missing or duplicated source score')
    scores={r['id']:r['score'] for r in results}
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in scores.values()):
        raise ValueError('Invalid source score')
    return [dict(id=c['candidate_id'],score=max((scores[e['source_id']] for e in c['evidence']),default=0.))
            for c in payload['candidates']]


def validate_sources(payload,aliases,papers,reviewers):
    candidates=expand_candidates(payload)
    if len(set(aliases))!=len(aliases) or sorted(c['candidate_id'] for c in candidates)!=list(range(len(aliases))):
        raise ValueError('Invalid candidate aliases')
    for candidate in candidates:
        owner=aliases[candidate['candidate_id']]
        for evidence in candidate['evidence']:
            pid=evidence['paper_id']
            if pid not in reviewers[owner]['papers']:
                raise ValueError('Cross-expert source in packed input')
            text=(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
            if text[evidence['start']:evidence['end']]!=evidence['text']:
                raise ValueError('Changed source span')
    return candidates


def run(dataset='SIGIR', model_name='ms-marco-TinyBERT-L-2-v2',mode='concat'):
    from flashrank import Ranker,RerankRequest
    from flashrank.Config import model_file_map
    import onnxruntime as ort
    if mode not in ('concat','source-max'):raise ValueError('Unknown cross-encoder mode')
    root,papers,reviewers,labels=load(dataset)
    inputs=root/'packing-shared-v2'
    protocol=json.loads((inputs/'protocol.json').read_text())
    out=root/('crossencoder-v1' if mode=='concat' else 'crossencoder-source-max-v1')/model_name
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    ranker=Ranker(model_name=model_name,cache_dir='artifacts/models/flashrank',max_length=512,log_level='WARNING')
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    ranker.session=ort.InferenceSession(str(ranker.model_dir/model_file_map[model_name]),sess_options=options,providers=['CPUExecutionProvider'])
    initialization=time.perf_counter()-start
    hashes={str(p.relative_to(ranker.model_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ranker.model_dir.rglob('*')) if p.is_file()}
    spec=dict(dataset=dataset,model=model_name,flashrank=importlib.metadata.version('flashrank'),
        onnxruntime=ort.__version__,max_length=512,threads=1,weights=hashes,
        source_protocol_sha256=hashlib.sha256((inputs/'protocol.json').read_bytes()).hexdigest(),
        condition='similarity',input='Original target text plus exact similarity-packed windows concatenated in supplied source order; no names or labels',
        note='Frozen pretrained cross-encoder; no task training. Same pools and source windows as Luna similarity condition. Model tokenizer can truncate query and passage at joint 512 tokens; not identical effective token exposure to Luna.')
    if mode=='source-max':
        spec.update(mode=mode,input='Score each unique source window once, then max over candidate-owned references; no names or labels')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:
        raise ValueError('Changed cross-encoder protocol')
    marker.write_text(json.dumps(spec,indent=2))
    rows=[]
    for qid in protocol['all_query_ids']:
        target=out/f'{qid}.json'
        payload_path=inputs/f'{qid}-similarity-input.json'
        source_row=json.loads((inputs/f'{qid}-similarity.json').read_text())
        digest=hashlib.sha256(payload_path.read_bytes()).hexdigest()
        aliases=source_row['aliases']
        alias_digest=hashlib.sha256(json.dumps(aliases).encode()).hexdigest()
        payload=json.loads(payload_path.read_text())
        candidates=validate_sources(payload,aliases,papers,reviewers)
        if target.exists():
            row=json.loads(target.read_text())
            if row['input_sha256']!=digest:raise ValueError('Changed packed input')
            if row.get('aliases_sha256',alias_digest)!=alias_digest:raise ValueError('Changed alias mapping')
            if 'aliases_sha256' not in row:
                row['aliases_sha256']=alias_digest
                row['mapping_note']='Digest added after independent source ownership and offset validation'
                target.write_text(json.dumps(row,indent=2))
            rows.append(row)
            continue
        passages=([dict(id=c['candidate_id'],text='\n\n'.join(e['text'] for e in c['evidence'])) for c in candidates]
                  if mode=='concat' else [dict(id=s['source_id'],text=s['text']) for s in payload['source_windows']])
        start=time.perf_counter()
        result=ranker.rerank(RerankRequest(query=payload['target_paper'],passages=passages))
        numeric=[dict(id=r['id'],score=float(r['score'])) for r in result]
        if mode=='source-max':numeric=source_max_scores(numeric,payload)
        ranking=checked_ranking(numeric,aliases)
        elapsed=time.perf_counter()-start
        row=dict(query_id=qid,input_sha256=digest,aliases_sha256=alias_digest,ranking=ranking,scores=numeric,seconds=elapsed,
            candidates=len(aliases),metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()})
        target.write_text(json.dumps(row,indent=2))
        rows.append(row)
    report=dict(dataset=dataset,model=model_name,mode=mode,queries=len(rows),initialization_seconds=initialization,
        metrics={task:{key:statistics.mean(r['metrics'][task][key] for r in rows) for key in rows[0]['metrics'][task]} for task in labels},
        mean_seconds=statistics.mean(r['seconds'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),
        warning=spec['note']+' CPU inference includes tokenization and final sorting, but excludes coarse retrieval and evidence preparation. Model load/download separate. One pass, not a repeated isolated latency study. Other experts outside released judged pool remain unjudged.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['SIGIR','NIPS'],default='SIGIR')
    parser.add_argument('--model',choices=['ms-marco-TinyBERT-L-2-v2','ms-marco-MiniLM-L-12-v2'],default='ms-marco-TinyBERT-L-2-v2')
    parser.add_argument('--mode',choices=['concat','source-max'],default='concat')
    args=parser.parse_args()
    run(args.dataset,args.model,args.mode)
