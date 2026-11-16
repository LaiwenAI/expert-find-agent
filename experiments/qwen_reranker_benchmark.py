"""Frozen Qwen3 control on the same CoF packed candidate/source inputs."""
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.cof_crossencoder import validate_sources,checked_ranking
from expert_search.qwen_reranker import QwenReranker,SYSTEM,TASK


def run(dataset='NIPS'):
    import mlx.core as mx
    memory_limit = 4 * 1024**3
    cache_limit = 128 * 1024**2
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(cache_limit)
    root,papers,reviewers,labels=load(dataset)
    inputs=root/'packing-shared-v2'
    parent=json.loads((inputs/'protocol.json').read_text())
    checkpoint=Path('artifacts/models/qwen3-reranker-0.6b')
    provenance=json.loads((checkpoint/'checkpoint-provenance.json').read_text())
    if 'files' not in provenance:raise ValueError('Checkpoint download and hash verification not completed')
    if any(hashlib.sha256((checkpoint/p).read_bytes()).hexdigest()!=h for p,h in provenance['files'].items()):
        raise ValueError('Changed checkpoint files')
    out=root/'qwen-reranker-mlx-v2';out.mkdir(exist_ok=True)
    spec=dict(dataset=dataset,checkpoint=provenance,
        memory_limit_bytes=memory_limit,cache_limit_bytes=cache_limit,
        versions={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers','numpy')},
        code_sha256=hashlib.sha256(Path('src/expert_search/qwen_reranker.py').read_bytes()).hexdigest(),
        source_protocol_sha256=hashlib.sha256((inputs/'protocol.json').read_bytes()).hexdigest(),
        system=SYSTEM,task=TASK,max_length=4096,batch_size=4,padding='right',
        mode='Concatenate the original similarity-packed candidate-owned windows in supplied order. No names or qrels in model input.',
        arithmetic='Official BF16 transformer, FP32 yes/no weight rows and last-real-position hidden states; sigmoid yes-minus-no. No full vocabulary or generated rationale needed.',
        warning='Apple GPU baseline, not directly hardware-comparable to CPU or hosted Luna. Same source windows do not imply identical effective token exposure. Timing includes tokenization, inference and sorting, excludes coarse retrieval, source packing, initialization and downloads. Frozen model; no own training, calibration or task-label tuning.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:raise ValueError('Changed Qwen protocol')
    marker.write_text(json.dumps(spec,indent=2))
    start=time.perf_counter();ranker=QwenReranker();initialization=time.perf_counter()-start
    validation=ranker.validate_projection()
    # Record batch/right-padding agreement on unequal length, real-position inputs.
    texts=['Robot motion control and path planning.','Marine biology studies the ecology and habitats of ocean organisms.']
    sequences=[ranker.encode('robot motion control',t)[0] for t in texts]
    serial=[ranker._batch([s])[0] for s in sequences]
    batched=ranker._batch(sequences)
    padding_error=max(abs(a-b) for a,b in zip(serial,batched))
    if padding_error>.01:raise ValueError('Batch padding probability discrepancy exceeds fixed .01 tolerance')
    validation.update(batch_serial_max_probability_error=padding_error,batch_serial_tolerance=.01,
        relevant_above_irrelevant=batched[0]>batched[1],weight_dtype=ranker.dtype,device=str(ranker.mx.default_device()))
    (out/'implementation-validation.json').write_text(json.dumps(validation,indent=2))
    rows=[]
    for qid in parent['all_query_ids']:
        source=json.loads((inputs/f'{qid}-similarity.json').read_text())
        payload_path=inputs/f'{qid}-similarity-input.json'
        payload=json.loads(payload_path.read_text());aliases=source['aliases']
        candidates=validate_sources(payload,aliases,papers,reviewers)
        digest=hashlib.sha256(payload_path.read_bytes()).hexdigest()
        alias_digest=hashlib.sha256(json.dumps(aliases).encode()).hexdigest()
        target=out/f'{qid}.json'
        if target.exists():
            row=json.loads(target.read_text())
            if row['input_sha256']!=digest or row['aliases_sha256']!=alias_digest:raise ValueError('Changed cached Qwen input')
            if checked_ranking(row['scores'],aliases)!=row['ranking']:raise ValueError('Changed cached Qwen rank')
        else:
            documents=['\n\n'.join(e['text'] for e in c['evidence']) for c in candidates]
            start=time.perf_counter();result=ranker.score(payload['target_paper'],documents)
            scores=[dict(id=c['candidate_id'],score=v) for c,v in zip(candidates,result['scores'])]
            ranking=checked_ranking(scores,aliases);elapsed=time.perf_counter()-start
            row=dict(query_id=qid,input_sha256=digest,aliases_sha256=alias_digest,scores=scores,ranking=ranking,
                seconds=elapsed,candidates=len(aliases),input_tokens=result['input_tokens'],
                untruncated_tokens=result['untruncated_tokens'],truncated_documents=result['truncated'],
                metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()})
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        if any(row['metrics'][task]!=precision_metrics(row['ranking'],truth[qid]) for task,truth in labels.items()):
            raise ValueError('Cached metrics differ from source labels')
        rows.append(row);print(dataset,qid,'complete',round(row['seconds'],3),flush=True)
    report=dict(dataset=dataset,queries=len(rows),initialization_seconds=initialization,validation=validation,
        metrics={task:{key:statistics.mean(r['metrics'][task][key] for r in rows) for key in rows[0]['metrics'][task]} for task in labels},
        mean_seconds=statistics.mean(r['seconds'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),
        truncated_documents=sum(r['truncated_documents'] for r in rows),total_documents=sum(r['candidates'] for r in rows),
        warning=spec['warning']+' One pass; candidate pools constrained by the released benchmark protocol. No full-catalog expert-quality claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    Path(f'docs/exploration/qwen-reranker-{dataset.lower()}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['NIPS','SIGIR'],default='NIPS')
    run(parser.parse_args().dataset)
