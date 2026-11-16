"""Fixed semantic-transport pilot; never use query labels in kernel construction."""
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
import numpy as np
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.semantic_profiles import mutual_kernel,TransportedProfiles
from expert_search.evidence_packing import STOP


def run(dataset='NIPS'):
    from fastembed import TextEmbedding
    fixed=json.loads(Path('docs/exploration/semantic-transport-protocol.json').read_text())
    root,papers,reviewers,labels=load(dataset)
    out=root/'semantic-transport-v1'
    out.mkdir(parents=True,exist_ok=True)
    pids=sorted({pid for r in reviewers.values() for pid in r['papers']})
    documents={pid:(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip() for pid in pids}
    reference=ExpertLanguageModels(documents,{eid:r['papers'] for eid,r in reviewers.items()})
    words=sorted(t for t in reference.background if t.isascii() and t.isalpha() and 3<=len(t)<=40 and t not in STOP)
    spec=dict(protocol=fixed,fastembed=importlib.metadata.version('fastembed'),words_sha256=hashlib.sha256(json.dumps(words).encode()).hexdigest(),
        source_sha256=hashlib.sha256(json.dumps([documents,reviewers],sort_keys=True).encode()).hexdigest())
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:raise ValueError('Changed semantic-transport input')
    marker.write_text(json.dumps(spec,indent=2))
    vectors_path=out/'word-vectors.npy'
    if not vectors_path.exists():
        model=TextEmbedding(model_name=fixed['models'][0],cache_dir='artifacts/models',threads=1)
        start=time.perf_counter()
        vectors=np.asarray(list(model.passage_embed(words,batch_size=64)),dtype=np.float32)
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-30)
        partial=vectors_path.with_suffix('.partial.npy');np.save(partial,vectors);partial.replace(vectors_path)
        (out/'word-encoding.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,words=len(words))))
    print(dataset,'word vectors ready',len(words),flush=True)
    kernel_path=out/'kernel.json'
    if not kernel_path.exists():
        start=time.perf_counter()
        kernel=mutual_kernel(words,np.load(vectors_path),fixed['neighbors'],fixed['self_mass'])
        kernel_path.write_text(json.dumps(kernel))
        (out/'kernel-cost.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start)))
    else:kernel=json.loads(kernel_path.read_text())
    start=time.perf_counter()
    semantic=TransportedProfiles(reference,kernel)
    compiled=CompiledCandidateLM(semantic)
    construction=time.perf_counter()-start
    print(dataset,'transport compiled',round(construction,3),flush=True)
    rows=[]
    for qid in sorted(labels['soft']):
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        available=sorted(compiled.available)
        start=time.perf_counter()
        scores=compiled.score(query,available)
        full=sorted(scores,key=lambda eid:(-scores[eid],eid))
        elapsed=time.perf_counter()-start
        expected=semantic.score(query,available)
        if scores!=expected:raise ValueError('Transport compilation changed scores')
        ranking=[eid for eid in full if eid in labels['soft'][qid]]
        row=dict(query_id=qid,ranking=ranking,seconds=elapsed,
                 metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()})
        (out/f'query-{qid}.json').write_text(json.dumps(row,indent=2));rows.append(row)
    report=dict(dataset=dataset,queries=len(rows),graph_words=len(words),compiled_array_bytes=compiled.array_bytes,
        construction_seconds=construction,word_vector_sha256=hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        metrics={task:{key:statistics.mean(r['metrics'][task][key] for r in rows) for key in rows[0]['metrics'][task]} for task in labels},
        mean_seconds=statistics.mean(r['seconds'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows),
        exact_compilation=True,warning='Exploratory fixed k8/self-mass .5/JM .5; no label tuning or neural training. Word-neighbor relatedness is not synonymy or evidence entailment. Both profile and background transported by one stochastic kernel. English-token prototype; Chinese tokens and OOV not semantically expanded. Warm scoring ALL profiles plus sorting timed; kernel construction and encoding separate. One pass, not an end-to-end speed claim. Translation models are prior work, not a new-theory claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    Path(f'docs/exploration/semantic-transport-{dataset.lower()}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=['NIPS','SIGIR'],default='NIPS')
    args=parser.parse_args();run(args.dataset)
