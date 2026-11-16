"""Same-cohort source-selection pilot; embedding similarity is never a label."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from expert_search.corpus import records
from expert_search.dense_retrieval import encode
from expert_search.source_counterfactual import GENERATOR,AUDITOR,build_views,execute


def prepare(output='artifacts/source-counterfactual-demand-ranked-v3',local=False):
    parent=Path('artifacts/source-counterfactual-v1/jobs.jsonl')
    jobs=[j for _,j in records(parent)]
    demands={d['id']:d for _,d in records('artifacts/demands/queries.jsonl')}
    wanted={j['expert_id'] for j in jobs}
    papers=defaultdict(list)
    for _,source in records('artifacts/corpus/evidence.jsonl'):
        if source['expert_id'] in wanted and source['kind']=='paper' and source['text'].strip():
            papers[source['expert_id']].append(dict(source,text=source['text'][:1200]))
    documents={s['id']:s for group in papers.values() for s in group}
    inputs=[dict(id=pid,text=s['text']) for pid,s in sorted(documents.items())]
    inputs += [dict(id=j['id'],text=demands[j['parent_demand']]['title']) for j in jobs]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Selection protocol is committed before embedding or generation outcomes.
    embedding_model=('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' if local else 'text-embedding-3-small')
    selection=dict(version=4 if local else 3,parent_jobs_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(json.dumps(inputs,ensure_ascii=False).encode()).hexdigest(),
        embedding_model=embedding_model,policy='Same six experts and real demands as v1. Rank all same-owner paper prefixes by cosine to the original demand title. Include the highest-ranked relational-title paper, then the three highest-ranked remaining papers. Retain original research-area fields. Remaining real papers supply matched replacements. No generated labels or audit results in source selection.',
        role='Exploratory source-policy change on a reused cohort, not held-out evaluation or an industrial-fit label.')
    marker=out/'selection-protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=selection:raise ValueError('Changed source-selection inputs')
    marker.write_text(json.dumps(selection,indent=2))
    if local:
        import importlib.metadata
        from fastembed import TextEmbedding
        model=TextEmbedding(model_name=embedding_model,cache_dir='artifacts/models',threads=1,max_length=128)
        local_spec=dict(model=embedding_model,fastembed=importlib.metadata.version('fastembed'),
            input_sha256=selection['input_sha256'],max_length=128,
            implementation='FastEmbed quantized ONNX export; embed without query/document prefixes.')
        embedding=out/'embedding';embedding.mkdir(exist_ok=True)
        marker_local=embedding/'local-protocol.json'
        if marker_local.exists() and json.loads(marker_local.read_text())!=local_spec:raise ValueError('Changed local encoder protocol')
        marker_local.write_text(json.dumps(local_spec,indent=2))
        if not (embedding/'vectors.npy').exists():
            matrix=np.asarray(list(model.embed([s['text'] for s in inputs],batch_size=32)),dtype=np.float32)
            if not np.isfinite(matrix).all() or (np.linalg.norm(matrix,axis=1)==0).any():raise ValueError('Invalid local embeddings')
            matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
            np.save(embedding/'vectors.npy',matrix)
            (embedding/'manifest.json').write_text(json.dumps(dict(ids=[s['id'] for s in inputs],model=embedding_model,
                input_sha256=selection['input_sha256'],dimensions=matrix.shape[1]),indent=2))
        model_directory=model.model._model_dir
        materialization=dict(snapshot=model_directory.name,
            files={str(p.relative_to(model_directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(model_directory.rglob('*')) if p.is_file()},
            effective_truncation=model.model.tokenizer.truncation,
            pooling_implementation=type(model.model).__name__,
            vectors_sha256=hashlib.sha256((embedding/'vectors.npy').read_bytes()).hexdigest())
        provenance=embedding/'encoder-materialization.json'
        if provenance.exists() and json.loads(provenance.read_text())!=materialization:
            raise ValueError('Changed local weights, tokenizer or cached vectors')
        provenance.write_text(json.dumps(materialization,indent=2))
    else:
        from laiwenai import LaiwenAI
        encode(inputs,out/'embedding',LaiwenAI(embedding_model=embedding_model),batch_size=32)
    meta=json.loads((out/'embedding/manifest.json').read_text())
    matrix=np.load(out/'embedding/vectors.npy')
    if (meta['ids']!=[s['id'] for s in inputs] or meta['input_sha256']!=selection['input_sha256'] or
            matrix.shape!=(len(inputs),meta['dimensions']) or not np.isfinite(matrix).all() or
            not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-5)):
        raise ValueError('Changed embedding mapping, shape or normalization')
    vector={eid:v for eid,v in zip(meta['ids'],matrix)}
    selected=[]
    for old in jobs:
        options=sorted(papers[old['expert_id']],key=lambda s:(-float(vector[s['id']]@vector[old['id']]),s['id']))
        relational=[s for s in options if re.search(r'\b(for|using|via|based)\b|基于|应用|用于|结合',s.get('title') or '',re.I)]
        if len(options)<5 or not relational:raise ValueError('Original cohort no longer eligible')
        first=relational[0]
        picked=[first]+[s for s in options if s['id']!=first['id']][:3]
        fields=[s for s in old['sources'].values() if s['kind']!='paper']
        sources={s['id']:s for s in fields+picked}
        if any(s['expert_id']!=old['expert_id'] for s in sources.values()):raise ValueError('Cross-owner source')
        job=dict(old,sources=sources,reserve=[s for s in options if s['id'] not in sources],
            selection=dict(protocol_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
                paper_cosines={s['id']:float(vector[s['id']]@vector[old['id']]) for s in options}))
        selected.append(job)
    serialized=''.join(json.dumps(j,ensure_ascii=False,sort_keys=True)+'\n' for j in selected)
    spec=dict(version=selection['version'],jobs=len(selected),jobs_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        generator=GENERATOR,auditor=AUDITOR,models=['gpt-4o','gpt-6-luna'],
        selection=selection,embedding_manifest=meta,
        warning='Same six-expert cohort with revised demand-title source ranking; model-audited source claims only. No new human labels, no industrial-fit negatives, no causal held-out improvement claim. Same-owner real replacements and all original structural/audit gates retained.')
    protocol=out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=spec:raise ValueError('Changed prepared controls')
    protocol.write_text(json.dumps(spec,indent=2));(out/'jobs.jsonl').write_text(serialized)
    return spec


def generate_only(output):
    """Retain generator feasibility evidence while an auditor is unavailable."""
    from laiwenai import LaiwenAI
    from expert_search.agent import ExpertAgent
    root=Path(output);protocol=json.loads((root/'protocol.json').read_text())
    if protocol['generator']!=GENERATOR or hashlib.sha256((root/'jobs.jsonl').read_bytes()).hexdigest()!=protocol['jobs_sha256']:
        raise ValueError('Changed prepared inputs or generator')
    helper=ExpertAgent.__new__(ExpertAgent);helper.client=LaiwenAI(model='gpt-4o')
    helper.request_options={}
    out=root/'generation-only';out.mkdir(exist_ok=True);statuses=[]
    for _,job in records(root/'jobs.jsonl'):
        fingerprint=hashlib.sha256(json.dumps([job,protocol],sort_keys=True).encode()).hexdigest()
        target=out/(job['id']+'.json')
        if target.exists():
            result=json.loads(target.read_text())
            if result['fingerprint']!=fingerprint:raise ValueError('Changed generator input')
        else:
            helper.trace=[];helper.usage={'prompt_tokens':0,'completion_tokens':0}
            result=dict(id=job['id'],fingerprint=fingerprint,status='unresolved',audited=False)
            try:
                payload=dict(original_demand=job['demand'],sources=[{k:s.get(k) for k in ('id','kind','field','title','text')} for s in job['sources'].values()])
                example=helper.ask('source_control_generation',GENERATOR,payload)
                result['example']=example
                if example.get('feasible') is False:result['status']='not_feasible'
                else:
                    result['views']=build_views(job,example)
                    result['status']='constructed_pending_independent_audit'
            except Exception as error:
                result.update(error_type=type(error).__name__,
                    status='rejected_structure' if 'example' in result else 'unresolved')
            result.update(trace=helper.trace,usage=helper.usage)
            partial=target.with_suffix('.partial');partial.write_text(json.dumps(result,ensure_ascii=False,indent=2));partial.replace(target)
        statuses.append(result['status']);print(job['id'],result['status'],flush=True)
    summary=dict(jobs=len(statuses),statuses={s:statuses.count(s) for s in sorted(set(statuses))},
        audited=False,warning='Generator feasibility only, no accepted silver labels or quality claim. Independent auditor currently unavailable; do not export these as labeled data.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--generate-only',action='store_true')
    parser.add_argument('--local',action='store_true',help='Explicit local bilingual encoder in a separate v4 cohort directory')
    args=parser.parse_args()
    output='artifacts/source-counterfactual-demand-ranked-local-v4' if args.local else 'artifacts/source-counterfactual-demand-ranked-v3'
    print(json.dumps(execute(output) if args.execute else generate_only(output) if args.generate_only else prepare(output,args.local),indent=2))
