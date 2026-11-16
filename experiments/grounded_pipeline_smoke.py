"""Real-source integration run, not an enterprise-quality benchmark."""
import hashlib
import json
import time
from pathlib import Path
import faiss
import numpy as np
from expert_search.expert_catalog import ExpertCatalog
from expert_search.grounded_pipeline import GroundedPipeline,DemandSpanPlanner
from expert_search.owned_vector_index import OwnedVectorIndex
from expert_search.qwen_embedding import QwenEmbedding
from expert_search.span_applicability import SpanApplicability

TASK='Find researchers whose documented research background could help address the technical requirement.'


class EnterpriseEncoder:
    def __init__(self,model):self.model=model
    def encode(self,texts,query=False):
        return self.model.encode(['Instruct: '+TASK+'\nQuery: '+t for t in texts] if query else texts,query=False)


def run():
    from laiwenai import LaiwenAI
    faiss.omp_set_num_threads(1);start=time.perf_counter();catalog=ExpertCatalog()
    eids=sorted(json.loads(p.read_text())['expert_id'] for p in Path('artifacts/expert-semantic-tree-v1').glob('CN_*.json'))
    sources=catalog.for_experts(eids);texts=[s['text'][:1200] for s in sources]
    jobs={r['id']:r for r in map(json.loads,Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines())}
    keys=json.loads(Path('artifacts/routing-requirements-v1/protocol.json').read_text())
    query_ids=sorted(p.stem for p in Path('artifacts/routing-requirements-v1').glob('demand-*.json'))
    out=Path('artifacts/grounded-pipeline-integration-v1');out.mkdir(exist_ok=True)
    code_paths=['src/expert_search/grounded_pipeline.py','src/expert_search/owned_vector_index.py','src/expert_search/span_applicability.py','src/expert_search/source_applicability.py','src/expert_search/source_routing.py','src/expert_search/qwen_embedding.py',__file__]
    protocol=dict(version=1,model='gpt-6-luna',task=TASK,experts=eids,queries=query_ids,
        inputs_sha256=hashlib.sha256(json.dumps([[(s['id'],s['expert_id'],t) for s,t in zip(sources,texts)],[jobs[k]['demand'] for k in query_ids]],sort_keys=True).encode()).hexdigest(),
        code_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_paths},
        source_max_chars=1200,candidates=3,k=3,max_length=4096,batch_size=8,
        scope='21-expert directory cohort, two reused development demands. All 296 owned sources eligible, no known-positive injection. Integration only; not full-corpus quality, independent labels or low-latency confirmation.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed integration inputs or code')
    marker.write_text(json.dumps(protocol,indent=2));model=QwenEmbedding(max_length=4096,batch_size=8);encoder=EnterpriseEncoder(model)
    initialization=time.perf_counter()-start;vp=out/'source-vectors.npy';vm=out/'source-vectors.json'
    if vp.exists():vectors=np.load(vp,allow_pickle=False);vector_metadata=json.loads(vm.read_text())
    else:
        begin=time.perf_counter();vectors=encoder.encode(texts);np.save(vp,vectors)
        vector_metadata=dict(offline_encoding_seconds=time.perf_counter()-begin,vectors_sha256=hashlib.sha256(vp.read_bytes()).hexdigest());vm.write_text(json.dumps(vector_metadata,indent=2))
    if vectors.shape!=(len(sources),1024) or hashlib.sha256(vp.read_bytes()).hexdigest()!=vector_metadata['vectors_sha256']:raise ValueError('Invalid integration vectors')
    begin=time.perf_counter();index=OwnedVectorIndex(vectors,[s['id'] for s in sources],[[s['expert_id']] for s in sources]);index_seconds=time.perf_counter()-begin
    client=LaiwenAI(model='gpt-6-luna',max_retries=0)
    pipeline=GroundedPipeline(encoder,index,catalog,DemandSpanPlanner(client),SpanApplicability(client));rows=[]
    for key in query_ids:
        target=out/(key+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            row=dict(id=key,status='unresolved');begin=time.perf_counter()
            try:
                events=pipeline.run(jobs[key]['demand'],k=3,candidates=3)
                row.update(status='executed' if events[-1]['stage']=='result' else 'unresolved',events=events,wall_seconds=time.perf_counter()-begin)
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None),wall_seconds=time.perf_counter()-begin)
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,ensure_ascii=False,indent=2));temporary.replace(target)
        rows.append(row);print(key,row['status'],row.get('wall_seconds'),flush=True)
    report=dict(experts=len(eids),sources=len(sources),queries=len(rows),executed=sum(r['status']=='executed' for r in rows),initialization_seconds=initialization,index_seconds=index_seconds,offline_vectors=vector_metadata,
        rows=[dict(id=r['id'],status=r['status'],wall_seconds=r['wall_seconds'],stages=[dict(stage=e['stage'],seconds=e.get('seconds'),total_seconds=e['total_seconds'],status=e.get('status')) for e in r.get('events',[])],final_status=r.get('events',[{}])[-1].get('status')) for r in rows],
        warning='Actual uncached query embedding and API planning/assessment in saved runs. Initialization and source encoding separated; saved reruns reuse prior results, not fresh timing. Model-valid outputs are not independently correct. Source view is prefix-limited; corpus is only 21 experts.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
