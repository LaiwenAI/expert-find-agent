"""Fixed top30 semantic shortlist then native Decisions evidence reranking."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.evidencebench_data import prepare,aspect_coverage
from expert_search.evidence_decisions import EvidenceDecisions,INSTRUCTION,CHOICES
from expert_search.qwen_embedding import QwenEmbedding
from expert_search.statistics import paired_interval
from evidencebench_pilot import TASK


def run():
    data=prepare();parent=Path('artifacts/evidencebench-pilot-v1');out=Path('artifacts/evidencebench-decisions-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,model='gpt-6-luna',instruction=INSTRUCTION,choices=CHOICES,shortlist=30,k=10,
        dataset_sha256=hashlib.sha256(Path('artifacts/public/evidencebench/dev.json').read_bytes()).hexdigest(),
        pilot_protocol_sha256=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path('src/expert_search/evidence_decisions.py').read_bytes()).hexdigest(),
        scope='All 37 development papers, full source context and up to 30 nonempty semantic candidate sentences; no gold aspect input. API errors fall back to the same query semantic top10 and are counted. Model initialization/offline indexing excluded; fresh query encoding included.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed Decisions evidence inputs')
    marker.write_text(json.dumps(protocol,indent=2));model=QwenEmbedding(max_length=512,batch_size=8);judge=EvidenceDecisions();rows=[]
    for key,row in data.items():
        target=out/(key+'.json')
        if target.exists():result=json.loads(target.read_text())
        else:
            vectors=np.load(parent/(key+'.npy'),allow_pickle=False);sentences=row['paper_as_candidate_pool']
            if vectors.shape!=(len(sentences),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid source vector inventory')
            start=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery: '+row['hypothesis']])[0]
            scores=vectors@q;ranking=sorted((i for i in range(len(sentences)) if sentences[i].strip()),key=lambda i:(-float(scores[i]),i));coarse_seconds=time.perf_counter()-start
            shortlist=ranking[:30];baseline=ranking[:10];result=dict(id=key,status='fallback',coarse_seconds=coarse_seconds)
            api_start=time.perf_counter()
            try:
                response=judge.score(row['hypothesis'],{'s'+str(i):sentences[i] for i in shortlist},{'sentences':sentences})
                position={i:r for r,i in enumerate(ranking)};selected=sorted(shortlist,key=lambda i:(-response['scores']['s'+str(i)],position[i],i))[:10]
                result.update(status='valid',decision=response)
            except Exception as error:
                selected=baseline;result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            result.update(api_wall_seconds=time.perf_counter()-api_start,total_online_seconds=time.perf_counter()-start,
                          selected=selected,metrics=aspect_coverage(row,selected),same_query_baseline=aspect_coverage(row,baseline),
                          shortlist_coverage=aspect_coverage(row,shortlist))
            temp=target.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2));temp.replace(target)
        rows.append(result);print(key,result['status'],round(result['total_online_seconds'],3),flush=True)
    report=dict(protocol=protocol,papers=len(rows),valid=sum(r['status']=='valid' for r in rows),fallbacks=sum(r['status']!='valid' for r in rows),
        aspect_recall=float(np.mean([r['metrics']['aspect_recall'] for r in rows])),same_query_baseline=float(np.mean([r['same_query_baseline']['aspect_recall'] for r in rows])),
        paired_difference=paired_interval([r['metrics']['aspect_recall']-r['same_query_baseline']['aspect_recall'] for r in rows]),
        mean_total_online_seconds=float(np.mean([r['total_online_seconds'] for r in rows])),
        warning='Auxiliary development evidence extraction only. All failures retained with semantic fallback. Native probabilities uncalibrated; no expert-fit accuracy, method originality, production SLA or held-out superiority claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
