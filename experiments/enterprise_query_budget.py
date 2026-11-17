"""Whole-demand versus source-blind query decomposition, fixed evidence budget."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from expert_search.grounded_pipeline import DemandSpanPlanner
from expert_search.span_applicability import source_spans
from enterprise_role_budget import DATA,TASK,citation_retention


def run():
    root=Path('artifacts/enterprise-query-budget-v1');root.mkdir(exist_ok=True)
    data=[json.loads(line) for line in DATA.read_text().splitlines()]
    demands={r['pair_id']:r['original_demand'] for r in data}
    protocol=dict(dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/enterprise_query_budget.py','src/expert_search/grounded_pipeline.py']},
                  methods=['whole_demand','literal_chunks_max','planned_max'],k=3,width=400,task=TASK,max_length=4096,
                  planning_input='Original demand only; no expert, condition, source text, label or gold citation. Existing DemandSpanPlanner max6 anchored requirements; planning failure falls back to whole demand.',
                  selection='Same source fragment inventory; max similarity across nonempty fixed400char demand chunks, or at most6 model-derived requirements. No fitted weights; stable original-position ties.',
                  scope='Five reused synthetic pairs, not independent validation or expert ranking. Diagnostic follows observed whole-query citation loss. No tuning against output, no significance claim.')
    marker=root/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed query-budget protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    from laiwenai import LaiwenAI
    planner=DemandSpanPlanner(LaiwenAI(model='gpt-6-luna',max_retries=0));plans={}
    for key,demand in sorted(demands.items()):
        path=root/(hashlib.sha256(key.encode()).hexdigest()+'-plan.json')
        if path.exists():result=json.loads(path.read_text())
        else:
            start=time.perf_counter();result=dict(status='fallback')
            try:result.update(status='planned',plan=planner.plan(demand))
            except Exception as error:result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            result['online_planning_seconds']=time.perf_counter()-start
            path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
        plans[key]=result;print(key,result['status'],flush=True)
    from expert_search.qwen_embedding import QwenEmbedding
    model=QwenEmbedding(max_length=4096,batch_size=8)
    inventory={}
    for row in data:
        for source in row['sources'].values():
            for start in range(0,len(source['text']),400):
                text=source['text'][start:start+400]
                if text.strip():inventory[text]=None
    texts=list(inventory);start=time.perf_counter();encoded=model.encode(texts)
    offline_seconds=time.perf_counter()-start;lookup=dict(zip(texts,encoded));results=[]
    for row in data:
        spans=list(source_spans(row['expert_id'],row['sources']).values());vectors=np.asarray([lookup[s['quote']] for s in spans])
        demand=row['original_demand'];plan=plans[row['pair_id']]
        queries=dict(whole_demand=[demand],literal_chunks_max=[demand[i:i+400] for i in range(0,len(demand),400) if demand[i:i+400].strip()],
                     planned_max=[r['text'] for r in plan['plan']['requirements'].values()] if plan['status']=='planned' else [demand])
        timings={};selected={}
        for method,qtexts in queries.items():
            start=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery: '+t for t in qtexts]);scores=(vectors@q.T).max(axis=1)
            selected[method]=sorted(range(len(spans)),key=lambda i:(-float(scores[i]),i))[:3];timings[method]=time.perf_counter()-start
        results.append(dict(id=row['id'],pair_id=row['pair_id'],view=row['view'],selected={m:[dict(spans[i]) for i in ids] for m,ids in selected.items()},
                            retention={m:citation_retention(row,spans,ids) for m,ids in selected.items()},query_counts={m:len(q) for m,q in queries.items()},
                            retrieval_seconds=timings,online_planning_seconds=plan['online_planning_seconds']))
    summary=dict(views=len(data),pairs=len(demands),plans=sum(p['status']=='planned' for p in plans.values()),offline_source_encoding_seconds=offline_seconds,
                 methods={m:dict(complete=sum(a['complete'] for r in results for a in r['retention'][m].values()),reviewer_quotes=sum(len(r['retention'][m]) for r in results),
                                mean_fraction=float(np.mean([a['fraction'] for r in results for a in r['retention'][m].values()])),
                                mean_retrieval_seconds=float(np.mean([r['retrieval_seconds'][m] for r in results])),
                                mean_total_seconds=float(np.mean([r['retrieval_seconds'][m]+(r['online_planning_seconds'] if m=='planned_max' else 0) for r in results]))) for m in protocol['methods']},
                 warning='All five constructed experts are supplied; no expert retrieval accuracy. Correlated silver citations, reused data. Requirement texts are model hypotheses, not audited complete plans. Planning time charged per view although requests cached for identical pair demand. Online assessment not performed, so not end-to-end expert-agent latency.')
    (root/'evaluation.json').write_text(json.dumps(results,indent=2,ensure_ascii=False));(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
