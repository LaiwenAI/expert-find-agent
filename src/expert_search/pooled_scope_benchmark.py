"""Pooled source-support evaluation; unknown is not actual expert irrelevance."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent
from .area_retrieval import AreaSearch
from .corpus import records
from .decision_cache import DecisionCache
from .fast_retrieval import FastSearch


POLICY='pool-v1:bm25-max5,bm25-vote5,area5;common-top3-plus-area;4000chars/source'


def prepare(source='artifacts/dataset-release/scoped-capabilities/pairs.jsonl',output='artifacts/pooled-scope'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    engine=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite');area=AreaSearch()
    profiles={r['expert_id']:r for _,r in records('artifacts/corpus/evidence.jsonl') if r.get('field')=='\u7814\u7a76\u9886\u57df_CN'}
    queries=[r for _,r in records(source)];contexts=[];runs=[]
    for query in queries:
        rankings={method:[r['id'] for r in engine.search(query['query'],5,method)]
                  for method in ('bm25-max','bm25-vote')}
        rankings['area-bm25']=[r['id'] for r in area.search(query['query'],5)['results']]
        pool=sorted({eid for ranking in rankings.values() for eid in ranking})
        selected=engine.evidence_for(query['query'],pool,3)
        for eid in pool:
            evidence=selected[eid]
            if eid in profiles and profiles[eid]['id'] not in {e['id'] for e in evidence}:
                evidence=evidence+[profiles[eid]]
            sources=[dict(id=e['id'],expert_id=e['expert_id'],source=e['source'],text=e['text'][:4000]) for e in evidence]
            if any(e['expert_id']!=eid for e in sources):raise ValueError('Mixed source ownership')
            contexts.append(dict(id=query['id']+':'+eid,query_id=query['id'],expert_id=eid,
                                 constraint=query['query'],sources=sources))
        runs.append(dict(query_id=query['id'],rankings=rankings,pool=pool))
    manifest=dict(policy=POLICY,source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(Path('artifacts/corpus/evidence.jsonl').read_bytes()).hexdigest(),
        contexts_sha256=hashlib.sha256(json.dumps(contexts,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        queries=len(queries),candidate_pairs=len(contexts))
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Pooled inputs or policy changed')
    path.write_text(json.dumps(manifest,indent=2))
    (out/'contexts.json').write_text(json.dumps(contexts,ensure_ascii=False))
    (out/'runs.json').write_text(json.dumps(runs,indent=2))
    return contexts,runs


def run(output='artifacts/pooled-scope'):
    contexts,runs=prepare(output=output);out=Path(output)
    pairs=[dict(constraint=c['constraint'],text='\n\n'.join(e['text'] for e in c['sources'])) for c in contexts]
    decisions={};stats={}
    for model in ('gpt-4o','gpt-6-luna'):
        agent=ExpertAgent(client=LaiwenAI(model=model))
        if model=='gpt-6-luna':agent.request_options={'reasoning_effort':'none'}
        cache=DecisionCache(str(out/(model+'.sqlite')),model+':pooled-topic-v1:'+str(getattr(agent,'request_options',{})),
            lambda instruction,payload:agent.ask('pooled_source_support',instruction+
                ' The query asks for research topics, considered separately, not sole authorship or a joint application. '
                'Profile fields and attributed projects/publications can support their stated topics. '
                'Copy one short contiguous exact quote without ellipsis for a supported/contradicted label. '
                'Judge the complete query, not merely one of its topics.',payload),batch_size=6)
        try:
            answers=[];failures=[];calls=0
            for start in range(0,len(pairs),6):
                batch=pairs[start:start+6]
                try:
                    result=cache.judge(batch);answers.extend(result['judgments']);calls+=result['network_calls']
                except ValueError as error:
                    # Invalid labels/quotes never become unknown or negative qrels.
                    failures.append(dict(start=start,count=len(batch),error=str(error)))
                    answers.extend(dict(label=None,quote=None,validation_error=str(error)) for _ in batch)
            decisions[model]=dict(judgments=answers)
            stats[model]=dict(successful_network_calls=calls,invalid_batches=failures)
        finally:
            if agent.trace:(out/(model+'-trace-'+str(time.time_ns())+'.json')).write_text(json.dumps(agent.trace,ensure_ascii=False))
    rows=[]
    for i,context in enumerate(contexts):
        judged={m:decisions[m]['judgments'][i] for m in decisions}
        for answer in judged.values():
            if answer['label'] in ('supported','contradicted') and not any(answer['quote'] in e['text'] for e in context['sources']):
                raise ValueError('Quote crosses source boundaries')
        labels={v['label'] for v in judged.values()}
        rows.append(dict(id=context['id'],query_id=context['query_id'],expert_id=context['expert_id'],
                         judgments=judged,agreed_label=next(iter(labels)) if len(labels)==1 else None))
    lookup={(r['query_id'],r['expert_id']):r['agreed_label'] for r in rows}
    methods={}
    for method in runs[0]['rankings']:
        per_query=[]
        for run in runs:
            ranking=run['rankings'][method];labels=[lookup[(run['query_id'],eid)] for eid in ranking]
            per_query.append(dict(query_id=run['query_id'],returned=len(ranking),
                supported=sum(v=='supported' for v in labels),unknown=sum(v=='unknown' for v in labels),
                contradicted=sum(v=='contradicted' for v in labels),unresolved=sum(v is None for v in labels)))
        methods[method]=dict(per_query=per_query,
            confirmed_supported_fraction_at5=sum(r['supported']/5 for r in per_query)/len(per_query),
            unresolved_fraction_at5=sum(r['unresolved']/5 for r in per_query)/len(per_query))
    summary=dict(queries=len(runs),pairs=len(rows),labels=dict(Counter(r['agreed_label'] or 'disagreement' for r in rows)),
        methods=methods,cache_stats=stats,
        warning='Pooled model-audited source support only. Unknown is not actual nonrelevance or inability; unpooled experts unjudged. All methods use the same lexical top3-plus-area source policy, which may favor lexical retrieval and omit support. Not human gold, an exhaustive corpus benchmark, or an independent test of generator-model bias.')
    (out/'judgments.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':print(json.dumps(run(),indent=2))
