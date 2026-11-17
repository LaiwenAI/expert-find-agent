"""Real industrial clause/evidence support pilot. No synthetic ability claims."""
import json
import re
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent
from .corpus import records
from .fast_retrieval import FastSearch
from .decision_cache import DecisionCache


def validate_clauses(demand, quotes):
    """Map exact, unique source spans; reject titles and marked background."""
    text=demand['query'];body_start=text.find('\n')+1
    if not isinstance(quotes,list) or len(quotes)>2:raise ValueError('Expected at most two clauses')
    headings=list(re.finditer('\uff08[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\uff09([^\\s\u3002]{1,20})',text))
    clauses=[]
    for quote in quotes:
        if not isinstance(quote,str) or not 25<=len(quote)<=400 or text.count(quote)!=1:
            raise ValueError('Clause must be a unique exact source quote of 25–400 characters')
        start=text.index(quote);end=start+len(quote)
        if start<body_start:raise ValueError('Title is not a technical requirement')
        prior=[h for h in headings if h.start()<=start]
        if prior and '\u80cc\u666f' in prior[-1].group(1):raise ValueError('Background is not a requirement')
        if any(start<c[1] and end>c[0] for c in clauses):raise ValueError('Overlapping clauses')
        clauses.append((start,end,quote))
    return clauses


def build(n_demands=5,output='artifacts/demand-evidence-pilot-v2'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    retrieval=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite');pairs=[];metadata=[];rejected=[]
    for _,demand in list(records('artifacts/demands/queries.jsonl'))[:n_demands]:
        extraction=out/(demand['id']+'-extraction.json')
        if extraction.exists():
            saved=json.loads(extraction.read_text())
            if saved['source_query']!=demand['query']:raise ValueError('Extraction source changed')
        else:
            extractor=ExpertAgent(client=LaiwenAI(model='gpt-4o'))
            result=extractor.ask('extract_requirements',
                'Select at most two concrete technical requirements from the ORIGINAL demand body. '
                'Copy each as one continuous exact quote of 25–400 characters; do not rewrite or join spans. '
                'Exclude the title, industry/business background, submission formats and generic aspirations. '
                'Prefer specific technical tasks or measurable constraints. Preserve every modifier in the selected clause. '
                'Return {"clauses":[str]}; return an empty list if no suitable requirement exists.',demand)
            saved=dict(source_query=demand['query'],clauses=result.get('clauses'),trace=extractor.trace)
            extraction.write_text(json.dumps(saved,ensure_ascii=False,indent=2))
        try:clauses=validate_clauses(demand,saved['clauses'])
        except ValueError as error:
            rejected.append(dict(demand_id=demand['id'],reason=str(error)));continue
        for start,end,clause in clauses:
            for expert in retrieval.search(clause,2,'bm25-max'):
                evidence=expert['evidence'][0];pairs.append({'constraint':clause,'text':evidence['text'][:4000]})
                metadata.append(dict(demand_id=demand['id'],demand_source=demand['source'],clause_start=start,clause_end=end,
                    expert_id=expert['id'],evidence_id=evidence['id'],evidence_source=evidence['source']))
    decisions={}
    for model in ['gpt-4o','gpt-6-luna']:
        agent=ExpertAgent(client=LaiwenAI(model=model))
        if model=='gpt-6-luna':agent.request_options={'reasoning_effort':'none'}
        cache=DecisionCache(str(out/(model+'.sqlite')),model+':none' if model=='gpt-6-luna' else model,
            lambda instruction,payload:agent.ask('support_audit',instruction,payload),batch_size=16)
        decisions[model]=cache.judge(pairs)
    rows=[]
    for i,(pair,meta) in enumerate(zip(pairs,metadata)):
        a=decisions['gpt-4o']['judgments'][i];b=decisions['gpt-6-luna']['judgments'][i]
        rows.append(dict(id=f'demand-evidence-{i}',**meta,clause=pair['constraint'],evidence_text=pair['text'],
            generator_judgment=a,verifier_judgment=b,status='model_agreement_silver' if a['label']==b['label'] else 'needs_review',
            label_semantics='Support of complete original clause by this evidence, not actual expert ability or overall fit'))
    (out/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary=dict(pairs=len(rows),rejected_extractions=rejected,agreed=sum(r['status']=='model_agreement_silver' for r in rows),
        agreed_labels={label:sum(r['status']=='model_agreement_silver' and r['verifier_judgment']['label']==label for r in rows) for label in ['supported','contradicted','unknown']},
        warning='Retrieved-evidence pilot; not human gold, complete demand relevance or all-expert annotation. Full source text kept locally only.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':print(build())
