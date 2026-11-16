"""Evidence-grounded silver-data pilot. Span validation is not semantic gold."""
import hashlib
import json
from pathlib import Path
from .agent import ExpertAgent
from .corpus import records


def validate(example, sources):
    errors=[]
    if not isinstance(example.get('query'),str) or not example['query'].strip(): errors.append('missing query')
    constraints=example.get('constraints',[])
    if not isinstance(constraints,list) or not constraints: return errors+['missing constraints']
    for c in constraints:
        if not isinstance(c,dict): errors.append('invalid constraint');continue
        if c.get('label') not in ['supported','contradicted','unknown']: errors.append('invalid label')
        if not isinstance(c.get('text'),str) or not c['text'].strip(): errors.append('missing constraint text')
        if c.get('label') in ['supported','contradicted']:
            eid=c.get('evidence_id'); quote=c.get('quote')
            if eid not in sources or not isinstance(quote,str) or not quote.strip() or quote not in sources[eid]['text']:
                errors.append('unverifiable evidence quote')
    return errors


def pilot(n=3, output='artifacts/synthesis-pilot', evidence_first=False):
    agent=ExpertAgent(); out=Path(output);out.mkdir(parents=True,exist_ok=True)
    demands=[r for _,r in records('artifacts/demands/queries.jsonl')][:n]
    for demand in demands:
        target=out/f"{demand['id']}.json"
        if target.exists(): continue
        candidates=agent.searcher.search(demand['query'],3,'bm25-vote')
        for row in candidates:
            for e in row['evidence']: e['text']=e['text'][:4000]
        sources={e['id']:e for row in candidates for e in row['evidence']}
        agent.trace=[]
        generation_rule=('Generate the synthetic query from the selected expert evidence FIRST, using the demand only for industry vocabulary. Every required capability in the synthetic query must have a direct original quote from this expert. Include at least two supported constraints. If this is impossible return an empty query so the sample is rejected. Do not copy unsupported details from the original demand. ' if evidence_first else '')
        generated=agent.ask('synthesize',
            generation_rule+
            'Construct one synthetic industrial expert-search query inspired by the real demand but limited to the capabilities in supplied evidence. '
            'Do not mention person names, institutions or paper titles in query. Give 2-4 atomic constraints, including one unknown constraint if evidence is incomplete. '
            'Schema: {"query":str,"constraints":[{"text":str,"label":"supported|contradicted|unknown","evidence_id":str|null,"quote":str|null}],'
            '"counterfactual":{"query":str,"changed_constraint":str,"expected_label":"unknown|contradicted","reason":str}}. '
            'A missing capability is unknown, never contradicted. Copy evidence quotes exactly. Do not infer collaborations or joint expertise across people.',
            {'real_demand':demand['query'],'candidate':candidates[0] if candidates else None})
        # Only quotes from the chosen expert are valid, regardless of retrieved distractors.
        chosen={e['id']:e for e in candidates[0]['evidence']} if candidates else {}
        errors=validate(generated,chosen)
        if evidence_first and sum(c.get('label')=='supported' for c in generated.get('constraints',[]) if isinstance(c,dict))<2:
            errors.append('fewer than two evidence-supported constraints')
        row=dict(parent_demand=demand['id'],demand_source=demand['source'],expert_id=candidates[0]['id'] if candidates else None,
            example=generated,sources=chosen,validation_errors=errors,status='span_valid_silver' if not errors else 'rejected',
            label_provenance='single generator; semantic verification pending; not gold',trace=agent.trace,
            family_id=hashlib.sha256(demand['id'].encode()).hexdigest()[:16])
        target.write_text(json.dumps(row,ensure_ascii=False,indent=2))
    return str(out)
