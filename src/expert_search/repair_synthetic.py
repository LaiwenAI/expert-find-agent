"""Feedback repair preserves failed originals; revised examples remain silver."""
import json
from pathlib import Path
from .agent import ExpertAgent
from .corpus import records
from .synthesis import validate


def run(source='artifacts/synthetic-evidence-first-selected/examples.jsonl',audits='artifacts/synthetic-query-audit',output='artifacts/synthetic-repaired'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);agent=ExpertAgent()
    for _,r in records(source):
        p=out/f"{r['parent_demand']}.json"
        if p.exists():continue
        audit=json.loads((Path(audits)/p.name).read_text())
        agent.trace=[]
        new=agent.ask('repair',
            'Repair this synthetic expert-search task using the audit. Remove every unsupported modifier, domain, combination, and application. '
            'Prefer a narrower but fully evidenced query; do NOT preserve industrial terminology at the cost of truth. '
            'Every requirement in query must be represented as one of 2-4 supported atomic constraints with exact quotes. '
            'Schema: {"query":str,"constraints":[{"text":str,"label":"supported","evidence_id":str,"quote":str}],'
            '"counterfactual":{"query":str,"changed_constraint":str,"expected_label":"unknown|contradicted","reason":str}}. '
            'The counterfactual must change just one requirement. If fewer than two supported constraints are possible, return empty query and constraints.',
            {'original_query':r['example']['query'],'audit':audit.get('audit'), 'sources':list(r['sources'].values())})
        errors=validate(new,r['sources'])
        if sum(c.get('label')=='supported' for c in new.get('constraints',[]))<2:errors.append('insufficient supported constraints')
        revised=dict(r,example=new,validation_errors=errors,status='rejected' if errors else 'span_valid_silver',
                     repair_trace=agent.trace,repair_provenance='Feedback from full-query audit; original artifact retained')
        p.write_text(json.dumps(revised,ensure_ascii=False,indent=2))
    return str(out)
