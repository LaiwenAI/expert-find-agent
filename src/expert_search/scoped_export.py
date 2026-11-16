"""Export scope-controlled silver pairs without source text or model traces."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from .synthesis_v2 import render_scoped_queries


def export(source='artifacts/synthesis-scope-v5',output='artifacts/dataset-release/scoped-capabilities'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[];statuses=Counter()
    for path in sorted(Path(source).glob('*candidate*.json')):
        record=json.loads(path.read_text());statuses[record['status']]+=1
        if record['status']!='silver_candidate':continue
        example=record['example'];audit=record['audit']
        query,counterfactual=render_scoped_queries(example['constraints'],example['counterfactual'])
        if query!=example['query'] or counterfactual!=example['counterfactual']['query']:
            raise ValueError('Stored query differs from canonical one-requirement rendering')
        if not(audit['a_supported'] is True and audit['unsupported_a']==[]
               and audit['b_label']=='unknown' and audit['one_requirement_changed'] is True
               and audit['demand_related'] is True):raise ValueError('Accepted record fails audit conditions')
        references=[]
        for requirement in audit['requirements_a']:
            evidence=record['sources'][requirement['evidence_id']];quote=requirement['quote']
            if not isinstance(quote,str) or not quote.strip() or quote not in evidence['text']:
                raise ValueError('Invalid audit source span')
            start=evidence['text'].index(quote)
            references.append(dict(requirement=requirement['text'],evidence_id=evidence['id'],
                source=evidence['source'],start=start,end=start+len(quote),
                source_snapshot_sha256=hashlib.sha256(evidence['text'].encode()).hexdigest()))
        rows.append(dict(id=record['id'],query=query,counterfactual_query=counterfactual,
            constructing_expert_id=record['expert_id'],parent_demand_id=record['parent_demand'],
            base_source_support='supported',counterfactual_source_support='unknown',
            changed_index=example['counterfactual']['changed_index'],
            replacement_text=example['counterfactual']['replacement_text'],
            constraints=[c['text'] for c in example['constraints']],references=references,
            original_demand_gaps=example['original_demand_gaps'],
            shared_technical_topic=audit['shared_technical_topic'],
            artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            label_quality='cross-model-audited silver, not human gold',
            label_scope='Only this expert and supplied source snapshots. Unknown is not inability; other experts are unjudged. Narrowed subtask does not establish full industrial fit.'))
    (out/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest=dict(pairs=len(rows),query_variants=2*len(rows),construction_statuses=dict(statuses),
        distinct_demands=len({r['parent_demand_id'] for r in rows}),
        distinct_experts=len({r['constructing_expert_id'] for r in rows}),
        source_text_included=False,model_traces_included=False,split='unsplit pilot',
        warning='Small silver capability diagnostic, not a complete industrial-relevance benchmark. No confidence intervals treating paired variants as independent. Future splits must connect shared demands, experts and source snapshots.',
        redistribution='Local review candidate; derived queries and provenance references require dataset-specific rights review before public redistribution.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':print(json.dumps(export(),indent=2))
