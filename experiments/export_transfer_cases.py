"""Export all outcomes, capability-only source pairs, and blind review packets."""
import hashlib
import json
from pathlib import Path
import random


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    source=Path('artifacts/demand-first-paper-jobs-v1/jobs.jsonl')
    jobs=[json.loads(s) for s in source.read_text().splitlines()]
    generation=Path('artifacts/demand-transfer-generation-v1');audit=Path('artifacts/demand-transfer-audit-v1');controls=Path('artifacts/demand-transfer-controls-v1')
    out=Path('artifacts/dataset-release/transfer-candidates-v1');out.mkdir(parents=True,exist_ok=True)
    cases=[];pairs=[];packets=[];parents={str(source):digest(source)}
    for job in jobs:
        key=job['id'];gp=generation/(key+'.json');g=json.loads(gp.read_text());parents[str(gp)]=digest(gp)
        a=None;c=None
        for root in [audit,controls]:
            p=root/(key+'.json')
            if p.exists():
                parents[str(p)]=digest(p)
                if root==audit:a=json.loads(p.read_text())
                else:c=json.loads(p.read_text())
        case=dict(id=key,parent_demand=job['parent_demand'],expert_id=job['expert_id'],original_demand=job['demand'],sources=job['sources'],generation_status=g['status'],screening_status=a['status'] if a else None,control_status=c['status'] if c else None,expert_fit=None,future_project_success=None,human_review=None)
        if g['status']=='proposal_pending_audit':
            case.update(capability=g['capability'],capability_citations=g['citations'],target_operation=g['target_operation'],target_citation=g['demand_citation'],generator_hypothesis=dict(mechanism=g['mechanism'],assumptions=g['assumptions'],remaining_demand_gaps=g['remaining_demand_gaps']))
            if a:
                case['model_fact_screen']=a.get('fact_screen');case['blind_reconstruction']=a.get('review')
                case['reconstructed_demand_citation']=a.get('resolved_demand');case['reconstructed_source_citation']=a.get('resolved_source')
        elif g['status']=='no_proposal':case['no_proposal_reason']=g['reason']
        else:case['generation_error']=g.get('error_type')
        cases.append(case)
        if c and c['status']=='native_source_control_candidate':
            for view,sources in c['views'].items():
                pairs.append(dict(id=key+'-'+view,pair_id=key,parent_demand=job['parent_demand'],expert_id=job['expert_id'],view=view,original_demand=job['demand'],capability=g['capability'],sources=sources,
                                  source_sha256={sid:hashlib.sha256(s['text'].encode()).hexdigest() for sid,s in sources.items()},
                                  model_fact_label=c['labels'][view],label_scope='capability source support only; unknown is not inability',
                                  conditional_transfer=a.get('review'),transfer_validity=None,expert_fit=None,human_review=None))
        packets.append(dict(original_demand=job['demand'],sources=job['sources'],
                            phase1_without_model_claim=dict(required_operations=None,documented_methods=None,source_citations=None,unresolved_information=None),
                            phase2_proposed_hypothesis=dict(capability=g.get('capability'),target_operation=g.get('target_operation'),mechanism=a.get('review',{}).get('mechanism') if a else None),
                            phase2_review=dict(fact_support=None,target_alignment=None,bridge_specificity=None,assumptions_preserved=None,reason=None),
                            source_case_id=key))
    random.Random(20261117).shuffle(packets)
    for i,packet in enumerate(packets):packet['blind_id']='transfer-review-'+str(i).zfill(3)
    outputs={'cases.jsonl':cases,'source-pairs.jsonl':pairs,'human-review.jsonl':packets}
    for name,rows in outputs.items():
        body=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows);p=out/name
        if p.exists() and p.read_text()!=body:raise ValueError('Changed exported transfer collection')
        p.write_text(body)
    manifest=dict(cases=len(cases),conditional_proposals=sum(c['generation_status']=='proposal_pending_audit' for c in cases),
                  model_screened_candidates=sum(c['screening_status']=='model_screened_transfer_candidate' for c in cases),
                  capability_source_pairs=len(pairs)//2,source_views=len(pairs),human_review_cases=len(packets),human_labels=0,
                  output_sha256={name:digest(out/name) for name in outputs},parents=parents,
                  scope='Local model-screened conditional-transfer research collection. Not independent expert ranking gold. Native labels apply only to documented capability facts; bridge, full-project fit and future success unlabeled. All original20 outcomes retained, no new independent test split.',
                  release_status='Local only: real enterprise demand and teacher paper prefixes retained. Not included in MIT source archive or publicly redistributed.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='parents'},indent=2))


if __name__=='__main__':run()
