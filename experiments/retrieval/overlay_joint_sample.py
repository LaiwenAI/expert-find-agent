"""Overlay a validated blinded subsample onto fixed parent-view paired bounds."""
import argparse
import json
from pathlib import Path
from .joint_review_bounds import calculate,joint_label,records,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path);p.add_argument('sample',type=Path);p.add_argument('reconciled',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.parent/'manifest.json').read_text());sample=json.loads((a.sample/'manifest.json').read_text());rec=json.loads((a.reconciled/'summary.json').read_text())
    if sha(a.parent/'manifest.json')!=sample['input_sha256']['parent_manifest'] or sha(a.sample/'manifest.json')!=rec['packet_sha256']:
        raise ValueError('Changed packet lineage')
    for root,manifest,names in ((a.parent,parent,('blind-review.jsonl','private-mapping.jsonl')),(a.sample,sample,('blind-review.jsonl','private-mapping.jsonl','private-design.jsonl'))):
        for name in names:
            if sha(root/name)!=manifest['outputs_sha256'][name]:raise ValueError('Changed packet contents')
    outcomes_path=a.reconciled/'private-outcomes.json'
    if sha(outcomes_path)!=rec['outcomes_sha256']:raise ValueError('Changed reconciliation')
    parent_forms={r['case_id']:r for r in records(a.parent/'blind-review.jsonl')}
    parent_map={r['case_id']:r for r in records(a.parent/'private-mapping.jsonl')}
    sample_forms=records(a.sample/'blind-review.jsonl');sample_map=records(a.sample/'private-mapping.jsonl')
    outcomes=json.loads(outcomes_path.read_text());design=records(a.sample/'private-design.jsonl')
    if len(sample_forms)!=sample['candidate_views'] or len(sample_map)!=len(sample_forms) or len(design)!=len(sample_forms):raise ValueError('Changed sample size')
    if len({r['case_id'] for r in sample_forms})!=len(sample_forms) or {r['case_id'] for r in sample_forms}!=set(outcomes):raise ValueError('Changed sample inventory')
    if {r['case_id'] for r in sample_map}!=set(outcomes) or {r['case_id'] for r in design}!=set(outcomes):raise ValueError('Changed mapping inventory')
    for row in sample_forms:
        if parent_forms.get(row['case_id'])!=row:raise ValueError('Changed sampled view')
    for row in sample_map:
        if parent_map.get(row['case_id'])!=row:raise ValueError('Changed sampled mapping')
    labels={}
    for row in sample_map:
        cell=(row['query_id'],row['expert_id'])
        if cell in labels:raise ValueError('Duplicate query-expert')
        labels[cell]=joint_label(outcomes[row['case_id']])
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');audit=Path('artifacts/fresh-lexical-pool-audit-v1/runs.jsonl')
    if sha(root/'pools.jsonl')!=parent['inputs_sha256']['pools.jsonl'] or sha(audit)!=parent['inputs_sha256']['lexical_runs']:
        raise ValueError('Changed rankings')
    dense={r['query_id']:r['dense_order'] for r in records(root/'pools.jsonl') if r['split']=='development'}
    lexical={r['query_id']:[x['expert_id'] for x in r['ranking']] for r in records(audit) if r['variant']=='full-demand'}
    parent_cells={(r['query_id'],r['expert_id']) for r in parent_map.values()}
    if any((q,e) not in parent_cells for q in dense for e in dense[q][:5]+lexical[q][:5]):raise ValueError('Ranked view missing from parent')
    initial=calculate(dense,lexical,{})
    result={'initial':initial,'after_sample':calculate(dense,lexical,labels),'sample_cases':len(labels),
            'definite_labels':sum(v is not None for v in labels.values()),'unknown_labels':sum(v is None for v in labels.values()),
            'parent_manifest_sha256':sha(a.parent/'manifest.json'),'sample_manifest_sha256':sha(a.sample/'manifest.json'),
            'reconciliation_sha256':sha(a.reconciled/'summary.json'),'script_sha256':sha(Path(__file__)),
            'scope':'Partial-label identification bounds conditional on exact joint source views, not design-weighted inference or a confidence interval. Unsampled, missing, uncertain and conflicting labels stay unknown. Source-scoped unsupported is not expert inability. Inputs contain private text and case mappings; no independent review or relevance claim is implied.'}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
