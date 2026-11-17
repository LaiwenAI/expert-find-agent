"""Create a blinded targeted review of numeric-provenance flags and controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from expert_search.bilingual.constraints import lossless_segments
from experiments.bilingual.audit_numeric_provenance import sha


def main(root,audit,output):
    data=json.loads((root/'manifest.json').read_text());summary=json.loads((audit/'summary.json').read_text());cases_path=audit/'private-cases.json'
    if sha(root/'manifest.json')!=summary['dataset_manifest_sha256'] or sha(cases_path)!=summary['cases_sha256'] or any(sha(root/n)!=data['outputs_sha256'][n] for n in ('real-queries.jsonl','synthetic-queries.jsonl')):raise ValueError('Changed audit or dataset')
    audits=json.loads(cases_path.read_text());derived={r['id']:r for r in map(json.loads,(root/'synthetic-queries.jsonl').read_text().splitlines())};parents={r['id']:r for r in map(json.loads,(root/'real-queries.jsonl').read_text().splitlines())}
    if len(audits)!=len(derived) or {r['id'] for r in audits}!=set(derived):raise ValueError('Changed audit inventory')
    flagged=[];controls=defaultdict(list)
    for case in audits:
        key=(case['intent_type'],case['status']);checks=case['checks']
        if any(x['absent_from_parent_values'] for x in checks.values()):flagged.append(case)
        elif any(x['detected_query_values'] for x in checks.values()):controls[key].append(case)
    if len(flagged)!=summary['overall']['either_absent_from_parent']:raise ValueError('Changed flag count')
    rng=random.Random(20261123);selected=[]
    for case in sorted(flagged,key=lambda x:x['id']):
        key=(case['intent_type'],case['status']);pool=controls[key]
        if not pool:raise ValueError('Insufficient matched controls')
        index=rng.randrange(len(pool));control=pool.pop(index)
        selected.extend([(case,'flagged'),(control,'control')])
    rng.shuffle(selected);forms=[];mapping=[]
    for i,(case,group) in enumerate(selected,1):
        row=derived[case['id']];parent=parents[row['parent_id']];segments={s['id']:s['zh'] for s in lossless_segments(parent['zh'])}
        if set(row['source_ids'])-set(segments):raise ValueError('Changed source IDs')
        key=f'case-{i:03d}';forms.append({'case_id':key,'intent_type':row['intent_type'],'original_need':parent['zh'],
            'cited_segments':[segments[s] for s in row['source_ids']],'chinese_request':row['zh'],'english_request':row['en'],
            'chinese_grounding':None,'english_grounding':None,'bilingual_equivalence':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':key,'query_id':row['id'],'parent_id':row['parent_id'],'audit_group':group,'original_status':row['status']})
    output.mkdir(parents=True,exist_ok=False)
    (output/'blind-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in forms))
    (output/'private-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    report={'cases':len(forms),'flagged':len(flagged),'controls':len(flagged),'seed':20261123,'dataset_manifest_sha256':sha(root/'manifest.json'),
            'audit_summary_sha256':sha(audit/'summary.json'),'output_sha256':sha(output/'blind-review.jsonl'),'private_mapping_sha256':sha(output/'private-mapping.json'),
            'scope':'Targeted numeric flag and intent/status-matched control packet; source and output text remain local. Audit group and model status hidden. Cases may be recognized by content. No human judgments or population error estimate.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/dataset-release/bilingual-query-construction-v1'));p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.audit,a.output)
