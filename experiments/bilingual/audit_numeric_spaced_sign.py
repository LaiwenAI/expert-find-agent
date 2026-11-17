"""Recheck frozen numeric provenance flags with ambiguous spaced signs."""
import argparse
from collections import Counter
import json
from pathlib import Path
from expert_search.bilingual.numeric_provenance_v2 import audit_numeric_provenance_v2
from experiments.bilingual.audit_numeric_provenance import sha


def main(root,prior,output):
    manifest=json.loads((root/'manifest.json').read_text());previous=json.loads((prior/'summary.json').read_text());cases_path=prior/'private-cases.json'
    if sha(root/'manifest.json')!=previous['dataset_manifest_sha256'] or sha(cases_path)!=previous['cases_sha256'] or any(sha(root/n)!=manifest['outputs_sha256'][n] for n in ('real-queries.jsonl','synthetic-queries.jsonl')):raise ValueError('Changed frozen inputs')
    parents={r['id']:r for r in map(json.loads,(root/'real-queries.jsonl').read_text().splitlines())};derived={r['id']:r for r in map(json.loads,(root/'synthetic-queries.jsonl').read_text().splitlines())};old=json.loads(cases_path.read_text())
    if len(old)!=len(derived) or {r['id'] for r in old}!=set(derived):raise ValueError('Changed inventory')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in old:
        row=derived[case['id']];parent=parents[row['parent_id']];checks={}
        for lang in ('zh','en'):
            check=audit_numeric_provenance_v2(parent['zh'],row['source_ids'],row[lang]);prior_check=case['checks'][lang]
            if check['absent_from_parent_v1']!=prior_check['absent_from_parent_values'] or check['absent_from_cited_v1']!=prior_check['elsewhere_in_parent_values']+prior_check['absent_from_parent_values']:
                raise ValueError('Original audit not reproduced')
            checks[lang]={k:v for k,v in check.items() if k!='scope'}
        rows.append({'id':row['id'],'intent_type':row['intent_type'],'status':row['status'],'checks':checks})
    (output/'private-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    def aggregate(subset):
        result=Counter(pairs=len(subset))
        for row in subset:
            for lang,x in row['checks'].items():
                result[lang+'_v1_parent_flag']+=x['absent_from_parent_v1']>0
                result[lang+'_ambiguous_spaced_sign']+=x['ambiguous_parent_spaced_sign_values']>0
                result[lang+'_still_parent_flag']+=x['still_absent_from_parent_values']>0
            result['either_v1_parent_flag']+=any(x['absent_from_parent_v1']>0 for x in row['checks'].values())
            result['either_ambiguous_spaced_sign']+=any(x['ambiguous_parent_spaced_sign_values']>0 for x in row['checks'].values())
            result['either_still_parent_flag']+=any(x['still_absent_from_parent_values']>0 for x in row['checks'].values())
        return dict(result)
    report={'overall':aggregate(rows),'by_status':{s:aggregate([r for r in rows if r['status']==s]) for s in sorted({r['status'] for r in rows})},
            'dataset_manifest_sha256':sha(root/'manifest.json'),'prior_summary_sha256':sha(prior/'summary.json'),'prior_cases_sha256':sha(cases_path),
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/bilingual/numeric_provenance_v2.py'),Path('src/expert_search/bilingual/constraints.py'))},
            'cases_sha256':sha(output/'private-cases.json'),
            'scope':'Spaced negative signs are possible alternatives, not verified minus signs: range dashes and bullets can match. Ambiguous cases are not cleared or promoted. All earlier flags and screening statuses remain unchanged; exact values and IDs stay private. Value presence does not prove grounding.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/dataset-release/bilingual-query-construction-v1'));p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.prior,a.output)
