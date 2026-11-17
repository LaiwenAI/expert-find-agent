"""Recheck frozen numeric flags for two post-hoc cross-language alternatives."""
import argparse
from collections import Counter
import json
from pathlib import Path
from expert_search.bilingual.numeric_provenance_v3 import audit_numeric_provenance_v3
from experiments.bilingual.audit_numeric_provenance import sha


def main(root,prior,output):
    manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((prior/'summary.json').read_text());case_path=prior/'private-cases.json'
    if sha(root/'manifest.json')!=summary['dataset_manifest_sha256'] or sha(case_path)!=summary['cases_sha256'] or any(sha(root/n)!=manifest['outputs_sha256'][n] for n in ('real-queries.jsonl','synthetic-queries.jsonl')):raise ValueError('Changed frozen inputs')
    parents={r['id']:r for r in map(json.loads,(root/'real-queries.jsonl').read_text().splitlines())};derived={r['id']:r for r in map(json.loads,(root/'synthetic-queries.jsonl').read_text().splitlines())};old=json.loads(case_path.read_text())
    if len(old)!=len(derived) or {r['id'] for r in old}!=set(derived):raise ValueError('Changed inventory')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in old:
        row=derived[case['id']];parent=parents[row['parent_id']];checks={}
        for lang in ('zh','en'):
            result=audit_numeric_provenance_v3(parent['zh'],row['source_ids'],row[lang])
            if result['still_absent_from_parent_v2']!=case['checks'][lang]['still_absent_from_parent_values']:raise ValueError('Changed v2 flags')
            checks[lang]={k:v for k,v in result.items() if k!='scope'}
        rows.append({'id':row['id'],'intent_type':row['intent_type'],'status':row['status'],'checks':checks})
    (output/'private-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    def aggregate(subset):
        count=Counter(pairs=len(subset))
        for row in subset:
            count['v2_flagged']+=any(v['still_absent_from_parent_v2']>0 for v in row['checks'].values())
            count['possible_cross_language']+=any(v['possible_cross_language_values']>0 for v in row['checks'].values())
            count['still_flagged']+=any(v['still_absent_after_hints']>0 for v in row['checks'].values())
        return dict(count)
    report={'overall':aggregate(rows),'by_status':{s:aggregate([r for r in rows if r['status']==s]) for s in sorted({r['status'] for r in rows})},
            'prior_summary_sha256':sha(prior/'summary.json'),'dataset_manifest_sha256':sha(root/'manifest.json'),'cases_sha256':sha(output/'private-cases.json'),
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/bilingual/numeric_provenance_v3.py'),Path('src/expert_search/bilingual/numeric_provenance_v2.py'))},
            'scope':'Two narrow cross-language literal hints chosen post hoc after inspecting flagged queries. Matches are ambiguous, not semantic verification; remaining flags are not proven errors. Earlier audits and dataset status are unchanged. No data or identifiers published.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/dataset-release/bilingual-query-construction-v1'));p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.prior,a.output)
