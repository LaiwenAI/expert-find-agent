"""Create and validate a private blind review of high-overlap intent pairs."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    result={r['case_id']:r for r in rows}
    if len(rows)!=len(result):raise ValueError('Duplicate review cases')
    return result


def check(root,review):
    manifest=json.loads((root/'manifest.json').read_text());baseline=root/'blind-review.jsonl'
    if sha(baseline)!=manifest['output_sha256']:raise ValueError('Changed blind export')
    expected=read_jsonl(baseline);actual=read_jsonl(review)
    if set(expected)!=set(actual) or len(expected)!=manifest['cases']:raise ValueError('Changed review cases')
    fields=('chinese_relation','english_relation');mutable=set(fields)|{'rationale','reviewer_id'}
    counts={f:{'same':0,'different':0,'uncertain':0,'missing':0} for f in fields}
    for case,row in actual.items():
        if row.keys()!=expected[case].keys() or any(row[k]!=expected[case][k] for k in row if k not in mutable):raise ValueError('Changed immutable review content')
        for f in fields:
            if row[f] not in (None,'same','different','uncertain'):raise ValueError('Invalid relation label')
            counts[f][row[f] or 'missing']+=1
        if any(row[f] is not None for f in fields) and any(not isinstance(row[k],str) or not row[k].strip() for k in ('rationale','reviewer_id')):
            raise ValueError('A judgment requires rationale and reviewer ID')
    return counts


def export(data,audit,output):
    summary=json.loads((audit/'summary.json').read_text());manifest=json.loads((data/'manifest.json').read_text())
    cases=audit/'private-cases.json';synthetic=data/'synthetic-queries.jsonl';real=data/'real-queries.jsonl'
    if sha(cases)!=summary['private_cases_sha256'] or sha(synthetic)!=summary['input_sha256'] or sha(data/'manifest.json')!=summary['manifest_sha256'] or sha(real)!=manifest['outputs_sha256'][real.name]:
        raise ValueError('Changed diversity audit inputs')
    rows={r['id']:r for r in map(json.loads,synthetic.read_text().splitlines())}
    parents={r['id']:r for r in map(json.loads,real.read_text().splitlines())}
    eligible=[r for r in json.loads(cases.read_text()) if any(r['overlap'][lang] is not None and r['overlap'][lang]>=.5 for lang in ('zh','en'))]
    if len({tuple(r['ids']) for r in eligible})!=len(eligible):raise ValueError('Duplicate pair')
    random.Random(20261123).shuffle(eligible);forms=[];mapping=[]
    for i,case in enumerate(eligible,1):
        a,b=(rows[x] for x in case['ids']);parent=parents[case['parent_id']]
        if a['parent_id']!=parent['id'] or b['parent_id']!=parent['id'] or parent['split']!='construction':raise ValueError('Changed parent binding')
        if i%2:a,b=b,a
        case_id=f'intent-{i:04d}'
        forms.append(dict(case_id=case_id,original_need=parent['zh'],request_a_zh=a['zh'],request_b_zh=b['zh'],request_a_en=a['en'],request_b_en=b['en'],
                          chinese_relation=None,english_relation=None,rationale=None,reviewer_id=None))
        mapping.append(dict(case_id=case_id,query_ids=[a['id'],b['id']],intent_types=[a['intent_type'],b['intent_type']],parent_id=parent['id']))
    output.mkdir(parents=True,exist_ok=False)
    (output/'blind-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms))
    (output/'private-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    report=dict(cases=len(forms),seed=20261123,selection='Any within-parent pair with Chinese or English character-four-gram Jaccard >= 0.5; post-hoc targeted sample.',
        audit_summary_sha256=sha(audit/'summary.json'),dataset_manifest_sha256=sha(data/'manifest.json'),output_sha256=sha(output/'blind-review.jsonl'),private_mapping_sha256=sha(output/'private-mapping.json'),script_sha256=sha(__file__),
        scope='Local private review packet. Reviewer sees both bilingual views and original need, but not type, status, source IDs or overlap scores. Labels same/different/uncertain are independent human judgments and initially missing. A targeted post-hoc sample cannot estimate semantic redundancy for the full population.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(cases=len(forms),review_counts=check(output,output/'blind-review.jsonl'),scope=report['scope']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('artifacts/dataset-release/bilingual-query-construction-v1'));p.add_argument('--audit',type=Path,default=Path('artifacts/bilingual-intent-diversity-v1'));p.add_argument('--output',type=Path,default=Path('artifacts/intent-diversity-human-review-v1'));p.add_argument('--review',type=Path);a=p.parse_args()
    if a.review:print(json.dumps(check(a.output,a.review),indent=2))
    else:export(a.data,a.audit,a.output)
