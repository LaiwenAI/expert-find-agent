"""Extract all compressed/full actionable disagreements, without model calls."""
import json
from pathlib import Path


def run():
    root=Path('artifacts/enterprise-span-assessment-v1')
    rows=json.loads((root/'evaluation.json').read_text())
    responses={r['request']:json.loads((root/'responses'/(r['request']+'.json')).read_text()) for r in rows}
    if any(r['status']!='structurally_valid' for r in responses.values()):
        raise ValueError('Disagreement protocol requires complete valid parent; do not silently drop failures')
    full={r['id']:responses[r['request']]['result'] for r in rows if r['method']=='full_source_view'}
    actionable=lambda a:a['level'] in ('direct_application','transferable_method')
    differences=[]
    for row in rows:
        if row['method']=='full_source_view':continue
        for rid,a in responses[row['request']]['result']['assessments'].items():
            b=full[row['id']]['assessments'][rid]
            if actionable(a)!=actionable(b):
                differences.append(dict(id=row['id'],method=row['method'],requirement_id=rid,compressed=a,full=b))
    path=root/'disagreements.json'
    if path.exists():
        if json.loads(path.read_text())!=differences:raise ValueError('Changed saved disagreements')
    else:path.write_text(json.dumps(differences,indent=2,ensure_ascii=False))
    print('All actionable disagreements:',len(differences))


if __name__=='__main__':run()
