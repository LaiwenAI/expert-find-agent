"""Prepare local model-blind review forms for complete-need translations."""
import hashlib
import json
import random
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=Path('artifacts/large-bilingual-data-v1');comparison=data/'translation-comparison-v1/summary.json'
    frozen=json.loads(comparison.read_text());root_direct=data/'original-translations-v1'
    protocol=json.loads((root_direct/'protocol.json').read_text());parents=protocol['selected_parent_ids'][:32]
    qfile=data/'queries-zh.jsonl'
    if sha(qfile)!=protocol['source_queries_sha256']:raise ValueError('Changed parent cohort')
    queries={r['id']:r for r in map(json.loads,qfile.read_text().splitlines())};cases=[];counts={}
    arms={'direct':'original-translations-v1','placeholders':'original-translations-protected-v1','fixed_blocks':'original-translations-gaps-v1'}
    for arm,name in arms.items():
        root=data/name
        if json.loads((root/'protocol.json').read_text())['selected_parent_ids'][:32]!=parents:raise ValueError('Changed cohort')
        counts[arm]=0
        for qid in parents:
            path=root/'parents'/(qid+'.json')
            if sha(path)!=frozen['inputs_sha256'][str(path)]:raise ValueError('Changed translation record')
            row=json.loads(path.read_text());parent=queries[qid]
            if row['parent_id']!=qid or row['split']!='construction' or row['source_text_sha256']!=parent['source']['text_sha256']:raise ValueError('Invalid parent identity')
            if row['en'] is None:continue
            segments=row['segments'];source=parent['zh']
            if not segments or ''.join(s['zh'] for s in segments)!=source or '\n'.join(s['en'] for s in segments)!=row['en']:raise ValueError('Incomplete segment assembly')
            if any(source[s['start']:s['end']]!=s['zh'] for s in segments):raise ValueError('Invalid source offsets')
            cases.append((arm,qid,source,row['en'],row['status']));counts[arm]+=1
    random.Random(20261126).shuffle(cases);forms=[];mapping=[]
    for i,(arm,qid,source,translation,status) in enumerate(cases,1):
        cid=f'case-{i:03d}'
        forms.append({'case_id':cid,'original_need':source,'english_translation':translation,'label':None,'reason':None,'reviewer_id':None})
        mapping.append({'case_id':cid,'parent_id':qid,'arm':arm,'screen_status':status})
    root=Path('artifacts/original-translation-human-review-v1');root.mkdir(exist_ok=False)
    review=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n').encode()
    (root/'blind-review.jsonl').write_bytes(review);(root/'private-mapping.json').write_bytes(private)
    manifest={'cases':len(forms),'parents':len(parents),'generated_by_arm':counts,'seed':20261126,
              'review_sha256':hashlib.sha256(review).hexdigest(),'mapping_sha256':hashlib.sha256(private).hexdigest(),
              'cohort_sha256':sha(qfile),'comparison_sha256':sha(comparison),'code_sha256':sha(Path(__file__)),
              'scope':'Local sensitive full original needs; model judgments and arm labels hidden. Same parent may recur and text may reveal identity. Incomplete generation attempts excluded from review but remain in original experiment denominator. No human judgments yet.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
