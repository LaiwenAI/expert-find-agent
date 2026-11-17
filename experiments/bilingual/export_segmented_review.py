"""Prepare blind paired full-translation review from the segmented pilot."""
import hashlib
import json
import random
from pathlib import Path

from .translate_segmented_pilot import sha


def main():
    data=Path('artifacts/large-bilingual-data-v1');root=data/'original-translations-segmented-v1';direct=data/'original-translations-v1'
    protocol=json.loads((root/'protocol.json').read_text());qfile=data/'queries-zh.jsonl'
    if sha(qfile)!=protocol['source_sha256'] or sha(direct/'protocol.json')!=protocol['direct_protocol_sha256']:raise ValueError('Changed source')
    queries={r['id']:r for r in map(json.loads,qfile.read_text().splitlines())};cases=[];attempts={};counts={'direct':0,'segmented':0}
    for qid in protocol['parent_ids']:
        for arm,folder in [('direct',direct),('segmented',root)]:
            path=folder/'parents'/(qid+'.json');attempts[str(path)]=sha(path)
            if arm=='direct' and attempts[str(path)]!=protocol['direct_record_sha256'][qid]:raise ValueError('Changed direct attempt')
            row=json.loads(path.read_text());parent=queries[qid]
            if row['parent_id']!=qid or row['split']!='construction' or row['source_text_sha256']!=parent['source']['text_sha256']:raise ValueError('Changed parent')
            if row['en'] is None:continue
            segments=row['segments'];source=parent['zh']
            if ''.join(s['zh'] for s in segments)!=source or '\n'.join(s['en'] for s in segments)!=row['en'] or any(source[s['start']:s['end']]!=s['zh'] for s in segments):raise ValueError('Invalid translation assembly')
            cases.append((qid,arm,source,row['en'],row['status']));counts[arm]+=1
    random.Random(20261127).shuffle(cases);forms=[];mapping=[]
    for i,(qid,arm,source,en,status) in enumerate(cases,1):
        cid=f'case-{i:03d}';forms.append({'case_id':cid,'original_need':source,'english_translation':en,'label':None,'reason':None,'reviewer_id':None})
        mapping.append({'case_id':cid,'parent_id':qid,'arm':arm,'screen_status':status})
    output=Path('artifacts/segmented-translation-human-review-v1');output.mkdir(exist_ok=False)
    review=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n').encode()
    (output/'blind-review.jsonl').write_bytes(review);(output/'private-mapping.json').write_bytes(private)
    manifest={'cases':len(forms),'parent_count':len(protocol['parent_ids']),'generated_by_arm':counts,'review_sha256':hashlib.sha256(review).hexdigest(),
              'mapping_sha256':hashlib.sha256(private).hexdigest(),'protocol_sha256':sha(root/'protocol.json'),'attempt_sha256':attempts,
              'scope':'Local sensitive text; model screen and arm hidden, repeated parents and source identity may be apparent. Missing generations excluded from forms but retained in trial denominators. No human labels.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='attempt_sha256'},indent=2))


if __name__=='__main__':main()
