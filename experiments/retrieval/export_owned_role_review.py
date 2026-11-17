"""Prepare local model-blind review forms for fixed-window role hints."""
import hashlib
import json
import random
from pathlib import Path

from expert_search.owned_role_windows import segment_source,source_key


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    roots=[Path('artifacts/owned-role-compile-pilot-v1'),Path('artifacts/owned-role-paper-pilot-v1')]
    cases={};input_hashes={}
    for root in roots:
        protocol=json.loads((root/'protocol.json').read_text());sources_path=root/'sources.json'
        if sha(sources_path)!=protocol['source_sha256']:raise ValueError('Changed source views')
        input_hashes[str(root/'protocol.json')]=sha(root/'protocol.json');input_hashes[str(sources_path)]=sha(sources_path)
        sources=json.loads(sources_path.read_text())
        if len(sources)!=protocol['sources']:raise ValueError('Changed source count')
        for i,source in enumerate(sources,1):
            path=root/(f'source-{i:03d}.json');record=json.loads(path.read_text());input_hashes[str(path)]=sha(path)
            if record['source_key']!=source_key(source,protocol['width']):raise ValueError('Changed source binding')
            windows=segment_source(source,protocol['width']);identity=json.dumps(record['source_key'],sort_keys=True,ensure_ascii=False)
            if identity not in cases:cases[identity]={'source':source,'windows':windows,'records':[]}
            elif cases[identity]['source']!=source:raise ValueError('Inconsistent duplicate source')
            cases[identity]['records'].append({'study':root.name,'status':record['status'],'role_record':record['role_record']})
    cases=list(cases.values());random.Random(20261124).shuffle(cases);forms=[];mapping=[]
    for i,case in enumerate(cases,1):
        source,windows=case['source'],case['windows']
        cid=f'case-{i:03d}'
        forms.append({'case_id':cid,'source_text':source['text'],
                      'windows':[{'id':w['id'],'start':w['start'],'end':w['end'],'text':w['text'],'label':None,'reason':None} for w in windows],
                      'reviewer_id':None})
        mapping.append({'case_id':cid,'source_id':source['id'],'expert_id':source['expert_id'],'records':case['records']})
    root=Path('artifacts/owned-role-human-review-v1');root.mkdir(exist_ok=False)
    review=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n').encode()
    (root/'blind-review.jsonl').write_bytes(review);(root/'private-mapping.json').write_bytes(private)
    report={'cases':len(forms),'windows':sum(len(r['windows']) for r in forms),'seed':20261124,
            'review_sha256':hashlib.sha256(review).hexdigest(),'mapping_sha256':hashlib.sha256(private).hexdigest(),
            'input_sha256':input_hashes,'code_sha256':sha(Path(__file__)),
            'scope':'Deduplicated local sensitive source views; hides model roles and source IDs, but source identity may be visible in text. No human judgments yet. Window roles do not establish personal authorship.'}
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
