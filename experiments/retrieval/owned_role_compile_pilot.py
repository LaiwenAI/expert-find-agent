"""Compile role hints for a small deterministic sample of retrieved sources."""
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.owned_role_compile import compile_owned_roles


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent=Path('artifacts/retrieved-coverage-pilot-v1');protocol=json.loads((parent/'protocol.json').read_text())
    sources=[];seen=set()
    for qid in protocol['query_ids']:
        path=parent/(qid+'-pool.json');raw=path.read_bytes();hits=json.loads(raw)
        for arm in protocol['arms']:
            result=json.loads((parent/(qid+'-'+arm+'.json')).read_text())
            if result['input_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Changed candidate pool')
        used=0
        for hit in hits:
            if not hit['evidence']:continue
            row=hit['evidence'][0];identity=(row['id'],row['expert_id'])
            if identity in seen:continue
            if row['expert_id']!=hit['id']:raise ValueError('Invalid source ownership')
            sources.append({'id':row['id'],'expert_id':row['expert_id'],'text':row['text'][:6000]});seen.add(identity);used+=1
            if used==3:break
        if used!=3:raise ValueError('Insufficient distinct sources')
    if len(sources)!=12:raise ValueError('Unexpected sample size')
    root=Path('artifacts/owned-role-compile-pilot-v1');root.mkdir(exist_ok=False)
    payload=json.dumps(sources,ensure_ascii=False,sort_keys=True).encode();(root/'sources.json').write_bytes(payload)
    manifest={'parent_protocol_sha256':sha(parent/'protocol.json'),'source_sha256':hashlib.sha256(payload).hexdigest(),
              'code_sha256':{p:sha(Path(p)) for p in ['experiments/retrieval/owned_role_compile_pilot.py','src/expert_search/owned_role_compile.py','src/expert_search/owned_role_windows.py','src/expert_search/source_roles.py']},
              'model':'gpt-6-luna','width':400,'sources':len(sources),'scope':'Development only; three distinct first retrieved source views per demand, capped at 6000 characters. Source-only model call; no demand or judgments sent. Role semantics and expert attribution remain unverified.'}
    (root/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
    load_dotenv('.env');client=LaiwenAI(model='gpt-6-luna',max_retries=0)
    for i,source in enumerate(sources,1):
        result=compile_owned_roles(source,client,width=400)
        (root/(f'source-{i:03d}.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f'source-{i:03d}',result['status'],flush=True)


if __name__=='__main__':main()
