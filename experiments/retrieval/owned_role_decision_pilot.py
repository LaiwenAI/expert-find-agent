"""Compare native role choices with an earlier fixed paper-source pilot."""
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.owned_role_decision import compile_owned_decision_roles


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent=Path('artifacts/owned-role-paper-pilot-v1');prior=json.loads((parent/'protocol.json').read_text())
    raw=(parent/'sources.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=prior['source_sha256']:raise ValueError('Changed source views')
    sources=json.loads(raw)
    if len(sources)!=prior['sources']:raise ValueError('Changed sample size')
    inputs={f'source-{i:03d}.json':sha(parent/(f'source-{i:03d}.json')) for i in range(1,len(sources)+1)}
    code=['experiments/retrieval/owned_role_decision_pilot.py','src/expert_search/owned_role_decision.py',
          'src/expert_search/owned_role_windows.py','src/expert_search/source_roles.py','src/expert_search/choice_judgments.py']
    root=Path('artifacts/owned-role-decision-pilot-v1');root.mkdir(exist_ok=False)
    protocol={'source_sha256':hashlib.sha256(raw).hexdigest(),'previous_protocol_sha256':sha(parent/'protocol.json'),
              'previous_attempt_sha256':inputs,'code_sha256':{path:sha(Path(path)) for path in code},'sources':len(sources),
              'width':prior['width'],'model':'gpt-6-luna',
              'scope':'Post-hoc sequential native-choice versus prior Chat on same development paper views. Prompts, transport and answer space differ. Model roles are not independent gold; no claims about semantic correctness or enterprise relevance.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    load_dotenv('.env');client=LaiwenAI(model='gpt-6-luna',max_retries=0)
    for i,source in enumerate(sources,1):
        result=compile_owned_decision_roles(source,client,width=prior['width'])
        (root/(f'source-{i:03d}.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f'source-{i:03d}',result['status'],flush=True)


if __name__=='__main__':main()
