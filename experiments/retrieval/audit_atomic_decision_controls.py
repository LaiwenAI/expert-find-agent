"""Check the atomic Decisions question on synthetic support controls."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_reason_atomic_claims import OPTIONS
from .audit_support_heldout_controls import cases


def main():
    root=Path('artifacts/anchored-atomic-decision-controls-v1');root.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    def one(case):
        row=dict(id=case['id'],expected=case['expected'],status='unresolved')
        try:
            payload=dict(passages=case['payload']['quotes'],claims={'claim_0':case['payload']['reason']})
            q=[dict(type='choice',name='claim_0',instructions='Do the passages explicitly support the exact factual statement named claim_0? Do not infer additional facts.',choices=OPTIONS)]
            answer=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),payload,q)
            row.update(status='valid',prediction=answer['labels']['claim_0'])
        except Exception as error:row['error_type']=type(error).__name__
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,cases()))
    counts=Counter((r['expected'],r.get('prediction','unresolved')) for r in rows)
    report=dict(cases=len(rows),counts=[dict(expected=a,prediction=b,count=n) for (a,b),n in sorted(counts.items())],
                scope='Post-hoc native atomic Decisions control on six previously inspected synthetic template pairs. No human gold or evidence that batched multi-claim judgments generalize.')
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
