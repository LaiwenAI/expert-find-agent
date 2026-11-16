"""Conservative pilot selection; preserves rejected rows and reasons."""
import json
from pathlib import Path


def select(source='artifacts/synthesis-pilot',audit='artifacts/synthesis-audit',output='artifacts/synthetic-selected'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);accepted=[];decisions=[]
    for p in sorted(Path(source).glob('*.json')):
        r=json.loads(p.read_text());check=Path(audit)/p.name;reasons=[]
        if r['validation_errors']:reasons.append('invalid source span')
        if not any(c['label']=='supported' for c in r['example']['constraints']):reasons.append('no supported constraint')
        if not check.exists():reasons.append('no independent model audit')
        elif json.loads(check.read_text())['status']!='model_agreement_silver':reasons.append('verifier disagreement or failure')
        decisions.append(dict(id=p.stem,accepted=not reasons,reasons=reasons))
        if not reasons:accepted.append(r)
    (out/'examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    (out/'selection.json').write_text(json.dumps(dict(accepted=len(accepted),total=len(decisions),decisions=decisions,
        label_quality='silver only; selection does not validate counterfactual labels or full query match'),ensure_ascii=False,indent=2))
    return decisions
