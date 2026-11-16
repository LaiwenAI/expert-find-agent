"""Encode offline profile cards, falling back to raw profiles on failures."""
import json
from pathlib import Path
from laiwenai import LaiwenAI
from .dense_retrieval import encode
from .cmu_baseline import text


def run(source='artifacts/public/cmu-profile-cards',output='artifacts/public/cmu-card-vectors'):
    source=Path(source)
    summary=json.loads((source/'summary.json').read_text())
    root=Path('artifacts/public/cmu-gold-upstream/evaluation_datasets/d_20_1/archives')
    rows=[];owners={};fallbacks=[]
    for p in sorted(root.glob('*.jsonl')):
        rid=p.stem.lstrip('~');r=json.loads((source/f'{rid}.json').read_text())
        if r['status']=='span_valid':texts=[c['text'] for c in r['cards']]
        else:
            pubs=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            texts=['\n'.join(text(x) for x in pubs)[:6000]];fallbacks.append(rid)
        for i,t in enumerate(texts):
            eid=f'{rid}:{i}';rows.append({'id':eid,'text':t});owners[eid]=rid
    encode(rows,output,LaiwenAI(),32)
    Path(output,'owners.json').write_text(json.dumps(dict(owners=owners,fallback_reviewers=fallbacks,source_summary=summary),indent=2))
    return dict(vectors=len(rows),fallback_profiles=len(fallbacks))

if __name__=='__main__':print(run())
