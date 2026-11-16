"""Offline capability-card ablation with fixed fusion weights and common labels."""
import json
import time
from pathlib import Path
import numpy as np
from .cmu_gold import load_annotations,pairwise_loss


def run(output='artifacts/public/cmu-card-results'):
    root=Path('artifacts/public');v=root/'cmu-card-vectors';raw=root/'cmu-dense'
    cm=json.loads((v/'manifest.json').read_text());pm=json.loads((raw/'manifest.json').read_text())
    if cm['model']!=pm['model']:raise ValueError('Card/query embedding model mismatch')
    cards=np.load(v/'vectors.npy');papers=np.load(raw/'vectors.npy');pidx={pid:i for i,pid in enumerate(pm['ids'])}
    ownership=json.loads((v/'owners.json').read_text());owner=ownership['owners'];groups={}
    for i,cid in enumerate(cm['ids']):groups.setdefault(owner[cid],[]).append(i)
    labels=load_annotations(root/'cmu-gold/data/evaluations.csv')
    mixture=json.loads((root/'cmu-dense-results/mean-max-50-50-predictions.json').read_text())
    pred={m:{} for m in ['card-max','card-raw-50-50']};start=time.perf_counter()
    for rid,grades in labels.items():
        if rid not in groups:raise ValueError('Missing reviewer cards or raw fallback')
        for m in pred:pred[m][rid]={}
        for pid in grades:
            score=float(np.max(cards[groups[rid]]@papers[pidx[pid]]))
            pred['card-max'][rid][pid]=score
            pred['card-raw-50-50'][rid][pid]=.5*(score+mixture[rid][pid])
    elapsed=time.perf_counter()-start
    result=dict(results={m:pairwise_loss(p,labels) for m,p in pred.items()},fallback_reviewers=ownership['fallback_reviewers'],local_seconds=elapsed,
        embedding_model=cm['model'],warning='Single d_20_1 profile realization; all embeddings precomputed; fusion weights fixed, exploratory comparisons, no model training. Source-span verification does not establish semantic correctness of cards.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for m,p in pred.items():(out/f'{m}-predictions.json').write_text(json.dumps(p,indent=2))
    (out/'summary.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
