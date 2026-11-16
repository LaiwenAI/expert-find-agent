"""Frozen CMU Gold profile representations, no test-label parameter tuning."""
import json
import time
from pathlib import Path
import numpy as np
from .cmu_gold import load_annotations,pairwise_loss
from .bounded_profile import BoundedProfile


def run(root='artifacts/public/cmu-gold-upstream',vectors='artifacts/public/cmu-dense',output='artifacts/public/cmu-dense-results'):
    root=Path(root);v=Path(vectors);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((v/'manifest.json').read_text());matrix=np.load(v/'vectors.npy')
    lookup={pid:i for i,pid in enumerate(meta['ids'])}
    profiles={}
    for p in (root/'evaluation_datasets/d_20_1/archives').glob('*.jsonl'):
        profiles[p.stem.lstrip('~')]=[json.loads(line)['id'] for line in p.read_text().splitlines() if line.strip()]
    labels=load_annotations(root/'data/evaluations.csv')
    predictions={method:{} for method in ['max-paper','bounded-max-paper','mean-profile','mean-max-50-50']};start=time.perf_counter()
    evaluated=0;bound_dots=0;full_dots=0;profile_cache={}
    for rid,grades in labels.items():
        for method in predictions:predictions[method][rid]={}
        for pid in grades:
            publications=[x for x in profiles[rid] if x!=pid]
            if not publications:raise ValueError('Empty profile')
            q=matrix[lookup[pid]];p=matrix[[lookup[x] for x in publications]]
            profile_key=tuple(publications)
            if profile_key not in profile_cache:profile_cache[profile_key]=BoundedProfile(p)
            bounded=profile_cache[profile_key].maximum(q)
            evaluated+=bounded['member_dot_products'];bound_dots+=bounded['bound_dot_products'];full_dots+=len(publications)
            maximum=float(np.max(p@q))
            if abs(maximum-bounded['score'])>1e-5:raise ValueError('Bound score mismatch')
            predictions['bounded-max-paper'][rid][pid]=bounded['score']
            centroid=p.mean(axis=0);norm=np.linalg.norm(centroid)
            mean=float(centroid@q/norm) if norm else 0.
            predictions['max-paper'][rid][pid]=maximum
            predictions['mean-profile'][rid][pid]=mean
            predictions['mean-max-50-50'][rid][pid]=.5*(mean+maximum)
    elapsed=time.perf_counter()-start
    summary={method:pairwise_loss(scores,labels) for method,scores in predictions.items()}
    for method,scores in predictions.items():(out/f'{method}-predictions.json').write_text(json.dumps(scores,indent=2))
    report=dict(model=meta['model'],profile='d_20_1',results=summary,local_total_seconds=elapsed,bound_diagnostics={'member_dots':evaluated,'bound_dots':bound_dots,'full_member_dots':full_dots},
        warning='All embeddings precomputed; excludes network and index cost. Fixed 50/50 is exploratory baseline, not tuned on labels. Single profile realization.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report
