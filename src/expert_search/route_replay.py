"""Label-blind profile-disagreement routing replay; not fresh online timing.
Uses fixed threshold 0.25 and whole-reviewer requests identical to recorded calls.
"""
import json
from itertools import combinations
from pathlib import Path
from .cmu_gold import load_annotations,pairwise_loss


def replay(threshold=.25,output='docs/exploration/route-replay.json'):
    root=Path('artifacts/public');labels=load_annotations(root/'cmu-gold/data/evaluations.csv')
    maximum=json.loads((root/'cmu-dense-results/max-paper-predictions.json').read_text());mean=json.loads((root/'cmu-dense-results/mean-profile-predictions.json').read_text())
    rows={p.stem:json.loads(p.read_text()) for p in (root/'cmu-evidence-results').glob('*.json') if p.name!='summary.json'}
    scoped={rid:labels[rid] for rid in rows};pred={};routes=[];seconds=0
    for rid,row in rows.items():
        papers=sorted(mean[rid]);pairs=list(combinations(papers,2))
        disagreements=sum((maximum[rid][a]-maximum[rid][b])*(mean[rid][a]-mean[rid][b])<0 for a,b in pairs)
        ratio=disagreements/len(pairs) if pairs else 0
        use_model=ratio>threshold
        if use_model and row['status']=='ok':
            pred[rid]=row['predictions'];seconds+=sum(t['seconds'] for t in row['trace'])
        else:pred[rid]=mean[rid]
        routes.append(dict(reviewer=rid,disagreement_fraction=ratio,use_model=use_model,request_status=row['status']))
    result=dict(threshold=threshold,completed_reviewers=len(rows),routed_requests=sum(r['use_model'] for r in routes),
        metrics=pairwise_loss(pred,scoped),mean_baseline=pairwise_loss({rid:mean[rid] for rid in scoped},scoped),recorded_selected_model_seconds=seconds,routes=routes,
        warning='Offline replay on completed reviewer requests. Does not prove actual online savings, robustness, or end-to-end query performance. Threshold exploratory and fixed, not optimized on gold labels.')
    Path(output).write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    r=replay();print({k:v for k,v in r.items() if k!='routes'})
