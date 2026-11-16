"""Compare completed CMU evidence requests to identical reviewers and papers."""
import json
from pathlib import Path
from .cmu_gold import load_annotations,pairwise_loss


def run(output='docs/exploration/cmu-evidence-comparison.json'):
    root=Path('artifacts/public');labels=load_annotations(root/'cmu-gold/data/evaluations.csv')
    files=sorted((root/'cmu-evidence-results').glob('*.json'));rows={p.stem:json.loads(p.read_text()) for p in files if p.name!='summary.json'}
    scoped={rid:labels[rid] for rid in rows};model={};failures=[]
    for rid,row in rows.items():
        if row['status']=='ok':model[rid]=row['predictions']
        else:
            # Tied prediction on failure avoids silently dropping failed cases.
            model[rid]={pid:0. for pid in labels[rid]};failures.append(rid)
    if not scoped:raise ValueError('No completed request artifacts')
    results={'model-top3':pairwise_loss(model,scoped)}
    sources={'lexical':root/'cmu-results/lexical-predictions.json','dense-max':root/'cmu-dense-results/max-paper-predictions.json',
             'dense-mean':root/'cmu-dense-results/mean-profile-predictions.json','dense-mixture':root/'cmu-dense-results/mean-max-50-50-predictions.json'}
    for name,path in sources.items():
        scores=json.loads(path.read_text());results[name]=pairwise_loss({rid:scores[rid] for rid in scoped},scoped)
    times=[sum(t['seconds'] for t in r.get('trace',[])) for r in rows.values() if r['status']=='ok']
    report=dict(completed_reviewers=len(rows),reviewer_ids=sorted(rows),failures=failures,results=results,
        model_call_seconds=times,mean_model_call_seconds=sum(times)/len(times) if times else None,
        scope='Only completed requests, same reviewer IDs across all methods; partial until requested experiment finishes.',
        failure_policy='Failed request predictions tied at zero; reports actual missing coverage. Does not measure query embedding latency.')
    Path(output).write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
