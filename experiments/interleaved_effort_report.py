"""Summarize the fixed six-reviewer, two-repeat paired transport experiment."""
import json
import random
import statistics
from pathlib import Path
from expert_search.cmu_gold import load_annotations, pairwise_loss


def report():
    root=Path('artifacts/public/cmu-interleaved-effort')
    protocol=json.loads((root/'protocol.json').read_text())
    labels=load_annotations('artifacts/public/cmu-gold/data/evaluations.csv')
    rows=[];conditions={}
    for effort in protocol['conditions']:
        times=[];tokens=[];repeat_metrics=[];failures=[]
        for repeat in range(protocol['repeats']):
            predictions={}
            for rid in protocol['reviewers']:
                path=root/f'r{repeat}'/rid/effort/(rid+'.json')
                if not path.exists():raise ValueError(f'Missing result: {path}')
                result=json.loads(path.read_text())
                if result['status']=='ok':
                    predictions[rid]=result['predictions']
                    seconds=sum(t['seconds'] for t in result['trace'])
                    times.append(seconds)
                    tokens.extend(t.get('reasoning_tokens') for t in result['trace'])
                else:
                    predictions[rid]={pid:0. for pid in labels[rid]}
                    failures.append(dict(repeat=repeat,reviewer=rid));seconds=None
                rows.append(dict(reviewer=rid,repeat=repeat,effort=effort,seconds=seconds,
                    loss=pairwise_loss({rid:predictions[rid]},{rid:labels[rid]})['weighted_pairwise_loss']))
            repeat_metrics.append(pairwise_loss(predictions,{rid:labels[rid] for rid in protocol['reviewers']}))
        conditions[effort]=dict(repeat_metrics=repeat_metrics,
            mean_loss=statistics.mean(m['weighted_pairwise_loss'] for m in repeat_metrics),
            mean_model_seconds=statistics.mean(times) if times else None,
            median_model_seconds=statistics.median(times) if times else None,
            reported_reasoning_tokens=tokens,failures=failures)
    differences=[]
    for rid in protocol['reviewers']:
        means={e:statistics.mean(r['loss'] for r in rows if r['reviewer']==rid and r['effort']==e)
               for e in protocol['conditions']}
        differences.append(means['none']-means['medium'])
    rng=random.Random(42)
    bootstrap=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(5000))
    result=dict(protocol=protocol,conditions=conditions,rows=rows,
        macro_reviewer_loss_difference_none_minus_medium=statistics.mean(differences),
        reviewer_bootstrap_95_interval=[bootstrap[124],bootstrap[4874]],
        warning='Six reviewers only, two stochastic repeats; interleaving reduces temporal confounding but does not establish general quality equivalence. Model-call latency excludes retrieval and input construction. Missing/failed outputs count as tied predictions.')
    Path('docs/exploration/interleaved-effort-results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','protocol')},indent=2))


if __name__=='__main__':report()
