"""Complete-cohort human-rating evaluation with explicit abstention bounds.

Refuses partial runs; pairwise accuracy CIs cluster by exact target-paper title.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from expert_search.merit_suitability import prompts,build_case,gold

ROOT=Path('artifacts/merit-suitability-comparison-v1');DATA=Path('data/public/merit-release-20261117');ARMS=['official-long','short-chat','native-choice']


def ratio(a,b):return a/b if b else None


def metrics(labels,predictions):
    tp=fp=tn=fn=up=un=0
    for actual,pred in zip(labels,predictions):
        if pred is None:
            if actual:up+=1
            else:un+=1
        elif pred==actual:
            if pred:tp+=1
            else:tn+=1
        elif pred:fp+=1
        else:fn+=1
    n=len(labels);p=sum(labels);negative=n-p;valid=tp+fp+tn+fn
    return dict(total=n,valid=valid,unresolved=up+un,tp=tp,fp=fp,tn=tn,fn=fn,unresolved_actual_positive=up,unresolved_actual_negative=un,
                valid_only=dict(accuracy=ratio(tp+tn,valid),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn),balanced_accuracy=(ratio(tp,tp+fn)+ratio(tn,tn+fp))/2 if tp+fn and tn+fp else None),
                full_cohort_bounds=dict(accuracy=[ratio(tp+tn,n),ratio(tp+tn+up+un,n)],precision=[ratio(tp,tp+fp+un),ratio(tp+up,tp+up+fp)],recall=[ratio(tp,p),ratio(tp+up,p)],f1=[ratio(2*tp,2*tp+fp+fn+up+un),ratio(2*(tp+up),2*(tp+up)+fp+fn)],balanced_accuracy=[(tp/p+tn/negative)/2,((tp+up)/p+(tn+un)/negative)/2] if p and negative else None))


def bootstrap_cluster(values,clusters):
    grouped=defaultdict(list)
    for v,c in zip(values,clusters):grouped[c].append(v)
    keys=sorted(grouped);sums=np.asarray([sum(grouped[k]) for k in keys]);sizes=np.asarray([len(grouped[k]) for k in keys]);rng=np.random.default_rng(20261117)
    samples=[]
    # Bounded batches avoid allocating a10000x509 array unnecessarily.
    for _ in range(100):
        indices=rng.integers(0,len(keys),size=(100,len(keys)));samples.extend((sums[indices].sum(axis=1)/sizes[indices].sum(axis=1)).tolist())
    return dict(mean=float(np.mean(values)),target_paper_clusters=len(keys),paper_cluster_bootstrap95=list(map(float,np.quantile(samples,[.025,.975]))))


def run():
    protocol=json.loads((ROOT/'protocol.json').read_text());source=DATA/'MERIT-Assessor/test/test.json'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=protocol['dataset_sha256']:raise ValueError('Changed released test data')
    prompt_path=DATA/'MERIT-Assessor/test/prompts.py'
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest()!=protocol['official_prompts_sha256']:raise ValueError('Changed official prompt asset')
    official=prompts(prompt_path);rows=json.loads(source.read_text());cases=[build_case(r,i,official) for i,r in enumerate(rows)];labels=[gold(r) for r in rows]
    if len(cases)!=610:raise ValueError('Wrong cohort')
    predictions={a:[] for a in ARMS};inputs={}
    for case in cases:
        for arm in ARMS:
            path=ROOT/'predictions'/(case['id']+'-'+arm+'.json')
            if not path.exists():raise RuntimeError('Incomplete610pair run: missing '+path.name)
            value=json.loads(path.read_text())
            if value['input_sha256']!=case['input_sha256'] or value['arm']!=arm or value['case_id']!=case['id']:raise ValueError('Prediction provenance mismatch')
            if value['status'] not in ('valid','unresolved') or not math.isfinite(value['seconds']) or value['seconds']<0:raise ValueError('Invalid prediction state/time')
            if value['status']=='valid' and (type(value['label']) is not int or value['label'] not in (0,1)):raise ValueError('Invalid binary label')
            if value['status']=='unresolved' and value['label'] is not None:raise ValueError('Failed label imputation')
            predictions[arm].append(value);inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    arms={}
    for arm,values in predictions.items():
        times=[v['seconds'] for v in values];usages=[v['response'].get('usage') or {} for v in values if 'response' in v];tokens=[u['completion_tokens'] for u in usages if type(u.get('completion_tokens')) is int]
        arms[arm]=dict(metrics(labels,[v['label'] for v in values]),mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),empirical_p95_seconds=float(np.quantile(times,.95)),mean_completion_tokens_observed=float(np.mean(tokens)) if tokens else None,token_observed_requests=len(tokens))
    paired={}
    for a,b in [('short-chat','official-long'),('native-choice','official-long'),('native-choice','short-chat')]:
        indices=[i for i in range(len(cases)) if predictions[a][i]['status']=='valid' and predictions[b][i]['status']=='valid'];differences=[int(predictions[a][i]['label']==labels[i])-int(predictions[b][i]['label']==labels[i]) for i in indices];clusters=[cases[i]['paper_cluster'] for i in indices];time_diffs=[predictions[a][i]['seconds']-predictions[b][i]['seconds'] for i in range(len(cases))]
        pair=dict(common_valid_pairs=len(indices),accuracy_difference_on_common_valid=bootstrap_cluster(differences,clusters) if indices else None,latency_difference_all_attempts=bootstrap_cluster(time_diffs,[c['paper_cluster'] for c in cases]),full_cohort_accuracy_difference_bounds=[arms[a]['full_cohort_bounds']['accuracy'][0]-arms[b]['full_cohort_bounds']['accuracy'][1],arms[a]['full_cohort_bounds']['accuracy'][1]-arms[b]['full_cohort_bounds']['accuracy'][0]])
        paired[a+'-minus-'+b]=pair
    report=dict(pairs=len(cases),positive_self_ratings=sum(labels),negative_self_ratings=len(labels)-sum(labels),exact_target_paper_clusters=len({c['paper_cluster'] for c in cases}),arms=arms,paired=paired,predictions_sha256=inputs,protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='All610released human-self-rated reviewer/paper test pairs. Rating4/5 positive, others negative; no labels in inference. Unresolved states are abstentions, not0. Identification bounds are not confidence intervals. Accuracy bootstrap clusters by normalized exact target title only, not semantic duplicates or unknown reviewer identity; no guaranteed pretraining non-contamination. All armsgpt-6-luna, not trained MERITcheckpoint. Auxiliary suitability classification, not enterprise transfer/ownership/citation gold or whole expert-search E2E.')
    (ROOT/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='predictions_sha256'},indent=2))


if __name__=='__main__':run()
