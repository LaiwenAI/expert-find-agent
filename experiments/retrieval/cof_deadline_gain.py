"""Apply time-integrated gain to a fully judged public reviewer-ranking proxy."""
import hashlib
import itertools
import json
import statistics
from pathlib import Path

from expert_search.cof_benchmark import load
from expert_search.cof_packing_experiment import CONDITIONS
from expert_search.evaluation_gain import actionable_gain_interval,time_integrated_gain_interval
from expert_search.packing_analysis import analyze


def main():
    dataset='SIGIR';serialization='shared';horizon=30.
    validated=analyze(dataset,serialization)
    if not validated['complete']:raise ValueError('Incomplete underlying comparison')
    root,_,_,labels=load(dataset);folder=root/'packing-shared-v2'
    protocol=json.loads((folder/'protocol.json').read_text());qids=protocol['all_query_ids']
    output={condition:{task:[] for task in labels} for condition in CONDITIONS};arrivals={condition:[] for condition in CONDITIONS}
    hashes={}
    for qid,condition in itertools.product(qids,CONDITIONS):
        path=folder/(qid+'-'+condition+'.json');data=path.read_bytes();hashes[path.name]=hashlib.sha256(data).hexdigest();row=json.loads(data)
        seconds=sum(row[name] for name in ('candidate_scoring_seconds','preparation_seconds','rate_wait_seconds','model_call_seconds'))
        arrivals[condition].append(seconds)
        for task in labels:
            grades={eid:(2 if score>0 else 0) for eid,score in labels[task][qid].items()}
            gain=actionable_gain_interval(row['ranking'],grades,k=5)
            if gain['lower']!=gain['upper']:raise ValueError('Unexpected unjudged candidate')
            output[condition][task].append(time_integrated_gain_interval([(seconds,gain['lower'],gain['upper'])],horizon)['lower'])
    summary={'queries':len(qids),'horizon_seconds':horizon,
             'conditions':{c:{'mean_arrival_seconds':statistics.mean(arrivals[c]),'completed_by_horizon':sum(t<=horizon for t in arrivals[c]),
                              'mean_time_integrated_gain':{task:statistics.mean(output[c][task]) for task in labels}} for c in CONDITIONS},
             'input_sha256':{'protocol':hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
                             **{task:hashlib.sha256((root/f'{dataset}_queries_test_{task}.json').read_bytes()).hexdigest() for task in labels}},
             'record_sha256':hashes,
             'scope':'Exploratory post-hoc 30-second horizon; original public reviewer-match labels, not evidence support or enterprise expert fit. Summed measured components include candidate scoring, preparation, rate wait and model request, but omit initialization, query encoding and other overhead. Failed requests use the experiment coarse fallback after failure; same original candidate sets and protocols, no new model calls. No statistical significance or end-to-end SLA claim.'}
    out=Path('artifacts/cof-deadline-gain-v1');out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='record_sha256'},indent=2))


if __name__=='__main__':main()
