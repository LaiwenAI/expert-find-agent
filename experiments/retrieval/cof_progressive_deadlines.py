"""Replay recorded progressive screening at fixed deadlines; no model calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from expert_search.cof_benchmark import load, precision_metrics
from expert_search.evaluation_gain import latest_gain_interval, time_integrated_gain_interval
from expert_search.statistics import paired_interval


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(pilot,deadlines=(0.01,0.5,1.,2.,5.,10.)):
    root,_,reviewers,labels=load('SIGIR');summary=json.loads((pilot/'summary.json').read_text()); protocol=json.loads((pilot/'protocol.json').read_text())
    for name,key in [('protocol.json','protocol_sha256'),('results.json','results_sha256'),('candidate-snapshots.json','candidate_snapshots_sha256')]:
        if sha(pilot/name)!=summary[key]:raise ValueError('Changed pilot artifact')
    if any(sha(root/name)!=digest for name,digest in protocol['input_sha256'].items()):raise ValueError('Changed benchmark data')
    rows=json.loads((pilot/'results.json').read_text())
    if [r['query_id'] for r in rows]!=protocol['query_ids']:raise ValueError('Incomplete query cohort')
    if any(type(d) not in (int,float) or not math.isfinite(d) or d<=0 for d in deadlines) or list(deadlines)!=sorted(set(deadlines)):raise ValueError('Invalid deadlines')
    traces=[]
    for row in rows:
        retrieval=row['retrieval_seconds'];screen=row['screen_seconds'];baseline=row['baseline'];qid=row['query_id']
        if any(not isinstance(x,(float,int)) or not math.isfinite(x) or x<0 for x in (retrieval,screen)):raise ValueError('Invalid event time')
        if len(set(baseline))!=len(baseline) or len(baseline)>5 or set(baseline)-set(reviewers):raise ValueError('Invalid baseline')
        if row['stage']=='screen':
            ranking=row['screened']
            if len(set(ranking))!=len(ranking) or len(ranking)>5 or set(ranking)-set(reviewers):raise ValueError('Invalid screen')
        else:ranking=None
        traces.append((qid,retrieval,screen,baseline,ranking))
    table=[]
    for deadline in deadlines:
        values={task:{'baseline':[],'progressive':[]} for task in labels}; initial=finished=0; returned=[]
        for qid,retrieval,screen,base,rank in traces:
            initial+=retrieval<=deadline; finished+=rank is not None and retrieval+screen<=deadline
            returned.append(len(rank) if rank is not None and retrieval+screen<=deadline else len(base) if retrieval<=deadline else 0)
            for task in labels:
                b=precision_metrics(base,labels[task][qid])['p@5'];event=[(retrieval,b,b)]
                if rank is not None:
                    s=precision_metrics(rank,labels[task][qid])['p@5'];event.append((retrieval+screen,s,s))
                values[task]['baseline'].append(latest_gain_interval(event[:1],deadline)['lower'])
                values[task]['progressive'].append(latest_gain_interval(event,deadline)['lower'])
        table.append({'deadline_seconds':deadline,'initial_available':initial,'screen_available':finished,'mean_returned':statistics.mean(returned),
                      'p_at_5':{t:{a:statistics.mean(values[t][a]) for a in ('baseline','progressive')} for t in labels}})
    integrated={}
    for task in labels:
        diffs=[];base_scores=[];progressive_scores=[]
        for qid,retrieval,screen,base,rank in traces:
            b=precision_metrics(base,labels[task][qid])['p@5'];events=[(retrieval,b,b)]
            if rank is not None:
                s=precision_metrics(rank,labels[task][qid])['p@5'];events.append((retrieval+screen,s,s))
            a=time_integrated_gain_interval(events[:1],deadlines[-1])['lower'];v=time_integrated_gain_interval(events,deadlines[-1])['lower']
            base_scores.append(a);progressive_scores.append(v);diffs.append(v-a)
        integrated[task]={'baseline':statistics.mean(base_scores),'progressive':statistics.mean(progressive_scores),'paired_difference':paired_interval(diffs)}
    return {'queries':len(rows),'deadlines':table,'integrated_through_seconds':deadlines[-1],'time_average_p_at_5':integrated,
            'protocol_sha256':summary['protocol_sha256'],'results_sha256':summary['results_sha256'],
            'scope':'Post-hoc replay on reused fully judged reviewer queries: per-query recorded retrieval plus screening durations are added. Retrieval and screen were measured in different phases, so this is not a measured streaming session. No explanation, initialization, queueing between phases or network variability is included. Latest complete result replaces the prior snapshot even if relevance falls. Reviewer topic relevance is not evidence support or industrial fit. Bootstrap assumes independent queries and is unadjusted.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=analyze(a.pilot);a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Choose a new output path')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
