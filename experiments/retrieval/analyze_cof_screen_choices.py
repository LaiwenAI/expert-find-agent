"""Analyze recorded screen labels against reviewer-level judgments post hoc."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from expert_search.cof_benchmark import load, precision_metrics
from expert_search.statistics import paired_interval


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(pilot):
    root,_,reviewers,labels=load('SIGIR')
    protocol=json.loads((pilot/'protocol.json').read_text()); summary=json.loads((pilot/'summary.json').read_text())
    if any(sha(pilot/name)!=summary[key] for name,key in [('protocol.json','protocol_sha256'),('results.json','results_sha256'),('candidate-snapshots.json','candidate_snapshots_sha256')]):
        raise ValueError('Pilot artifact hash mismatch')
    if any(sha(root/name)!=digest for name,digest in protocol['input_sha256'].items()):raise ValueError('Benchmark input mismatch')
    rows=json.loads((pilot/'results.json').read_text()); snapshots=json.loads((pilot/'candidate-snapshots.json').read_text())
    if [r['query_id'] for r in rows]!=protocol['query_ids'] or set(snapshots)!=set(protocol['query_ids']):raise ValueError('Incomplete query cohort')
    if any(r['stage']!='screen' for r in rows):raise ValueError('Incomplete screening; failures must not be treated as labels')
    grades={'direct_application':0,'transferable_method':1}; allowed=set(grades)|{'topic_only','insufficient',None}
    out={}
    for task in ('soft','hard'):
        categories=Counter(); choice_by_truth=Counter(); diffs=[]; overlap=0
        for row in rows:
            judgments=row['judgments']; ids=[j['expert_id'] for j in judgments]; choices=[j['label'] for j in judgments]
            if len(ids)!=len(set(ids)) or set(ids)-set(reviewers) or any(c not in allowed for c in choices):raise ValueError('Invalid recorded decisions')
            if ids[:5]!=row['baseline']:raise ValueError('Baseline differs from judgment order')
            expected=[ids[i] for i in sorted((i for i,c in enumerate(choices) if c in grades),key=lambda i:(grades[choices[i]],i))[:5]]
            if expected!=row['screened']:raise ValueError('Recorded ranking inconsistent with choices')
            truth=labels[task][row['query_id']]; screened=set(row['screened']); overlap+=len(set(row['baseline'])&screened)
            for eid,choice in zip(ids,choices):choice_by_truth[(str(choice), 'relevant' if truth[eid]>0 else 'nonrelevant')]+=1
            for eid,choice in zip(ids[:5],choices[:5]):
                if truth[eid]>0 and eid not in screened:
                    categories['not_screened_no_evidence' if choice is None else 'screen_negative' if choice not in grades else 'positive_below_five']+=1
            b=precision_metrics(row['baseline'],truth)['p@5'];s=precision_metrics(row['screened'],truth)['p@5']
            if abs(b-row['baseline_metrics'][task]['p@5'])>1e-12 or abs(s-row['screened_metrics'][task]['p@5'])>1e-12:raise ValueError('Metric mismatch')
            diffs.append(s-b)
        out[task]={'paired_p_at_5':paired_interval(diffs),'relevant_baseline_top_five_absences':dict(categories),
                   'choice_by_reviewer_label':{f'{choice}|{truth}':count for (choice,truth),count in sorted(choice_by_truth.items())},
                   'mean_top_five_intersection':overlap/len(rows)}
    return {'queries':len(rows),'analysis':out,'protocol_sha256':summary['protocol_sha256'],'results_sha256':summary['results_sha256'],
            'scope':'Post-hoc repeat on previously studied public reviewer queries. Screen choices pertain to one publication view; reviewer labels pertain to the reviewer, not necessarily that view. Counts do not establish decision error, individual contribution or industrial expert fit. Bootstrap assumes independent queries, with no multiple-comparison adjustment.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze(args.pilot);args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Choose a new output path')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
