"""Separate returned reviewer candidates from model-positive choices."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from expert_search.cof_benchmark import load


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows,labels):
    valid=[r for r in rows if r['status'] in ('complete','partial','empty')];positive={'direct_application','transferable_method'}
    if not valid:raise ValueError('No completed decisions')
    totals=Counter();scores={t:[] for t in labels}
    for row in valid:
        if len(row['ranking'])!=len(row['levels']) or len(set(row['ranking']))!=len(row['ranking']) or len(row['ranking'])>3:raise ValueError('Invalid ranking')
        for task in labels:
            truth=labels[task][row['query_id']]
            if set(row['ranking'])-set(truth):raise ValueError('Unjudged reviewer')
            scores[task].append(sum(truth[e]>0 and level in positive for e,level in zip(row['ranking'],row['levels']))/3)
        totals.update(row['levels']);totals['returned']+=len(row['ranking']);totals['model_positive']+=sum(level in positive for level in row['levels'])
    return {'valid_queries':len(valid),'counts':dict(totals),'mean_model_positive_returned':totals['model_positive']/len(valid),
            'model_positive_reviewer_p_at_3':{t:statistics.mean(values) for t,values in scores.items()},
            'scope':'Post-hoc model-positive gate at fixed denominator three; reviewer labels concern whole profiles and do not validate source spans or model choice. Unknown/insufficient is not actual inability. Valid-only, reused development-like sample.'}


def main(pilot,output):
    summary=json.loads((pilot/'summary.json').read_text());protocol=json.loads((pilot/'protocol.json').read_text())
    if sha(pilot/'protocol.json')!=summary['protocol_sha256'] or sha(pilot/'results.json')!=summary['results_sha256']:raise ValueError('Changed pilot')
    root,_,_,labels=load('SIGIR')
    if any(sha(root/name)!=h for name,h in protocol['input_sha256'].items()):raise ValueError('Changed benchmark')
    rows=json.loads((pilot/'results.json').read_text())
    if [r['query_id'] for r in rows]!=protocol['query_ids']:raise ValueError('Changed cohort')
    report=summarize(rows,labels);report.update(pilot_protocol_sha256=sha(pilot/'protocol.json'),pilot_results_sha256=sha(pilot/'results.json'))
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Choose a new output path')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.pilot,a.output)
