"""Post-hoc decomposition of recorded top-five rankings, without model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from expert_search.cof_benchmark import load, precision_metrics
from expert_search.statistics import paired_interval


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(pilot):
    root, _, reviewers, labels = load('SIGIR')
    protocol = json.loads((pilot/'protocol.json').read_text())
    summary = json.loads((pilot/'summary.json').read_text())
    if sha(pilot/'protocol.json') != summary['protocol_sha256'] or sha(pilot/'results.json') != summary['results_sha256']:
        raise ValueError('Pilot files changed')
    if any(sha(root/name) != digest for name,digest in protocol['input_sha256'].items()):
        raise ValueError('Benchmark inputs changed')
    rows = json.loads((pilot/'results.json').read_text())
    if [r['query_id'] for r in rows] != protocol['query_ids'] or any(r['stage'] != 'screen' for r in rows):
        raise ValueError('Require the complete originally planned successful pilot')
    result = {}
    for task in ('soft','hard'):
        diffs=[]; removed=added=0; changes={'improved':0,'unchanged':0,'worsened':0}; common=0; missing=0
        for row in rows:
            qid=row['query_id']; baseline=row['baseline']; screen=row['screened']; truth=labels[task][qid]
            if (len(baseline)>5 or len(screen)>5 or len(set(baseline))!=len(baseline) or len(set(screen))!=len(screen)
                or (set(baseline)|set(screen))-set(reviewers)):
                raise ValueError('Invalid saved ranking')
            b=precision_metrics(baseline,truth)['p@5']; s=precision_metrics(screen,truth)['p@5']
            if abs(b-row['baseline_metrics'][task]['p@5'])>1e-12 or abs(s-row['screened_metrics'][task]['p@5'])>1e-12:
                raise ValueError('Recorded metric mismatch')
            delta=s-b; diffs.append(delta)
            changes['improved' if delta>1e-12 else 'worsened' if delta < -1e-12 else 'unchanged']+=1
            removed+=sum(truth[eid]>0 for eid in set(baseline)-set(screen))
            added+=sum(truth[eid]>0 for eid in set(screen)-set(baseline))
            common+=len(set(baseline)&set(screen)); missing+=5-len(screen)
        if abs((added-removed)/5/len(rows)-sum(diffs)/len(rows))>1e-12:
            raise ValueError('Set decomposition mismatch')
        result[task]={'paired_p_at_5':paired_interval(diffs), 'query_changes':changes,
                      'relevant_baseline_only':removed,'relevant_screen_only':added,
                      'mean_top_five_intersection':common/len(rows),'missing_screen_slots':missing}
    return {'dataset':'CoF SIGIR','queries':len(rows),'analysis':result,
            'protocol_sha256':sha(pilot/'protocol.json'),'results_sha256':sha(pilot/'results.json'),
            'scope':'Post-hoc analysis on reused public queries. Query bootstrap assumes independence and is not corrected for multiple comparisons. Baseline-only means absent from screened top five; its screen label and presence below rank five are unknown. Relevance is reviewer-level, not evidence entailment or enterprise fit.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',type=Path,default=Path('artifacts/cof-progressive-screen-pilot-v1'));parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze(args.pilot); args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise FileExistsError('Choose a new output path')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
