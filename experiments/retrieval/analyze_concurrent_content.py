"""Summarize nonempty and empty evidence-span agreement separately."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root,output):
    summary=json.loads((root/'summary.json').read_text());rows_path=root/'private-results.json'
    if sha(root/'protocol.json')!=summary['protocol_sha256'] or sha(rows_path)!=summary['private_results_sha256']:raise ValueError('Changed pilot inputs')
    rows=json.loads(rows_path.read_text());pairs=[]
    if len(rows)!=summary['queries']:raise ValueError('Changed query count')
    for row in rows:
        a=row['arms']['sequential']['decisions'];b=row['arms']['concurrent']['decisions']
        if set(a)!=set(b):raise ValueError('Changed candidates')
        pairs.extend((a[e],b[e]) for e in a if a[e]['status']=='structurally_valid' and b[e]['status']=='structurally_valid')
    if len(pairs)!=summary['jointly_valid_candidates']:raise ValueError('Changed valid count')
    classes=Counter();levels=Counter()
    for a,b in pairs:
        x,y=a['citations'],b['citations'];levels[a['level']+' -> '+b['level']]+=1
        if not x and not y:classes['both_empty']+=1
        elif bool(x)!=bool(y):classes['one_empty']+=1
        elif x==y:classes['equal_nonempty']+=1
        else:classes['different_nonempty']+=1
    report=dict(queries=len(rows),jointly_valid_candidates=len(pairs),same_level=sum(a['level']==b['level'] for a,b in pairs),citation_comparison=dict(classes),level_transitions=dict(levels),
        pilot_summary_sha256=sha(root/'summary.json'),pilot_results_sha256=sha(rows_path),script_sha256=sha(__file__),
        scope='Independent serial and concurrent model outputs on the same candidate sources. Empty citations do not establish evidence support; matching IDs/offsets establish structural agreement only, not entailment or correctness. Order, timing, sampling and concurrency are confounded. Small reused cohort and no human gold.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
