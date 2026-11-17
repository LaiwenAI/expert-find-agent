"""Validate and reconcile independent copies of a private blinded review packet."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.check_prioritized_fresh_review import check,read
from expert_search.review_agreement import reconcile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(packet,first,second,output,adjudication=None):
    reviews=[first,second]+([adjudication] if adjudication else [])
    checks=[check(packet,path) for path in reviews]
    first_rows=read(first);second_rows=read(second);third_rows=read(adjudication) if adjudication else None
    report=reconcile(first_rows,second_rows,third_rows)
    output.mkdir(parents=True,exist_ok=False)
    consensus=report.pop('outcomes');path=output/'consensus.json'
    path.write_text(json.dumps(consensus,indent=2)+'\n')
    source=read(packet/'blind-review.jsonl');merged=[]
    for case in source:
        row=deepcopy(source[case]);state=consensus[case]['status']
        if state=='agreed':row['annotation']=deepcopy(first_rows[case]['annotation'])
        elif state=='adjudicated':row['annotation']=deepcopy(third_rows[case]['annotation'])
        merged.append(row)
    review_path=output/'consensus-review.jsonl'
    review_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
    check(packet,review_path)
    report.update(packet_manifest_sha256=sha(packet/'manifest.json'),review_sha256=[x['review_sha256'] for x in checks],consensus_sha256=sha(path),consensus_review_sha256=sha(review_path))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True)
    p.add_argument('--adjudication',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.packet,a.first,a.second,a.output,a.adjudication)
