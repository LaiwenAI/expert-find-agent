"""Summarize two independently completed intent-pair review copies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .export_intent_diversity_review import check,read_jsonl


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root,first,second):
    check(root,first);check(root,second)
    a,b=read_jsonl(first),read_jsonl(second)
    fields=('chinese_relation','english_relation');out={}
    for field in fields:
        counts=Counter();first_labels=Counter();second_labels=Counter()
        for case in a:
            x,y=a[case][field],b[case][field];first_labels[x or 'missing']+=1;second_labels[y or 'missing']+=1
            if x is not None and y is not None:
                if a[case]['reviewer_id']==b[case]['reviewer_id']:raise ValueError('Two judgments must use different reviewer IDs')
            if x is None or y is None:counts['missing_at_least_one']+=1
            elif x=='uncertain' or y=='uncertain':counts['uncertain_at_least_one']+=1
            elif x==y:counts['determinate_agree']+=1
            else:counts['determinate_disagree']+=1
        n=counts['determinate_agree']+counts['determinate_disagree']
        out[field]=dict(joint_determinate=n,agreement=counts['determinate_agree']/n if n else None,counts=dict(counts),first_labels=dict(first_labels),second_labels=dict(second_labels))
    return dict(cases=len(a),fields=out,first_review_sha256=sha(first),second_review_sha256=sha(second),export_manifest_sha256=sha(root/'manifest.json'),
        scope='Agreement among jointly determinate independent reviewer IDs only; neither missing nor uncertain is negative. Independence of review process and qualification are not mechanically verifiable. Targeted high-overlap packet is not population-representative; adjudication remains separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--root',type=Path,default=Path('artifacts/intent-diversity-human-review-v1'));a=p.parse_args();print(json.dumps(analyze(a.root,a.first,a.second),indent=2))
