"""Summarize two independent partial CoF translation-review copies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .check_cof_translation_review import check,read


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(packet,first,second):
    check(packet,first);check(packet,second);a,b=read(first),read(second);counts=Counter();first_counts=Counter();second_counts=Counter()
    for case in a:
        x,y=a[case]['equivalence'],b[case]['equivalence'];first_counts[x or 'missing']+=1;second_counts[y or 'missing']+=1
        if x is not None and y is not None and a[case]['reviewer_id']==b[case]['reviewer_id']:raise ValueError('Independent judgments require distinct reviewer IDs')
        if x is None or y is None:counts['missing_at_least_one']+=1
        elif x=='uncertain' or y=='uncertain':counts['uncertain_at_least_one']+=1
        elif x==y:counts['determinate_agree']+=1
        else:counts['determinate_disagree']+=1
    joint=counts['determinate_agree']+counts['determinate_disagree']
    manifest=json.loads((packet/'manifest.json').read_text())
    return dict(cases=len(a),missing_translations=manifest['missing_translation_cases'],joint_determinate=joint,agreement=counts['determinate_agree']/joint if joint else None,counts=dict(counts),first_labels=dict(first_counts),second_labels=dict(second_counts),
                first_sha256=sha(first),second_sha256=sha(second),packet_manifest_sha256=sha(packet/'manifest.json'),
                scope='Agreement on jointly determinate judgments only. Missing and uncertain remain unknown; the missing translation is outside the packet. Reviewer IDs do not prove independent review or qualification. No relevance, capability or population accuracy conclusion follows.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('packet',type=Path);p.add_argument('first',type=Path);p.add_argument('second',type=Path);a=p.parse_args();print(json.dumps(analyze(a.packet,a.first,a.second),indent=2))
