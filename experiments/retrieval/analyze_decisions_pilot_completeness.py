"""Audit the returned-result completeness of the Decisions pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    protocol=json.loads((root/'protocol.json').read_text()); summary=json.loads((root/'summary.json').read_text())
    if sha(root/'protocol.json')!=summary['protocol_sha256']:raise ValueError('Changed pilot protocol')
    counts=Counter();labels=Counter();returned=rejected=capacity=0;attempt_hashes={}
    for qid in protocol['query_ids']:
        path=root/(qid+'.json');row=json.loads(path.read_text());attempt_hashes[qid]=sha(path)
        if row['query_id']!=qid:raise ValueError('Changed query ID')
        labels.update(row.get('decision_labels',{}).values())
        if row['status']=='abstained':counts['screen_abstained']+=1;continue
        if row['status']!='valid':counts[row['status']]+=1;continue
        result=row['result'];selected=row['selected_candidates'];wanted=min(protocol['explanation_k'],selected)
        if len(result['results'])!=row['returned'] or len(result['candidate_ids'])!=selected or row['rejected']!=result['rejected_results']:
            raise ValueError('Changed explanation counts')
        if len(result['results'])>wanted:raise ValueError('Too many outputs')
        for r in result['results']:
            if not r['evidence'] or any(e['expert_id']!=r['expert']['id'] or e['text'][e['start']:e['end']]!=e['quote'] for e in r['evidence']):
                raise ValueError('Invalid quote anchor')
        n=len(result['results']);counts['empty' if n==0 else 'complete' if n==wanted else 'partial']+=1
        returned+=n;rejected+=row['rejected'];capacity+=wanted
    report={'queries':len(protocol['query_ids']),'completion_states':dict(sorted(counts.items())),'decision_labels':dict(sorted(labels.items())),
            'requested_explanation_slots':capacity,'returned_anchored_results':returned,'rejected_results':rejected,
            'input_sha256':{'protocol':sha(root/'protocol.json'),'summary':sha(root/'summary.json'),'attempts':attempt_hashes},
            'code_sha256':sha(Path(__file__)),
            'scope':'Post-hoc structural audit of four previously studied development needs. Empty results can be a model abstention; no missing result is fabricated. Exact quote and ownership checks do not establish entailment, independent expert fit, personal contribution or reason for omission. Model decisions and returned candidates are unjudged.'}
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
