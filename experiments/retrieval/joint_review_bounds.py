"""Bound paired source-scoped P@5 differences without imputing missing reviews."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def records(path):return [json.loads(line) for line in path.read_text().splitlines()]


def joint_label(fields):
    """Positive requires both fields; unsupported is scoped to displayed view."""
    labels=[fields[f]['label'] for f in ('method_fit','constraint_support')]
    if any(v not in (None,'uncertain','supported','not_supported_by_view') for v in labels):raise ValueError('Invalid reconciled labels')
    if 'not_supported_by_view' in labels:return 0
    if labels==['supported','supported']:return 1
    if all(v in (None,'uncertain','supported') for v in labels):return None
    raise ValueError('Invalid reconciled labels')


def calculate(first,second,judgments,k=5):
    """Sharp shared-label bounds for a mean fixed-denominator paired difference."""
    if type(k)!=int or k<1 or not first or set(first)!=set(second):raise ValueError('Invalid rankings')
    n=len(first);coefficients={}
    for q in first:
        for ranking,sign in ((first[q],1),(second[q],-1)):
            if not isinstance(ranking,list) or len(ranking)<k or len(ranking)!=len(set(ranking)):
                raise ValueError('Invalid ranking')
            for e in ranking[:k]:coefficients[q,e]=coefficients.get((q,e),0)+sign/(k*n)
    known=0.;low=0.;high=0.;unknown=0
    for cell,c in coefficients.items():
        if abs(c)<1e-15:continue
        label=judgments.get(cell)
        if label is None:
            unknown+=1;low+=min(c,0);high+=max(c,0)
        elif type(label) is int and label in (0,1):known+=c*label
        else:raise ValueError('Invalid judgment')
    return {'lower':known+low,'upper':known+high,'unknown_nonzero_cells':unknown,'queries':n,'k':k}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('packet',type=Path);p.add_argument('reconciled',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    packet=json.loads((a.packet/'manifest.json').read_text());recon=json.loads((a.reconciled/'summary.json').read_text())
    mapping_path=a.packet/'private-mapping.jsonl';outcomes_path=a.reconciled/'private-outcomes.json'
    if sha(mapping_path)!=packet['outputs_sha256']['private-mapping.jsonl'] or sha(a.packet/'manifest.json')!=recon['packet_sha256'] or sha(outcomes_path)!=recon['outcomes_sha256']:
        raise ValueError('Changed packet or reconciliation')
    mapping=records(mapping_path);outcomes=json.loads(outcomes_path.read_text())
    if len(mapping)!=packet['candidate_views'] or {m['case_id'] for m in mapping}!=set(outcomes):raise ValueError('Changed cases')
    labels={}
    for m in mapping:
        cell=(m['query_id'],m['expert_id'])
        if cell in labels:raise ValueError('Duplicate query-expert')
        labels[cell]=joint_label(outcomes[m['case_id']])
    root=Path('artifacts/dataset-release/fresh-enterprise-eval-v1');audit=Path('artifacts/fresh-lexical-pool-audit-v1/runs.jsonl')
    for name in ('pools.jsonl',):
        if sha(root/name)!=packet['inputs_sha256'][name]:raise ValueError('Changed frozen pool')
    if sha(audit)!=packet['inputs_sha256']['lexical_runs']:raise ValueError('Changed lexical run')
    dense={r['query_id']:r['dense_order'] for r in records(root/'pools.jsonl') if r['split']=='development'}
    lexical={r['query_id']:[x['expert_id'] for x in r['ranking']] for r in records(audit) if r['variant']=='full-demand'}
    if set(dense)!=set(lexical):raise ValueError('Cohort mismatch')
    if any((q,e) not in labels for q in dense for e in dense[q][:5]+lexical[q][:5]):raise ValueError('Ranked candidate missing review view')
    result={'dense_minus_lexical':calculate(dense,lexical,labels),'candidate_views':len(labels),
            'joint_known_positive':sum(v==1 for v in labels.values()),'joint_known_unsupported':sum(v==0 for v in labels.values()),
            'joint_unknown':sum(v is None for v in labels.values()),
            'packet_sha256':sha(a.packet/'manifest.json'),'reconciliation_sha256':sha(a.reconciled/'summary.json'),'script_sha256':sha(Path(__file__)),
            'scope':'Identification bounds on source-scoped conjunction of method and constraint support in displayed candidate views, not expert ability, complete corpus recall, confidence intervals or causal accuracy. One unsupported field gives zero for this explicitly scoped indicator; missing/uncertain fields stay unknown. The shared label for identical query/expert/view cancels in the paired difference. Reviews and provenance must remain bound to this exact packet; no old judgments are imported.'}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
