"""Report fixed-cohort fusion sensitivity to known duplicate and screen status."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from expert_search.statistics import paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    root=Path('artifacts/cof-remaining-equal-budget-v1');summary=json.loads((root/'summary.json').read_text());results=root/'results.json'
    if sha(results)!=summary['results_sha256']:raise ValueError('Changed evaluation rows')
    overlap=Path('artifacts/cof-exact-overlap-audit-v1/summary.json');audit=json.loads(overlap.read_text());excluded=set(audit['fields']['abstract']['matched_query_ids'])
    rows=json.loads(results.read_text())
    if len(rows)!=summary['queries'] or len({r['query_id'] for r in rows})!=len(rows) or not excluded<=set(r['query_id'] for r in rows):raise ValueError('Changed query inventory or known overlap')
    def aggregate(items):
        if not items:raise ValueError('Empty cohort')
        return {task:{metric:dict(english=statistics.mean(r['metrics'][task]['english'][metric] for r in items),rrf=statistics.mean(r['metrics'][task]['rrf'][metric] for r in items),difference=paired_interval([r['metrics'][task]['rrf'][metric]-r['metrics'][task]['english'][metric] for r in items])) for metric in ('candidate_recall','p5','ndcg10')} for task in ('soft','hard')}
    groups={'all':rows,'without_exact_abstract_overlap':[r for r in rows if r['query_id'] not in excluded],'same_model_screened':[r for r in rows if r['status']=='model_screened']}
    report=dict(cohorts={k:dict(queries=len(v),metrics=aggregate(v)) for k,v in groups.items()},excluded_query_count=len(excluded),input_sha256={str(p):sha(p) for p in (root/'summary.json',results,overlap)},script_sha256=sha(__file__),
        scope='Post-hoc sensitivity analyses on the remaining globally explored CoF queries, not a sealed evaluation. Exact abstract overlap exclusion cannot rule out other leakage. Same-model screen is not independent translation verification; conditioning on it changes selection. Query bootstrap assumes independence and is unadjusted for multiple analyses. Reviewer labels do not establish enterprise fit or source entailment.')
    output.mkdir(parents=True,exist_ok=False);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
