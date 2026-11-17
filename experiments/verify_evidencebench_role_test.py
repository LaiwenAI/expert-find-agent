"""Recompute returned metrics and conservative normalization, without API calls."""
import hashlib
import json
from pathlib import Path
from expert_search.evidencebench_data import aspect_coverage
from expert_search.source_roles import normalize_roles
from expert_search.statistics import clustered_paired_interval


def run():
    root=Path('artifacts/evidencebench-role-test-v1')
    rows=json.loads((root/'evaluation.json').read_text())
    data=json.loads(Path('artifacts/public/evidencebench/test.json').read_text())
    if len(rows)!=len(data) or {r['id'] for r in rows}!=set(data):raise ValueError('Incomplete test results')
    for row in rows:
        raw=json.loads((root/'roles'/(row['id']+'.json')).read_text())
        source=data[row['id']]
        if row['metrics']!=aspect_coverage(source,row['selected']) or row['baseline']!=aspect_coverage(source,row['baseline_selected']):raise ValueError('Metric mismatch')
        if raw['status']=='normalized':
            parsed=json.loads(raw['response']['choices'][0]['message']['content'])
            if normalize_roles(parsed,len(source['paper_as_candidate_pool']))['roles']!=raw['roles']:raise ValueError('Role normalization mismatch')
        elif row['selected']!=row['baseline_selected']:raise ValueError('Failed compilation changed baseline')
    result=clustered_paired_interval([r['metrics']['aspect_recall']-r['baseline']['aspect_recall'] for r in rows],[data[r['id']]['paper_id'] for r in rows])
    (root/'cluster-analysis.json').write_text(json.dumps(result,indent=2))
    verification=dict(records=len(rows),all_metrics_recomputed=True,all_normalized_roles_recomputed=True,all_failed_records_preserve_baseline=True,
                      limitation='Checks saved selections and role references, not model attribution truth or reconstruction of complete CUDA vector ranking.',
                      artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'protocol.json',root/'evaluation.json',root/'summary.json',root/'cluster-analysis-protocol.json',root/'cluster-analysis.json']})
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
