"""Apply shared-label time-integrated bounds to frozen output-budget calls."""
import hashlib
import json
import math
from pathlib import Path
from expert_search.evaluation_gain import paired_time_integrated_actionable_gain_interval


def run():
    root = Path('artifacts/focused-partial-comparison-v1')
    protocol_path = root / 'protocol.json'; protocol = json.loads(protocol_path.read_text())
    jobs_path = Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl')
    labels_path = Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl')
    jobs = {r['id']: r for r in map(json.loads, jobs_path.read_text().splitlines())}
    grades = {}
    for row in map(json.loads, labels_path.read_text().splitlines()):
        grades.setdefault(row['query_id'], {})[row['expert_id']] = row['grade']
    horizon = 10.0; results = []; hashes = {}
    for qid in protocol['query_ids']:
        path = root / (qid + '.json'); value = json.loads(path.read_text())
        hashes[qid] = hashlib.sha256(path.read_bytes()).hexdigest()
        events = {}; complete = {}
        for arm in ('focused', 'full'):
            response = value['arms'][arm]; elapsed = response['seconds']; review = response['review']
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid elapsed time')
            if review['status'] == 'structurally_valid':
                ranking = [entry['expert_id'] for entry in review['result']['ranking']][:3]
                if any(key not in jobs[qid]['candidates'] for key in ranking):
                    raise ValueError('Foreign candidate')
                events[arm] = [(elapsed, ranking)]; complete[arm] = elapsed <= horizon
            else:
                events[arm] = []; complete[arm] = False
        result = paired_time_integrated_actionable_gain_interval(events['focused'], events['full'], grades[qid], horizon, k=3)
        results.append(dict(query_id=qid, contrast=result, completed_by_deadline=complete))
    n = len(results)
    mean = lambda key: sum(r['contrast'][key] for r in results) / n
    report = dict(queries=n, horizon_seconds=horizon, shared_bounds=[mean('lower'), mean('upper')],
                  independent_bounds=[sum(r['contrast']['independent_bounds'][i] for r in results)/n for i in (0, 1)],
                  completed_by_deadline={a:sum(r['completed_by_deadline'][a] for r in results) for a in ('focused','full')},
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), input_sha256=hashes,
                  scope='Post-hoc ten-second request-time horizon on 20 previously studied development queries. Same raw candidate views, same-model silver grades and shared unjudged labels; failed or late arms contribute no completed result before the deadline. No retrieval, initialization, streaming or independent human labels. Bounds are identification intervals, not confidence intervals or proof of expert ability.')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__': run()
