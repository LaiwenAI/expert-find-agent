"""Recompute shared-label bounds after validated independent reviews."""
import argparse
from collections import defaultdict
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.evaluation_gain import actionable_gain_interval, paired_actionable_gain_interval
from experiments.check_prioritized_fresh_review import check, read


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay(grades, mapping, reviewed):
    updated = {qid: dict(values) for qid, values in grades.items()}; seen = set(); seen_views = set(); applied = 0
    for entry in mapping:
        case, qid, expert = entry['case_id'], entry['query_id'], entry['expert_id']
        key = (qid, expert)
        if case in seen or key in seen_views or qid not in updated or expert not in updated[qid] or updated[qid][expert] is not None:
            raise ValueError('Duplicate or non-unjudged mapped view')
        seen.add(case); seen_views.add(key)
        grade = reviewed[case]['annotation']['grade']
        if grade is not None:
            updated[qid][expert] = grade; applied += 1
    if seen != set(reviewed): raise ValueError('Changed case mapping')
    return updated, applied


def bounds(rankings, grades):
    values = [paired_actionable_gain_interval(rankings[qid]['qwen-paper-max'], rankings[qid]['bm25-max'], grades[qid], 5) for qid in sorted(rankings)]
    return {'lower': statistics.mean(v['lower'] for v in values), 'upper': statistics.mean(v['upper'] for v in values)}


def main(packet, review, priority, output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); base = Path('artifacts/fresh-evaluation-baselines-v1')
    checked = check(packet, review); packet_manifest = json.loads((packet / 'manifest.json').read_text()); priority_report = json.loads((priority / 'summary.json').read_text())
    if sha(priority / 'summary.json') != packet_manifest['priority_summary_sha256'] or sha(priority / 'private-priority.json') != priority_report['priority_sha256']:
        raise ValueError('Changed priority')
    if sha(data / 'manifest.json') != priority_report['input_sha256']['dataset_manifest'] or sha(base / 'evaluation.json') != priority_report['input_sha256']['baseline_evaluation'] or sha(base / 'summary.json') != priority_report['input_sha256']['baseline_summary']:
        raise ValueError('Changed frozen inputs')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('queries.jsonl', 'pools.jsonl', 'judgments.jsonl', 'snapshots.jsonl', 'human-review.jsonl'):
        if sha(data / name) != manifest['files_sha256'][name]: raise ValueError('Changed dataset member')
    queries = {r['id'] for r in map(json.loads, (data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development'}
    grades = {qid: {} for qid in queries}
    for r in map(json.loads, (data / 'judgments.jsonl').read_text().splitlines()):
        if r['query_id'] in grades:
            if r['expert_id'] in grades[r['query_id']]: raise ValueError('Duplicate source judgment')
            grades[r['query_id']][r['expert_id']] = r['grade']
    rankings = defaultdict(dict)
    for r in json.loads((base / 'evaluation.json').read_text()):
        qid, method = r['query_id'], r['method']
        if qid not in grades or method in rankings[qid] or actionable_gain_interval(r['ranking'], grades[qid], 5) != r['actionable_gain']:
            raise ValueError('Changed baseline ranking')
        rankings[qid][method] = r['ranking']
    if set(rankings) != queries: raise ValueError('Incomplete rankings')
    mapping = json.loads((packet / 'private-mapping.json').read_text())
    selected = json.loads((priority / 'private-priority.json').read_text())[:len(mapping)]
    if {(r['query_id'], r['expert_id']) for r in mapping} != {(r['query_id'], r['expert_id']) for r in selected} or len({(r['query_id'], r['expert_id']) for r in mapping}) != len(mapping):
        raise ValueError('Changed selected views')
    updated, applied = overlay(grades, mapping, read(review))
    before, after = bounds(rankings, grades), bounds(rankings, updated)
    if after['lower'] < before['lower'] - 1e-12 or after['upper'] > before['upper'] + 1e-12:
        raise ValueError('Updated interval must be nested')
    report = {'queries': len(queries), 'review_cases': len(mapping), 'applied_grades': applied,
              'review_grade_counts': checked['grades'], 'before': before, 'after': after,
              'interval_width_reduction': before['upper'] - before['lower'] - after['upper'] + after['lower'],
              'input_sha256': {'dataset_manifest': sha(data / 'manifest.json'), 'baseline_evaluation': sha(base / 'evaluation.json'),
                               'priority_summary': sha(priority / 'summary.json'), 'packet_manifest': sha(packet / 'manifest.json'), 'review': sha(review)},
              'code_sha256': sha(Path(__file__)),
              'scope': 'Post-hoc fixed-view development bounds using original same-model silver plus any submitted review grades. No original data changed. Annotation structure does not prove independent review, source entailment or expert ability. Missing grades remain unknown; labels are not representative or held-out, and bounds are not confidence intervals.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--packet', type=Path, required=True); p.add_argument('--review', type=Path, required=True)
    p.add_argument('--priority', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); main(args.packet, args.review, args.priority, args.output)
