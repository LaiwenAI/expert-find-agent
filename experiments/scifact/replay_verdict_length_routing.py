"""Calibrate an abstract-length router and replay measured SciFact verdicts."""
import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

from expert_search.statistics import clustered_paired_interval

ARCHIVE = Path('artifacts/public/scifact-original-data.tar.gz')
CASCADE = Path('artifacts/scifact-open-verdict-cascade-v2')
BALANCED = 'scifact-binary-checks-train-v1'
UNKNOWN = 'scifact-binary-checks-v1'
FULL, SHORT = 'early', 'short_then_full'
THRESHOLDS = (None, 5, 8, 12, 16)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route(row, threshold):
    return SHORT if threshold is not None and row['sentences'] >= threshold else FULL


def metrics(rows, arms):
    records = [row['arms'][arm] for row, arm in zip(rows, arms, strict=True)]
    seconds = [a['seconds'] for a in records]
    return {
        'claims': len(rows),
        'short_route': arms.count(SHORT),
        'fallbacks': sum(a['fallback'] for a in records),
        'correct': sum(a['verdict'] == row['gold'] for row, a in zip(rows, records)),
        'wrong_committed': sum(a['verdict'] not in ('unknown', row['gold']) for row, a in zip(rows, records)),
        'mean_ms': 1000 * statistics.mean(seconds),
        'p95_ms': 1000 * sorted(seconds)[math.ceil(.95 * len(seconds))-1],
    }


def main(output):
    manifest = json.loads((CASCADE/'summary.json').read_text())
    if sha(ARCHIVE) != manifest['archive_sha256'] or sha(CASCADE/'rows.json') != manifest['rows_sha256']:
        raise ValueError('Changed cascade input')
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        member = next(m for m in archive.getmembers() if Path(m.name).name == 'corpus.jsonl')
        with archive.extractfile(member) as stream:
            document_sizes = {str(r['doc_id']): len(r['abstract']) for r in map(json.loads, stream)}
    papers = {}
    for cohort in (BALANCED, UNKNOWN):
        parent = Path('artifacts')/cohort/'rows.json'
        if sha(parent) != manifest['cohorts'][cohort]['parent_rows_sha256']:
            raise ValueError('Changed parent cohort')
        papers[cohort] = {r['claim']: r['paper'] for r in json.loads(parent.read_text())}
    rows = json.loads((CASCADE/'rows.json').read_text())
    for row in rows:
        row['paper'] = papers[row['cohort']][row['claim']]
        row['sentences'] = document_sizes[row['paper']]
    balanced = [r for r in rows if r['cohort'] == BALANCED]
    other = [r for r in rows if r['cohort'] == UNKNOWN]
    if len({r['paper'] for r in balanced}) != 90:
        raise ValueError('Expected 90 distinct papers')
    calibration, held_out = [], []
    for label in ('support','contradict','unknown'):
        subset = sorted((r for r in balanced if r['gold'] == label),
                        key=lambda r: hashlib.sha256(f"routing-v1:{r['claim']}".encode()).digest())
        if len(subset) != 30:
            raise ValueError('Expected balanced cohort')
        calibration.extend(subset[:15])
        held_out.extend(subset[15:])
    if {r['paper'] for r in calibration} & {r['paper'] for r in held_out + other}:
        raise ValueError('Calibration paper leakage')
    candidates = [(t, metrics(calibration, [route(r,t) for r in calibration])) for t in THRESHOLDS]
    baseline = candidates[0][1]
    eligible = [(threshold, m) for threshold, m in candidates if
                m['correct'] >= baseline['correct'] and m['wrong_committed'] <= baseline['wrong_committed']
                and m['p95_ms'] <= baseline['p95_ms']]
    strict_choice = min(eligible, key=lambda item: item[1]['mean_ms'])[0]
    mean_choice = min(candidates, key=lambda item: item[1]['mean_ms'])[0]
    report = {
        'source_run': manifest['job'], 'script_sha256': sha(Path(__file__)),
        'cascade_rows_sha256': sha(CASCADE/'rows.json'),
        'selected_strict_threshold': strict_choice, 'selected_mean_only_threshold': mean_choice,
        'calibration': [{'threshold': t, **m} for t, m in candidates],
        'evaluation': {},
        'scope': 'Post-hoc replay of live paired verifier measurements. One hash-split stratification of the previously explored balanced set. Only calibration queries are used to select thresholds. The other cohort is shifted towards the benchmark unknown label. None means always full. Routed selection and queueing overhead are not timed. Descriptive comparison, not an untouched confirmatory test.',
    }
    for name, cohort in (('balanced_held_out', held_out), ('unknown_shift', other)):
        arms = {
            'always_full': [FULL for _ in cohort],
            'always_short': [SHORT for _ in cohort],
            'strict': [route(r, strict_choice) for r in cohort],
            'mean_only': [route(r, mean_choice) for r in cohort],
        }
        results = {method: metrics(cohort, used) for method, used in arms.items()}
        for method in ('strict', 'mean_only'):
            differences = [1000*(r['arms'][arm]['seconds']-r['arms'][FULL]['seconds'])
                           for r, arm in zip(cohort, arms[method])]
            results[method]['delta_ms_vs_full'] = clustered_paired_interval(differences, [r['paper'] for r in cohort])
        report['evaluation'][name] = results
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
