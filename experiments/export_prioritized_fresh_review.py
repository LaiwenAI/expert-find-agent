"""Create a blinded local review packet from a frozen priority list."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(priority, output, count):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1')
    manifest = json.loads((data / 'manifest.json').read_text())
    summary = json.loads((priority / 'summary.json').read_text())
    if sha(data / 'manifest.json') != summary['input_sha256']['dataset_manifest'] or sha(priority / 'private-priority.json') != summary['priority_sha256']:
        raise ValueError('Changed priority or dataset')
    for name in ('human-review.jsonl', 'snapshots.jsonl'):
        if sha(data / name) != manifest['files_sha256'][name]: raise ValueError('Changed review source')
    selected = json.loads((priority / 'private-priority.json').read_text())
    if type(count) is not int or not 1 <= count <= len(selected): raise ValueError('Invalid review count')
    snapshots = {(r['query_id'], r['expert_id']): r['id'] for r in map(json.loads, (data / 'snapshots.jsonl').read_text().splitlines()) if r['split'] == 'development'}
    reviews = {r['snapshot_id']: r for r in map(json.loads, (data / 'human-review.jsonl').read_text().splitlines()) if r['split'] == 'development'}
    cases = []
    for r in selected[:count]:
        sid = snapshots[(r['query_id'], r['expert_id'])]; row = reviews[sid]
        if row['annotation']['grade'] is not None: raise ValueError('Unexpected prefilled label')
        cases.append((r, row))
    random.Random(20261119).shuffle(cases); forms = []; mapping = []
    for i, (priority_row, row) in enumerate(cases, 1):
        case = f'case-{i:03d}'
        forms.append({**row, 'snapshot_id': case})
        mapping.append({'case_id': case, 'query_id': priority_row['query_id'], 'expert_id': priority_row['expert_id'],
                        'snapshot_id': row['snapshot_id'], 'bound_width_reduction': priority_row['bound_width_reduction']})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'blind-review.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in forms))
    (output / 'private-mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    report = {'cases': count, 'seed': 20261119, 'blind_sha256': sha(output / 'blind-review.jsonl'),
              'mapping_sha256': sha(output / 'private-mapping.json'), 'priority_summary_sha256': sha(priority / 'summary.json'),
              'code_sha256': sha(Path(__file__)),
              'scope': 'Local private review packet: model labels, method names, ranks and priority weights hidden from review form; source text may reveal identities. Influence-selected cases are not representative; no human labels are supplied. Fixed snapshot content must remain unchanged and a separate completed copy should be reviewed independently.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--priority', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--count', type=int, default=50)
    args = parser.parse_args(); main(args.priority, args.output, args.count)
