"""Compare second-abstract decisions on annotated SciFact-Open source pairs."""
import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path
from statistics import mean

SEED = 20261118
POLARITIES = {'SUPPORT': 'support', 'CONTRADICT': 'contradict'}
STRATA = ('same_support', 'same_refutation', 'mixed')
POLICIES = ('first_only', 'always_second', 'second_after_support', 'second_after_uncertainty')
SOURCE = 'https://scifact.s3.us-west-2.amazonaws.com/scifact-open/latest/scifact_open.tar.gz'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(archive, checksum):
    if digest(archive) != checksum:
        raise ValueError('Changed SciFact-Open archive')
    groups = []
    required = set()
    abstracts = {}
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            if member.name == 'data/claims.jsonl':
                claims = [json.loads(line) for line in stream.extractfile(member)]
                if len(claims) != 279 or len({row['id'] for row in claims}) != len(claims):
                    raise ValueError('Changed claim inventory')
                for claim in claims:
                    by_label = {label: [] for label in POLARITIES.values()}
                    for doc, meta in claim['evidence'].items():
                        by_label[POLARITIES[meta['label']]].append(doc)
                    randomizer = random.Random(f'{SEED}:{claim["id"]}')
                    pools = {}
                    for label, docs in by_label.items():
                        order = sorted(docs)
                        randomizer.shuffle(order)
                        pools[label] = order
                    candidates = {
                        'same_support': pools['support'][:2],
                        'same_refutation': pools['contradict'][:2],
                        'mixed': pools['support'][:1] + pools['contradict'][:1]
                        if all(pools.values()) else [],
                    }
                    for stratum, docs in candidates.items():
                        if len(docs) != 2:
                            continue
                        required.update(docs)
                        groups.append({
                            'claim_id': claim['id'], 'stratum': stratum, 'claim': claim['claim'],
                            'docs': [{
                                'id': doc, 'gold': POLARITIES[claim['evidence'][doc]['label']],
                                'provenance': claim['evidence'][doc]['provenance'],
                            } for doc in docs],
                        })
            if member.name == 'data/corpus.jsonl':
                if not required:
                    raise ValueError('Expected claims before corpus')
                for line in stream.extractfile(member):
                    row = json.loads(line)
                    doc_id = str(row['doc_id'])
                    if doc_id in required:
                        if doc_id in abstracts or not row['abstract'] or not all(
                            isinstance(sentence, str) for sentence in row['abstract']
                        ):
                            raise ValueError(f'Invalid abstract: {doc_id}')
                        abstracts[doc_id] = row['abstract']
                break
    if abstracts.keys() != required:
        raise ValueError('Missing selected documents')
    return {'source': SOURCE, 'archive_sha256': checksum, 'groups': groups, 'abstracts': abstracts}


def use_second(policy, first):
    if policy in ('first_only', 'always_second'):
        return policy == 'always_second'
    if policy == 'second_after_support':
        return first['majority'] == 'support'
    if policy == 'second_after_uncertainty':
        return first['majority'] == 'unknown' or len(set(first['votes'])) > 1
    raise ValueError(f'Unknown policy: {policy}')


def decision(first, second, policy):
    observations = [first, second] if use_second(policy, first) else [first]
    labels = {obs['majority'] for obs in observations} - {'unknown'}
    if len(labels) == 2:
        verdict = 'conflict'
    else:
        verdict = next(iter(labels), 'unknown')
    return {
        'verdict': verdict, 'checks': len(observations),
        'milliseconds': 1000 * sum(o['elapsed_seconds']['majority'] for o in observations),
    }


def summarize(rows):
    stats = {}
    for stratum in STRATA:
        subset = [r for r in rows if r['stratum'] == stratum]
        stats[stratum] = {'pairs': len(subset), 'policies': {}}
        stats[stratum]['missed_support_from_both'] = sum(
            all(d['majority'] != 'support' for d in r['predictions']) for r in subset
        ) if stratum == 'same_support' else None
        stats[stratum]['double_false_support'] = sum(
            all(d['majority'] == 'support' for d in r['predictions']) for r in subset
        ) if stratum == 'same_refutation' else None
        if stratum == 'same_refutation':
            stats[stratum]['any_false_support'] = sum(
                any(d['majority'] == 'support' for d in r['predictions']) for r in subset
            )
        for policy in POLICIES:
            # Each claim contributes equally through both possible reading orders.
            forward = [decision(*r['predictions'], policy) for r in subset]
            reverse = [decision(*r['predictions'][::-1], policy) for r in subset]
            metrics = {}
            for kind, results in [('first_listed', forward), ('reversed', reverse), ('both', forward + reverse)]:
                metrics[kind] = {
                    'replays': len(results),
                    'conflict': sum(r['verdict'] == 'conflict' for r in results),
                    'support': sum(r['verdict'] == 'support' for r in results),
                    'contradict': sum(r['verdict'] == 'contradict' for r in results),
                    'unknown': sum(r['verdict'] == 'unknown' for r in results),
                    'mean_checks': mean(r['checks'] for r in results),
                    'mean_verifier_milliseconds': mean(r['milliseconds'] for r in results),
                }
            stats[stratum]['policies'][policy] = metrics
    return stats


def run(cases, checksum, output):
    import os
    import torch
    from experiments.scifact.cross_document_verifier_errors import make_verifier

    if digest(cases) != checksum:
        raise ValueError('Changed case inventory')
    fixture = json.loads(cases.read_text())
    verify, model = make_verifier()
    verify('Synthetic warmup claim.', ['An unrelated synthetic abstract.'])
    seen = {}
    rows = []
    for group in fixture['groups']:
        predictions = []
        for doc in group['docs']:
            key = group['claim_id'], doc['id']
            if key not in seen:
                seen[key] = verify(group['claim'], fixture['abstracts'][doc['id']])
            predictions.append(seen[key])
        rows.append({k: v for k, v in group.items() if k != 'claim'} | {'predictions': predictions})

    output.mkdir(parents=True, exist_ok=False)
    detail = output / 'rows.json'
    detail.write_text(json.dumps(rows, indent=2) + '\n')
    report = {
        'job_id': os.environ['SLURM_JOB_ID'],
        'gpu': torch.cuda.get_device_name(),
        'dataset_sha256': fixture['archive_sha256'],
        'case_sha256': checksum,
        'model_manifest_sha256': model,
        'distinct_claims': len({r['claim_id'] for r in rows}),
        'distinct_claim_document_pairs': len(seen),
        'script_sha256': digest(Path(__file__)),
        'rows_sha256': digest(detail),
        'stats': summarize(rows),
        'scope': (
            'Given annotated abstract pairs only, not corpus retrieval or expert matching. '
            'Citation and pooled annotations may overlap SciFact. Gold labels choose strata '
            'but are hidden from verifier and policy. Measured verifier durations are summed '
            'over replayed actions; no retrieval or service time. Mixed pairs are evaluated '
            'in both directions; opposite annotator labels do not identify adjudicated truth.'
        ),
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == 'prepare':
        cases = prepare(args.input, args.sha256)
        with args.output.open('x') as handle:
            json.dump(cases, handle, ensure_ascii=False, sort_keys=True)
        print(digest(args.output))
    else:
        run(args.input, args.sha256, args.output)


if __name__ == '__main__':
    main()
