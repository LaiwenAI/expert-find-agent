"""Audit what blind fit and citation reviews can identify about a paired ranking."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

from experiments.retrieval.cof_rank_disagreement_review import (
    ARM, FIELDS, archived_rows, sha, verify_packet, verify_submission, write_json,
)

COMPATIBLE = {('direct', 'direct_support'), ('transferable', 'transfer_support')}
K = 5


def agreed_labels(packet, submissions, dimension, cases):
    if submissions is None:
        return dict.fromkeys(cases)
    first, first_ids = verify_submission(packet, dimension, submissions[0])
    second, second_ids = verify_submission(packet, dimension, submissions[1])
    if first_ids & second_ids:
        raise ValueError('Use independent reviewers')
    labels = {}
    for key in cases:
        label = first[key]['label']
        labels[key] = label if label == second[key]['label'] and label != 'insufficient' else None
    return labels


def joint(fit, citation):
    compatible = {(a, b) in COMPATIBLE for a, b in product(
        [fit] if fit is not None else FIELDS['fit'],
        [citation] if citation is not None else FIELDS['citation'],
    )}
    return int(compatible.pop()) if len(compatible) == 1 else None


def audit(packet, run, fit_files=None, citation_files=None):
    manifest, mapping = verify_packet(packet)
    summary, archived = archived_rows(run)
    if summary['rows_sha256'] != manifest['source_rows_sha256']:
        raise ValueError('Packet and rankings were created from different runs')
    if len(mapping) != manifest['candidate_cases']:
        raise ValueError('Unexpected packet inventory')
    fit = agreed_labels(packet, fit_files, 'fit', mapping)
    citation = agreed_labels(packet, citation_files, 'citation', mapping)
    cells = {}
    for case, row in mapping.items():
        key = row['query_id'], row['expert_id']
        if key in cells or key[0] not in archived:
            raise ValueError('Duplicate or unexpected pair')
        orders = archived[key[0]]
        ranks = {arm: order.index(key[1]) + 1 if key[1] in order else None
                 for arm, order in orders.items()}
        stratum = 'same' if set(orders['dense']) == set(orders[ARM]) else 'changed'
        if row['ranks'] != ranks or row['stratum'] != stratum or not any(ranks.values()):
            raise ValueError('Packet and ranking mismatch')
        cells[key] = joint(fit[case], citation[case])
    selected = {q for q, _ in cells}
    if len(selected) != manifest['queries']:
        raise ValueError('Unexpected selected query count')
    totals = {name: Counter() for name in ('sample', 'full')}
    strata = Counter()
    for q, arms in archived.items():
        first, second = map(set, (arms[ARM], arms['dense']))
        if not all(len(order) == len(set(order)) == K for order in arms.values()):
            raise ValueError('Expected distinct top-five ranks')
        strata['same' if first == second else 'changed'] += 1
        union = first | second
        if q in selected and any((q, e) not in cells for e in union):
            raise ValueError('Sampled query has incomplete pooling')
        for e in union:
            sign = (e in first) - (e in second)
            label = cells.get((q, e))
            for scope in ('sample', 'full') if q in selected else ('full',):
                counts = totals[scope]
                if not sign:
                    counts['shared'] += 1
                elif label is None:
                    counts['unknown_plus' if sign > 0 else 'unknown_minus'] += 1
                else:
                    counts['known_exclusive'] += 1
                    counts['known_difference'] += sign * label
    if any(strata[key] != info['population'] for key, info in manifest['strata'].items()):
        raise ValueError('Unexpected archived strata')
    output = {}
    for scope, n in (('sample', len(selected)), ('full', len(archived))):
        c = totals[scope]
        denom = K * n
        unknown = c['unknown_minus'] + c['unknown_plus']
        output[scope] = {
            'queries': n,
            'shared_pairs': c['shared'],
            'known_exclusive_pairs': c['known_exclusive'],
            'unresolved_exclusive_pairs': unknown,
            'lower': (c['known_difference'] - c['unknown_minus']) / denom,
            'upper': (c['known_difference'] + c['unknown_plus']) / denom,
            'resolutions_needed_for_width_at_most_10pp': max(0, unknown - denom // 10),
        }
    return {
        'sample_candidate_pairs': len(cells),
        'sample_known_joint_pairs': sum(x is not None for x in cells.values()),
        'scopes': output,
        'packet_manifest_sha256': sha(packet / 'custodian' / 'manifest.json'),
        'archived_rows_sha256': summary['rows_sha256'],
        'submission_sha256': {key: [sha(path) for path in paths] if paths else None
                              for key, paths in (('fit', fit_files), ('citation', citation_files))},
        'scope': 'Differences in evidence-scoped top-five expert-and-citation concordance. '
                 'A positive pairs direct fit with direct support or transferable fit with transfer support. '
                 'Each query-expert pair uses the same profile and fixed first citation across arms. '
                 'A missing, insufficient, or disputed component stays unresolved unless the other component '
                 'rules out concordance. The full-cohort bounds treat unreviewed queries as unknown. '
                 'Each resolved exclusive pair reduces width by one top-five slot. '
                 'Resolution counts assume an informative outcome and are not a stopping or publication rule. '
                 'These are identification bounds, not confidence intervals or evidence of expert ability.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fit', type=Path, nargs=2)
    parser.add_argument('--citation', type=Path, nargs=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.packet, args.run, args.fit, args.citation))
