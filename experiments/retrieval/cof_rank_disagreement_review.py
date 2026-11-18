"""Export blind fit and citation reviews from reproduced CoF ranking disagreements."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3

SEED = 20261122
ARM = 'complementary'
MAX_QUERIES = {'changed': 12, 'same': 8}
FIELDS = {
    'fit': {'direct', 'transferable', 'poor_fit', 'insufficient'},
    'citation': {'direct_support', 'transfer_support', 'contradicts', 'insufficient'},
}
SUPPORT = {'direct', 'transferable', 'direct_support', 'transfer_support', 'contradicts'}
TRANSFER = {'transferable', 'transfer_support'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {row['case']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate case label')
    return result


def archived_rows(run):
    summary = json.loads((run / 'summary.json').read_text())
    path = run / 'rows.json'
    checkpoint = run / f'{ARM}.npz'
    if sha(path) != summary['rows_sha256'] or sha(checkpoint) != summary['models'][ARM]['sha256']:
        raise ValueError('Changed reference rows or re-ranker')
    rows = {r['query']: {m: r['arms'][m]['experts'] for m in ('dense', ARM)}
            for r in json.loads(path.read_text())}
    return summary, rows


def cohort(rows):
    groups = {k: [] for k in MAX_QUERIES}
    for query, arms in sorted(rows.items()):
        key = 'same' if set(arms['dense']) == set(arms[ARM]) else 'changed'
        groups[key].append(query)
    rng = random.Random(SEED)
    selected = {q: key for key, values in groups.items()
                for q in rng.sample(values, min(len(values), MAX_QUERIES[key]))}
    return selected, {key: {'population': len(values),
                            'sample': sum(x == key for x in selected.values())}
                      for key, values in groups.items()}


def reconstruct(run, archive, queries_to_keep, temp):
    import numpy as np
    import torch

    from expert_search.cof_benchmark import load
    from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, normalize_text
    from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare
    from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
    from experiments.retrieval.cof_source_feature_rerank import (
        SOURCES, assemble, candidate_sources, features, metadata, retrieve, source_exclusions,
    )

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA to replay retrieved documents')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    path = EMBEDDING / 'checkpoint-provenance.json'
    if sha(path) != archive['encoder_sha256']:
        raise ValueError('Changed encoder revision')
    for name, expected in json.loads(path.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed encoder file')
    engine = QueryEncoderVariants()
    engine.configure('no_cache')
    encoder = engine.encoder
    root, papers, _, _ = load('KDD')
    inventory = json.loads((root / 'manifest.json').read_text())
    omit = set(inventory['query_ids_in_profiles']) | set(inventory['exact_query_texts_in_profiles'])
    ranker, db, train, labels, info = prepare('KDD', temp, path, exclude_queries=omit)
    if info['vectors_sha256'] != archive['train']['vectors_sha256']:
        raise ValueError('Changed KDD document index')
    with sqlite3.connect(db) as connection:
        x, _, titles, used = assemble(ranker, connection, encoder, train, labels, papers)
    db.unlink()
    del ranker
    if len(x) != archive['train_pairs'] or len(used) != archive['train_sources']:
        raise ValueError('Changed supervised source exposure')
    remove_q, remove_s = source_exclusions((
        set(train), {normalize_text(papers[key]['title']) for key in train}, used, titles))
    ranker, db, queries, _, inventory = prepare(
        DATASET, temp, path, exclude_queries=remove_q, exclude_sources=remove_s)
    for key in ('queries', 'sources', 'owners', 'ownership_edges', 'vectors_sha256'):
        if inventory[key] != archive['evaluation'][key]:
            raise ValueError(f'Changed evaluation inventory: {key}')
    with np.load(run / f'{ARM}.npz', allow_pickle=False) as saved:
        model = {k: saved[k] for k in ('center', 'scale', 'weight', 'bias')}

    result = {}
    with sqlite3.connect(db) as connection:
        for qid in sorted(queries_to_keep):
            query = queries[qid]
            score, owners = retrieve(ranker, encoder.encode([query], query=True)[0])
            candidates = candidate_sources(ranker, score, owners, SOURCES)
            signals = features(ranker, connection, query, candidates, score, SOURCES)
            adjusted = ((signals - model['center']) / model['scale']) @ model['weight'] + model['bias']
            selections = {'dense': np.arange(5), ARM: np.argsort(-adjusted, kind='stable')[:5]}
            linked = {}
            for offset in sorted(set(i for top in selections.values() for i in top)):
                owner, top_sources, _, _ = candidates[offset]
                expert = ranker.index.owner_ids[owner]
                ids = [ranker.index.source_ids[i] for i in top_sources]
                text = metadata(ranker, connection, set(top_sources))
                linked[expert] = [{'id': pid, 'title': text[pid][0], 'excerpt': text[pid][1]} for pid in ids]
            ranking = {m: [ranker.index.owner_ids[candidates[i][0]] for i in top]
                       for m, top in selections.items()}
            result[qid] = (query, ranking, linked)
    db.unlink()
    return result


def export(run, output):
    summary, rows = archived_rows(run)
    selected, strata = cohort(rows)
    output.mkdir(parents=True, exist_ok=False)
    result = reconstruct(run, summary, selected, output)
    public = output / 'reviewer'
    private = output / 'custodian'
    public.mkdir()
    private.mkdir()
    fit, evidence, mapping = [], [], {}
    rng = random.Random(SEED + 1)
    picked = list(sorted(result))
    rng.shuffle(picked)
    for i, qid in enumerate(picked, start=1):
        request, rank, links = result[qid]
        if rank != rows[qid]:
            raise ValueError('Reconstructed ranking differs from the reference run')
        candidates = sorted(links)
        rng.shuffle(candidates)
        for j, expert in enumerate(candidates, start=1):
            case = f'q{i:02d}-x{j:02d}'
            sources = links[expert]
            citation = {k: sources[0][k] for k in ('title', 'excerpt')}
            candidates_view = [{k: r[k] for k in ('title', 'excerpt')} for r in sources]
            rng.shuffle(candidates_view)
            common = {'case': case, 'query': request, 'reviewer': None,
                      'label': None, 'quote': None, 'explanation': None, 'adaptation': None}
            fit.append({**common, 'profile': candidates_view})
            evidence.append({**common, 'citation': citation})
            mapping[case] = {'query_id': qid, 'stratum': selected[qid], 'expert_id': expert,
                             'source_ids': [r['id'] for r in sources],
                             'ranks': {m: (order.index(expert) + 1 if expert in order else None)
                                       for m, order in rank.items()}}
    rng.shuffle(fit)
    rng.shuffle(evidence)
    fit_path, citation_path = public / 'fit.jsonl', public / 'citation.jsonl'
    write_rows(fit_path, fit)
    write_rows(citation_path, evidence)
    map_path = private / 'mapping.json'
    write_json(map_path, mapping)
    write_json(private / 'manifest.json', {
        'queries': len(selected), 'candidate_cases': len(mapping), 'strata': strata, 'seed': SEED,
        'source_job': summary['job_id'], 'replay_job': os.environ['SLURM_JOB_ID'],
        'source_rows_sha256': summary['rows_sha256'], 'source_encoder_sha256': summary['encoder_sha256'],
        'hashes': {'fit': sha(fit_path), 'citation': sha(citation_path), 'mapping': sha(map_path)},
        'scope': 'Conditional reviewer fit from up to three owned excerpts and support from the exact '
                 'displayed dense citation. Arm, rank, and known relevance are blinded. Quotes may '
                 'still reveal identity. Per-query strata are sampled without reading relevance grades.',
    })
    (public / 'REVIEW.md').write_text(
        '# Blind expert and citation review\n\n'
        'Copy only this reviewer folder. Keep individual submissions separate. Judge candidate fit using '
        'the up to three displayed profile excerpts first; do not open the citation form until fit is locked. '
        'Then judge whether the single displayed citation supports a match to the query. Work independently.\n\n'
        'Fit labels: direct, transferable, poor_fit, insufficient. Citation labels: direct_support, '
        'transfer_support, contradicts, insufficient. A transferable match requires a concrete demonstrated '
        'capability, its application to this query and a material adaptation or constraint that needs '
        'checking. Keyword overlap and coauthorship alone are insufficient. Use insufficient when the '
        'available text cannot support a decision. Poor fit requires clear lack of fit within the visible '
        'profile, rather than merely missing detail.\n\n'
        'For each judged row enter your stable reviewer pseudonym, label, and explanation. For positive fit, '
        'positive citation support, and contradiction, provide an exact quote from the displayed excerpt. '
        'For transfer judgments also describe the required adaptation. Leave unreviewed rows empty. Do not '
        'modify the case, query or displayed material.\n')


def verify_submission(packet, dimension, path):
    original_path = packet / 'reviewer' / f'{dimension}.jsonl'
    template, actual = read_rows(original_path), read_rows(path)
    editable = {'reviewer', 'label', 'quote', 'explanation', 'adaptation'}
    identities = set()
    if set(template) != set(actual):
        raise ValueError('Changed case inventory')
    for key, row in actual.items():
        reference = template[key]
        if set(row) != set(reference) or any(row[f] != reference[f] for f in reference.keys() - editable):
            raise ValueError('Modified prompt or evidence')
        value = row['label']
        if value is None:
            if any(row[field] is not None for field in editable - {'reviewer', 'label'}):
                raise ValueError('Unlabeled rows must have no partial judgment')
            continue
        if value not in FIELDS[dimension] or not all(
            isinstance(row[field], str) and row[field].strip() for field in ('reviewer', 'explanation')
        ):
            raise ValueError('Judged rows require a permitted label, reviewer ID and explanation')
        identities.add(row['reviewer'].strip())
        if value in SUPPORT:
            excerpts = [row['citation']] if dimension == 'citation' else row['profile']
            quote = row['quote']
            if not isinstance(quote, str) or not quote.strip() or not any(quote in p['excerpt'] for p in excerpts):
                raise ValueError('Exact quote is required from a displayed excerpt')
        if value in TRANSFER and (not isinstance(row['adaptation'], str) or not row['adaptation'].strip()):
            raise ValueError('Transfer judgments require adaptation')
    if len(identities) > 1:
        raise ValueError('One submission per reviewer')
    return actual, identities


def verify_packet(packet):
    private = packet / 'custodian'
    manifest = json.loads((private / 'manifest.json').read_text())
    mapping = private / 'mapping.json'
    expected = {'mapping': mapping, **{m: packet / 'reviewer' / f'{m}.jsonl' for m in FIELDS}}
    if any(sha(path) != manifest['hashes'][name] for name, path in expected.items()):
        raise ValueError('Changed packet')
    return manifest, json.loads(mapping.read_text())


def compare(packet, dimension, first, second, output):
    manifest, mapping = verify_packet(packet)
    a, a_ids = verify_submission(packet, dimension, first)
    b, b_ids = verify_submission(packet, dimension, second)
    if a_ids & b_ids:
        raise ValueError('Use independent reviewers')
    if len(a) != manifest['candidate_cases'] or set(a) != set(mapping):
        raise ValueError('Mismatch against blind mapping')
    support = {'direct', 'transferable'} if dimension == 'fit' else {'direct_support', 'transfer_support'}
    counts = Counter()
    arms = {m: Counter() for m in ('dense', ARM)}
    by_query = {}
    for case, private in mapping.items():
        labels = a[case]['label'], b[case]['label']
        if None in labels:
            status = 'missing'
        elif labels[0] != labels[1]:
            status = 'disputed'
        else:
            status = labels[0]
        counts[status] += 1
        flag = 1 if status in support else (None if status in ('disputed', 'missing', 'insufficient') else 0)
        difference = int(private['ranks'][ARM] is not None) - int(private['ranks']['dense'] is not None)
        lower, upper = ((min(0, difference), max(0, difference)) if flag is None
                        else (difference * flag, difference * flag))
        for method, rank in private['ranks'].items():
            if rank is not None:
                arms[method]['top5_slots'] += 1
                arms[method]['agreed_positive'] += int(flag == 1)
                arms[method]['unresolved'] += int(flag is None)
        group = by_query.setdefault(private['query_id'], [private['stratum'], 0, 0])
        group[1] += lower / 5
        group[2] += upper / 5
    strata_bounds = {}
    for name, inventory in manifest['strata'].items():
        selected = [r[1:] for r in by_query.values() if r[0] == name]
        if len(selected) != inventory['sample']:
            raise ValueError('Changed query sample')
        strata_bounds[name] = [sum(r[i] for r in selected) / len(selected) for i in range(2)] if selected else None
    write_json(output, {'dimension': dimension, 'counts': dict(counts),
                        'sample_arm_counts': {m: dict(x) for m, x in arms.items()},
                        'by_stratum_paired_difference_bounds': strata_bounds,
                        'scope': 'Sample coverage and exact agreement. Unknown and disputed slots '
                                 'remain unknown. Shared candidate outcomes cancel in paired bounds. '
                                 'No population interval or cross-dimension joint measure.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--run', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--packet', type=Path, required=True)
    c.add_argument('--dimension', choices=tuple(FIELDS), required=True)
    c.add_argument('--first', type=Path, required=True)
    c.add_argument('--second', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.run, args.output)
    else:
        compare(args.packet, args.dimension, args.first, args.second, args.output)


if __name__ == '__main__':
    main()
