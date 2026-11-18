"""Compare locked shallow encoders on a separately filtered SciRepEval cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch
from safetensors.torch import load_file

from experiments.retrieval.cof_matched_finetune_control import NATIVE
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_backbone_finetune import ROOT, ARMS
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_rank_distillation import ANCHORS, RIDGE_ROOT
from experiments.retrieval.cof_shallow_alignment import (
    AlignedEncoder, CALIBRATION_ROWS, FIT_ROWS, benchmark, normalize_text,
    public_training_texts, summarize,
)
from expert_search.cof_benchmark import load

DATASET = 'SciRepEval'


def exclusions():
    items = public_training_texts(ANCHORS, include_ids=True)
    split = FIT_ROWS + CALIBRATION_ROWS
    ridge = json.loads((RIDGE_ROOT / 'summary.json').read_text())
    distillation = json.loads((ROOT / 'summary.json').read_text())
    for records, expected in (
        (items[:split], ridge['training']['training_queries_sha256']),
        (items[split:], distillation['anchors_sha256']),
    ):
        texts = [s for s, _ in records]
        if hashlib.sha256(json.dumps(texts).encode()).hexdigest() != expected:
            raise ValueError('Changed training documents')
    training_ids = {key for _, key in items}
    training_titles = {normalize_text(s.split('\n', 1)[0]) for s, _ in items}
    reviewed_ids = set()
    reviewed_titles = set()
    for dataset in ('SIGIR', 'NIPS'):
        _, papers, _, labels = load(dataset)
        reviewed_ids.update(labels['soft'])
        reviewed_titles.update(normalize_text(papers[q]['title']) for q in labels['soft'])
    root, papers, reviewers, labels = load(DATASET)
    all_sources = {doc for reviewer in reviewers.values() for doc in reviewer['papers']}
    source_titles = {normalize_text(papers[doc]['title']) for doc in all_sources}
    removed_sources = {doc for doc in all_sources
                       if doc in training_ids or normalize_text(papers[doc]['title']) in training_titles}
    profile_collision = set()
    earlier_test = set()
    fit_collision = set()
    profile_manifest = json.loads((root / 'manifest.json').read_text())
    exact_profile_overlap = (set(profile_manifest['exact_query_texts_in_profiles'])
                             | set(profile_manifest['query_ids_in_profiles']))
    for query in labels['soft']:
        title = normalize_text(papers[query]['title'])
        if query in exact_profile_overlap or query in all_sources or title in source_titles:
            profile_collision.add(query)
        if query in reviewed_ids or title in reviewed_titles:
            earlier_test.add(query)
        if query in training_ids or title in training_titles:
            fit_collision.add(query)
    removed_queries = profile_collision | earlier_test | fit_collision
    if not removed_queries < labels['soft'].keys():
        raise ValueError('No retained queries')
    return removed_queries, removed_sources, {
        'original_queries': len(labels['soft']), 'profile_overlap': len(profile_collision),
        'earlier_test_overlap': len(earlier_test), 'training_overlap': len(fit_collision),
        'removed_queries': len(removed_queries), 'removed_sources': len(removed_sources),
    }


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm and CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed model file: {name}')
    orig = json.loads((ROOT / 'summary.json').read_text())
    native = json.loads((NATIVE / 'summary.json').read_text())
    projection = ROOT / 'projection.npz'
    fine_weights = NATIVE / 'query_blocks_20_21.safetensors'
    if not (sha(checkpoint) == native['checkpoint_manifest_sha256'] == orig['checkpoint_manifest_sha256']
            and sha(projection) == orig['projector_sha256'] and sha(fine_weights) == native['weights_sha256']):
        raise ValueError('Changed locked models')
    with np.load(projection, allow_pickle=False) as saved:
        arrays = saved['weights'], saved['bias']
    frozen = QueryEncoderVariants()
    fine = QueryEncoderVariants()
    fine.configure('layers_21')
    with torch.no_grad():
        parameters = dict(fine.encoder.model.named_parameters())
        for name, value in load_file(str(fine_weights)).items():
            parameters[name].copy_(value)
    encoders = {
        'projection_only': AlignedEncoder(frozen.encoder, *arrays),
        'backbone_finetune': AlignedEncoder(fine.encoder, *arrays),
    }
    remove_q, remove_s, exclusions_report = exclusions()
    output.mkdir(parents=True, exist_ok=False)
    ranker, db, queries, labels, metadata = prepare(
        DATASET, output, checkpoint, exclude_queries=remove_q, exclude_sources=remove_s
    )
    eligible = {}
    for query in queries:
        if labels['soft'][query].keys() != labels['hard'][query].keys():
            raise ValueError('Different judged reviewer pools by label type')
        eligible[query] = labels['soft'][query].keys() & ranker.index.owners.keys()
    stats = {}
    availability = {
        'under_five': sum(len(pool) < 5 for pool in eligible.values()),
        'empty': sum(not pool for pool in eligible.values()),
    }
    try:
        with sqlite3.connect(db) as connection:
            for name, pool in (('global', None), ('judged_pool', eligible)):
                cohort = queries if pool is None else {q: queries[q] for q, entries in pool.items() if entries}
                rows = benchmark(frozen, encoders, ranker, connection, cohort, labels, arms=ARMS, eligible=pool)
                path = output / f'{name}-rows.json'
                path.write_text(json.dumps(rows, indent=2) + '\n')
                stats[name] = {'rows_sha256': sha(path), 'results': summarize(rows, arms=ARMS)}
    finally:
        db.unlink()
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'model_sha256': sha(checkpoint),
        'projection_sha256': sha(projection), 'fine_weights_sha256': sha(fine_weights),
        'index_metadata': metadata, 'exclusions': exclusions_report,
        'judged_pool_availability': availability, 'results': stats,
        'scope': (
            'Frozen SciRepEval candidate profile snapshot. Models and epoch choices were locked '
            'on earlier KDD/SIGIR/NIPS work. Exact normalized-title and ID exclusions are not '
            'semantic deduplication. A source known as a training paper is removed before all '
            'ranking. Global ranking has incomplete grades. Judged-pool ranking conditions on '
            'existing assessment membership, without quality values at inference. Empty judged '
            'pools are excluded only in that arm, and short pools keep a denominator of five. Both protocols '
            'share the same exact scoring, evidence-store access and warm request harness.'
        ),
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
