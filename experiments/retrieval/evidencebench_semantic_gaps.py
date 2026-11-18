"""Test marginal embedding coverage under matched EvidenceBench reading budgets."""
import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path

import numpy as np

from experiments.retrieval.evidencebench_gap_selection import CAPS
from experiments.retrieval.evidencebench_character_budget import ROOT, sha
from expert_search.evidencebench_data import aspect_coverage, prepare
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval


def choose(similarity, costs, budget, adaptive):
    spent = 0
    chosen = []
    covered = np.zeros(similarity.shape[1])
    remaining = list(range(len(costs)))
    while affordable := [i for i in remaining if costs[i] + spent <= budget]:
        scores = np.maximum(similarity[affordable] - covered, 0).sum(axis=1)
        i = affordable[int(np.argmax(scores / costs[affordable]))]
        chosen.append(i)
        spent += costs[i]
        remaining.remove(i)
        if adaptive:
            covered = np.maximum(covered, similarity[i])
    return chosen


def main(output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside a Slurm allocation')
    checkpoint = Path('artifacts/models/qwen3-embedding-0.6b')
    model_manifest = checkpoint / 'checkpoint-provenance.json'
    for name, expected in json.loads(model_manifest.read_text())['files'].items():
        with (checkpoint / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Changed model file: {name}')
    verified = json.loads((ROOT/'verification.json').read_text())['artifacts']
    protocol, evaluation = ROOT/'protocol.json', ROOT/'evaluation.json'
    for path in (protocol, evaluation):
        if sha(path) != verified[str(path)]:
            raise ValueError(f'Changed artifact: {path}')
    data = prepare('test')
    dataset = Path('artifacts/public/evidencebench/test.json')
    if sha(dataset) != json.loads(protocol.read_text())['dataset_sha256']:
        raise ValueError('Changed test data')
    records = json.loads(evaluation.read_text())
    if len(records) != len(data) or {r['id'] for r in records} != set(data):
        raise ValueError('Changed query cohort')

    model = QwenTorchEmbedding(max_length=256, batch_size=32)
    rows = []
    for record in records:
        query = data[record['id']]
        hypothesis = query['hypothesis']
        fragments = [part.strip() for part in re.split(r';|\band\b|\bwhereas\b', hypothesis, flags=re.I)]
        fragments = list(dict.fromkeys(part for part in fragments if len(part.split()) >= 4))
        aspects = fragments if len(fragments) >= 2 else [hypothesis]
        ranking = record['selected']
        texts = [query['paper_as_candidate_pool'][i].strip() for i in ranking]
        costs = np.array([len(text) + 1 for text in texts])
        started = time.perf_counter()
        vectors = model.encode([text or '[empty sentence]' for text in texts] + [
            'Instruct: Retrieve a sentence containing concrete evidence for this statement.\nQuery: ' + a
            for a in aspects
        ])
        encode_seconds = time.perf_counter() - started
        similarity = np.maximum(vectors[:len(texts)] @ vectors[len(texts):].T, 0)
        row = {'id': record['id'], 'aspects': len(aspects), 'paper_id': query['paper_id'],
               'encode_seconds': encode_seconds, 'caps': {}}
        for cap in CAPS:
            row['caps'][str(cap)] = {}
            for arm, adaptive in (('static', False), ('marginal', True)):
                picked = choose(similarity, costs, cap + 1, adaptive)
                used = sum(int(costs[i]) for i in picked) - bool(picked)
                if used > cap:
                    raise ValueError('Exceeded budget')
                row['caps'][str(cap)][arm] = {
                    **aspect_coverage(query, [ranking[i] for i in picked]), 'characters': used}
        rows.append(row)

    summary = {'queries': len(rows), 'multiple_fragments': sum(r['aspects'] > 1 for r in rows),
               'job_id': os.environ['SLURM_JOB_ID'], 'gpu': model.torch.cuda.get_device_name(),
               'model_manifest_sha256': sha(model_manifest), 'dataset_sha256': sha(dataset),
               'script_sha256': sha(Path(__file__)), 'evaluation_sha256': sha(evaluation),
               'mean_combined_encode_seconds': statistics.mean(r['encode_seconds'] for r in rows),
               'caps': {}, 'scope': 'Post-hoc reused EvidenceBench test cohort. Same role-prioritized pool, approximate query fragments, Qwen embeddings truncated to 256 tokens and no gold facets in selection. Marginal coverage tracks best cosine already obtained for each fragment. Character budget excludes prompts and metadata. Combined encoding includes source and fragment encoding but not model initialization, retrieval or verification. Cluster bootstrap intervals are unadjusted.'}
    for cap in CAPS:
        entries = [r['caps'][str(cap)] for r in rows]
        summary['caps'][str(cap)] = {
            arm: {key: statistics.mean(entry[arm][key] for entry in entries)
                  for key in ('aspect_recall', 'characters')} for arm in ('static', 'marginal')}
        summary['caps'][str(cap)]['paired_difference'] = clustered_paired_interval(
            [entry['marginal']['aspect_recall'] - entry['static']['aspect_recall'] for entry in entries],
            [r['paper_id'] for r in rows])
    output.mkdir(parents=True, exist_ok=False)
    (output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary['rows_sha256'] = sha(output/'rows.json')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
