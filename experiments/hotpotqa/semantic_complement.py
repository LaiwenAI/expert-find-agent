"""Compare semantic second-passage policies on a fixed HotpotQA retrieval pool."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import mean

from experiments.hotpotqa.second_document_complement import (
    CANDIDATE_LIMIT,
    DATA_SHA256,
    assess,
    bm25,
    sha,
)
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval

CHECKPOINT = Path('artifacts/models/qwen3-embedding-0.6b')
MAX_TOKENS = 512
INSTRUCTION = 'Instruct: Retrieve a passage containing evidence needed to answer this question.\nQuery: '
ARMS = ('bm25', 'missing_terms', 'dense', 'dense_residual')


def verify_model():
    manifest = CHECKPOINT / 'checkpoint-provenance.json'
    for name, expected in json.loads(manifest.read_text())['files'].items():
        with (CHECKPOINT / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Changed model file: {name}')
    return sha(manifest)


def evaluate(record, reference, model):
    titles = [title for title, _ in record['context']]
    passages = ['\n'.join([title, *sentences]) for title, sentences in record['context']]
    first, second, *others = bm25(record['question'], passages)[:CANDIDATE_LIMIT]
    feasible = [second, *(i for i in others if len(passages[i]) <= len(passages[second]))]
    arms = {
        'bm25': reference['arms']['bm25'],
        'missing_terms': reference['arms']['complement'],
    }
    elapsed = 0.0
    if len(feasible) == 1:
        arms.update(dense=arms['bm25'], dense_residual=arms['bm25'])
    else:
        started = time.perf_counter()
        question, first_passage, *choices = model.encode([
            INSTRUCTION + record['question'],
            passages[first],
            *(passages[i] for i in feasible),
        ])
        # Remove the query component aligned with the first passage.
        residual = question - max(0.0, float(question @ first_passage)) * first_passage
        picks = {
            'dense': max(range(len(choices)), key=lambda i: float(choices[i] @ question)),
            'dense_residual': max(range(len(choices)), key=lambda i: float(choices[i] @ residual)),
        }
        elapsed = time.perf_counter() - started
        # Gold labels only score the resulting indices.
        supporting = {title for title, _ in record['supporting_facts']}
        for arm, picked in picks.items():
            index = feasible[picked]
            total_chars = len(passages[first]) + len(passages[index]) + 1
            if total_chars > arms['bm25']['chars']:
                raise ValueError('Character cap exceeded')
            arms[arm] = {
                'all_supporting_docs': supporting <= {titles[first], titles[index]},
                'chars': total_chars,
            }
    return {
        'id': record['_id'],
        'type': record['type'],
        'has_alternative': len(feasible) > 1,
        'joint_encode_and_score_seconds': elapsed,
        'arms': arms,
    }


def summarize(rows):
    baseline = 'missing_terms'
    measures = {}
    for arm in ARMS:
        entries = [row['arms'][arm] for row in rows]
        measures[arm] = {
            'complete': sum(entry['all_supporting_docs'] for entry in entries),
            'mean_chars': mean(entry['chars'] for entry in entries),
        }
    paired = {}
    for a, b in [('dense', baseline), ('dense_residual', 'dense'), ('dense_residual', baseline)]:
        difference = [
            int(row['arms'][a]['all_supporting_docs']) - int(row['arms'][b]['all_supporting_docs'])
            for row in rows
        ]
        paired[f'{a}_minus_{b}'] = {
            **paired_interval(difference, repeats=2000),
            'improved': difference.count(1),
            'degraded': difference.count(-1),
        }
    return {'questions': len(rows), 'arms': measures, 'paired': paired}


def main(dataset, output):
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        raise RuntimeError('Run inside a Slurm allocation')
    if output.exists():
        raise FileExistsError(output)
    if sha(dataset) != DATA_SHA256:
        raise ValueError('Changed HotpotQA distractor development file')
    model_sha = verify_model()
    model = QwenTorchEmbedding(max_length=MAX_TOKENS, batch_size=CANDIDATE_LIMIT + 1)
    model.encode(['Warm up the embedding model.'])
    rows = []
    excluded = []
    for record in json.loads(dataset.read_text()):
        reference = assess(record)
        if reference is None:
            excluded.append(record['_id'])
            continue
        rows.append(evaluate(record, reference, model))
        if len(rows) % 1000 == 0:
            print(f'Processed {len(rows)} questions', flush=True)

    eligible = [row for row in rows if row['has_alternative']]
    if len(rows) != 7404 or len(excluded) != 1:
        raise ValueError('Changed query cohort')
    totals = summarize(rows)
    if (totals['arms']['bm25']['complete'], totals['arms']['missing_terms']['complete']) != (2330, 2562):
        raise ValueError('The lexical controls do not reproduce the original replay')
    summary = {
        'job_id': job_id,
        'gpu': model.torch.cuda.get_device_name(),
        'model_manifest_sha256': model_sha,
        'script_sha256': sha(Path(__file__)),
        'dataset_sha256': DATA_SHA256,
        'excluded_invalid_support_indices': excluded,
        'embedding_max_tokens': MAX_TOKENS,
        'all': totals,
        'with_alternative': summarize(eligible),
        'mean_joint_encode_and_score_seconds': mean(row['joint_encode_and_score_seconds'] for row in eligible),
        'scope': (
            'Given HotpotQA distractor paragraphs, fixed first BM25 passage, and the same '
            'length-capped choices. Frozen embedding cosine versus projection residual. '
            'Timing covers the joint embedding and scoring of both semantic arms on '
            'queries with alternatives, excluding BM25, I/O, and model initialization. '
            'Paired intervals resample queries, ignoring possible shared articles.'
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary['rows_sha256'] = sha(output / 'rows.json')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.output)
