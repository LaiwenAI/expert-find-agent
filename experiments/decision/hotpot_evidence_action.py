"""Train a budget-constrained document selection policy on public HotpotQA labels."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval
from experiments.hotpotqa.second_document_complement import CANDIDATE_LIMIT, DATA_SHA256, bm25, words

HF_REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
TRAIN_SHA256 = {
    'train-00000-of-00002.parquet': '76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68',
    'train-00001-of-00002.parquet': '713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f',
}
SEED = 20261119
FEATURES = (
    'bm25_rank', 'question_terms_covered', 'missing_terms_covered',
    'first_passage_terms_shared', 'title_term_overlap', 'length_vs_bm25',
    'question_term_count', 'first_passage_question_coverage', 'candidate_term_count',
)
EPOCHS = 16
LIMITS = {'train': 12000, 'validation': 2000}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(question):
    return ' '.join(question.casefold().split())


def candidate_actions(question, context):
    titles, sentences = zip(*context, strict=True)
    text = ['\n'.join([title, *parts]) for title, parts in zip(titles, sentences, strict=True)]
    first, second, *remaining = bm25(question, text)[:CANDIDATE_LIMIT]
    cap = len(text[second])
    actions = [second] + [p for p in remaining if len(text[p]) <= cap]
    rank = {doc: i + 2 for i, doc in enumerate([second, *remaining])}
    return titles, text, first, actions, rank


def action_features(question, titles, text, first, actions, rank):
    query = words(question)
    consumed = words(text[first])
    missing = query - consumed
    cap = len(text[actions[0]])
    features = np.zeros((len(actions), len(FEATURES)), dtype=np.float32)
    for slot, doc in enumerate(actions):
        tokens = words(text[doc])
        features[slot] = [
            rank[doc], len(query & tokens) / max(len(query), 1),
            len(missing & tokens) / max(len(missing), 1),
            len(consumed & tokens) / max(len(tokens), 1),
            len(words(titles[doc]) & query) / max(len(query), 1),
            len(text[doc]) / max(cap, 1), len(query),
            len(consumed & query) / max(len(query), 1), len(tokens),
        ]
    return features


def examples(row):
    question = row['question']
    context = list(row['context'])
    available = {title: len(sentences) for title, sentences in context}
    if len(available) != len(context):
        return None
    facts = {tuple(fact) for fact in row['supporting_facts']}
    if not facts or any(title not in available or type(i) is not int or not 0 <= i < available[title]
                        for title, i in facts):
        return None
    target = {title for title, _ in facts}
    if len(target) < 2:
        return None

    titles, text, first, actions, rank = candidate_actions(question, context)
    count = len(actions)
    features = np.zeros((CANDIDATE_LIMIT - 1, len(FEATURES)), dtype=np.float32)
    features[:count] = action_features(question, titles, text, first, actions, rank)
    outcomes = np.zeros(CANDIDATE_LIMIT - 1, dtype=np.float32)
    lengths = np.zeros(CANDIDATE_LIMIT - 1, dtype=np.int32)
    for slot, doc in enumerate(actions):
        outcomes[slot] = target <= {titles[first], titles[doc]}
        lengths[slot] = len(text[first]) + 1 + len(text[doc])
    return row['_id'], features, outcomes, lengths, count, int(features[:count, 2].argmax())


def training_rows(parquets):
    import pyarrow.parquet as pq

    for path in sorted(parquets):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=2048, columns=['id', 'question', 'context', 'supporting_facts']
        ):
            for row in batch.to_pylist():
                context, facts = row['context'], row['supporting_facts']
                yield {'_id': row['id'], 'question': row['question'],
                       'context': zip(context['title'], context['sentences'], strict=True),
                       'supporting_facts': zip(facts['title'], facts['sent_id'], strict=True)}


def prepare(dev, parquets, output):
    if sha(dev) != DATA_SHA256 or len(parquets) != 2:
        raise ValueError('Unexpected dev data or training inventory')
    training_hashes = {p.name: sha(p) for p in parquets}
    if training_hashes != TRAIN_SHA256:
        raise ValueError('Changed training shards')
    development = json.loads(dev.read_text())
    forbidden = {normalize(r['question']) for r in development}
    dev_ids = {r['_id'] for r in development}
    folds = {k: [] for k in (*LIMITS, 'dev')}
    seen = set()
    for row in training_rows(parquets):
        question = normalize(row['question'])
        if question in forbidden or question in seen or row['_id'] in dev_ids:
            continue
        digest = hashlib.sha256(f'{SEED}:{question}'.encode()).digest()
        if digest[1] >= 55:
            continue
        split = 'validation' if digest[0] % 6 == 0 else 'train'
        if len(folds[split]) >= LIMITS[split]:
            continue
        record = examples(row)
        seen.add(question)
        if record is not None:
            folds[split].append(record)
        if all(len(folds[k]) == n for k, n in LIMITS.items()):
            break
    folds['dev'] = [x for row in development if (x := examples(row)) is not None]
    arrays = {}
    for fold, records in folds.items():
        if len(records) != {'dev': 7404, **LIMITS}[fold]:
            raise ValueError(f'Unexpected {fold} size: {len(records)}')
        for key, values in zip(('ids', 'features', 'outcomes', 'chars', 'count', 'complement'),
                               zip(*records, strict=True), strict=True):
            arrays[f'{fold}_{key}'] = np.array(values)
    arrays['metadata'] = np.array(json.dumps({
        'revision': HF_REVISION, 'inputs_sha256': {dev.name: DATA_SHA256, **training_hashes},
        'script_sha256': sha(Path(__file__)), 'features': FEATURES,
        'scope': 'Public distractor passages only. Exclude exact normalized development questions '
                 'and dev IDs from train/validation. Gold is used only for supervision and evaluation.',
    }))
    np.savez_compressed(output, **arrays)


def baseline_metrics(y, chars, chosen):
    indices = np.arange(len(chosen))
    if np.any(chars[indices, chosen] > chars[:, 0]):
        raise ValueError('Context budget violation')
    return y[indices, chosen], {
        'complete': int(y[indices, chosen].sum()),
        'mean_chars': float(chars[indices, chosen].mean()),
        'changed_actions': int(np.count_nonzero(chosen)),
    }


def fit(pack, output):
    import torch

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a Slurm CUDA allocation')
    torch.manual_seed(SEED)
    data = np.load(pack, allow_pickle=False)
    metadata = json.loads(str(data['metadata']))
    if metadata['script_sha256'] != sha(Path(__file__)):
        raise ValueError('Prepared data and training code differ')
    tensors = {}
    for fold in (*LIMITS, 'dev'):
        features, values, count = (data[f'{fold}_{key}'] for key in ('features', 'outcomes', 'count'))
        feasible = np.arange(features.shape[1])[None, :] < count[:, None]
        tensors[fold] = [torch.from_numpy(x).to('cuda') for x in (features, values, feasible)]
    train_x, train_y, feasible = tensors['train']
    observed = train_x[feasible]
    center, scale = observed.mean(0), observed.std(0).clamp_min(1e-6)
    for fold, (x, y, mask) in tensors.items():
        tensors[fold] = [(x - center) / scale, y, mask]
    train_x, train_y, feasible = tensors['train']

    dev_y = data['dev_outcomes']
    dev_chars = data['dev_chars']
    dev_mask = np.arange(dev_y.shape[1])[None, :] < data['dev_count'][:, None]
    choices = {
        'bm25': np.zeros(len(dev_y), dtype=int), 'missing_terms': data['dev_complement'],
        'oracle': np.where(dev_mask, dev_y, -1).argmax(1),
    }
    output.mkdir(parents=True, exist_ok=False)

    def predict(model, fold):
        x, labels, mask = tensors[fold]
        with torch.inference_mode():
            picked = model(x).squeeze(-1).masked_fill(~mask, -torch.inf).argmax(1)
            found = labels.gather(1, picked[:, None]).squeeze(1)
        return picked.cpu().numpy(), int(found.sum())

    models = {}
    for name in ('linear', 'mlp'):
        head = (torch.nn.Linear(len(FEATURES), 1) if name == 'linear' else torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), 24), torch.nn.ReLU(), torch.nn.Linear(24, 1)
        )).to('cuda')
        initial = [parameter.detach().clone() for parameter in head.parameters()]
        optimizer = torch.optim.AdamW(head.parameters(), lr=.003)
        checkpoint, history, best = None, [], -1
        for epoch in range(1, EPOCHS + 1):
            for ii in torch.randperm(len(train_x), device='cuda').split(256):
                mask = feasible[ii]
                logits = head(train_x[ii]).squeeze(-1)[mask]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y[ii][mask])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            _, completed = predict(head, 'validation')
            history.append({'epoch': epoch, 'complete': completed})
            if completed > best:
                best = completed
                checkpoint = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(checkpoint)
        choices[name], _ = predict(head, 'dev')
        weights = output / f'{name}.npz'
        np.savez(weights, center=center.cpu().numpy(), scale=scale.cpu().numpy(),
                 **{k: v.cpu().numpy() for k, v in checkpoint.items()})
        with np.load(weights) as exported:
            head.load_state_dict({k: torch.from_numpy(exported[k]) for k in checkpoint})
        assert np.array_equal(choices[name], predict(head, 'dev')[0])
        models[name] = {
            'validation': history, 'selected_epoch': max(history, key=lambda e: e['complete'])['epoch'],
            'parameters': sum(p.numel() for p in head.parameters()),
            'changed_parameters': sum(int(torch.count_nonzero(a != b)) for a, b in zip(initial, head.parameters(), strict=True)),
            'weights_sha256': sha(weights),
        }

    scores = {name: baseline_metrics(dev_y, dev_chars, choice) for name, choice in choices.items()}
    outcomes = {name: metrics for name, (_, metrics) in scores.items()}
    comparison = {}
    for name in ('linear', 'mlp'):
        for base in ('bm25', 'missing_terms') + (('linear',) if name == 'mlp' else ()):
            diff = (scores[name][0] - scores[base][0]).tolist()
            comparison[f'{name}_minus_{base}'] = {
                'improved': diff.count(1), 'degraded': diff.count(-1),
                'paired': paired_interval(diff, repeats=2000),
            }
    rows = output / 'replay.npz'
    np.savez_compressed(rows, ids=data['dev_ids'], **choices)
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'input_sha256': sha(pack), 'script_sha256': sha(Path(__file__)), 'input': metadata,
        'training_questions': len(train_x), 'validation_questions': len(tensors['validation'][0]),
        'development_questions': len(dev_y), 'models': models, 'outcomes': outcomes,
        'compared': comparison, 'replay_sha256': sha(rows),
        'scope': 'Offline action supervision on official training partitions. Fixed first BM25 '
                 'document and matched per-question two-document character cap. Previously explored '
                 'official development benchmark; no answer generation, expert judgments or serving timing.',
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dev', type=Path, required=True)
    prep.add_argument('--train', type=Path, nargs=2, required=True)
    prep.add_argument('--output', type=Path, required=True)
    trainer = sub.add_parser('fit')
    trainer.add_argument('--input', type=Path, required=True)
    trainer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.dev, args.train, args.output)
    else:
        fit(args.input, args.output)
