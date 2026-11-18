"""Measure trained source-aware reranking in a fixed, globally retrieved CoF pool."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.retrieval import tokens
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions, normalize_text
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, serialize_hits, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, compare_labels

SHORTLIST = 20
SOURCES = 3
METHODS = ('dense', 'dense_trained', 'one_source', 'complementary')
DIMENSIONS = {'dense_trained': 4, 'one_source': 7, 'complementary': 11}
STOP = set('the and for are that with from this into using has use can not our which these such'.split())
SEED = 20261121


def terms(text):
    return {token for token in tokens(text) if len(token) >= 3 and token not in STOP}


def retrieve(ranker, vector):
    raw = ranker.score(vector).cpu().numpy()
    idx = ranker.index
    maxima = np.maximum.reduceat(raw[idx.owner_rows], idx.owner_starts)
    owners = np.argsort(-maxima, kind='stable')[:SHORTLIST]
    return raw, [(int(o), float(maxima[o])) for o in owners]


def candidate_sources(ranker, values, owners, width):
    slots = []
    for owner, score in owners:
        indices = np.asarray(ranker.index.owners[ranker.index.owner_ids[owner]])
        local = values[indices]
        order = np.argsort(-local, kind='stable')[:width]
        slots.append((owner, indices[order].tolist(), score, len(indices)))
    return slots


def metadata(ranker, conn, source_ids):
    ids = [ranker.index.source_ids[i] for i in sorted(source_ids)]
    params = ','.join('?' for _ in ids)
    return {key: (title, excerpt) for key, title, excerpt in conn.execute(
        f'SELECT id, title, excerpt FROM source WHERE id IN ({params})', ids
    )}


def features(ranker, conn, query, candidates, values, width):
    query_terms = terms(query)
    size = max(1, len(query_terms))
    if width:
        documents = metadata(ranker, conn, {i for _, sources, _, _ in candidates for i in sources})
        tokenized = {key: (terms(title), terms(excerpt)) for key, (title, excerpt) in documents.items()}
    rows = []
    for position, (_, sources, maximum, profile_size) in enumerate(candidates):
        signals = [maximum, maximum - candidates[0][2], position / SHORTLIST, np.log1p(profile_size)]
        if width:
            title_terms, first_terms = tokenized[ranker.index.source_ids[sources[0]]]
            seen = query_terms & first_terms
            signals += [len(seen) / size, len(query_terms & title_terms) / size, np.log1p(len(first_terms))]
            if width > 1:
                missing = query_terms - seen
                alternatives = [(i, tokenized[ranker.index.source_ids[i]][1]) for i in sources[1:]]
                if alternatives:
                    best, extra = max(alternatives, key=lambda item: len(missing & item[1]))
                    signals += [len(extra & missing) / size, len((seen | extra) & query_terms) / size,
                                len(first_terms & extra) / max(1, len(first_terms | extra)),
                                maximum - float(values[best])]
                else:
                    signals += [0., len(seen) / size, 0., 0.]
        rows.append(signals)
    return np.asarray(rows, dtype=np.float32)


def fit(x, y, width):
    torch.manual_seed(SEED)
    x = torch.tensor(x[:, :width], device='cuda')
    y = torch.tensor(y, device='cuda', dtype=torch.float32)
    avg, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    transformed = (x - avg) / scale
    net = torch.nn.Linear(width, 1).cuda()
    start = torch.cat([p.detach().ravel().clone() for p in net.parameters()])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01)
    for _ in range(24):
        for inds in torch.randperm(len(x), device='cuda').split(256):
            prediction = net(transformed[inds]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, y[inds])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    current = torch.cat([p.detach().ravel() for p in net.parameters()])
    return {'center': avg.cpu().numpy(), 'scale': scale.cpu().numpy(),
            'weight': net.weight.detach().cpu().numpy()[0], 'bias': net.bias.detach().cpu().numpy()[0],
            'updated': int(torch.count_nonzero(start != current))}


def assemble(ranker, conn, encoder, queries, labels, papers):
    rows, target, exposed = [], [], set()
    for qid, query in queries.items():
        values, owners = retrieve(ranker, encoder.encode([query], query=True)[0])
        candidates = candidate_sources(ranker, values, owners, SOURCES)
        f = features(ranker, conn, query, candidates, values, SOURCES)
        for row, (owner, sources, _, _) in zip(f, candidates, strict=True):
            expert = ranker.index.owner_ids[owner]
            if expert not in labels['hard'][qid]:
                continue
            rows.append(row)
            target.append(int(labels['hard'][qid][expert] == 1))
            exposed.update(ranker.index.source_ids[i] for i in sources)
    return np.asarray(rows), np.asarray(target), {normalize_text(papers[key]['title']) for key in exposed}, exposed


def respond(ranker, conn, encoder, query, method, model=None):
    started = time.perf_counter()
    values, owners = retrieve(ranker, encoder.encode([query], query=True)[0])
    width = 3 if method == 'complementary' else 1
    candidates = candidate_sources(ranker, values, owners, width)
    if model is not None:
        read_width = width if method != 'dense_trained' else 0
        x = features(ranker, conn, query, candidates, values, read_width)
        score = ((x - model['center']) / model['scale']) @ model['weight'] + model['bias']
        order = np.argsort(-score, kind='stable')[:5]
    else:
        order = np.arange(5)
    top = [(candidates[i][0], candidates[i][1][0], candidates[i][2]) for i in order]
    _, ids = serialize_hits(ranker, conn, top)
    return ids, (time.perf_counter() - started) * 1000


def evaluate(ranker, conn, encoder, queries, labels, models):
    order = list(METHODS)
    warm = next(iter(queries.values()))
    for method in order:
        respond(ranker, conn, encoder, warm, method, models.get(method))
    rows = []
    for i, (qid, query) in enumerate(queries.items()):
        offset = i % len(order)
        output = {method: respond(ranker, conn, encoder, query, method, models.get(method))
                  for method in order[offset:] + order[:offset]}
        reference = output['dense'][0]
        rows.append({'query': qid, 'arms': {
            method: {'ms': latency, 'experts': ranking, 'label_metrics': {
                t: compare_labels(reference, ranking, by_query[qid]) for t, by_query in labels.items()
            }} for method, (ranking, latency) in output.items()
        }})
    return rows


def summarize(rows):
    summary = {}
    for method in METHODS:
        arms = [row['arms'][method] for row in rows]
        metric = {}
        for label in ('soft', 'hard'):
            values = [a['label_metrics'][label] for a in arms]
            metric[label] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            metric[label]['paired_known_difference'] = paired_interval(
                [v['known_difference'] for v in values], repeats=4000
            )
        med, p95 = np.quantile([a['ms'] for a in arms], [.5, .95])
        summary[method] = {'p50_ms': float(med), 'p95_ms': float(p95), 'labels': metric}
    return summary


def source_exclusions(train_queries):
    remove_q, remove_s, _ = exclusions()
    _, papers, _, labels = load(DATASET)
    train_ids, train_titles, source_ids, source_titles = train_queries
    for q in labels['soft']:
        if q in train_ids or normalize_text(papers[q]['title']) in train_titles:
            remove_q.add(q)
    for pid, paper in papers.items():
        if pid in source_ids or normalize_text(paper['title']) in source_titles:
            remove_s.add(pid)
    return remove_q, remove_s


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed checkpoint')
    setup = QueryEncoderVariants()
    setup.configure('no_cache')
    encoder = setup.encoder
    output.mkdir(parents=True, exist_ok=False)
    root, papers, _, _ = load('KDD')
    inventory = json.loads((root / 'manifest.json').read_text())
    omit = set(inventory['query_ids_in_profiles']) | set(inventory['exact_query_texts_in_profiles'])
    ranker, db, queries, labels, train_meta = prepare('KDD', output, checkpoint, exclude_queries=omit)
    with sqlite3.connect(db) as conn:
        x, y, source_titles, source_ids = assemble(ranker, conn, encoder, queries, labels, papers)
    db.unlink()
    del ranker
    query_titles = {normalize_text(papers[q]['title']) for q in queries}
    remove_q, remove_s = source_exclusions((set(queries), query_titles, source_ids, source_titles))
    models, saved = {}, {}
    for name, dimension in DIMENSIONS.items():
        model = fit(x, y, dimension)
        path = output / f'{name}.npz'
        np.savez(path, **model)
        models[name] = model
        saved[name] = {'sha256': sha(path), 'updated': model['updated']}
    print('Trained', len(x), 'labeled candidate actions', flush=True)
    ranker, db, eval_queries, eval_labels, eval_meta = prepare(
        DATASET, output, checkpoint, exclude_queries=remove_q, exclude_sources=remove_s
    )
    with sqlite3.connect(db) as conn:
        rows = evaluate(ranker, conn, encoder, eval_queries, eval_labels, models)
    db.unlink()
    path = output / 'rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    summary = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
               'rows_sha256': sha(path), 'script_sha256': sha(Path(__file__)), 'encoder_sha256': sha(checkpoint),
               'train': train_meta, 'evaluation': eval_meta, 'train_pairs': len(y), 'train_positives': int(y.sum()),
               'train_sources': len(source_ids), 'models': saved, 'metrics': summarize(rows),
               'scope': 'KDD hard reviewer ratings supervise 20 globally retrieved candidates. '
                        'Held-out filtered SciRepEval uses incomplete reviewer ratings. '
                        'Remove exact normalized-title or ID overlap with training query/features. '
                        'Evidence content is title and up to 500 characters; best dense citation '
                        'still returned for each expert. Timing includes GPU encoding, retrieval, '
                        'candidate feature reads and rendering; no internet or evidence verification.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
