"""Test a small learned switch between flat ranking and offline article links."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.hotpotqa.semantic_complement import INSTRUCTION, verify_model
from experiments.hotpotqa.shared_source_navigation import (
    CHAR_CAP, DATA_SHA256, SEED, build_links, build_sources, choose, frozen_pool, measure, normalize, sha, terms,
)

HOLDOUT = 512
PLACEBOS = 32
COMPARISON = re.compile(
    r'\b(both|either|neither|same|older|younger|earlier|later|longer|shorter|more|less|larger|smaller|'
    r'first|last|further|closer|higher|lower|more recent|which of)\b', re.I
)
METHODS = ('dense', 'links', 'comparison_rule', 'learned')
FEATURES = ('comparison_pattern', 'question_terms', 'first_similarity', 'first_to_dense_gap',
            'dense_to_link_gap', 'number_of_links', 'link_in_question', 'dense_in_question',
            'two_alternatives_term_overlap', 'linkable_candidates_share')


def support_titles(row):
    return {normalize(title) for title, _ in row['supporting_facts']}


def make_cohorts(data):
    pool, train = frozen_pool(data)
    forbidden_titles = set().union(*(support_titles(r) for r in train))
    forbidden_questions = {normalize(r['question']) for r in train}
    remainder = pool[len(train):]
    test = [row for row in remainder if not support_titles(row) & forbidden_titles
            and normalize(row['question']) not in forbidden_questions][:HOLDOUT]
    if len(test) != HOLDOUT or {row['_id'] for row in train} & {row['_id'] for row in test}:
        raise ValueError('Could not form disjoint label cohorts')
    return pool, train, test


def paper_clusters(rows):
    representative = {}
    titles_per_question = []
    for row in rows:
        a, b = sorted(support_titles(row))
        representative.setdefault(a, a)
        representative.setdefault(b, b)
        titles_per_question.append(a)

        def root(title):
            while title != representative[title]:
                title = representative[title]
            return title

        representative[root(b)] = root(a)
    result = []
    for title in titles_per_question:
        while title != representative[title]:
            title = representative[title]
        result.append(title)
    return result


def query_state(row, model, embeddings, lengths, eligible_first):
    started = perf_counter()
    q = model.encode([INSTRUCTION + row['question']])[0]
    dense = torch.mv(embeddings, torch.tensor(q, device='cuda')).cpu().numpy()
    first = choose(dense, eligible_first)
    allowed = lengths + lengths[first] + 1 <= CHAR_CAP
    allowed[first] = False
    candidates = np.flatnonzero(allowed)
    base = choose(dense, candidates)
    elapsed = (perf_counter() - started) * 1000
    return dense, first, allowed, candidates, base, elapsed


def linked_choice(dense, first, allowed, links, base):
    neighbors = links[first][allowed[links[first]]]
    return (choose(dense, neighbors) if len(neighbors) else base), neighbors


def signal(question, titles, scores, first, base, linked, n_links, n_candidates):
    q = set(terms(question))
    b, link = (set(terms(titles[i])) for i in (base, linked))
    return [
        int(bool(COMPARISON.search(question))), len(q), float(scores[first]), float(scores[first] - scores[base]),
        float(scores[base] - scores[linked]), float(np.log1p(n_links)), len(q & link) / max(1, len(link)),
        len(q & b) / max(1, len(b)), len(b & link) / max(1, len(b | link)), n_links / n_candidates,
    ]


def evidence(row, first, second, titles, passages, sentences):
    n = len(passages[first]) + len(passages[second]) + 1
    return measure(row, first, second, titles, sentences, n)


def fit(examples):
    x = np.array([x for x, _, _ in examples], dtype=np.float64)
    y = np.array([int(link['both_sources']) - int(base['both_sources']) for _, base, link in examples])
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    x = np.column_stack((np.ones(len(x)), (x - center) / scale))
    weights = np.linalg.solve(x.T @ x + np.diag([0.] + [10.] * len(center)), x.T @ y)
    return center, scale, weights


def run(dataset, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    if sha(dataset) != DATA_SHA256:
        raise ValueError('Changed source benchmark')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model_hash = verify_model()
    pool, training, testing = make_cohorts(json.loads(dataset.read_text()))
    titles, passages, sentences = build_sources(pool)
    started = perf_counter()
    links, _ = build_links(titles, passages)
    graph_seconds = perf_counter() - started
    lengths = np.array([len(p) for p in passages])
    eligible_first = np.flatnonzero(lengths + int(lengths.min()) + 1 <= CHAR_CAP)
    model = QwenTorchEmbedding(max_length=384, batch_size=32)
    model.model.config.use_cache = False
    started = perf_counter()
    embeddings = torch.tensor(model.encode(passages), device='cuda')
    torch.cuda.synchronize()
    index_seconds = perf_counter() - started
    model.encode([INSTRUCTION + training[0]['question']])

    def state(row):
        return query_state(row, model, embeddings, lengths, eligible_first)

    examples = []
    skip = 0
    for row in training:
        dense, first, allowed, candidates, base, _ = state(row)
        linked, neighbors = linked_choice(dense, first, allowed, links, base)
        if linked == base:
            skip += 1
            continue
        features = signal(row['question'], titles, dense, first, base, linked, len(neighbors), len(candidates))
        examples.append((features, evidence(row, first, base, titles, passages, sentences),
                         evidence(row, first, linked, titles, passages, sentences)))
    started = perf_counter()
    center, scale, weights = fit(examples)
    fitting_seconds = perf_counter() - started
    rng = np.random.default_rng(SEED + 1)
    records = []
    for row in testing:
        dense, first, allowed, candidates, base, common = state(row)
        order = rng.permutation(METHODS)
        arms = {}
        linked_reference, neighbors = linked_choice(dense, first, allowed, links, base)
        for method in order:
            started = perf_counter()
            selection = base
            if method != 'dense':
                linked, linked_pool = linked_choice(dense, first, allowed, links, base)
                if method == 'links' or (method == 'comparison_rule' and not COMPARISON.search(row['question'])):
                    selection = linked
                elif method == 'learned' and linked != base:
                    features = signal(row['question'], titles, dense, first, base, linked,
                                      len(linked_pool), len(candidates))
                    score = weights[0] + ((features - center) / scale) @ weights[1:]
                    if score > 0:
                        selection = linked
            payload = '\n'.join(passages[i] for i in (first, selection)).encode()
            elapsed_ms = (perf_counter() - started) * 1000
            arms[method] = {**evidence(row, first, selection, titles, passages, sentences),
                            'bytes': len(payload), 'decision_ms': elapsed_ms}
        gold = {title for title, _ in row['supporting_facts']}
        random_graph = []
        for _ in range(PLACEBOS):
            candidate = choose(dense, rng.choice(candidates, size=len(neighbors), replace=False)) if len(neighbors) else base
            random_graph.append({titles[first], titles[candidate]} == gold)
        records.append({
            'id': row['_id'], 'type': row['type'], 'first': first, 'common_ms': common,
            'linked_available': len(neighbors), 'arms_differ': linked_reference != base,
            'arms': arms, 'matched_degree_random_links': random_graph,
        })
    clusters = paper_clusters(testing)
    results = {}
    for arm in METHODS:
        selected = [r['arms'][arm] for r in records]
        success = [int(x['both_sources']) for x in selected]
        results[arm] = {
            'complete': sum(success),
            'all_supporting_sentences': sum(r['all_supporting_sentences'] for r in selected),
            'mean_chars': mean(r['characters'] for r in selected),
            'p95_total_ms': float(np.quantile([r['common_ms'] + x['decision_ms']
                                               for r, x in zip(records, selected, strict=True)], .95)),
            'per_question_type': {t: [sum(r['arms'][arm]['both_sources'] for r in records if r['type'] == t),
                                      sum(r['type'] == t for r in records)] for t in ('bridge', 'comparison')},
            'paired_vs_flat': clustered_paired_interval(
                [y - int(r['arms']['dense']['both_sources']) for y, r in zip(success, records, strict=True)],
                clusters, repeats=4000),
        }
    changed = [r for r in records if r['arms_differ']]
    selected_links = sum(r['arms']['learned']['second'] == r['arms']['links']['second'] for r in changed)
    random_gate = []
    for _ in range(1000):
        ids = set(rng.choice(len(changed), size=selected_links, replace=False).tolist())
        random_gate.append(sum(r['arms']['links' if i in ids else 'dense']['both_sources']
                               - r['arms']['dense']['both_sources'] for i, r in enumerate(changed)))
    default_hits = results['dense']['complete']
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'rows.json'
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    summary = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(), 'program_sha256': sha(Path(__file__)),
        'dataset_sha256': DATA_SHA256, 'model_sha256': model_hash, 'rows_sha256': sha(path),
        'pool_sha256': hashlib.sha256(json.dumps(passages).encode()).hexdigest(),
        'sources': len(titles), 'train_questions': len(training), 'fit_decisions': len(examples), 'skip_equal': skip,
        'heldout_questions': len(testing), 'heldout_support_clusters': len(set(clusters)), 'char_cap': CHAR_CAP,
        'build_links_seconds': graph_seconds, 'build_dense_seconds': index_seconds, 'fit_seconds': fitting_seconds,
        'features': FEATURES, 'center': center.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(),
        'learned_link_choices_where_arms_differ': selected_links, 'disagreement_queries': len(changed),
        'oracle_either_arm': sum(any(r['arms'][a]['both_sources'] for a in ('dense', 'links')) for r in records),
        'matched_rate_random_gate': {
            'mean_hits': default_hits + mean(random_gate),
            'central_95': (np.quantile(random_gate, [.025, .975]) + default_hits).tolist(),
        },
        'matched_degree_random_links': {
            'mean_hits': sum(sum(r['matched_degree_random_links']) for r in records) / PLACEBOS,
            'repeats': PLACEBOS,
        },
        'learned_minus_links': clustered_paired_interval(
            [r['arms']['learned']['both_sources'] - r['arms']['links']['both_sources'] for r in records],
            clusters, repeats=4000),
        'outcomes': results,
        'scope': 'Previously reused HotpotQA development benchmark, new questions for this policy, with '
                 'disjoint normalized gold titles and exact questions between fitting and held-out grading. '
                 'A common source corpus includes both cohorts and additional supplied contexts. The labels '
                 'are used for split hygiene and outcome training, not as router inputs. No enterprise fit, '
                 'full-Wikipedia retrieval, citation entailment, or WAN latency.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
