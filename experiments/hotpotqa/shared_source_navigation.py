"""Evaluate offline article links for cross-source retrieval on a shared HotpotQA pool."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.hotpotqa.second_document_complement import DATA_SHA256, sha
from experiments.hotpotqa.semantic_complement import INSTRUCTION, verify_model
from experiments.retrieval.evidencebench_block_navigation import Sparse, terms

POOL_QUESTIONS = 1536
EVAL_QUESTIONS = 512
CHAR_CAP = 1600
MAX_TITLE_WORDS = 6
LINK_BONUS = .06
SEED = 20261122
ARMS = ('dense', 'dense_residual', 'bm25_bridge', 'links_first', 'links_bonus')


def normalize(text):
    return ' '.join(text.casefold().split())


def frozen_pool(data):
    valid = []
    for row in data:
        positions = {title: sentences for title, sentences in row['context']}
        facts = {tuple(fact) for fact in row['supporting_facts']}
        if len(positions) != len(row['context']) or len({title for title, _ in facts}) != 2:
            continue
        if any(title not in positions or type(i) is not int or not 0 <= i < len(positions[title])
               for title, i in facts):
            continue
        valid.append(row)
    ordered = sorted(valid, key=lambda r: hashlib.sha256(f"{SEED}:{r['_id']}".encode()).digest())
    selected = ordered[:POOL_QUESTIONS]
    if len(selected) != POOL_QUESTIONS or len({r['_id'] for r in selected}) != len(selected):
        raise ValueError('Invalid shared pool sample')
    return selected, selected[:EVAL_QUESTIONS]


def build_sources(pool):
    documents = {}
    for row in pool:
        for title, sentences in row['context']:
            text = '\n'.join([title, *sentences]).strip()
            candidate = (len(text), text, tuple(sentences))
            if title not in documents or candidate > documents[title]:
                documents[title] = candidate
    titles = sorted(documents)
    passages = [documents[t][1] for t in titles]
    sentences = [{normalize(s) for s in documents[t][2]} for t in titles]
    return titles, passages, sentences


def build_links(titles, passages):
    aliases = defaultdict(list)
    for i, title in enumerate(titles):
        alias = tuple(terms(title))
        if 2 <= len(alias) <= MAX_TITLE_WORDS:
            aliases[alias].append(i)
    usable = {alias: ids[0] for alias, ids in aliases.items() if len(ids) == 1}
    prefixes = {alias[0] for alias in usable}
    sizes = sorted({len(alias) for alias in usable})
    links = []
    for own, passage in enumerate(passages):
        content = terms(passage.split('\n', 1)[1] if '\n' in passage else '')
        matches = {usable[alias] for start, token in enumerate(content) if token in prefixes
                   for size in sizes if (alias := tuple(content[start:start + size])) in usable}
        links.append(np.array(sorted(matches - {own}), dtype=np.int32))
    return links, len(usable)


def choose(scores, feasible):
    return int(feasible[np.argmax(scores[feasible])])


def measure(row, first, next_, titles, sentences, characters):
    target = {title for title, _ in row['supporting_facts']}
    cited = {titles[first]: first, titles[next_]: next_}
    original = dict(row['context'])
    facts = set(map(tuple, row['supporting_facts']))
    recovered = sum(t in cited and normalize(original[t][i]) in sentences[cited[t]] for t, i in facts)
    if characters > CHAR_CAP:
        raise ValueError('Evidence read budget exceeded')
    return {'second': next_, 'characters': characters, 'both_sources': target <= cited.keys(),
            'all_supporting_sentences': recovered == len(facts), 'sentence_recall': recovered / len(facts)}


def run(dataset, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    if sha(dataset) != DATA_SHA256:
        raise ValueError('Changed source benchmark')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model_hash = verify_model()
    pool, evaluation = frozen_pool(json.loads(dataset.read_text()))
    started = perf_counter()
    titles, passages, sentences = build_sources(pool)
    source_seconds = perf_counter() - started
    title_position = {name: i for i, name in enumerate(titles)}
    lengths = np.array([len(text) for text in passages])
    started = perf_counter()
    links, aliases = build_links(titles, passages)
    link_seconds = perf_counter() - started
    started = perf_counter()
    sparse = Sparse(passages)
    sparse_seconds = perf_counter() - started
    model = QwenTorchEmbedding(max_length=384, batch_size=32)
    model.model.config.use_cache = False
    started = perf_counter()
    encoded = model.encode(passages)
    embeddings = torch.tensor(encoded, device='cuda')
    torch.cuda.synchronize()
    embedding_seconds = perf_counter() - started
    shortest = int(np.min(lengths))
    eligible_first = np.flatnonzero(lengths + shortest + 1 <= CHAR_CAP)
    warmed = model.encode([INSTRUCTION + evaluation[0]['question']])[0]
    torch.mv(embeddings, torch.tensor(warmed, device='cuda')).cpu()
    rows = []
    rng = random.Random(SEED)
    for row in evaluation:
        started = perf_counter()
        q = model.encode([INSTRUCTION + row['question']])[0]
        query = torch.tensor(q, device='cuda')
        dense = torch.mv(embeddings, query).cpu().numpy()
        first = choose(dense, eligible_first)
        allowed = lengths + lengths[first] + 1 <= CHAR_CAP
        allowed[first] = False
        candidates = np.flatnonzero(allowed)
        if not len(candidates):
            raise ValueError('No feasible second source')
        common_seconds = perf_counter() - started
        expected = {title_position[t] for t, _ in row['supporting_facts']}
        missing = expected - {first}
        available_links = links[first][allowed[links[first]]]
        info = {
            'id': row['_id'], 'first': first, 'first_is_supporting': first in expected,
            'global_feasible_second': len(missing) == 1 and all(allowed[i] for i in missing),
            'graph_feasible_second': len(missing) == 1 and missing.issubset(available_links),
            'links_available': len(available_links), 'common_ms': common_seconds * 1000, 'arms': {},
        }
        order = list(ARMS)
        rng.shuffle(order)
        for arm in order:
            started = perf_counter()
            if arm == 'dense':
                second = choose(dense, candidates)
            elif arm == 'dense_residual':
                residual = query - max(0., float(dense[first])) * embeddings[first]
                second = choose(torch.mv(embeddings, residual).cpu().numpy(), candidates)
            elif arm == 'bm25_bridge':
                tokens = dict.fromkeys(terms(row['question']) + terms(titles[first]))
                second = choose(sparse.score(tokens)[0], candidates)
            else:
                neighbors = links[first][allowed[links[first]]]
                if arm == 'links_first':
                    second = choose(dense, neighbors if len(neighbors) else candidates)
                else:
                    boosted = dense.copy()
                    boosted[neighbors] += LINK_BONUS
                    second = choose(boosted, candidates)
            payload = '\n'.join(passages[i] for i in (first, second))
            _ = payload.encode()
            elapsed = (perf_counter() - started) * 1000
            result = measure(row, first, second, titles, sentences, len(payload))
            info['arms'][arm] = {**result, 'second_stage_ms': elapsed}
        rows.append(info)
    outcomes = {}
    for arm in ARMS:
        selected = [r['arms'][arm] for r in rows]
        outcomes[arm] = {
            'both_sources': sum(item['both_sources'] for item in selected),
            'all_supporting_sentences': sum(item['all_supporting_sentences'] for item in selected),
            'mean_sentence_recall': mean(item['sentence_recall'] for item in selected),
            'mean_characters': mean(item['characters'] for item in selected),
            'p95_total_ms': float(np.quantile([r['common_ms'] + item['second_stage_ms']
                                               for r, item in zip(rows, selected, strict=True)], .95)),
            'p95_second_stage_ms': float(np.quantile([item['second_stage_ms'] for item in selected], .95)),
            'source_difference_vs_dense': paired_interval(
                [int(r['arms'][arm]['both_sources']) - int(r['arms']['dense']['both_sources']) for r in rows],
                repeats=4000),
        }
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    summary = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'dataset_sha256': DATA_SHA256, 'model_sha256': model_hash, 'script_sha256': sha(Path(__file__)),
        'rows_sha256': sha(path), 'source_text_sha256': hashlib.sha256(json.dumps(passages).encode()).hexdigest(),
        'queries': len(rows), 'pool_questions': len(pool), 'sources': len(titles), 'char_cap': CHAR_CAP,
        'source_selection_seconds': source_seconds, 'build_links_seconds': link_seconds,
        'build_bm25_seconds': sparse_seconds, 'build_dense_seconds': embedding_seconds,
        'unique_aliases': aliases, 'link_edges': sum(map(len, links)),
        'sources_with_links': sum(bool(len(a)) for a in links),
        'first_supporting': sum(row['first_is_supporting'] for row in rows),
        'reachable_with_shared_index': sum(row['global_feasible_second'] for row in rows),
        'reachable_with_links_only': sum(row['graph_feasible_second'] for row in rows),
        'queries_with_eligible_links': sum(row['links_available'] > 0 for row in rows),
        'arms': outcomes,
        'scope': 'Reused development benchmark. Each query searches one shared title-deduplicated corpus, '
                 'including its supplied passages and other questions’ passages. First source and budget are '
                 'fixed across arms. Gold titles and exact normalized supporting sentence matches are checked '
                 'against returned passages. No official fullwiki evaluation, expert labels or semantic verification.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
