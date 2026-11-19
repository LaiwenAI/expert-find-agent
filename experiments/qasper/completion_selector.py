"""Fit a lightweight complementary-evidence selector at fixed retrieval budget."""

import argparse
from collections import Counter
from hashlib import sha256
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.adaptive_evidence import digest, evidence_sets, offline, query_input, read_data
from experiments.qasper.completion_retrieval import dense_order
from experiments.qasper.evidence_gap_audit import choose
from experiments.qasper.evidence_shortlist import terms
from experiments.qasper.remaining_evidence import TASK, excerpt, holdout
from experiments.qasper.set_repair import read_train


BUDGET = 4
REGULARIZATION = .25
FEATURE_NAMES = (
    "dense_cosine", "inverse_rank", "question_overlap", "candidate_length",
    "anchor_sim_max", "anchor_sim_mean", "same_section", "nearest_distance",
    "uncovered_question_overlap", "shared_anchor_overlap", "new_anchor_terms",
)
ABLATION = 4


def features(question, paragraphs, sections, bank, dense, shown, pool):
    word_sets = [set(terms(text)) for text in paragraphs]
    query = set(terms(question))
    seen = set().union(*(word_sets[i] for i in shown))
    missing = query - seen
    frequency = Counter(word for words in word_sets for word in words)
    idf = {word: math.log1p((len(paragraphs) - frequency[word] + .5) / (frequency[word] + .5))
           for word in query | seen}
    query_weight = max(sum(idf[w] for w in sorted(query)), 1e-12)
    seen_weight = max(sum(idf[w] for w in sorted(seen)), 1e-12)
    sims = bank[pool] @ bank[shown].T
    rows = []
    for rank, idx in enumerate(pool, start=1):
        words = word_sets[idx]
        rows.append([
            dense[idx], 1 / rank,
            sum(idf[w] for w in sorted(words & query)) / query_weight,
            math.log1p(len(words)),
            sims[rank - 1].max(), sims[rank - 1].mean(),
            any(sections[idx] == sections[a] for a in shown),
            math.log1p(min(abs(idx - a) for a in shown)),
            sum(idf[w] for w in sorted(words & missing)) / query_weight,
            sum(idf[w] for w in sorted(words & seen)) / seen_weight,
            math.log1p(len(words - seen)),
        ])
    return np.asarray(rows, dtype=np.float64)


def observe(encoder, reader, pid, qa, meta, bank):
    paragraphs, sections, start, end = meta
    ids = {text: i for i, text in enumerate(paragraphs)}
    refs = {frozenset(ids[text] for text in ref) for ref in evidence_sets(qa, paragraphs)}
    view = holdout(pid, refs)
    if view is None:
        raise ValueError("Expected an incomplete anchor set")
    shown, witnesses = view
    question = qa["question"]
    started = perf_counter()
    vectors = bank[start:end]
    dense = vectors @ encoder.encode([query_input(question)])[0]
    candidates = [i for i in range(len(paragraphs)) if i not in shown]
    pool = dense_order(dense, candidates)[:BUDGET]
    dense_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    x = features(question, paragraphs, sections, vectors, dense, shown, pool)
    feature_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    anchors = "\n".join(excerpt(reader.tokenizer, paragraphs[i], 128) for i in shown)
    anchored = question + "\nEvidence already available:\n" + anchors
    chunks = [(anchored, f"{sections[i]}\n{paragraphs[i]}") for i in pool]
    margins, clipped = reader.score_pairs(chunks, task=TASK)
    reader_ms = (perf_counter() - started) * 1000
    return dict(x=x, margins=np.asarray(margins), pool=np.asarray(pool),
                labels=np.asarray([i in witnesses for i in pool], dtype=bool),
                dense_ms=dense_ms, feature_ms=feature_ms, reader_ms=reader_ms, clipped=clipped)


def fit(rows, columns):
    diffs, weights = [], []
    for row in rows:
        matrix = row["x"][:, :columns]
        labels = row["labels"]
        if labels.any() and not labels.all():
            pair_diffs = (matrix[labels, None, :] - matrix[None, ~labels, :]).reshape(-1, columns)
            diffs.extend(pair_diffs)
            weights.extend([1 / len(pair_diffs)] * len(pair_diffs))
    x = np.asarray(diffs)
    w = np.asarray(weights)
    if len(x) < 12:
        raise ValueError("Too few pairs with contrasting labels")
    scale = np.maximum(np.sqrt(np.average(x ** 2, weights=w, axis=0)), 1e-8)
    z = x / scale
    system = z.T @ (z * w[:, None]) + REGULARIZATION * w.sum() * np.eye(columns)
    model = np.linalg.solve(system, z.T @ w) / scale
    return model, dict(pairs=len(x), comparable_questions=int(round(w.sum())), weights=model.tolist())


def select(values, indices):
    return min(range(len(values)), key=lambda idx: (-values[idx], indices[idx]))


def summarize(rows, question_model, complement_model):
    predictions = {name: [] for name in ("dense", "frozen", "learned_question", "learned_complement")}
    times = {name: [] for name in predictions}
    reachable = sum(bool(row["labels"].any()) for row in rows)
    for row in rows:
        pred = {"dense": 0, "frozen": select(row["margins"], row["pool"])}
        extra_ms = {"dense": 0, "frozen": row["reader_ms"]}
        for name, model in (("learned_question", question_model), ("learned_complement", complement_model)):
            started = perf_counter()
            values = row["x"][:, :len(model)] @ model
            pred[name] = select(values, row["pool"])
            extra_ms[name] = row["feature_ms"] + (perf_counter() - started) * 1000
        for name, idx in pred.items():
            predictions[name].append(int(row["labels"][idx]))
            times[name].append(row["dense_ms"] + extra_ms[name])
    results = {}
    for name, labels in predictions.items():
        outcomes = np.asarray(labels)
        results[name] = dict(hit=int(outcomes.sum()), selection_misses=reachable - int(outcomes.sum()),
                             serial_p95_ms=float(np.quantile(times[name], .95)),
                             difference_vs_frozen=paired_interval(outcomes - predictions["frozen"]),
                             difference_vs_question_model=paired_interval(outcomes - predictions["learned_question"]))
    return dict(questions=len(rows), reachable=reachable, frozen_truncations=sum(row["clipped"] for row in rows),
                methods=results)


def run(data, train_archive, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    embed = verify_checkpoint(embedding)
    reader = Reranker(reranker)
    if (embed["repository"] != "Qwen/Qwen3-Embedding-0.6B"
            or reader.provenance["repository"] != "Qwen/Qwen3-Reranker-0.6B"):
        raise ValueError("Expected frozen Qwen3 checkpoints")
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader.score_pairs([("Warm up", "A paragraph")] * BUDGET, task=TASK)
    encoder.encode([query_input("Warm up")])
    datasets = {"train": read_train(train_archive)}
    datasets.update({split: read_data(data, split) for split in ("dev", "test")})
    if len(set().union(*datasets.values())) != sum(map(len, datasets.values())):
        raise ValueError("Paper splits overlap")
    results = {}
    weights = {}
    for split, dataset in datasets.items():
        panel = choose(dataset, multi=True)
        meta, _, bank, index = offline({pid: dataset[pid] for pid, _ in panel}, encoder)
        print(f"{split}: indexed {len(panel)} papers", flush=True)
        rows = []
        for i, (pid, qa) in enumerate(panel, start=1):
            rows.append(observe(encoder, reader, pid, qa, meta[pid], bank))
            if i % 50 == 0:
                print(f"{split}: {i}/{len(panel)}", flush=True)
        if split == "train":
            question_model, weights["question"] = fit(rows, ABLATION)
            complement_model, weights["complement"] = fit(rows, len(FEATURE_NAMES))
        results[split] = dict(index=index,
                              panel_sha256=sha256("\n".join(qa["question_id"] for _, qa in panel).encode()).hexdigest(),
                              **summarize(rows, question_model, complement_model))
    write_json(output, dict(dataset="QASPER v0.3", job=os.environ["SLURM_JOB_ID"],
                            gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
                            embedding=embed["revision"], reranker=reader.provenance["revision"],
                            regularization=REGULARIZATION, feature_names=FEATURE_NAMES,
                            weights=weights, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "train_archive", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
