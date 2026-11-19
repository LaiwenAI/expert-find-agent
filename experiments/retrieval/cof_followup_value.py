"""Allocate a shared source-check budget using counterfactual expert-ranking gains."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import os
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, select
from experiments.retrieval.cof_live_prefix_stop import TASK


DATASETS = ("SIGIR", "KDD", "SciRepEval")
METHODS = ("head", "round_robin", "one_hit", "margin", "learned", "hindsight")
FOLLOWUP = 5
TOP = 5
DEPTH = 3
SCORE_BATCH = 8
PENALTY = 1.


class Request:
    def __init__(self, hits, labels, margins):
        self.hits = hits
        self.labels = labels
        self.margins = margins
        self.initial = {docs[0] for _, docs in hits}
        self.sources = sorted(margins)
        self.owners = defaultdict(list)
        for rank, (_, docs) in enumerate(hits):
            for position, doc in enumerate(docs):
                self.owners[doc].append((rank, position))

    def ranks(self, observed):
        support = [
            sum(observed.get(doc, -np.inf) >= 0 for doc in docs)
            for _, docs in self.hits
        ]
        confirmed = [n >= 2 for n in support]
        ranking = sorted(range(len(support)), key=lambda i: (not confirmed[i], i))[:TOP]
        return support, confirmed, ranking

    def quality(self, observed):
        _, confirmed, ranking = self.ranks(observed)
        labels = [self.labels.get(self.hits[i][0]) for i in ranking]
        pairs = list(zip(ranking, labels, strict=True))
        return dict(positive=sum(label == 1 for label in labels),
                    judged=sum(label is not None for label in labels),
                    cited_positive=sum(label == 1 and confirmed[i] for i, label in pairs),
                    cited_negative=sum(label == 0 and confirmed[i] for i, label in pairs),
                    cited_unknown=sum(label is None and confirmed[i] for i, label in pairs))

    def expert_set(self, observed):
        return {self.hits[i][0] for i in self.ranks(observed)[2]}

    def reward(self, observed):
        score = self.quality(observed)
        return score["positive"] + 2 * score["cited_positive"] - PENALTY * score["cited_negative"]

    def features(self, doc, observed, state):
        support, confirmed, ranking = state
        owners = self.owners[doc]
        first_rank, best_position = min(owners)
        pending = [(rank, position) for rank, position in owners if not confirmed[rank]]
        active = pending or owners
        best_active_rank = min(rank for rank, _ in active)
        nearest = [(rank, pos) for rank, pos in active if support[rank] == 1]
        prior = [observed[d] for rank, _ in active for d in self.hits[rank][1] if d in observed]
        margin = float(np.clip(max(prior, default=-8.), -8., 8.) / 8.)
        return np.array([
            1., first_rank / len(self.hits), best_position / DEPTH, len(owners) / len(self.hits),
            best_active_rank / len(self.hits), bool(nearest),
            min((rank for rank, _ in nearest), default=len(self.hits)) / len(self.hits),
            sum(rank in ranking for rank, _ in nearest) / TOP,
            sum(confirmed[i] for i in ranking) / TOP, margin,
            margin * bool(nearest), len(observed) / (len(self.initial) + FOLLOWUP),
        ])

    def ordered_options(self, observed):
        return [d for d in self.sources if d not in observed]

    def initial_values(self):
        return {d: self.margins[d] for d in sorted(self.initial)}

    def choose(self, method, observed, options, weights):
        if method == "head":
            return min(options, key=lambda d: (min(self.owners[d]), d))
        if method == "round_robin":
            return min(options, key=lambda d: (min((pos, rank) for rank, pos in self.owners[d]), d))
        state = self.ranks(observed)
        if method == "hindsight":
            return max(options, key=lambda d: (self.reward(observed | {d: self.margins[d]}), -min(self.owners[d])[0]))
        features = np.stack([self.features(doc, observed, state) for doc in options])
        if method == "one_hit":
            priority = list(zip(-features[:, 5], features[:, 6], features[:, 1], features[:, 2], strict=True))
            return options[min(range(len(options)), key=lambda i: priority[i])]
        if method == "margin":
            priority = list(zip(-features[:, 5], -features[:, 9], features[:, 1], features[:, 2], strict=True))
            return options[min(range(len(options)), key=lambda i: priority[i])]
        return options[int(np.argmax(features @ weights))]


def select_documents(request, method, weights):
    observed = request.initial_values()
    times = []
    for _ in range(FOLLOWUP):
        options = request.ordered_options(observed)
        if not options:
            break
        start = perf_counter()
        chosen = request.choose(method, observed, options, weights)
        times.append((perf_counter() - start) * 1e3)
        observed[chosen] = request.margins[chosen]
    return observed, sum(times)


def fit(requests):
    features, targets = [], []
    for request in requests:
        observed = request.initial_values()
        for _ in range(FOLLOWUP):
            options = request.ordered_options(observed)
            if not options:
                break
            state = request.ranks(observed)
            value = request.reward(observed)
            for doc in options:
                features.append(request.features(doc, observed, state))
                targets.append(request.reward(observed | {doc: request.margins[doc]}) - value)
            chosen = request.choose("one_hit", observed, options, None)
            observed[chosen] = request.margins[chosen]
    x, y = np.stack(features), np.asarray(targets)
    beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y)
    return beta, dict(states=len(y), nonzero=int(np.count_nonzero(y)),
                      positive=int((y > 0).sum()), negative=int((y < 0).sum()))


def measure(requests, weights):
    result = defaultdict(list)
    for request in requests:
        raw = request.quality({})
        initial = request.quality(request.initial_values())
        reference, _ = select_documents(request, "one_hit", weights)
        reference_ids = request.expert_set(reference)
        for method in METHODS:
            observed, cpu_ms = select_documents(request, method, weights)
            expected = min(len(request.margins), len(request.initial) + FOLLOWUP)
            if len(observed) != expected:
                raise AssertionError("A policy deviated from the common unique-read budget")
            stats = request.quality(observed)
            base = dict(raw_positive=raw["positive"], initial_cited_positive=initial["cited_positive"])
            predicted = request.expert_set(observed)
            gained, dropped = predicted - reference_ids, reference_ids - predicted
            positive_delta = sum(request.labels.get(e) == 1 for e in gained) - sum(request.labels.get(e) == 1 for e in dropped)
            result[method].append(dict(stats, **base, reads=len(observed), policy_cpu_ms=cpu_ms,
                                       lower=(positive_delta - len(dropped - request.labels.keys())) / TOP,
                                       upper=(positive_delta + len(gained - request.labels.keys())) / TOP))
    baseline = result["one_hit"]
    return {method: dict(
        **{name: sum(r[name] for r in rows) for name in
           ("positive", "judged", "cited_positive", "cited_negative", "cited_unknown",
            "raw_positive", "initial_cited_positive")},
        mean_reads=mean(row["reads"] for row in rows),
        incomplete_label_bounds=[mean(row[k] for row in rows) for k in ("lower", "upper")],
        policy_cpu_p95_ms=float(np.quantile([row["policy_cpu_ms"] for row in rows], .95)),
        known_positive_p5_vs_one_hit=paired_interval(
            [(row["positive"] - ref["positive"]) / TOP for row, ref in zip(rows, baseline, strict=True)]),
        cited_positive_p5_vs_one_hit=paired_interval(
            [(row["cited_positive"] - ref["cited_positive"]) / TOP for row, ref in zip(rows, baseline, strict=True)]),
    ) for method, rows in result.items()}


def collect(dataset, encoder, verifier, provenance, *, select_queries=select):
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (row.get("title", "") + "\n" + row.get("abstract", "")).strip() for pid, row in papers.items()}
    chosen = ids if dataset == "SIGIR" else select_queries(dataset, ids, texts)
    if dataset == "SIGIR" and any(set(labels[qid]) != set(reference.owner_ids) for qid in chosen):
        raise ValueError("Expected full SIGIR reviewer judgments")
    index = ProfileIndex(reference)
    rows = []
    truncated = 0
    for i, qid in enumerate(chosen):
        hits = index.search_query(texts[qid], encoder, "mean_max", depth=DEPTH)
        if not hits or any(not docs for _, docs in hits):
            raise ValueError("Empty expert or source pool")
        docs = sorted({doc for _, sources in hits for doc in sources})
        margins = {}
        for start in range(0, len(docs), SCORE_BATCH):
            batch = docs[start:start + SCORE_BATCH]
            values, clipped = verifier.score(texts[qid], texts, batch, task=TASK)
            if not all(np.isfinite(values)):
                raise ValueError("Invalid verifier margins")
            truncated += clipped
            margins.update(zip(batch, values, strict=True))
        rows.append((qid, Request(hits, labels[qid], margins)))
        if (i + 1) % 40 == 0:
            print(dataset, "completed", i + 1, flush=True)
    titles = {re.sub(r"[^a-z0-9]+", " ", papers[qid]["title"].casefold()).strip() for qid in chosen}
    return root, rows, len(reference.owner_ids), len(reference.source_ids), truncated, titles


def run(embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh result and a GPU compute allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    reports = {}
    weights = None
    train_ids, fit_info = [], {}
    previous_titles = set()
    for dataset in DATASETS:
        root, data, experts, sources, truncated, titles = collect(dataset, encoder, verifier, provenance)
        if previous_titles & titles:
            raise ValueError("Duplicate query title across panels")
        previous_titles.update(titles)
        if dataset == "SIGIR":
            sorted_rows = sorted(data, key=lambda row: sha256(f"cof-followup-v1:{row[0]}".encode()).digest())
            split = 2 * len(data) // 3
            train_ids = [qid for qid, _ in sorted_rows[:split]]
            weights, fit_info = fit([request for _, request in sorted_rows[:split]])
            data = sorted_rows[split:]
        reports[dataset] = dict(
            queries=len(data), experts=experts, sources=sources,
            qids_sha256=sha256("\n".join(q for q, _ in data).encode()).hexdigest(),
            labels_sha256=digest(root / f"{dataset}_queries_test_hard.json"),
            mean_full_pool_checks=mean(len(request.sources) for _, request in data),
            truncated_pairs_in_collected_pool=truncated,
            full_pool={key: sum(req.quality(req.margins)[key] for _, req in data)
                       for key in ("positive", "judged", "cited_positive", "cited_negative", "cited_unknown")},
            report=measure([request for _, request in data], weights),
        )
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        training_queries=len(train_ids), training_ids_sha256=sha256("\n".join(train_ids).encode()).hexdigest(),
        model_weights=weights.tolist(), training=fit_info, followup_budget=FOLLOWUP, report=reports,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
