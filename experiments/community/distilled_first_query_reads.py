"""Check whether an offline distilled selector buys broader retrieval without extra reads."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.retrieval import tokens
from experiments.community.first_query_owner_routing import (
    QwenTorchEmbedding, Reranker, clustered_paired_interval, connected_groups,
    cross_site_overlap, digest, independent, load_panel, verify_checkpoint, write_json,
)
from experiments.community.held_window_routing import EXPLICIT


DEPTH = 10
ANCHORS = 2
ARMS = {"dense3": 3, "overlap3": 3, "student3": 3, "dense5": 5, "student5": 5, "dense10": 10}
FEATURES = ("similarity", "reciprocal_rank", "query_overlap", "idf_query_overlap",
            "answer_overlap", "answer_terms")


class EvidenceIndex:
    def __init__(self, data, encoder):
        ids = sorted(map(int, data["documents"]))
        records = [data["documents"][str(doc)] for doc in ids]
        self.docs = [row["answer"] for row in records]
        self.owners = [row["owner"] for row in records]
        self.embeddings = encoder.encode(self.docs)
        self.windows = [text[:850] for text in self.docs]
        self.terms = [set(tokens(text)) for text in self.windows]
        counts = Counter(term for terms in self.terms for term in terms)
        self.idf = {word: float(np.log((len(self.docs) + 1) / (n + 1))) for word, n in counts.items()}
        self.ids = dict(zip(ids, range(len(ids)), strict=True))

    def nominate(self, query, encoder):
        vector = encoder.encode([query], query=True)[0]
        similarities = self.embeddings @ vector
        seen = set()
        candidates = []
        for doc in np.argsort(-similarities, kind="stable"):
            owner = self.owners[doc]
            if owner not in seen:
                candidates.append(int(doc))
                seen.add(owner)
            if len(candidates) == DEPTH:
                return candidates, similarities
        raise ValueError("Insufficient distinct candidate accounts")

    def features(self, query, candidates, similarities):
        qterms = set(tokens(query[:950]))
        weight = sum(self.idf[t] for t in sorted(qterms) if t in self.idf)
        features = []
        for rank, doc in enumerate(candidates, 1):
            answer_terms = self.terms[doc]
            shared = qterms & answer_terms
            features.append((
                similarities[doc], 1 / rank, len(shared) / max(len(qterms), 1),
                sum(self.idf[t] for t in sorted(shared)) / max(weight, 1e-9),
                len(shared) / max(len(answer_terms), 1), np.log1p(len(answer_terms)),
            ))
        return np.array(features, dtype=np.float64)

    def read(self, query, indices, reader):
        scores, clipped = reader.score_pairs(
            [(query[:950], self.windows[doc]) for doc in indices], task=EXPLICIT,
        )
        if len(scores) != len(indices) or not np.isfinite(scores).all():
            raise ValueError("Incomplete teacher verification")
        return scores, clipped


def question_text(query):
    return f"{query['title']}\n{query['query']}"


def fit(queries, index, encoder, reader):
    start = perf_counter()
    features, margins = [], []
    clipped = 0
    for query in queries:
        text = question_text(query)
        candidates, similarities = index.nominate(text, encoder)
        scores, truncations = index.read(text, candidates, reader)
        features.extend(index.features(text, candidates, similarities)[ANCHORS:])
        margins.extend(scores[ANCHORS:])
        clipped += truncations
    x = np.stack(features)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    x = np.column_stack((np.ones(len(x)), (x - center) / scale))
    # Preserve the teacher's confidence ordering while limiting leverage from extreme margins.
    targets = np.clip(margins, -8, 8)
    penalty = 5 * np.eye(x.shape[1])
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ targets)
    return (center, scale, weights), dict(
        queries=len(queries), teacher_pairs=len(queries) * DEPTH, fitted_tail_rows=len(x),
        weights=weights.tolist(), center=center.tolist(), scale=scale.tolist(),
        clipped=clipped, fit_and_label_ms=(perf_counter() - start) * 1000,
    )


def choose(arm, index, model, text, candidates, similarities):
    budget = ARMS[arm]
    if arm.startswith("dense"):
        return candidates[:budget]
    x = index.features(text, candidates, similarities)[ANCHORS:]
    if arm == "overlap3":
        scores = x[:, FEATURES.index("idf_query_overlap")]
    else:
        center, scale, weights = model
        scores = (x - center) / scale @ weights[1:] + weights[0]
    order = np.argsort(-scores, kind="stable")[:budget - ANCHORS]
    return [*candidates[:ANCHORS], *(candidates[ANCHORS + i] for i in order)]


def paired_delta(rows, baseline, groups, matched=False):
    differences = []
    for (observed, known, *_), (reference, *_) in zip(rows, baseline, strict=True):
        cutoff = min(len(reference), len(observed)) if matched else max(len(reference), len(observed))
        change = len(known.intersection(observed[:cutoff])) - len(known.intersection(reference[:cutoff]))
        differences.append(change / 5)
    return clustered_paired_interval(differences, groups)


def evaluate(queries, index, encoder, reader, model, offset):
    names = tuple(ARMS)
    records = defaultdict(list)
    for i, query in enumerate([queries[0], *queries]):
        text = question_text(query)
        known = {index.ids[doc] for doc in query["answers"]}
        shift = (i + offset) % len(names)
        for arm in names[shift:] + names[:shift]:
            start = perf_counter()
            candidates, similarities = index.nominate(text, encoder)
            bought = choose(arm, index, model, text, candidates, similarities)
            scores, clipped = index.read(text, bought, reader)
            issued = [bought[j] for j in np.argsort(-np.array(scores), kind="stable") if scores[j] > 0][:5]
            elapsed = (perf_counter() - start) * 1000
            if i:
                records[arm].append((issued, known, bought, clipped, elapsed))
    groups = connected_groups(queries)
    report = {}
    for arm, rows in records.items():
        report[arm] = dict(
            verified_pairs=sum(len(bought) for _, _, bought, _, _ in rows),
            issued=sum(len(issued) for issued, *_ in rows),
            known=sum(len(known.intersection(issued)) for issued, known, *_ in rows),
            known_available=sum(len(known.intersection(bought)) for _, known, bought, *_ in rows),
            clipped=sum(row[-2] for row in rows),
            p50_p95_ms=np.quantile([row[-1] for row in rows], [.50, .95]).tolist(),
            versus_dense5=paired_delta(rows, records["dense5"], groups),
            versus_dense5_matched_issue=paired_delta(rows, records["dense5"], groups, matched=True),
        )
        if ARMS[arm] in (3, 5):
            reference = records[f"dense{ARMS[arm]}"]
            report[arm]["same_read"] = paired_delta(rows, reference, groups)
            report[arm]["same_read_matched_issue"] = paired_delta(rows, reference, groups, matched=True)
    return dict(queries=len(queries), clusters=len(set(groups)), arms=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("stats2024", "stats2025", "mathoverflow", "embedding", "reranker", "reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a Slurm GPU and a fresh output path")
    old, old_sha = load_panel(args.stats2024)
    later, later_sha = load_panel(args.stats2025)
    overflow, overflow_sha = load_panel(args.mathoverflow)
    held, later_exclusions = independent(old, later)
    overflow, overflow_exclusions = cross_site_overlap((old, held), overflow)
    reference = json.loads(args.reference.read_bytes())["runs"][0]["panels"]
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    model, training = fit(old["queries"], EvidenceIndex(old, encoder), encoder, reader)
    panels = {}
    for name, data, sha in (("stats2025", held, later_sha), ("mathoverflow", overflow, overflow_sha)):
        result = evaluate(data["queries"], EvidenceIndex(data, encoder), encoder, reader, model, args.order_offset)
        baseline = reference[name]["metrics"]["source_dense"]
        observed = result["arms"]["dense5"]
        if (observed["known"], observed["issued"], observed["known_available"]) != (
            baseline["supported"], baseline["issued"], baseline["selected"]
        ):
            raise ValueError(f"Frozen dense-five reference did not reproduce for {name}")
        panels[name] = dict(snapshot_sha256=sha, **result)
        print(f"{name}: dense-five reference reproduced", flush=True)
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)), order_offset=args.order_offset,
        embedding_revision=checkpoint["revision"], reranker_revision=reader.provenance["revision"],
        training_snapshot_sha256=old_sha, reference_sha256=digest(args.reference),
        exclusions=dict(stats2025=later_exclusions, mathoverflow=overflow_exclusions),
        features=FEATURES, anchors=ANCHORS, depth=DEPTH, training=training, panels=panels,
    ))


if __name__ == "__main__":
    main()
