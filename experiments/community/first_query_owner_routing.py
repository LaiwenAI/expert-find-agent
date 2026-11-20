"""Compare profile-first and source-first expert search with live answer verification."""

import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import os

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.community.evaluate_crossvalidated_witness import connected_groups
from experiments.community.held_window_routing import EXPLICIT, independent, load
from experiments.community.witness_matched_issuance import cross_site_overlap
from experiments.credit.vector_hybrid_rrf import RankFusion
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


ARMS = ("profile", "source_dense", "source_hybrid", "source_bounded")
SLOTS = 5
PREFIX = 512
BLOCK = 16
# Conservative slack for the FP32 dot/norm computations on unit-length vectors.
SLACK = 0.0003


def best_sources(order, owners):
    selected = {}
    for doc in order:
        selected.setdefault(owners[doc], doc)
        if len(selected) == SLOTS:
            break
    return list(selected.values())


class AnswerIndex:
    def __init__(self, docs, owners, encoder, scratch):
        self.owners = owners
        self.vectors = encoder.encode(docs)
        if self.vectors.shape[1] != 1024 or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1):
            raise ValueError("Expected unit-normalized 1024-dimensional embeddings")
        self.prefix = np.ascontiguousarray(self.vectors[:, :PREFIX])
        self.tail_norm = np.linalg.norm(self.vectors[:, PREFIX:], axis=1)
        groups = defaultdict(list)
        for doc, owner in enumerate(owners):
            groups[owner].append(doc)
        self.postings = [np.array(rows) for rows in groups.values()]
        means = np.stack([self.vectors[rows].mean(axis=0) for rows in self.postings])
        self.profiles = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-8)
        lexical_input = {}
        for doc, body in enumerate(docs):
            while body in lexical_input:
                body += "\n"
            lexical_input[body] = doc
        self.fusion = RankFusion(range(len(docs)), build_lexical(lexical_input, scratch))

    def bounded(self, vector):
        upper = self.prefix @ vector[:PREFIX] + self.tail_norm * np.linalg.norm(vector[PREFIX:]) + SLACK
        order = np.argsort(-upper, kind="stable")
        scores = np.full(len(self.owners), -np.inf, dtype=np.float32)
        for start in range(0, len(order), BLOCK):
            batch = order[start:start + BLOCK]
            scores[batch] = self.vectors[batch] @ vector
            end = start + len(batch)
            seen = order[:end]
            ranked = seen[np.lexsort((seen, -scores[seen]))]
            selected = best_sources(ranked, self.owners)
            if len(selected) == SLOTS and (end == len(order) or scores[selected[-1]] > upper[order[end]]):
                return selected, end
        raise ValueError("Unable to select the required number of distinct owners")

    def select(self, arm, text, vector):
        if arm == "profile":
            order = np.argsort(-(self.profiles @ vector), kind="stable")[:SLOTS]
            selected = []
            for owner_index in order:
                docs = self.postings[owner_index]
                selected.append(int(docs[np.argmax(self.vectors[docs] @ vector)]))
            return selected, sum(len(self.postings[i]) for i in order)
        if arm == "source_bounded":
            return self.bounded(vector)
        scores = self.vectors @ vector
        if arm == "source_hybrid":
            ranking = self.fusion.rank(scores, text, "array_part")
            if len(best_sources(ranking, self.owners)) != SLOTS:
                raise ValueError("Hybrid search depth cannot fill five slots")
        else:
            ranking = np.argsort(-scores, kind="stable")
        return best_sources(ranking, self.owners), len(scores)


def load_panel(folder):
    snapshot = folder / "snapshot.json"
    return load(snapshot, folder / "manifest.json"), digest(snapshot)


def evaluate(data, encoder, reader, scratch, offset):
    ids = sorted(map(int, data["documents"]))
    records = [data["documents"][str(doc)] for doc in ids]
    owners = [record["owner"] for record in records]
    docs = [record["answer"] for record in records]
    lookup = {doc: i for i, doc in enumerate(ids)}
    start = perf_counter()
    index = AnswerIndex(docs, owners, encoder, scratch)
    index_ms = (perf_counter() - start) * 1000
    queries = data["queries"]
    results = defaultdict(list)
    for i, query in enumerate([queries[0], *queries]):
        text = f"{query['title']}\n{query['query']}"
        known = {lookup[doc] for doc in query["answers"]}
        known_owners = {owners[doc] for doc in known}
        shift = (i + offset) % len(ARMS)
        observed = {}
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            vector = encoder.encode([text], query=True)[0]
            after_embedding = perf_counter()
            selected, scored = index.select(arm, text, vector)
            after_search = perf_counter()
            margins, clipped = reader.score_pairs(
                [(text[:950], docs[doc][:850]) for doc in selected], task=EXPLICIT,
            )
            if len(selected) != SLOTS or not np.isfinite(margins).all():
                raise ValueError("Incomplete or invalid verification batch")
            issued = [int(doc) for doc, margin in zip(selected, margins, strict=True) if margin > 0]
            end = perf_counter()
            observed[arm] = (list(map(int, selected)), issued)
            if i:
                results[arm].append(dict(
                    selected=len(known.intersection(selected)), supported=len(known.intersection(issued)),
                    known_owners=sum(owners[doc] in known_owners for doc in selected),
                    issued=len(issued), clipped=clipped, score_rows=int(scored),
                    search_ms=1000 * (after_search - after_embedding), request_ms=1000 * (end - start),
                ))
        if observed["source_dense"] != observed["source_bounded"]:
            raise ValueError("Bounded search changed nominations or issued evidence")

    groups = connected_groups(queries)
    baseline = results["source_dense"]
    metrics = {}
    for arm, rows in results.items():
        totals = {key: sum(row[key] for row in rows)
                  for key in ("selected", "supported", "known_owners", "issued", "clipped", "score_rows")}
        deltas = [(row["supported"] - base["supported"]) / SLOTS for row, base in zip(rows, baseline, strict=True)]
        metrics[arm] = dict(
            **totals, supported_p5_delta=clustered_paired_interval(deltas, groups),
            p50_p95_search_ms=np.quantile([row["search_ms"] for row in rows], [.5, .95]).tolist(),
            p50_p95_request_ms=np.quantile([row["request_ms"] for row in rows], [.5, .95]).tolist(),
        )
    return dict(queries=len(queries), source_groups=len(set(groups)), docs=len(docs),
                owners=len(set(owners)), index_build_ms=index_ms, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("stats2024", "stats2025", "mathoverflow", "embedding", "reranker", "output"):
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
    embedding = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    panels = {}
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        for name, data, sha in (("stats2024", old, old_sha), ("stats2025", held, later_sha),
                                ("mathoverflow", overflow, overflow_sha)):
            root = Path(scratch) / name
            root.mkdir()
            panels[name] = dict(snapshot_sha256=sha, **evaluate(data, encoder, reader, root, args.order_offset))
            print(f"{name}: {len(data['queries'])} requests", flush=True)
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)), order_offset=args.order_offset,
        embedding_revision=embedding["revision"], reranker_revision=reader.provenance["revision"],
        exclusions=dict(stats2025=later_exclusions, mathoverflow=overflow_exclusions),
        slots=SLOTS, prefix=PREFIX, block=BLOCK, panels=panels,
    ))


if __name__ == "__main__":
    main()
