"""Test surface-scoped exclusion on released ComLQ passage relevance."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from expert_search.comlq import SHA256
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from expert_search.statistics import paired_interval
from experiments.qasper.adaptive_evidence import digest
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("bm25", "hybrid", "positive_only", "scoped_exclusion")
TASK = "Retrieve passages that provide evidence relevant to the question."
RRF_OFFSET = 60
POOL = 100
DAMP = .5
EXCLUSION = re.compile(
    r"\b(?:but\s+(?:(?:is|are|was|were|do|does|did|can|could|include|includes)\s+)?"
    r"(?:not|never)|except(?: for)?|excluding)\s+", re.I)
NESTED = re.compile(r"\b(?:not|never|except|excluding|instead|unless|only|rather)\b", re.I)


def clauses(query):
    matches = list(EXCLUSION.finditer(query))
    if len(matches) != 1:
        return None
    match = matches[0]
    if any(query[:match.start()].count(mark) % 2 for mark in ('"',)):
        return None
    if query[:match.start()].count("(") != query[:match.start()].count(")"):
        return None
    left = query[:match.start()].strip(" ,;")
    right = query[match.end():].strip(" ?!.,;")
    if min(len(tokens(left)), len(tokens(right))) < 3 or NESTED.search(right):
        return None
    return left, right


def load(archive):
    if digest(archive) != SHA256:
        raise ValueError("Unexpected ComLQ archive")
    with ZipFile(archive) as upstream:
        documents = [(row["_id"], row["text"]) for row in map(json.loads, upstream.read("dataset/corpus.jsonl").splitlines())]
        queries = [(row["_id"], row["text"], row["type"])
                   for row in map(json.loads, upstream.read("dataset/queries.jsonl").splitlines())]
        qrels = defaultdict(dict)
        for row in csv.DictReader(StringIO(upstream.read("dataset/qrels/test.tsv").decode()), delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    if len(documents) != 11251 or len(queries) != 2909 or set(qrels) != {qid for qid, _, _ in queries}:
        raise ValueError("Unexpected benchmark inventory")
    return sorted(documents), queries, qrels


def recall(ranking, target, k):
    return sum(doc in target for doc in ranking[:k]) / len(target)


def ndcg(ranking, target, k):
    discounts = [1 / math.log2(rank + 2) for rank in range(k)]
    got = sum(target.get(doc, 0) * weight for doc, weight in zip(ranking[:k], discounts))
    cap = sum(grade * weight for grade, weight in zip(sorted(target.values(), reverse=True), discounts))
    return got / cap


def metrics(ranking, gold):
    return {"ndcg10": ndcg(ranking, gold, 10), "recall100": recall(ranking, gold, 100)}


def rank_percentile(values):
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order / max(len(order) - 1, 1)


def scope_kind(query_type):
    return "negation" if "n" in query_type.split("_", 1)[0] else "other"


def build_index(texts):
    # Pinned collection may contain duplicate text under distinct document IDs.
    # Preserve each entry's identity in the lexical corpus without changing tokens.
    documents = {}
    for i, source in enumerate(texts):
        key = source
        while key in documents:
            key += "\n"
        documents[key] = i
    with TemporaryDirectory() as root:
        return build_lexical(documents, Path(root))


class Retrieval:
    def __init__(self, documents, embedder, vectors, index):
        self.embedder = embedder
        self.index = index
        self.ids = [doc for doc, _ in documents]
        self.torch = embedder.torch
        self.vectors = self.torch.as_tensor(vectors, dtype=self.torch.float32, device="cuda")

    def lexical(self, query):
        scores = self.index.document_scores(query)
        return [int(i) for i in sorted(scores, key=lambda i: (-scores[i], self.ids[int(i)]))[:POOL]]

    def run(self, query, method):
        start = perf_counter()
        lexical = self.lexical(query)
        if method == "bm25":
            return [self.ids[i] for i in lexical], (perf_counter() - start) * 1000, False, None
        parsed = clauses(query) if method in ("positive_only", "scoped_exclusion") else None
        inputs = [query]
        if parsed:
            inputs.append(parsed[0])
            if method == "scoped_exclusion":
                inputs.append(parsed[1])
        queries = self.embedder.encode([f"Instruct: {TASK}\nQuery: {value}" for value in inputs])
        with torch.inference_mode():
            q = torch.as_tensor(queries, dtype=torch.float32, device="cuda")
            scores = (self.vectors @ q.T).cpu().numpy()
        dense = np.argsort(-scores[:, 0], kind="stable")[:POOL]
        fused = defaultdict(float)
        for ranking in (lexical, dense):
            for position, i in enumerate(ranking, 1):
                fused[int(i)] += 1 / (RRF_OFFSET + position)
        membership = np.array(sorted(fused, key=lambda i: (-fused[i], self.ids[i]))[:POOL], dtype=np.int64)
        positive_scores = None
        if parsed:
            baseline = 1 - np.arange(len(membership)) / max(len(membership) - 1, 1)
            positive = rank_percentile(scores[membership, 1])
            updated = baseline + DAMP * positive
            if method == "scoped_exclusion":
                excluded = rank_percentile(scores[membership, 2])
                positive_scores = [self.ids[int(i)] for i in membership[np.argsort(-updated, kind="stable")]]
                updated -= DAMP * excluded
            membership = membership[np.argsort(-updated, kind="stable")]
        return ([self.ids[int(i)] for i in membership],
                (perf_counter() - start) * 1000, parsed is not None, positive_scores)


def summarize(rows):
    result = {}
    for cohort in ("all", "negation", "other", "routed_negation", "routed_other"):
        if cohort == "all":
            group = rows
        elif cohort.startswith("routed_"):
            kind = cohort.removeprefix("routed_")
            group = [row for row in rows if row["type_group"] == kind and row["routed"]]
        else:
            group = [row for row in rows if row["type_group"] == cohort]
        if not group:
            continue
        metrics_by_arm = {}
        for arm in METHODS:
            records = [row["arms"][arm] for row in group]
            metrics_by_arm[arm] = {
                "ndcg10": mean(row["ndcg10"] for row in records),
                "recall100": mean(row["recall100"] for row in records),
                "p50_p95_ms": list(map(float, np.quantile([row["ms"] for row in records], [.5, .95]))),
            }
        result[cohort] = {
            "queries": len(group), "routed": sum(row["routed"] for row in group),
            "metrics": metrics_by_arm,
            "scoped_minus_hybrid_ndcg10": paired_interval([
                row["arms"]["scoped_exclusion"]["ndcg10"] - row["arms"]["hybrid"]["ndcg10"] for row in group
            ]),
            "scoped_minus_positive_ndcg10": paired_interval([
                row["arms"]["scoped_exclusion"]["ndcg10"] - row["arms"]["positive_only"]["ndcg10"] for row in group
            ]),
        }
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    documents, queries, qrels = load(args.archive)
    checkpoint = json.loads((args.model / "checkpoint-provenance.json").read_bytes())
    for name, check in checkpoint["files"].items():
        if digest(args.model / name) != check:
            raise ValueError(f"Changed embedding artifact: {name}")
    model = QwenTorchEmbedding(args.model)
    model.model.config.use_cache = False
    texts = [text for _, text in documents]
    start = perf_counter()
    vectors = model.encode(texts)
    offline = perf_counter() - start
    print("Encoded passages", len(documents), "in", offline, "seconds", flush=True)
    engine = Retrieval(documents, model, vectors, build_index(texts))
    for arm in METHODS:
        engine.run("Find an example of a technical tool but not a programming language.", arm)

    rows = []
    for i, (qid, text, kind) in enumerate(queries):
        arms = {}
        routed = False
        shift = i % len(METHODS)
        for arm in METHODS[shift:] + METHODS[:shift]:
            ranked, ms, parsed, ablated = engine.run(text, arm)
            arms[arm] = dict(**metrics(ranked, qrels[qid]), ms=ms, top10=ranked[:10])
            if arm == "scoped_exclusion":
                routed = parsed
                if parsed:
                    arms[arm]["same_call_positive_only_ndcg10"] = ndcg(ablated, qrels[qid], 10)
        rows.append(dict(query_id=qid, query_type=kind, type_group=scope_kind(kind), routed=routed, arms=arms))
        if len(rows) % 400 == 0:
            print("Completed queries", len(rows), "/", len(queries), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "requests.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "archive_sha256": SHA256, "rows_sha256": digest(path),
        "script_sha256": digest(Path(__file__)), "gpu": torch.cuda.get_device_name(),
        "checkpoint": checkpoint, "corpus": len(documents), "query_kind_counts": dict(Counter(row[2] for row in queries)),
        "offline_encode_seconds": offline, "candidate_cap": POOL, "fusion_offset": RRF_OFFSET,
        "scope": "Released graded relevance only. No per-passage violation labels.",
        "metrics": summarize(rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/qwen3-embedding-0.6b"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
