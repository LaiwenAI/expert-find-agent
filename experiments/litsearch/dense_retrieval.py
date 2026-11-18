"""Compare exact dense, BM25 and fused retrieval with matched LitSearch verification."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
from heapq import nsmallest
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.source_withholding import POOL, QUERY_SHA256, REVISION, TASK, calibrate, split
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


METHODS = ("bm25", "dense", "fusion")
FUSION_DEPTH = 24
PREVIOUS_POLICY = {"score": 6.875, "margin": 1.75}
VIEWS = ("present", "withheld")


def reciprocal_fusion(left, right):
    scores = defaultdict(float)
    for ranking in (left, right):
        for position, doc in enumerate(ranking, 1):
            scores[doc] += 1 / (60 + position)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:POOL]


class Search:
    def __init__(self, papers, index, vectors, encoder, reader):
        self.ids = [r["corpusid"] for r in papers]
        self.positions = {doc: i for i, doc in enumerate(self.ids)}
        self.documents = {r["corpusid"]: paper_text(r) for r in papers}
        self.index = index
        self.vectors = vectors
        self.encoder = encoder
        self.reader = reader

    def lexical(self, text, hidden, depth):
        scores = self.index.document_scores(text)
        order = nsmallest(depth, (i for i in scores if self.ids[int(i)] not in hidden),
                          key=lambda i: (-scores[i], self.ids[int(i)]))
        return [self.ids[int(i)] for i in order]

    @torch.inference_mode()
    def dense(self, text, hidden, depth):
        encoded = self.encoder.encode([f"Instruct: {TASK}\nQuery: {text}"])[0]
        query = torch.from_numpy(encoded).to(self.vectors.device)
        scores = self.vectors @ query
        if hidden:
            scores[[self.positions[doc] for doc in hidden]] = -torch.inf
        order = torch.argsort(scores, descending=True, stable=True)[:depth].tolist()
        return [self.ids[i] for i in order]

    def run(self, question, view, method):
        start = perf_counter()
        text = question["query"]
        hidden = set(question["corpusids"]) if view == "withheld" else set()
        if method == "bm25":
            pool = self.lexical(text, hidden, POOL)
        elif method == "dense":
            pool = self.dense(text, hidden, POOL)
        else:
            pool = reciprocal_fusion(self.lexical(text, hidden, FUSION_DEPTH),
                                     self.dense(text, hidden, FUSION_DEPTH))
        retrieval_ms = (perf_counter() - start) * 1000
        observation = (None, 0., 0.)
        truncated = 0
        if pool:
            margins, truncated = self.reader.score(text, self.documents, pool, task=TASK)
            order = sorted(range(len(pool)), key=lambda i: (-margins[i], i))
            first = order[0]
            gap = margins[first] - margins[order[1]] if len(order) > 1 else 0.
            observation = (pool[first], margins[first], gap)
        return dict(pool=pool, reranker=observation, retrieval_ms=retrieval_ms,
                    end_to_end_ms=(perf_counter() - start) * 1000, truncated=truncated)


def evaluate(endpoint, questions):
    values = {method: [] for method in METHODS}
    for i, question in enumerate(questions):
        for offset in range(len(METHODS)):
            method = METHODS[(i + offset) % len(METHODS)]
            outcomes = {}
            for view in VIEWS[i % 2:] + VIEWS[:i % 2]:
                outcomes[view] = endpoint.run(question, view, method)
            values[method].append(dict(id=sha256(question["query"].encode()).hexdigest(),
                                       cohort=question["query_set"], targets=question["corpusids"], views=outcomes))
    return values


def target_groups(rows):
    components = {}

    def head(doc):
        while doc in components:
            doc = components[doc]
        return doc

    for row in rows:
        base = head(row["targets"][0])
        for doc in row["targets"][1:]:
            other = head(doc)
            if other != base:
                components[other] = base
    return [head(row["targets"][0]) for row in rows]


def summarize(rows, policies, reference):
    summary = {}
    for view in VIEWS:
        counts = Counter()
        latencies = defaultdict(list)
        hits = []
        reference_hits = []
        for row, baseline in zip(rows, reference, strict=True):
            if row["id"] != baseline["id"]:
                raise ValueError("Different evaluation order")
            rec = row["views"][view]
            targets = set(row["targets"])
            known_hits = len(targets.intersection(rec["pool"]))
            hits.append(bool(known_hits))
            reference_hits.append(bool(targets.intersection(baseline["views"][view]["pool"])))
            counts["known_target_links"] += len(targets)
            counts["known_target_links_in_pool"] += known_hits
            counts["at_least_one_target_in_pool"] += bool(known_hits)
            counts["top1_target"] += bool(rec["pool"] and rec["pool"][0] in targets)
            top = rec["reranker"][0]
            counts["reranked_top1_target"] += top in targets
            counts["truncated_documents"] += rec["truncated"]
            for name, policy in policies.items():
                if accepts(rec["reranker"], policy):
                    counts[f"{name}_accepted"] += 1
                    counts[f"{name}_known"] += top in targets
                    counts[f"{name}_unjudged"] += top not in targets
            for key in ("retrieval_ms", "end_to_end_ms"):
                latencies[key].append(rec[key])
        p95 = {key: float(np.quantile(values, .95)) for key, values in latencies.items()}
        means = {key: mean(values) for key, values in latencies.items()}
        summary[view] = dict(counts, requests=len(rows), p95=p95, mean=means,
                             hit_delta=clustered_paired_interval(
                                 [int(a) - int(b) for a, b in zip(hits, reference_hits, strict=True)],
                                 target_groups(rows),
                             ))
    return summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with gzip.open(args.projection, "rb") as stream:
        data = json.load(stream)
    if data["revision"] != REVISION or data["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch input")
    papers = data["papers"]
    projection_sha256 = digest(args.projection)
    provenance = verify_checkpoint(args.embedding)
    with np.load(args.cache, allow_pickle=False) as cached:
        if (str(cached["projection_sha256"]) != projection_sha256
                or json.loads(str(cached["provenance"])) != provenance
                or cached["corpus_ids"].tolist() != [r["corpusid"] for r in papers]):
            raise ValueError("Vector cache does not match this corpus or checkpoint")
        vectors = torch.from_numpy(cached["vectors"]).cuda()
        offline = dict(seconds=float(cached["offline_seconds"]), gpu=str(cached["gpu"]),
                       bytes=int(vectors.numel() * vectors.element_size()))
    torch.backends.cuda.matmul.allow_tf32 = False
    mapping = {}
    for i, paper in enumerate(papers):
        value = paper_text(paper)
        while value in mapping:
            value += "\n"
        mapping[value] = i
    started = perf_counter()
    with TemporaryDirectory(dir=args.projection.parent) as temporary:
        index = build_lexical(mapping, Path(temporary))
    index_seconds = perf_counter() - started
    print("Prepared lexical index", round(index_seconds, 2), "seconds", flush=True)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    endpoint = Search(papers, index, vectors, encoder, reader)
    validation, test, _ = split(data["questions"])
    for method in METHODS:
        endpoint.run(validation[0], "present", method)
    calibration = evaluate(endpoint, validation)
    selected = {method: calibrate(rows, "reranker", .10) for method, rows in calibration.items()}
    print("Frozen calibration policies", selected, flush=True)
    tests = evaluate(endpoint, test)
    report = {}
    for method, rows in tests.items():
        if any(set(row["views"]["withheld"]["pool"]) & set(row["targets"]) for row in rows):
            raise ValueError("A withheld target reached the reranker")
        report[method] = summarize(rows, dict(shared=PREVIOUS_POLICY, own=selected[method]["policy"]), tests["bm25"])
    write_json(args.output, dict(
        projection_sha256=projection_sha256, cache_sha256=digest(args.cache),
        script_sha256=digest(Path(__file__)), corpus=len(papers), offline=offline, index_seconds=index_seconds,
        embedding=provenance, reranker=reader.provenance, gpu=torch.cuda.get_device_name(),
        selected=selected, shared_policy=PREVIOUS_POLICY, calibration=calibration, tests=tests, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "cache", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
