"""Measure author discovery and exact accepted-answer citation together."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.community.crossvalidated_witness import write
from experiments.retrieval.lrbench_frozen_baselines import build_lexical

TOP_K = 5
FUSION_K = 60


def rank(values):
    return np.argsort(-np.asarray(values), kind="stable").tolist()


def group_witnesses(order, owners):
    witnesses = defaultdict(list)
    for index in order:
        witnesses[owners[index]].append(index)
    return list(witnesses.items())


def counts(rows, known, owners, width, top_k=TOP_K):
    people = {owners[doc] for doc in known}
    return dict(
        exact_people=sum(owner in people for owner, _ in rows[:top_k]),
        grounded_people=sum(bool(known.intersection(docs[:width])) for _, docs in rows[:top_k]),
        citations=sum(len(docs[:width]) for _, docs in rows[:top_k]),
    )


def connected_groups(queries):
    roots = {}

    def root(node):
        while roots.get(node, node) != node:
            node = roots[node]
        return node

    for query in queries:
        nodes = query["answers"]
        anchor = root(nodes[0])
        for node in nodes[1:]:
            other = root(node)
            if other != anchor:
                roots[other] = anchor
    return [root(query["answers"][0]) for query in queries]


def run(snapshot, manifest_path, model_path, output):
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("Run on Slurm with a fresh result file")
    manifest = json.loads(manifest_path.read_bytes())
    if sha256(snapshot.read_bytes()).hexdigest() != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot differs from the frozen manifest")
    data = json.loads(snapshot.read_bytes())
    docs = data["documents"]
    ids = sorted(int(aid) for aid in docs)
    if ids != manifest["candidate_answers"]:
        raise ValueError("Candidate manifest disagrees with snapshot")
    index_for_id = {aid: i for i, aid in enumerate(ids)}
    owners = [docs[str(aid)]["owner"] for aid in ids]
    documents = [f"{docs[str(aid)]['question']}\n{docs[str(aid)]['question_text']}\n{docs[str(aid)]['answer']}"
                 for aid in ids]
    if len(set(documents)) != len(documents):
        raise ValueError("Duplicate evidence texts would collapse lexical IDs")
    encoder = QwenTorchEmbedding(model_path)
    encoder.model.config.use_cache = False
    provenance = verify_checkpoint(model_path)
    started = perf_counter()
    vectors = encoder.encode(documents)
    index_ms = (perf_counter() - started) * 1000
    with TemporaryDirectory(dir=output.parent, prefix="cv-lexical-") as scratch:
        start = perf_counter()
        lexical = build_lexical(dict(zip(documents, range(len(ids)), strict=True)), Path(scratch))
        search_build_ms = (perf_counter() - start) * 1000
    queries = data["queries"]
    timings = defaultdict(list)
    outputs = defaultdict(list)
    for i, query in enumerate([queries[0], *queries]):
        text = f"{query['title']}\n{query['query']}"
        known = {index_for_id[aid] for aid in query["answers"]}
        begin = perf_counter()
        sparse_scores = lexical.document_scores(text)
        sparse = sorted((int(i) for i, score in sparse_scores.items() if score > 0),
                        key=lambda i: (-sparse_scores[str(i)], i))
        sparse_rows = group_witnesses(sparse, owners)
        sparse_ms = (perf_counter() - begin) * 1000
        begin = perf_counter()
        vector = encoder.encode([text], query=True)[0]
        dense = rank(vectors @ vector)
        dense_rows = group_witnesses(dense, owners)
        dense_ms = (perf_counter() - begin) * 1000
        begin = perf_counter()
        combined_scores = np.zeros(len(docs))
        for order in (sparse, dense):
            for rank_index, index in enumerate(order, 1):
                combined_scores[index] += 1 / (FUSION_K + rank_index)
        combined = rank(combined_scores)
        combined_rows = group_witnesses(combined, owners)
        combined_ms = sparse_ms + dense_ms + (perf_counter() - begin) * 1000
        if i:
            for name, rows, ms in (("bm25", sparse_rows, sparse_ms), ("dense", dense_rows, dense_ms),
                                   ("rrf", combined_rows, combined_ms)):
                outputs[name].append({
                    "one": counts(rows, known, owners, 1),
                    "two": counts(rows, known, owners, 2),
                    "ten_people": counts(rows, known, owners, 1, 10),
                })
                timings[name].append(ms)
    groups = connected_groups(queries)
    summary = {}
    for name, rows in outputs.items():
        arms = {}
        for variant in ("one", "two", "ten_people"):
            arm = [row[variant] for row in rows]
            totals = {key: sum(row[key] for row in arm) for key in arm[0]}
            arms[variant] = dict(
                **totals, queries_known_owner=sum(row["exact_people"] > 0 for row in arm),
                queries_known_witness=sum(row["grounded_people"] > 0 for row in arm),
                grounded_p5=totals["grounded_people"] / (TOP_K * len(queries)) if variant != "ten_people" else None,
                witness_hit_delta_vs_bm25=clustered_paired_interval(
                    [int(a["grounded_people"] > 0) - int(b[variant]["grounded_people"] > 0)
                     for a, b in zip(arm, outputs["bm25"], strict=True)], groups,
                ),
            )
        summary[name] = dict(arms=arms, local_p95_ms=float(np.quantile(timings[name], .95)))
    write(output, dict(job=os.environ["SLURM_JOB_ID"], queries=len(queries), owners=len(set(owners)),
                      evidence=len(docs), query_groups=len(set(groups)), encoding_ms=index_ms,
                      lexical_index_ms=search_build_ms, manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                      code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                      model_revision=provenance["revision"], metrics=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "manifest_path", "model_path", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
