"""Evaluate safe verifier corrections against the same exhaustive CoF ranker."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.bounded_adjustment import adjusted_ranking, normalized_dense
from expert_search import bounded_adjustment
from expert_search.cof_benchmark import load
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_live_prefix_stop import DEPTH, TOP, TASK, WIDTH, p5, verify_inputs


PRIMARY_WEIGHT = .10
WEIGHTS = (.025, .05, PRIMARY_WEIGHT, .20, .40, .80)
METHODS = ("dense", "bounded", "exhaustive")


def summarize(rows):
    report = {}
    for arm in METHODS:
        parts = [r["arms"][arm] for r in rows]
        report[arm] = dict(
            mean_checks=mean(p["checks"] for p in parts),
            p50_p95_ms=np.quantile([p["ms"] for p in parts], [.50, .95]).tolist(),
            same_ordered_top_k_as_exhaustive=sum(p["hits"] == r["arms"]["exhaustive"]["hits"]
                                                for r, p in zip(rows, parts, strict=True)),
            hard_p5=mean(p["p5"]["hard"] for p in parts),
            soft_p5=mean(p["p5"]["soft"] for p in parts),
            hard_delta_vs_dense=paired_interval([
                p["p5"]["hard"] - r["arms"]["dense"]["p5"]["hard"]
                for r, p in zip(rows, parts, strict=True)
            ])
        )
    for weight in WEIGHTS:
        values = [r["frontier"][weight] for r in rows]
        report[f"trace_{weight}"] = dict(hard_p5=mean(p["p5"]["hard"] for p in values),
                                         soft_p5=mean(p["p5"]["soft"] for p in values),
                                         mean_checks=mean(p["checks"] for p in values))
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    root, papers, reviewers, labels = load("SIGIR")
    ids = verify_inputs(args.manifest)
    if len(ids) != 73 or any(set(ids) != x.keys() for x in labels.values()):
        raise ValueError("Unexpected judged query set")
    ownership = defaultdict(list)
    for expert, profile in reviewers.items():
        for pid in set(profile["papers"]):
            ownership[pid].append(expert)
    pids = sorted(ownership)
    vectors = np.load(root / "qwen-embedding-torch-v1" / "vectors.npy", allow_pickle=False)
    provenance = verify_checkpoint(args.embedding)
    protocol = json.loads((root / "qwen-embedding-torch-v1" / "protocol.json").read_bytes())
    if vectors.shape != (len(pids) + len(ids), 1024) or provenance != protocol["checkpoint"] or set(ids) & ownership.keys():
        raise ValueError("Unexpected source vector snapshot")
    torch.backends.cuda.matmul.allow_tf32 = False
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(vectors[:len(pids)], pids, [ownership[p] for p in pids]))
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reader)
    docs = {pid: (record.get("title", "") + "\n" + record.get("abstract", "")).strip()
            for pid, record in papers.items()}

    def measure(qid, method):
        start = perf_counter()
        text = docs[qid]
        vector = encoder.encode([text], query=True)[0]
        found = index.search_vector(vector, k=DEPTH)["hits"]
        pool = [(h["id"], h["evidence"][0]["id"]) for h in found]
        if len(pool) != DEPTH or any(expert not in ownership[source] for expert, source in pool):
            raise ValueError("Unexpected candidate-source pairs")
        base = normalized_dense([h["score"] for h in found])
        observed = []

        def score(a, b):
            margins, truncated = reader.score(text, docs, [source for _, source in pool[a:b]], task=TASK)
            if truncated:
                raise ValueError("Verifier truncated a candidate source")
            observed.extend(margins)
            return margins

        weight = 0. if method == "dense" else PRIMARY_WEIGHT
        ranking, checks = adjusted_ranking(base, score, weight=weight, top=TOP, block_size=WIDTH,
                                           exhaust=method == "exhaustive")
        hits = [pool[i] for i in ranking]
        return dict(hits=hits, checks=checks, ms=(perf_counter() - start) * 1000,
                    pool=pool, base=base.tolist(), margins=observed)

    for method in METHODS:
        measure(ids[0], method)
    rows = []
    for i, query in enumerate(ids):
        grades = {task: values[query] for task, values in labels.items()}
        if any(set(x) != set(reviewers) for x in grades.values()):
            raise ValueError("Incomplete reviewer labels")
        offset = i % len(METHODS)
        arms = {m: measure(query, m) for m in METHODS[offset:] + METHODS[:offset]}
        reference = arms["exhaustive"]
        bounded = arms["bounded"]
        if any(a["pool"] != reference["pool"] or a["base"] != reference["base"] for a in arms.values()):
            raise AssertionError("Candidate ranking changed between policies")
        if bounded["margins"] != reference["margins"][:bounded["checks"]] or bounded["hits"] != reference["hits"]:
            raise AssertionError("Bounded result differs from the identical-block reference")
        for arm in arms.values():
            arm["p5"] = p5(arm["hits"], grades)
        frontier = {}
        for weight in WEIGHTS:
            base = reference["base"]
            margins = reference["margins"]

            def replay(a, b):
                return margins[a:b]

            rank, checks = adjusted_ranking(base, replay, weight=weight, top=TOP, block_size=WIDTH)
            full_rank, _ = adjusted_ranking(base, replay, weight=weight, top=TOP, block_size=WIDTH, exhaust=True)
            if rank != full_rank:
                raise AssertionError("Trace rank differs from full reference")
            frontier[weight] = dict(checks=checks, p5=p5([reference["pool"][j] for j in rank], grades))
        rows.append(dict(query=query, arms=arms, frontier=frontier))
        if (i + 1) % 20 == 0:
            print("Completed", i + 1, "of", len(ids), flush=True)
    write_json(args.output, dict(job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                primary_weight=PRIMARY_WEIGHT, input_sha256=digest(args.manifest),
                                script_sha256=digest(Path(__file__)),
                                ranking_sha256=digest(Path(bounded_adjustment.__file__)), embedding=provenance,
                                reranker=reader.provenance, report=summarize(rows), rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "embedding", "reader", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
