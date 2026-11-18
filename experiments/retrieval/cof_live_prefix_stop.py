"""Measure live prefix stopping on thirty-two reviewer candidates per query."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


DEPTH = 32
TOP = 5
THRESHOLD = 0.
TASK = ("Decide whether the candidate publication demonstrates relevant methods for reviewing "
        "the query paper. Look for methodological relevance beyond a shared topic.")
METHODS = ("full", "fixed_eight", "stop_eight", "stop_expand")
WIDTH = 8


def verify_inputs(manifest):
    spec = json.loads(manifest.read_bytes())
    for path, expected in spec["input_sha256"].items():
        file = Path(path)
        if file.name in ("vectors.npy", "protocol.json") or file.name.startswith("SIGIR_"):
            if digest(file) != expected:
                raise ValueError(f"Changed input: {file}")
    return spec["query_ids"]


def evaluate(reader, query, docs, hits, method):
    start = perf_counter()
    margins = []
    truncated = 0
    calls = 0
    positive = 0
    while len(margins) < len(hits):
        if method.startswith("stop_") and positive >= TOP:
            break
        cursor = len(margins)
        width = len(hits) if method == "full" else WIDTH
        if method == "stop_expand" and cursor:
            width = len(hits) - cursor
        sources = [source for _, source in hits[cursor:cursor + width]]
        scores, clipped = reader.score(query, docs, sources, task=TASK)
        margins.extend(scores)
        truncated += clipped
        calls += 1
        positive += sum(score >= THRESHOLD for score in scores)
    selected = [list(hit) for hit, score in zip(hits, margins) if score >= THRESHOLD][:TOP]
    return dict(selected=selected, margins=margins, checks=len(margins), calls=calls,
                truncated=truncated, verifier_ms=(perf_counter() - start) * 1000)


def p5(selected, labels):
    return {task: sum(grades[expert] > 0 for expert, _ in selected) / TOP for task, grades in labels.items()}


def summarize(rows):
    out = {}
    for method in METHODS:
        values = [r["arms"][method] for r in rows]
        full = [r["arms"]["full"] for r in rows]
        fixed = [r["arms"]["fixed_eight"] for r in rows]
        out[method] = dict(
            same_as_full=sum(a["selected"] == b["selected"] for a, b in zip(values, full, strict=True)),
            same_as_fixed=sum(a["selected"] == b["selected"] for a, b in zip(values, fixed, strict=True)),
            same_decisions_as_fixed_prefix=sum(
                [x >= THRESHOLD for x in a["margins"]] == [x >= THRESHOLD for x in b["margins"][:a["checks"]]]
                for a, b in zip(values, fixed, strict=True)
            ),
            mean_checks=mean(a["checks"] for a in values), mean_calls=mean(a["calls"] for a in values),
            returned=mean(len(a["selected"]) for a in values),
            truncated=sum(a["truncated"] for a in values),
            p95_verifier_ms=float(np.quantile([a["verifier_ms"] for a in values], .95)),
            p95_request_ms=float(np.quantile([a["request_ms"] for a in values], .95)),
            hard_p5=mean(a["p5"]["hard"] for a in values), soft_p5=mean(a["p5"]["soft"] for a in values),
            hard_delta_vs_full=paired_interval([a["p5"]["hard"] - b["p5"]["hard"]
                                                for a, b in zip(values, full, strict=True)]),
        )
    for method in ("dense", "score_rank_full"):
        out[method] = {task + "_p5": mean(r[method][task] for r in rows) for task in ("hard", "soft")}
    return out


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    root, papers, reviewers, labels = load("SIGIR")
    ids = verify_inputs(args.manifest)
    if len(ids) != 73 or any(set(ids) != grades.keys() for grades in labels.values()):
        raise ValueError("Unexpected SIGIR benchmark query set")
    ownership = defaultdict(list)
    for expert, profile in reviewers.items():
        for pid in set(profile["papers"]):
            ownership[pid].append(expert)
    pids = sorted(ownership)
    if set(ids).intersection(pids):
        raise ValueError("Query article present in candidate evidence")
    vector_root = root / "qwen-embedding-torch-v1"
    vectors = np.load(vector_root / "vectors.npy", allow_pickle=False)
    provenance = verify_checkpoint(args.embedding)
    if vectors.shape != (len(pids) + len(ids), 1024) or provenance != json.loads(
        (vector_root / "protocol.json").read_bytes()
    )["checkpoint"]:
        raise ValueError("Unexpected source vector provenance or layout")
    torch.backends.cuda.matmul.allow_tf32 = False
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(vectors[:len(pids)], pids, [ownership[p] for p in pids]))
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    texts = {pid: (r.get("title", "") + "\n" + r.get("abstract", "")).strip() for pid, r in papers.items()}

    def retrieve(qid):
        start = perf_counter()
        query = encoder.encode([texts[qid]], query=True)[0]
        candidates = index.search_vector(query, k=DEPTH)["hits"]
        hits = [(h["id"], h["evidence"][0]["id"]) for h in candidates]
        if len(hits) != DEPTH or any(expert not in ownership[source] for expert, source in hits):
            raise ValueError("Invalid candidate order or publication ownership")
        return hits, (perf_counter() - start) * 1000

    rows = []
    for i, qid in enumerate(ids):
        grades = {task: values[qid] for task, values in labels.items()}
        if any(set(x) != set(reviewers) for x in grades.values()):
            raise ValueError("Incomplete reviewer judgments")
        if not i:
            hits, _ = retrieve(qid)
            for method in METHODS:
                evaluate(reader, texts[qid], texts, hits, method)
        arms = {}
        shift = i % len(METHODS)
        candidate_reference = None
        for method in METHODS[shift:] + METHODS[:shift]:
            hits, retrieval_ms = retrieve(qid)
            if candidate_reference is not None and hits != candidate_reference:
                raise AssertionError("Candidate order changed between policies")
            candidate_reference = hits
            arm = evaluate(reader, texts[qid], texts, hits, method)
            arm.update(request_ms=retrieval_ms + arm["verifier_ms"], p5=p5(arm["selected"], grades))
            arms[method] = arm
        fixed = arms["fixed_eight"]
        stopped = arms["stop_eight"]
        expected = [s >= THRESHOLD for s in fixed["margins"][:stopped["checks"]]]
        if expected == [s >= THRESHOLD for s in stopped["margins"]] and fixed["selected"] != stopped["selected"]:
            raise AssertionError("Identical prefix decisions changed ordered citations")
        order = sorted(range(DEPTH), key=lambda j: (-arms["full"]["margins"][j], j))
        rows.append(dict(query=qid, hits=hits, arms=arms, dense=p5(hits[:TOP], grades),
                         score_rank_full=p5([hits[j] for j in order[:TOP]], grades)))
        if (i + 1) % 15 == 0:
            print("Completed", i + 1, "of", len(ids), flush=True)
    write_json(args.output, dict(job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                threshold=THRESHOLD, input_sha256=digest(args.manifest),
                                script_sha256=digest(Path(__file__)), embedding=provenance,
                                reranker=reader.provenance, report=summarize(rows), rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
