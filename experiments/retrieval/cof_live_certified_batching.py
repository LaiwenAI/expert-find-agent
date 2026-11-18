"""Time certified batching with fixed-shape live CoF source verification."""

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
from expert_search.qwen_reranker import pack_tokens
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_live_prefix_stop import TASK, TOP, verify_inputs


WIDTH = 4
LENGTH_BUCKET = 64
CAP = 1024
POOL = 10
SUPPORT = (1, 2)
METHODS = ("exhaustive", "one_per_expert", "fill_idle_slots")
PREFIX_CONTROL = ("exhaustive", "one_per_expert", "prefix_two", "prefix_three")
PER_EXPERT = {"one_per_expert": 1, "prefix_two": 2, "prefix_three": 3}


class WaveReader:
    def __init__(self, reader, query, documents, sources):
        self.reader = reader
        self.inputs = {}
        self.truncated = 0
        self.maximum = 0
        for source in sorted(sources):
            message = f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {documents[source]}"
            body = reader.tokenizer.encode(message, add_special_tokens=False)
            tokens, raw = pack_tokens(reader.prefix, body, reader.suffix, CAP)
            self.inputs[source] = dict(input_ids=tokens)
            self.truncated += raw > len(tokens)
            self.maximum = max(self.maximum, len(tokens))
        self.maximum = ((self.maximum + LENGTH_BUCKET - 1) // LENGTH_BUCKET) * LENGTH_BUCKET

    @torch.inference_mode()
    def score(self, jobs):
        batch = [self.inputs[doc] for _, doc in jobs]
        filled = batch + [batch[0]] * (WIDTH - len(batch))
        inputs = self.reader.tokenizer.pad(filled, padding="max_length", max_length=self.maximum,
                                          return_tensors="pt").to("cuda")
        logits = self.reader.model(**inputs, use_cache=False, logits_to_keep=1).logits
        values = logits[:len(jobs), -1, self.reader.labels].float()
        return (values[:, 0] - values[:, 1]).cpu().tolist()


def schedule(ordered, reader, count, method):
    pending = dict(ordered)
    seen = {e: 0 for e in pending}
    accepted = {e: [] for e in pending}
    rank = {e: i for i, e in enumerate(pending)}
    margins = {}
    reads = set()
    calls = 0
    while True:
        confirmed = [e for e in pending if len(accepted[e]) == count]
        if method == "exhaustive":
            jobs = [(e, doc) for e in pending for doc in pending[e][seen[e]:]][:WIDTH]
        else:
            boundary = rank[confirmed[TOP - 1]] if len(confirmed) >= TOP else len(rank)
            eligible = [e for e in pending if rank[e] < boundary and len(accepted[e]) < count
                        and len(accepted[e]) + len(pending[e]) - seen[e] >= count]
            if method == "fill_idle_slots":
                jobs = [(e, pending[e][seen[e]]) for e in eligible[:WIDTH]]
                if len(jobs) < WIDTH:
                    suffix = [(e, doc) for e in eligible for doc in pending[e][seen[e] + 1:]]
                    jobs.extend(suffix[:WIDTH - len(jobs)])
            else:
                jobs = [(e, doc) for e in eligible
                        for doc in pending[e][seen[e]:seen[e] + PER_EXPERT[method]]][:WIDTH]
        if not jobs:
            order = confirmed + [e for e in pending if len(accepted[e]) < count]
            selected = [[e, accepted[e] if e in confirmed else []] for e in order[:TOP]]
            return dict(selected=selected, checks=len(margins), calls=calls,
                        unique_reads=len(reads), margins=margins)
        calls += 1
        for (expert, source), score in zip(jobs, reader.score(jobs), strict=True):
            margins[expert, source] = score
            seen[expert] += 1
            reads.add(source)
            if len(accepted[expert]) < count and score >= 0.:
                accepted[expert].append(source)


def summarize(rows, methods):
    report = {}
    for count in SUPPORT:
        values = [row["support"][count] for row in rows]
        reference = [row["exhaustive"] for row in values]
        report[count] = {}
        for method in methods:
            arms = [row[method] for row in values]
            report[count][method] = dict(
                exact_experts_and_sources=sum(a["selected"] == b["selected"]
                                              for a, b in zip(arms, reference, strict=True)),
                score_differences=sum(a["score_differences"] for a in arms),
                decision_differences=sum(a["decision_differences"] for a in arms),
                p95_ms=float(np.quantile([a["ms"] for a in arms], .95)),
                p50_ms=float(np.median([a["ms"] for a in arms])),
                average={key: mean(arm[key] for arm in arms)
                         for key in ("calls", "checks", "unique_reads", "hard_p5", "confirmed_hard_p5")},
                latency_saved_vs_one_per_expert=paired_interval([
                    row["one_per_expert"]["ms"] - row[method]["ms"] for row in values
                ]),
            )
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    root, papers, reviewers, labels = load("SIGIR")
    ids = verify_inputs(args.manifest)
    if len(ids) != 73 or any(set(ids) != grades.keys() for grades in labels.values()):
        raise ValueError("Unexpected SIGIR query set")
    ownership = defaultdict(list)
    for owner, profile in reviewers.items():
        for doc in set(profile["papers"]):
            ownership[doc].append(owner)
    source_ids = sorted(ownership)
    cache = root / "qwen-embedding-torch-v1"
    vectors = np.load(cache / "vectors.npy", allow_pickle=False)
    provenance = verify_checkpoint(args.embedding)
    if (vectors.shape != (len(source_ids) + len(ids), 1024)
            or provenance != json.loads((cache / "protocol.json").read_bytes())["checkpoint"]
            or set(ids).intersection(ownership)):
        raise ValueError("Changed source-vector snapshot")
    torch.backends.cuda.matmul.allow_tf32 = False
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(vectors[:len(source_ids)], source_ids,
                                                    [ownership[doc] for doc in source_ids]))
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reranker = Reranker(args.reranker)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}

    def measure(qid, count, method):
        start = perf_counter()
        query = texts[qid]
        hits = index.search_vector(encoder.encode([query], query=True)[0], k=POOL)["hits"]
        ordered = [(row["id"], [w["id"] for w in row["evidence"]]) for row in hits]
        if len(ordered) != POOL or any(e not in ownership[d] for e, docs in ordered for d in docs):
            raise ValueError("Unexpected source ownership")
        reader = WaveReader(reranker, query, texts, {d for _, docs in ordered for d in docs})
        result = schedule(ordered, reader, count, method)
        result["ms"] = (perf_counter() - start) * 1000
        result["truncated"] = reader.truncated
        return ordered, result

    methods = PREFIX_CONTROL if args.prefix_control else METHODS
    for count in SUPPORT:
        for method in methods:
            measure(ids[0], count, method)
    rows = []
    for i, qid in enumerate(ids):
        by_count = {}
        offset = i % len(methods)
        for count in SUPPORT[::1 if i % 2 == 0 else -1]:
            ordered = None
            arms = {}
            for method in methods[offset:] + methods[:offset]:
                pool, arm = measure(qid, count, method)
                if ordered is not None and pool != ordered:
                    raise ValueError("Candidate inventory drift")
                ordered = pool
                arms[method] = arm
            expected = arms["exhaustive"]["margins"].copy()
            for arm in arms.values():
                actual = arm.pop("margins")
                arm["score_differences"] = sum(value != expected[key] for key, value in actual.items())
                arm["decision_differences"] = sum((value >= 0.) != (expected[key] >= 0.)
                                                  for key, value in actual.items())
                supported = labels["hard"][qid]
                arm["hard_p5"] = sum(supported[e] > 0 for e, _ in arm["selected"]) / TOP
                arm["confirmed_hard_p5"] = sum(supported[e] > 0 and bool(docs)
                                                 for e, docs in arm["selected"]) / TOP
            by_count[count] = arms
        rows.append(dict(id=qid, support=by_count))
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1}/{len(ids)}", flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(rows),
        script_sha256=digest(Path(__file__)), input_sha256=digest(args.manifest),
        embedding=provenance, reranker=reranker.provenance, report=summarize(rows, methods), rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--prefix-control", action="store_true")
    run(parser.parse_args())
