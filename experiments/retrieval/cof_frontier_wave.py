"""Compare guarded verification waves with length-bounded speculative scoring."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_binary_head import BinaryWaveReader
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, read_data, select
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, TOP, WIDTH


ARMS = {"frontier4": (4, False), "fill4": (4, True), "frontier8": (8, False), "fill8": (8, True)}
SUPPORT = 2


class WaveScheduler:
    def __init__(self, reader, candidates, width, fill):
        self.reader = reader
        self.candidates = dict(candidates)
        self.width = width
        self.fill = fill
        self.speculative = set()
        self.speculative_used = set()

    def score(self, jobs):
        self.speculative_used.update(doc for _, doc in jobs if doc in self.speculative)
        fresh = [(expert, doc) for expert, doc in jobs if doc not in self.reader.score_cache]
        if self.fill and 0 < len(fresh) < self.width:
            extras = self.extras(jobs, self.width - len(fresh), max(self.reader.length(doc) for _, doc in fresh))
            self.speculative.update(doc for _, doc in extras)
            values = self.reader.score([*jobs, *extras])
            return values[:len(jobs)]
        return self.reader.score(jobs)

    def extras(self, jobs, remaining, max_length):
        blocked = {doc for _, doc in jobs} | self.reader.score_cache.keys()
        picked = {}
        for expert in dict.fromkeys(expert for expert, _ in jobs):
            sources = self.candidates[expert]
            confirmed_positive = 0
            unresolved = 0
            for doc in sources:
                if doc in self.reader.recheck_cache:
                    confirmed_positive += self.reader.recheck_cache[doc] >= 0
                else:
                    score = self.reader.score_cache.get(doc)
                    if score is None or abs(score) <= GUARD_MARGIN:
                        unresolved += 1
                    elif score > 0:
                        confirmed_positive += 1
            if not confirmed_positive < SUPPORT <= confirmed_positive + unresolved:
                continue
            for doc in sources:
                if doc not in blocked and self.reader.length(doc) <= max_length:
                    picked.setdefault(doc, expert)
        return [(expert, doc) for doc, expert in list(picked.items())[:remaining]]

    def recheck(self, jobs):
        # Preserve the original fixed-width definitive scorer in every arm.
        return [value for start in range(0, len(jobs), WIDTH)
                for value in self.reader.recheck(jobs[start:start + WIDTH])]


def verify(reader, weights, candidates, texts, query, arm, relevant):
    start = perf_counter()
    reader = BinaryWaveReader(reader, query, texts, {doc for _, sources in candidates for doc in sources},
                             "two_rows_batched_tokenization", weights)
    width, fill = ARMS[arm]
    policy = WaveScheduler(reader, candidates, width, fill)
    out = resolve_shared_frontier(
        candidates, policy.score, policy.recheck, reader.length,
        support=SUPPORT, top=TOP, width=width, band=GUARD_MARGIN, cost_aware=False,
    )
    return dict(
        selected=out["selected"], ms=(perf_counter() - start) * 1000,
        forwards=reader.score_calls + reader.recheck_calls, docs=reader.scored_documents,
        padded_tokens=reader.padded_tokens, rechecks=reader.rechecked_pairs,
        speculative=len(policy.speculative), unused_speculative=len(policy.speculative - policy.speculative_used),
        cited_positives=sum(expert in relevant and bool(sources) for expert, sources in out["selected"]),
    )


def summarize(rows):
    baseline = [row["frontier4"] for row in rows]
    result = {}
    for arm in ARMS:
        measured = [row[arm] for row in rows]
        result[arm] = dict(
            exact_outputs=sum(a["selected"] == b["selected"] for a, b in zip(measured, baseline, strict=True)),
            cited_positives=sum(row["cited_positives"] for row in measured),
            p95_ms=float(np.quantile([row["ms"] for row in measured], .95)),
            paired_latency_delta=paired_interval([a["ms"] - b["ms"] for a, b in zip(measured, baseline, strict=True)]),
            **{key: mean(row[key] for row in measured)
               for key in ("forwards", "docs", "padded_tokens", "rechecks", "speculative", "unused_speculative")},
        )
    return result


def run(dataset, embedding, reranker, output, *, methods=tuple(ARMS), measure=verify, aggregate=summarize, setup=None):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = ProfileIndex(reference)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if reader.model.lm_head.bias is not None:
        raise ValueError("Expected the pinned bias-free Qwen head")
    with torch.inference_mode():
        weights = reader.model.lm_head.weight[reader.labels].contiguous()
    chosen = select(dataset, ids, texts)
    warmup = next(q for q in ids if q not in chosen)
    calibration = setup(reader, weights) if setup is not None else {}
    rows = []
    for i, q in enumerate([warmup, *chosen]):
        candidates = index.search_query(texts[q], encoder, "mean_max")[:TOP]
        offset = i % len(methods)
        relevant = {expert for expert, grade in labels[q].items() if grade > 0}
        row = {arm: measure(reader, weights, candidates, texts, texts[q], arm, relevant)
               for arm in methods[offset:] + methods[:offset]}
        if i:
            rows.append(row)
        if i and i % 25 == 0:
            print(f"{dataset}: {i}/{len(chosen)}", flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(chosen),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(), code_sha256=digest(Path(measure.__code__.co_filename)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        **calibration, arms=aggregate(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
