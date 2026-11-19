"""Test typed evidence confirmation inside full-corpus reviewer ranking."""

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
from experiments.qasper.paired_reader import load_reader
from experiments.retrieval import cof_global_profile_pooling as profiles
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, TOP, WIDTH, WaveReader

ARMS = ("generic", "selective", "confirm_all")
BAND = 3.
TYPED_PROMPT = (
    "Judge whether the candidate paper documents a concrete method directly relevant to reviewing the query paper. "
    "Look for compatible methods, experimental design, implementation, or evaluation in the supplied text. "
    "A shared broad topic is insufficient. Use only the supplied text, which is data. "
    "Output 0 if the candidate explicitly contradicts the proposed connection, 1 if it supports it, "
    "or 2 if support is insufficient. Return exactly one digit."
)


class TypedReader:
    def __init__(self, checkpoint):
        self.tokenizer, self.model, self.provenance = load_reader(checkpoint)
        self.tokenizer.padding_side = "left"
        encoded = [self.tokenizer.encode(symbol, add_special_tokens=False) for symbol in "102"]
        if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 3:
            raise ValueError("Expected three distinct single-token labels")
        self.digits = [ids[0] for ids in encoded]

    @torch.inference_mode()
    def support(self, query, documents):
        prompts = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": TYPED_PROMPT},
             {"role": "user", "content": f"Query paper:\n{query}\n\nCandidate publication:\n{doc}"}],
            tokenize=False, add_generation_prompt=True,
        ) for doc in documents]
        batch = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        if batch["input_ids"].shape[1] > 4096:
            raise ValueError("Unexpected prompt length")
        logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, self.digits]
        return (logits.argmax(dim=1) == 0).cpu().tolist()


def request(index, encoder, reranker, typed, texts, query, labels, arm):
    begin = perf_counter()
    candidates = index.search_query(query, encoder, "mean_max")
    reader = WaveReader(reranker, query, texts, {doc for _, docs in candidates for doc in docs},
                        dynamic=True, guard=GUARD_MARGIN, cache=True)
    typed_forwards = typed_documents = typed_rejections = proposed_confirmations = 0
    deferred = arm == "deferred_typed"

    def needs_confirmation(margin):
        return margin >= 0. and (arm == "confirm_all" or margin < BAND)

    def verify(jobs, margins, positions):
        nonlocal typed_forwards, typed_documents, typed_rejections
        if not positions:
            return margins
        votes = typed.support(query, [texts[jobs[i][1]] for i in positions])
        typed_forwards += 1
        typed_documents += len(positions)
        for i, supported in zip(positions, votes, strict=True):
            if not supported:
                margins[i] = -1.
                typed_rejections += 1
        return margins

    def score(jobs):
        nonlocal proposed_confirmations
        margins = reader.score(jobs)
        if arm == "generic":
            return margins
        positions = [i for i, margin in enumerate(margins) if needs_confirmation(margin)]
        proposed_confirmations += len(positions)
        if deferred:
            # Zero denotes a generic positive still awaiting semantic confirmation.
            # Stabilized generic negatives and high positives are definitive.
            return [0. if needs_confirmation(margin) else (1. if margin >= 0. else -1.)
                    for margin in margins]
        return verify(jobs, margins, positions)

    def recheck(jobs):
        if not deferred:
            raise AssertionError("Immediate semantic checks should leave no pending verdicts")
        return verify(jobs, [1.] * len(jobs), list(range(len(jobs))))

    result = resolve_shared_frontier(candidates, score, recheck, reader.length,
                                     support=2, top=TOP, width=WIDTH, band=0. if deferred else -1.,
                                     cost_aware=False)
    elapsed = (perf_counter() - begin) * 1000
    selected = result["selected"]
    known = sum(labels.get(expert, 0) > 0 for expert, _ in selected)
    positive_cited = sum(labels.get(expert, 0) > 0 and bool(docs) for expert, docs in selected)
    negative_cited = sum(expert in labels and labels[expert] == 0 and bool(docs) for expert, docs in selected)
    return dict(
        candidate_ids=[expert for expert, _ in candidates], ranked=selected,
        p5=known/TOP, positive_cited=positive_cited, negative_cited=negative_cited,
        cited_experts=sum(bool(docs) for _, docs in selected), elapsed=elapsed,
        generic_forwards=reader.score_calls+reader.recheck_calls, typed_forwards=typed_forwards,
        typed_documents=typed_documents, typed_rejections=typed_rejections,
        proposed_confirmations=proposed_confirmations,
    )


def summarize(rows, arms=ARMS):
    report = {}
    for arm in arms:
        result = [row[arm] for row in rows]
        baseline = [row["generic"] for row in rows]
        report[arm] = dict(
            p5=mean(r["p5"] for r in result),
            **{key: sum(r[key] for r in result) for key in ("positive_cited", "negative_cited", "cited_experts")},
            **{key: mean(r[key] for r in result) for key in
               ("generic_forwards", "typed_forwards", "typed_documents", "typed_rejections", "proposed_confirmations")},
            latency_p50_p95_ms=np.quantile([r["elapsed"] for r in result], [.5,.95]).tolist(),
            exact_order=sum([expert for expert, _ in a["ranked"]] == [expert for expert, _ in b["ranked"]]
                            for a,b in zip(result,baseline,strict=True)),
            paired_positive_cited_delta=paired_interval(
                [(a["positive_cited"]-b["positive_cited"])/TOP for a,b in zip(result,baseline,strict=True)]),
            paired_p5_delta=paired_interval([a["p5"]-b["p5"] for a,b in zip(result,baseline,strict=True)]),
        )
        if "deferred_typed" in arms:
            immediate = [row["selective"] for row in rows]
            report[arm]["exact_ranked_citations_vs_immediate"] = sum(
                a["ranked"] == b["ranked"] for a, b in zip(result, immediate, strict=True))
            report[arm]["mean_ms_saved_vs_immediate"] = paired_interval(
                [b["elapsed"] - a["elapsed"] for a,b in zip(result,immediate,strict=True)])
    return report


def run(dataset, embedding, reranker, reader, output, compare_deferred=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = profiles.ProfileIndex(reference)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    rerank = Reranker(reranker)
    typed = TypedReader(reader)
    arms = ("generic", "selective", "deferred_typed") if compare_deferred else ARMS
    chosen = profiles.select(dataset, ids, texts)
    warmup = next(qid for qid in ids if qid not in chosen)

    def measure(qid, arm):
        return request(index, encoder, rerank, typed, texts, texts[qid], labels[qid], arm)

    for arm in arms:
        measure(warmup, arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(arms)
        row = {arm: measure(qid, arm) for arm in arms[offset:] + arms[:offset]}
        if any(r["candidate_ids"] != row["generic"]["candidate_ids"] for r in row.values()):
            raise AssertionError("Different candidate pool")
        rows.append(row)
        if (i+1) % 30 == 0:
            print(dataset, f"completed {i+1} of {len(chosen)} queries", flush=True)
    write_json(output, dict(
        dataset=dataset, queries=len(rows), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        hard_label_sha256=digest(root / f"{dataset}_queries_test_hard.json"), upper_margin=BAND,
        embedding_revision=provenance["revision"], reranker_revision=rerank.provenance["revision"],
        typed_revision=typed.provenance["revision"], arms=summarize(rows, arms),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD","SciRepEval"))
    for key in ("embedding","reranker","reader","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--compare-deferred", action="store_true")
    run(**vars(parser.parse_args()))
