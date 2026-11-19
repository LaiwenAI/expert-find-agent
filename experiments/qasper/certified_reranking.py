"""Measure exact early stopping under a bounded reranker on QASPER full text."""

import argparse
from hashlib import sha256
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.adaptive_evidence import ASSETS, TASK, digest, evidence_sets, offline, quality, query_input, read_data


TOP = 5
BLOCK = 8
SAMPLE = 96
SALT = "qasper-certified-reranking-v1"
WEIGHT = 0.25
FIXED_LIMITS = {"dense": 0, "fixed8": 8, "fixed16": 16, "fixed32": 32}
ARMS = (*FIXED_LIMITS, "exhaustive", "certified")


def rank(cosines, score, arm):
    dense = np.argsort(-cosines, kind="stable")
    if arm == "dense":
        return dense[:TOP].tolist(), 0, 0
    base = (1.0 - WEIGHT) * cosines
    checked, truncated, top = 0, 0, []
    limit = FIXED_LIMITS.get(arm, len(dense))
    for start in range(0, min(limit, len(dense)), BLOCK):
        # Every arm evaluates the same block composition up to its stopping point.
        indices = dense[start:start + BLOCK].tolist()
        upper = float(base[indices[0]]) + WEIGHT
        if arm == "certified" and len(top) == TOP and upper < -top[-1][0]:
            break
        margins, clipped = score(indices)
        if not all(math.isfinite(margin) for margin in margins):
            raise ValueError("Nonfinite reranker margin")
        probabilities = [1 / (1 + math.exp(-max(-40, min(40, margin)))) for margin in margins]
        top.extend((-float(base[index] + WEIGHT * probability), index)
                   for index, probability in zip(indices, probabilities, strict=True))
        top = sorted(top)[:TOP]
        checked += len(indices)
        truncated += clipped
    return [index for _, index in top], checked, truncated


def panel(dataset):
    def key(identity):
        return sha256(f"{SALT}:{identity}".encode()).digest()
    papers = sorted(dataset, key=key)[:SAMPLE]
    return [(pid, min(dataset[pid]["qas"], key=lambda qa: key(qa["question_id"]))) for pid in papers]


def summarize(rows):
    answerable = [row for row in rows if row["arms"]["dense"]["human_evidence"] is not None]
    return {
        "questions": len(rows),
        "evidence_evaluable": len(answerable),
        "mean_paragraphs": mean(row["paragraphs"] for row in rows),
        "arms": {
            arm: dict(
                exact_exhaustive=sum(row["arms"][arm]["order"] == row["arms"]["exhaustive"]["order"] for row in rows),
                mean_checks=mean(row["arms"][arm]["checked"] for row in rows),
                checked=sum(row["arms"][arm]["checked"] for row in rows),
                truncated=sum(row["arms"][arm]["truncated"] for row in rows),
                latency_ms=list(map(float, np.quantile([row["arms"][arm]["ms"] for row in rows], [.5, .95]))),
                complete_evidence=mean(row["arms"][arm]["human_evidence"]["complete_evidence"] for row in answerable),
                evidence_f1=mean(row["arms"][arm]["human_evidence"]["f1"] for row in answerable),
                complete_delta_vs_dense=clustered_paired_interval([
                    row["arms"][arm]["human_evidence"]["complete_evidence"]
                    - row["arms"]["dense"]["human_evidence"]["complete_evidence"] for row in answerable
                ], [row["paper_id"] for row in answerable]),
            ) for arm in ARMS
        },
    }


def run(data, embedding, reranker, output):
    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm GPU allocation is required")
    split = "test"
    dataset = read_data(data, split)
    selection = panel(dataset)
    embedding_info = verify_checkpoint(embedding)
    if (embedding_info["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
            or not selection):
        raise ValueError("Unexpected embedding checkpoint or empty panel")
    encoder = QwenTorchEmbedding(embedding, max_length=512, batch_size=32)
    if not encoder.torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (reader.provenance["repository"] != "Qwen/Qwen3-Reranker-0.6B"
            or reader.provenance["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473"):
        raise ValueError("Unexpected reranker checkpoint")
    encoder.encode([query_input("What method is used?")])
    reader.score("What method is used?", ["The method is linear regression."] * BLOCK, range(BLOCK), task=TASK)
    metadata, _, vectors, offline_times = offline({pid: dataset[pid] for pid, _ in selection}, encoder)
    rows = []
    for i, (pid, qa) in enumerate(selection):
        paragraphs, sections, lo, hi = metadata[pid]
        context = [f"{section}\n{text}" for section, text in zip(sections, paragraphs, strict=True)]
        bank = np.ascontiguousarray(vectors[lo:hi], dtype=np.float64)
        gold = evidence_sets(qa, paragraphs)
        question = qa["question"]
        score = lambda indices: reader.score(question, context, indices, task=TASK)
        rotation = i % len(ARMS)
        arms = {}
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            started = perf_counter()
            query = np.asarray(encoder.encode([query_input(question)])[0], dtype=np.float64)
            order, checked, truncated = rank(bank @ query, score, arm)
            ms = (perf_counter() - started) * 1000
            arms[arm] = dict(order=order, checked=checked, truncated=truncated, ms=ms,
                             human_evidence=quality([paragraphs[index] for index in order], gold))
        if arms["certified"]["order"] != arms["exhaustive"]["order"]:
            raise AssertionError("Certified route changed the full reranker output")
        rows.append(dict(paper_id=pid, question_id=qa["question_id"], paragraphs=hi-lo, arms=arms))
        if len(rows) % 16 == 0:
            print("Completed matched requests:", len(rows), flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=encoder.torch.cuda.get_device_name(),
        data_sha256=ASSETS[f"qasper-{split}-v0.3.json"], script_sha256=digest(Path(__file__)),
        embedding=embedding_info, reranker=reader.provenance, offline=offline_times,
        config=dict(top=TOP, block=BLOCK, sample_salt=SALT, reranker_weight=WEIGHT),
        report=summarize(rows), rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
