"""Calibrate retrieval-order issuance and verifier reranking on separate RELISH data."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.crosspage_certificate_ablation import candidate_list
from experiments.csfcube.relish_risk_routing import RelishIndex, retrieve
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


DEPTHS = (0, 1, 2, 4)
POLICIES = {
    0: ("retrieve",), 1: ("top_only",), 2: ("rerank", "first_passing"),
    4: ("rerank", "first_passing"),
}


def collect(questions, index, encoder, reader, offset, timed):
    result = defaultdict(list)
    for i, question in enumerate([questions[0], *questions]):
        shift = (i + offset) % len(DEPTHS)
        for depth in DEPTHS[shift:] + DEPTHS[:shift]:
            clock = perf_counter()
            row = retrieve(question, index, encoder, reader, ())
            order = row["docs"]
            clips = 0
            checked = {}
            if depth:
                margins, clips = index.check(reader, row["focal"], order[:depth], question["facet"])
                checked = {doc: {"margin": float(score), "total": float(score)}
                           for doc, score in zip(order[:depth], margins, strict=True)}
            base = dict(query=question["query_id"], order=order, prior=float(row["cosine"][0]),
                        checked=checked)
            json.dumps(base)
            elapsed = (perf_counter() - clock) * 1000
            if i:
                result[depth].append(base | dict(reads=depth, clipped=clips, ms=elapsed if timed else None))
    return result


def score(choices, labels, cutoff):
    picked = [next((labels[i][doc] for doc, value in row if value > cutoff), None)
              for i, row in enumerate(choices)]
    return picked, dict(
        complete=picked.count(2), partial=sum(p in (0, 1) for p in picked),
        grade_zero=picked.count(0), grade_one=picked.count(1), issued=sum(p is not None for p in picked),
    )


def fit_threshold(choices, labels):
    """Maximize known strong returns under the predeclared development risk cap."""
    scores = {value for row in choices for _, value in row}
    if not scores:
        raise ValueError("No calibration scores")
    candidates = [float(np.nextafter(min(scores), -np.inf)), *sorted(scores)]
    feasible = []
    limit = len(choices) // 10
    for cutoff in candidates:
        _, counts = score(choices, labels, cutoff)
        if counts["partial"] <= limit:
            feasible.append((counts["complete"], -counts["partial"], -cutoff, counts))
    _, _, neg_cutoff, counts = max(feasible, key=lambda item: item[:3])
    return -neg_cutoff, counts


def evaluate(calibration, held, index, encoder, reader, offset):
    development = collect(calibration, index, encoder, reader, offset, False)
    cal_labels = [row["grades"] for row in calibration]
    frozen = {
        (depth, policy): fit_threshold([candidate_list(row, policy) for row in development[depth]], cal_labels)
        for depth in DEPTHS for policy in POLICIES[depth]
    }
    test = collect(held, index, encoder, reader, offset, True)
    labels = [row["grades"] for row in held]
    reference, counts = frozen[(0, "retrieve")]
    pure, pure_counts = score([candidate_list(row, "retrieve") for row in test[0]], labels, reference)
    result = {}
    for depth in DEPTHS:
        routes = {}
        for policy in POLICIES[depth]:
            cutoff, before = frozen[(depth, policy)]
            picks, after = score([candidate_list(row, policy) for row in test[depth]], labels, cutoff)
            routes[policy] = dict(
                threshold=cutoff, calibration=before, held=after,
                complete_gain_vs_retrieve=clustered_paired_interval(
                    [int(a == 2) - int(b == 2) for a, b in zip(picks, pure, strict=True)],
                    [q["query_id"] for q in held],
                ),
                held_grades=picks,
            )
        rows = test[depth]
        result[str(depth)] = dict(
            reads=sum(row["reads"] for row in rows), clips=sum(row["clipped"] for row in rows),
            p50_p95_ms=np.quantile([row["ms"] for row in rows], [.5,.95]).tolist(), policies=routes,
        )
    if result["0"]["policies"]["retrieve"]["held"] != pure_counts or counts["partial"] > len(calibration) // 10:
        raise AssertionError("Pure retrieval does not reproduce its frozen calibration")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "manifest", "embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["panel_sha256"] != digest(args.panel):
        raise ValueError("Panel differs from manifest")
    panel = json.loads(args.panel.read_bytes())
    split = panel["panels"]
    risk, held = split["risk"], split["held"]
    identities = lambda rows: {doc for row in rows for doc in [row["query_id"], *row["grades"]]}
    if identities(risk) & identities(held):
        raise ValueError("Calibration and held sources overlap")
    revision = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    index = RelishIndex(panel["documents"], risk + held, encoder)
    result = evaluate(risk, held, index, encoder, reader, args.offset)
    write_json(args.output, dict(
        calibration_queries=len(risk), held_queries=len(held), panel_sha256=manifest["panel_sha256"],
        gpu=torch.cuda.get_device_name(), embedding_revision=revision["revision"],
        reranker_revision=reader.provenance["revision"], script_sha256=digest(Path(__file__)),
        offset=args.offset, results=result,
    ))


if __name__ == "__main__":
    main()
