"""Test live adaptive source verification on fully judged CSFCube candidate pools."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.csfcube.frozen_confirmation import JudgedPoolEndpoint, STRONG_RELEVANCE, load
from experiments.csfcube.grouped_calibration import partition
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.adaptive_depth import RoutedConfirmation, STATES, fit
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.learned_confirmation import evaluate
from experiments.litsearch.source_withholding import calibrate
from experiments.litsearch.top_hit_confirmation import DEPTHS, paired_rows


BUDGETS = (.05, .10, .20)
METHODS = (*DEPTHS, "gap", "hash")


class Endpoint(RoutedConfirmation):
    retrieve = JudgedPoolEndpoint.retrieve


def policies_for(rows, budget):
    sample = [dict(targets=row["targets"], views={
        state: {method: row["views"][state][method]["scored"] for method in DEPTHS}
        for state in STATES
    }) for row in rows]
    return {
        **{method: calibrate(sample, method, budget)["policy"] for method in DEPTHS},
        **{method: fit(rows, budget, method) for method in ("gap", "hash")},
    }


def measure(endpoint, question, reference, method, state, policy):
    if method in DEPTHS:
        arm = endpoint.run(question, state, method)
        decision, depth = arm["scored"], DEPTHS[method]
        chosen = decision[0] if accepts(decision, policy) else None
        compared = reference["views"][state][method]
    else:
        arm = endpoint.execute(question, state, reference["id"], policy)
        decision = arm["score"]
        depth = DEPTHS[arm["method"]]
        chosen = arm["accepted"]
        compared = reference["views"][state][arm["method"]]
    if arm["pool"] != compared["pool"] or tuple(decision) != tuple(compared["scored"]):
        raise ValueError("Live decision score or candidate pool changed")
    grade = question["grades"][chosen] if chosen is not None else None
    if state == "withheld" and grade is not None and grade >= STRONG_RELEVANCE:
        raise ValueError("Leaked withheld strong candidate")
    return dict(grade=grade, depth=depth, ms=arm["ms"])


def summarize(results, groups):
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for state in STATES:
            batch = [row[state][method] for row in results]
            grades = Counter(arm["grade"] for arm in batch)
            depths = Counter(arm["depth"] for arm in batch)
            times = [arm["ms"] for arm in batch]
            baseline = [row[state]["two"] for row in results]
            key = (lambda arm: int((arm["grade"] or 0) >= STRONG_RELEVANCE)) if state == "present" else (
                lambda arm: int(arm["grade"] is not None)
            )
            summary[method][state] = dict(
                grade_counts=[grades[g] for g in range(4)], abstain=grades[None], depths=dict(depths),
                scored_documents=sum(depth * n for depth, n in depths.items()),
                p95_ms=float(np.quantile(times, .95)),
                mean_ms=float(np.mean(times)),
                delta_vs_two=clustered_paired_interval(
                    [key(a) - key(b) for a, b in zip(batch, baseline, strict=True)], groups
                ),
            )
    return summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    papers, questions, hashes = load(args.data)
    prior = json.loads(args.reference.read_bytes())
    provenance = verify_checkpoint(args.embedding)
    reader = Reranker(args.reranker)
    if hashes != prior["source_hashes"] or provenance != prior["embedding"] or reader.provenance != prior["reranker"]:
        raise ValueError("Frozen benchmark inputs changed")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    vectors = torch.from_numpy(encoder.encode([paper_text(row) for row in papers])).cuda()
    endpoint = Endpoint(papers, vectors, encoder, None, reader)
    for method in DEPTHS:
        endpoint.run(questions[0], "present", method)
    rows = list(paired_rows(evaluate(endpoint, questions, tuple(DEPTHS))))
    for row, original, control, six in zip(rows, prior["test"]["cosine"], prior["test"]["cosine_two"],
                                           prior["test"]["reader_six"], strict=True):
        if row["id"] != original["id"]:
            raise ValueError("CSFCube requests moved")
        for state in STATES:
            for method, previous in (("dense", original), ("two", control), ("six", six)):
                expected, current = previous["views"][state], row["views"][state][method]
                if expected["pool"] != current["pool"] or tuple(expected["scored"]) != tuple(current["scored"]):
                    raise ValueError("Existing ranking or fixed-depth scores changed")
    groups = [str(question["query_id"]) for question in questions]
    folds = partition([(group, None) for group in groups])
    report, raw = {}, {}
    for budget in BUDGETS:
        results = [None] * len(rows)
        selected = []
        for fold in folds:
            withheld = set(fold)
            calibration = [row for i, row in enumerate(rows) if i not in withheld]
            policies = policies_for(calibration, budget)
            selected.append(dict(calibration_size=len(calibration), test_size=len(fold), policies=policies))
            for i in fold:
                order = i % len(METHODS)
                results[i] = {
                    state: {
                        method: measure(endpoint, questions[i], rows[i], method, state, policies[method])
                        for method in METHODS[order:] + METHODS[:order]
                    } for state in STATES[::1 if i % 2 == 0 else -1]
                }
        key = f"{budget:.2f}"
        report[key] = dict(policies=selected, results=summarize(results, groups))
        raw[key] = results
        print(key, {m: (a["present"]["grade_counts"], a["withheld"]["grade_counts"])
                    for m, a in report[key]["results"].items()}, flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), reference_sha256=digest(args.reference),
        source_hashes=hashes, embedding=provenance, reranker=reader.provenance,
        groups=len(set(groups)), queries=len(rows), report=report, raw=raw,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "reference", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
