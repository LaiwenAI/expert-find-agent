"""Test role-definition comparison before joining paper-scoped contributors."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_cross_source_search import phrase_bank, required_roles
from experiments.credit.engineering_joint_calibration import SUPPORT, judge, open_calibration
from experiments.credit.engineering_role_banks import encode, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


TASK = ("Determine whether the research activity in the query is an instance of the single "
        "CRediT contributor role described in the document.")
COMPARISONS = ("ridge", "prototype", "shortlist", "full")


def scores_for(phrases, definitions, encoder, banks, reader):
    start = perf_counter()
    matrix = encode(encoder, phrases)
    dense = {name: matrix @ banks[name].T for name in ("ridge", "prototype")}
    dense_ms = (perf_counter() - start) * 1000
    rows = dict(dense)
    costs = dict.fromkeys(dense, (dense_ms, 0))
    for arm in ("shortlist", "full"):
        start = perf_counter()
        if arm == "shortlist":
            top = [np.argsort(-values, axis=1, kind="stable")[:, :2] for values in dense.values()]
            pools = [sorted(set(a) | set(b)) for a, b in zip(*top, strict=True)]
        else:
            pools = [list(range(len(ROLES))) for _ in phrases]
        pairs = [(phrase, definitions[i]) for phrase, pool in zip(phrases, pools, strict=True) for i in pool]
        margins, truncated = reader.score_pairs(pairs, task=TASK)
        if truncated:
            raise ValueError("Role definition comparison unexpectedly truncated")
        scores = np.full((len(phrases), len(ROLES)), -np.inf)
        cursor = 0
        for row, pool in zip(scores, pools, strict=True):
            row[pool] = margins[cursor:cursor + len(pool)]
            cursor += len(pool)
        rows[arm] = scores
        costs[arm] = ((perf_counter() - start) * 1000 + (dense_ms if arm == "shortlist" else 0), len(pairs))
    return rows, costs


def selected(matrix):
    first, second = np.argsort(-matrix, axis=1, kind="stable")[:, :2].T
    gap = matrix[np.arange(len(matrix)), first] - matrix[np.arange(len(matrix)), second]
    return tuple(ROLES[i] for i in first), float(gap.min())


def offers(articles, papers, required, role_labels):
    left, right = [articles[paper]["roles"] for paper in papers]
    return {owner: (1., role_labels) for owner in left.keys() & right.keys()
            if required[0] <= set(left[owner]) and required[1] <= set(right[owner])}


def single_role_offers(articles, row, matrix):
    roles, confidence = selected(matrix)
    required = [{role} for role in roles]
    return offers(articles, row["papers"], required, roles), confidence


def calibrate(articles, cases, glossary_scores):
    fit = {}
    for arm in COMPARISONS:
        matrix = glossary_scores[arm]
        rows = [single_role_offers(articles, row, matrix[[SUPPORT[role] for role in row["roles"]]]) for row in cases]
        scored = [judge(row, returned, 0.) for row, (returned, _) in zip(cases, rows, strict=True)]
        best = None
        for cutoff in sorted({0., *(max(0., value) for _, value in rows)}):
            counts = Counter()
            for value, (_, margin) in zip(scored, rows, strict=True):
                if margin > cutoff:
                    counts.update(value)
            if counts["adverse"] <= len(cases) // 20:
                key = counts["exact_proof"], -counts["adverse"], cutoff
                if best is None or key > best[0]:
                    best = key, dict(cutoff=cutoff, counts={field: counts[field] for field in scored[0]})
        fit[arm] = best[1]
    return fit


def evaluate(articles, cases, phrases, definitions, encoder, banks, reader, bands, fit, reverse):
    counts, times, workloads = {}, {}, {}
    cases = sorted(cases, key=lambda row: (row["family"], row["variant"]), reverse=reverse)
    for i, row in enumerate([cases[0], *cases]):
        prompts = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        scores, costs = scores_for(prompts, definitions, encoder, banks, reader)
        for arm in (*COMPARISONS, "ridge_band", "prototype_band"):
            start = perf_counter()
            base = arm.removesuffix("_band")
            if arm.endswith("_band"):
                required = required_roles(row, scores, arm, bands)
                roles = tuple(row["roles"]) if all(r in v for r, v in zip(row["roles"], required, strict=True)) else ("", "")
                chosen = offers(articles, row["papers"], required, roles)
            else:
                chosen, confidence = single_role_offers(articles, row, scores[arm])
                if confidence <= fit[arm]["cutoff"]:
                    chosen = {}
                required = [{role} for role in selected(scores[arm])[0]]
            report = judge(row, chosen, 0.)
            json.dumps(report)
            elapsed = costs[base][0] + (perf_counter() - start) * 1000
            if i:
                counts.setdefault(arm, Counter()).update(
                    report, swap_issued=row["variant"] == "swap" and report["issued"],
                    correct_role_slots=sum(role in kept for role, kept in zip(row["roles"], required, strict=True)),
                )
                times.setdefault(arm, []).append(elapsed)
                workloads.setdefault(arm, []).append(costs[base][1])
    return {arm: dict(totals=counts[arm], p95_ms=float(np.quantile(times[arm], .95)),
                      mean_comparisons=float(np.mean(workloads[arm]))) for arm in counts}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, _, calibration = open_calibration(args.plos)
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    reader = Reranker(args.reranker)
    glossary_scores, _ = scores_for(VIEWS["original"], definitions, encoder, banks, reader)
    fit = calibrate(articles, calibration, glossary_scores)
    panel = json.loads(args.panel.read_bytes())
    bands = json.loads(args.published.read_bytes())["runs"][0]["bands"]
    result = evaluate(panel["articles"], panel["queries"], phrase_bank(wording), definitions,
                      encoder, banks, reader, bands, fit, args.reverse)
    write_json(args.output, dict(
        panel_sha256=digest(args.panel), calibration_sha256=digest(args.plos),
        script_sha256=digest(Path(__file__)), embedding_revision=checkpoint["revision"],
        reranker=reader.provenance, gpu=torch.cuda.get_device_name(),
        calibration_cases=len(calibration), target_cases=len(panel["queries"]),
        reverse=args.reverse, fit=fit, results=result,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "plos", "niso", "training", "published", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
