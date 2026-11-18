"""Calibrate a dense-gap router and measure the selected verification depth live."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.litsearch.dense_retrieval import target_groups
from experiments.litsearch.learned_confirmation import observation
from experiments.litsearch.source_withholding import QUERY_SHA256, REVISION, TASK, split
from experiments.litsearch.top_hit_confirmation import DenseConfirmation


BUDGETS = ("0.02", "0.05")
STAGES = ("single", "two")
STATES = ("present", "withheld")
QUANTILES = (.25, .50, .75)


def routing_signal(kind, key, dense_gap):
    return dense_gap if kind == "gap" else int(key[:13], 16) / (16**13)


def profiles(sample, stage, cap):
    if not sample:
        return {(0, 0): (0, None)}
    values = [row["views"][state][stage]["scored"] for row, state in sample]
    scores = np.asarray([v[1] for v in values])
    margins = np.asarray([v[2] for v in values])
    visible = np.asarray([state == "present" for _, state in sample])
    known = np.asarray([v[0] in row["targets"] for v, (row, _) in zip(values, sample, strict=True)])
    scored = np.unique(np.quantile(scores, np.linspace(0, 1, 31)))
    gapped = np.unique(np.r_[0., np.quantile(margins, np.linspace(0, 1, 16))])
    masks = (scores[None, None, :] >= scored[:, None, None]) & (
        margins[None, None, :] >= gapped[None, :, None]
    )
    counts = (masks[..., ~visible].sum(-1), masks[..., visible & ~known].sum(-1),
              masks[..., visible & known].sum(-1))
    result = {(0, 0): (0, None)}
    for i, j in zip(*np.where((counts[0] <= cap) & (counts[1] <= cap) & (counts[2] > 0)), strict=True):
        errors = (int(counts[0][i, j]), int(counts[1][i, j]))
        hits = int(counts[2][i, j])
        policy = dict(score=float(scored[i]), margin=float(gapped[j]))
        current_hits, current_policy = result.get(errors, (0, None))
        if hits > current_hits or (hits == current_hits and
                (policy["score"], policy["margin"]) > (current_policy["score"], current_policy["margin"])):
            result[errors] = hits, policy
    return result


def fit(rows, budget, kind):
    requests = [(r, view) for r in rows for view in STATES]
    signals = [routing_signal(kind, r["id"], r["views"][view]["dense"]["scored"][2])
               for r, view in requests]
    thresholds = np.unique(np.quantile(signals, QUANTILES))
    limit = int(len(rows) * float(budget))
    best = None
    for threshold in thresholds:
        routed = {method: [] for method in STAGES}
        for request, signal in zip(requests, signals, strict=True):
            routed["single" if signal >= threshold else "two"].append(request)
        one, two = (profiles(routed[method], method, limit) for method in STAGES)
        for (w1, u1), (h1, p1) in one.items():
            for (w2, u2), (h2, p2) in two.items():
                if w1 + w2 > limit or u1 + u2 > limit:
                    continue
                rank = h1 + h2, -(w1 + w2), -(u1 + u2), len(routed["single"])
                if best is None or rank > best[0]:
                    best = rank, dict(kind=kind, threshold=float(threshold), single=p1, two=p2)
    rank, selected = best
    return dict(**selected, calibration_known=rank[0], calibration_withheld=-rank[1],
                calibration_unjudged=-rank[2], calibration_single_calls=rank[3])


class RoutedConfirmation(DenseConfirmation):
    def execute(self, question, state, key, policy):
        start = perf_counter()
        pool, _, cosine = self.retrieve(question, state)
        gap = float(cosine[0] - cosine[1])
        if policy["kind"] == "fixed":
            method = policy["method"]
        else:
            signal = routing_signal(policy["kind"], key, gap)
            method = "single" if signal >= policy["threshold"] else "two"
        depth = 1 if method == "single" else 2
        scores, truncated = self.reranker.score(question["query"], self.docs, pool[:depth], task=TASK)
        result, _ = observation(pool[:depth], scores)
        if method == "single":
            result = result[0], result[1], gap
        accepted = result[0] if accepts(result, policy[method]) else None
        return dict(pool=pool, score=result, accepted=accepted, method=method,
                    truncated=truncated, ms=(perf_counter() - start) * 1000)


def summarize(rows, policies):
    groups = target_groups(rows)
    results = {}
    for name in policies:
        results[name] = {}
        for state in STATES:
            arms = [row["views"][state][name] for row in rows]
            counts = Counter(a["method"] for a in arms)
            hits = [int(a["accepted"] in row["targets"]) for row, a in zip(rows, arms, strict=True)]
            unknown = [int(a["accepted"] is not None) - hit for a, hit in zip(arms, hits, strict=True)]
            latency = np.array([a["ms"] for a in arms])
            baseline = [row["views"][state]["fixed_two"] for row in rows]
            base_hits = [int(a["accepted"] in row["targets"]) for row, a in zip(rows, baseline, strict=True)]
            base_unknown = [int(a["accepted"] is not None) - hit
                            for a, hit in zip(baseline, base_hits, strict=True)]
            results[name][state] = dict(
                known=sum(hits), unjudged=sum(unknown), requests=len(rows), single=counts["single"],
                two=counts["two"], documents=counts["single"] + 2 * counts["two"],
                mean_ms=float(latency.mean()), p50_ms=float(np.median(latency)),
                p95_ms=float(np.quantile(latency, .95)),
                known_delta=clustered_paired_interval(np.subtract(hits, base_hits), groups),
                unjudged_delta=clustered_paired_interval(np.subtract(unknown, base_unknown), groups),
            )
    return results


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    earlier = json.loads(args.reference.read_bytes())
    with gzip.open(args.projection, "rb") as stream:
        data = json.load(stream)
    if data["revision"] != REVISION or data["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch projection")
    provenance = verify_checkpoint(args.embedding)
    if (digest(args.projection) != earlier["projection_sha256"]
            or digest(args.cache) != earlier["cache_sha256"] or provenance != earlier["embedding"]):
        raise ValueError("Frozen retrieval inputs changed")
    _, test, _ = split(data["questions"])
    papers = data["papers"]
    with np.load(args.cache, allow_pickle=False) as cache:
        vectors = torch.from_numpy(cache["vectors"]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    if reader.provenance != earlier["reranker"]:
        raise ValueError("Frozen verification model changed")
    endpoint = RoutedConfirmation(papers, vectors, encoder, None, reader)

    reports, raw = {}, {}
    for budget in BUDGETS:
        fixed = earlier["budgets"][budget]["calibration"]
        policies = {
            **{f"fixed_{stage}": dict(kind="fixed", method=stage, **{stage: fixed[stage]["policy"]})
               for stage in STAGES},
            **{kind: fit(earlier["validation"], budget, kind) for kind in ("gap", "hash")},
        }
        names = tuple(policies)
        for policy in policies.values():
            endpoint.execute(test[0], "present", earlier["test"][0]["id"], policy)
        rows = []
        for i, (question, reference) in enumerate(zip(test, earlier["test"], strict=True)):
            key = sha256(question["query"].encode()).hexdigest()
            if key != reference["id"] or question["corpusids"] != reference["targets"]:
                raise ValueError("Test request alignment changed")
            views = {}
            j = i % len(names)
            for state in STATES[::1 if i % 2 == 0 else -1]:
                views[state] = {}
                for name in names[j:] + names[:j]:
                    arm = endpoint.execute(question, state, key, policies[name])
                    expected = reference["views"][state][arm["method"]]
                    if arm["pool"] != expected["pool"] or list(arm["score"]) != expected["scored"]:
                        raise ValueError(f"Stage changed for {name} {key} {state}")
                    views[state][name] = arm
            rows.append(dict(id=key, targets=reference["targets"], views=views))
        reports[budget] = dict(policies=policies, test=summarize(rows, policies))
        raw[budget] = rows
        print(budget, {k: (r["present"]["known"], r["withheld"]["unjudged"])
                       for k, r in reports[budget]["test"].items()}, flush=True)
    write_json(args.output, dict(
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)),
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        budgets=reports, raw=raw,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "cache", "embedding", "reranker", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
