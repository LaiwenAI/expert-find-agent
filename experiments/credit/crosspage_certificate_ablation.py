"""Separate candidate order from the reader's permission to issue evidence."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.calibrated_topic_evidence import margins
from experiments.credit.crosspage_competing_search import complete_at_partial_caps, postings_index, select
from experiments.credit.crosspage_condition_search import TASK, page_index, score_held
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ROUTES = (
    ("whole", 0), ("lexical", 0), ("hybrid", 0),
    ("whole", 1), ("whole", 2), ("whole", 4),
    ("hybrid", 1), ("hybrid", 2), ("hybrid", 4),
)
FIXED_CUTOFFS = (0., -4.)


def observe(query, route, depth, index, postings, encoder, reader):
    begin = perf_counter()
    checked = clipped = 0
    candidates = {}
    texts, vectors, physical, masks = index
    if route == "whole":
        order, prior, _, _, _ = score_held(query, *index, encoder, reader, "whole_dense")
    else:
        order, prior, chosen, _, _ = select(
            query["conditions"], sorted(query["grades"]), texts, vectors, masks, postings,
            encoder, reader, route, 0,
        )
    if depth:
        if route == "whole":
            _, _, chosen, _, _ = select(
                query["conditions"], order[:depth], texts, vectors, masks, postings,
                encoder, reader, "hybrid", 0,
            )
        prompts = [f"The report substantively covers {condition}." for condition in query["conditions"]]
        pairs = [(prompt, texts[doc][page]) for doc in order[:depth]
                 for prompt, page in zip(prompts, chosen[doc], strict=True)]
        values, clipped = margins(reader, pairs, TASK)
        arity = len(prompts)
        for i, doc in enumerate(order[:depth]):
            group = values[i * arity:(i + 1) * arity]
            candidates[doc] = dict(margin=float(min(group)), total=float(sum(group)),
                                   pages=[physical[doc][page] for page in chosen[doc]])
        checked = len(pairs)
    json.dumps({"top": order[0], "checked": candidates})
    elapsed = (perf_counter() - begin) * 1000
    return dict(
        query=query["query_id"], order=order, prior=float(prior), checked=candidates,
        reads=checked, clipped=clipped, ms=elapsed,
    )


def candidate_list(row, policy):
    """Encode the policy's precedence as a list of document/margin pairs."""
    if policy == "retrieve":
        return [(row["order"][0], row["prior"])]
    checked = row["checked"]
    order = [doc for doc in row["order"] if doc in checked]
    if policy == "rerank":
        order = [min(order, key=lambda doc: (-checked[doc]["margin"], -checked[doc]["total"], doc))]
    elif policy == "top_only":
        order = order[:1]
    if policy == "first_passing":
        # A later document can never be emitted if an earlier one attains
        # at least its margin. Keep only the strictly increasing frontier.
        frontier = []
        best = -float("inf")
        for doc in order:
            if checked[doc]["margin"] > best:
                best = checked[doc]["margin"]
                frontier.append(doc)
        order = frontier
    return [(doc, checked[doc]["margin"]) for doc in order]


def audit_policy(rows, targets, policy):
    choices = [candidate_list(row, policy) for row in rows]
    gold = [targets[row["query"]] for row in rows]
    provisional = [dict(rank=1 if options[0][0] == doc else 2, margin=options[0][1])
                   for options, doc in zip(choices, gold, strict=True)]

    def at(cutoff):
        picked = [next((doc for doc, margin in options if margin > cutoff), None) for options in choices]
        return dict(
            complete=sum(pick == doc for pick, doc in zip(picked, gold, strict=True)),
            partial=sum(pick is not None and pick != doc for pick, doc in zip(picked, gold, strict=True)),
        )

    if policy == "first_passing":
        # Changing the threshold can change the selected document, so a
        # confidence sort of one frozen choice would give the wrong frontier.
        thresholds = {-float("inf")} | {margin for row in choices for _, margin in row}
        curve = [at(value) for value in thresholds]
        caps = {str(cap): max(item["complete"] for item in curve if item["partial"] <= cap)
                for cap in (5, 10)}
    else:
        caps = complete_at_partial_caps(provisional)
    report = dict(
        provisional_complete_first=sum(item["rank"] == 1 for item in provisional),
        best_at_observed_partial_cap=caps,
    )
    if policy != "retrieve":
        report["fixed_cutoff"] = {str(value): at(value) for value in FIXED_CUTOFFS}
    return report


def evaluate(queries, index, postings, encoder, reader, offset):
    records = defaultdict(list)
    for i, query in enumerate([queries[0], *queries]):
        shift = (i + offset) % len(ROUTES)
        for route, depth in ROUTES[shift:] + ROUTES[:shift]:
            row = observe(query, route, depth, index, postings, encoder, reader)
            if i:
                records[(route, depth)].append(row)
    targets = {row["query_id"]: next(doc for doc, grade in row["grades"].items() if grade == row["arity"])
               for row in queries}
    results = {}
    for (route, depth), rows in records.items():
        policies = ("retrieve",) if depth == 0 else ("rerank", "top_only", "first_passing")
        results[f"{route}_{depth}"] = dict(
            p50_p95_ms=np.quantile([r["ms"] for r in rows], [.5, .95]).tolist(),
            reads=sum(r["reads"] for r in rows), clipped=sum(r["clipped"] for r in rows),
            outcomes={policy: audit_policy(rows, targets, policy) for policy in policies},
            evidence=[{k: v for k, v in row.items() if k != "ms"} for row in rows],
        )
    return results


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
    revision = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    index = page_index(panel["documents"], encoder)
    postings = postings_index(index[0])
    result = evaluate(panel["queries"], index, postings, encoder, reader, args.offset)
    write_json(args.output, dict(
        queries=len(panel["queries"]), documents=len(index[0]), gpu=torch.cuda.get_device_name(),
        embedding_revision=revision["revision"], reranker_revision=reader.provenance["revision"],
        panel_sha256=manifest["panel_sha256"], script_sha256=digest(Path(__file__)),
        offset=args.offset, results=result,
    ))


if __name__ == "__main__":
    main()
