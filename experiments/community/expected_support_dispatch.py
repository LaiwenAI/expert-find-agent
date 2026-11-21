"""Compare static and state-dependent same-person evidence dispatch."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import transformers

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.community import joint_answer_witness as base
from experiments.credit.joint_byline_depth import pair_key
from experiments.credit.live_joint_source_request import verify_embedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker

NEW_ARMS = ("additive", "pair_static", "pair_adaptive")
REFERENCE = "staged"


def completion_probability(sources, weights, owners):
    """Probability that distinct sources satisfy both conditions for one person."""
    left, right = ([doc for doc in sources if weight.get(doc, 0) > 0] for weight in weights)
    result = 0.0
    for mask in range(1, 1 << len(left)):
        subset = [doc for i, doc in enumerate(left) if mask & (1 << i)]
        first = math.prod(weights[0][doc] if mask & (1 << i) else 1 - weights[0][doc]
                          for i, doc in enumerate(left))
        if first:
            possible = [doc for doc in right if any(doc != other and owners[doc] == owners[other]
                                                     for other in subset)]
            result += first * (1 - math.prod(1 - weights[1][doc] for doc in possible))
    return result


def best_seed(ranks, owners, fallback):
    right_by_owner = {}
    for doc in ranks[1]:
        right_by_owner.setdefault(owners[doc], []).append(doc)
    choices = ((left, right) for left in ranks[0] for right in right_by_owner.get(owners[left], [])
               if left != right)
    return min(choices, key=lambda pair: (ranks[0][pair[0]] * ranks[1][pair[1]], pair_key(pair, ranks)),
               default=tuple(fallback))


def next_source(seed, pool, weights, owners, ranks):
    return min((doc for doc in pool if doc not in seed), key=lambda doc: (
        -completion_probability((*seed, doc), weights, owners),
        min(rank.get(doc, base.DEPTH + 1) for rank in ranks), doc,
    ))


def make_plans(queries, documents, owners, requests, original, cache, reader):
    jobs = []
    seed_reads = set()
    for pair, ranks, anchors, _, _, _ in requests:
        pool = sorted(set().union(*ranks))
        weights = tuple({doc: 1 / rank for doc, rank in order.items()} for order in ranks)
        additive = sorted(pool, key=lambda doc: (-sum(weight.get(doc, 0) for weight in weights), doc))[:3]
        seed = best_seed(ranks, owners, anchors)
        seed_reads.update(product(pair, seed))
        jobs.append((pair, ranks, pool, weights, additive, seed))
    base.score_pairs(reader, queries, documents, seed_reads, cache)
    needed = set()
    plans = []
    for (pair, ranks, pool, weights, additive, seed), baseline in zip(jobs, original, strict=True):
        updated = tuple({**weight, **{doc: float(cache[q, doc] > 0) for doc in seed if doc in weight}}
                        for q, weight in zip(pair, weights, strict=True))
        proposal = {
            "additive": additive,
            "pair_static": [*seed, next_source(seed, pool, weights, owners, ranks)],
            "pair_adaptive": [*seed, next_source(seed, pool, updated, owners, ranks)],
        }
        for sources in proposal.values():
            if len(set(sources)) != 3:
                raise ValueError("Unmatched read budget")
            needed.update(product(pair, sources))
        plans.append(baseline | proposal)
    base.score_pairs(reader, queries, documents, needed, cache)
    return plans


def compare(queries, owners, requests, plans, cache, independent):
    outcomes = {arm: [] for arm in (*base.ARMS, *NEW_ARMS)}
    for (pair, ranks, *_), plan in zip(requests, plans, strict=True):
        target = [queries[q]["known"] for q in pair]

        def known(proof):
            return proof is not None and all(doc in truth for doc, truth in zip(proof, target, strict=True))

        for arm, sources in plan.items():
            proofs = base.paired_sources(ranks, sources, owners)
            shown, confidence = base.choose(pair, ranks, proofs, cache)
            outcomes[arm].append((pair, any(map(known, proofs)), shown is not None, known(shown), confidence))
    reference = outcomes[REFERENCE]
    summaries = {}
    for arm, rows in outcomes.items():
        natural = [row for row in rows if row[2]]
        matched = min(len(natural), sum(row[2] for row in reference))
        # Use the same confidence-based matching as the archived baseline.
        top = lambda candidates: sum(row[3] for row in sorted(
            (row for row in candidates if row[2]), key=lambda row: (-row[4], row[0])
        )[:matched])
        record = Counter(available=sum(row[1] for row in rows), issued=len(natural), known=sum(row[3] for row in rows))
        summaries[arm] = dict(
            record, reads=3 * len(rows), checks=6 * len(rows), disjoint_known=sum(
                row[3] for row in rows if row[0] in independent
            ), matched_quota=matched, matched_known=top(rows), matched_staged_known=top(reference),
        )
    signature = sha256(json.dumps(outcomes, sort_keys=True).encode()).hexdigest()
    return summaries, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stats2024", "stats2025", "mathoverflow", "embedding", "reranker", "baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output on a Slurm GPU")
    archive = json.loads(args.baseline.read_text())
    revision = verify_embedding(args.embedding)
    embed, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    if revision != archive["embedding_checkpoint"] or reader.provenance["revision"] != archive["reranker_checkpoint"]:
        raise ValueError("Changed checkpoints")
    panels = [("stats2024", args.stats2024, False), ("stats2025", args.stats2025, False),
              ("mathoverflow_original", args.mathoverflow, True)]
    results = {}
    with TemporaryDirectory(dir=args.output.parent) as temp:
        for name, folder, canonical in (reversed(panels) if args.reverse else panels):
            old = archive["panels"][name]
            raw, documents, snapshot = base.load_snapshot(folder)
            if snapshot != old["snapshot_sha256"]:
                raise ValueError("Changed snapshot")
            _, _, independent, _ = base.freeze_cases(raw, documents, canonical)

            def observe(queries, documents, owners, requests, plans, cache):
                plans = make_plans(queries, documents, owners, requests, plans, cache, reader)
                scores, signature = compare(queries, owners, requests, plans, cache, independent)
                results[name] = dict(scores=scores, signature=signature)

            reproduced = base.evaluate(folder, canonical, embed, reader, Path(temp) / name,
                                       int(args.reverse), observe=observe)
            for field in ("paired_requests", "pool_sources", "disjoint_question_pairs"):
                if old[field] != reproduced[field]:
                    raise ValueError(f"Changed panel structure: {field}")
            differences = {}
            for arm in base.ARMS:
                natural = reproduced["natural"][arm]
                evaluated = results[name]["scores"][arm]
                differences[arm] = {
                    field: natural[field] - old["natural"][arm][field]
                    for field in ("known_available", "issued", "known", "unresolved")
                }
                if [natural[key] for key in ("known_available", "issued", "known")] != [
                    evaluated[key] for key in ("available", "issued", "known")
                ]:
                    raise ValueError(f"{name} {arm} changed in the paired replay")
            results[name]["reference_deltas"] = differences
            print(name, results[name], flush=True)
    write_json(args.output, dict(
        panels=results, script_sha256=digest(Path(__file__)), baseline_sha256=digest(args.baseline),
        embedding=revision, reranker=reader.provenance["revision"], gpu=torch.cuda.get_device_name(),
        slurm=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__),
    ))


if __name__ == "__main__":
    main()
