"""Replay separate document-read and expert-verification budgets on TREC."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean

from expert_search.statistics import paired_interval
from experiments.trec_enterprise.expert_evidence_budget import FILES, POOL_SIZE, TOP_K, load, metrics, prior


BUDGETS = ((5, 10), (10, 20), (20, 40))
METHODS = ("round_robin", "cached_first", "shared_source")
MATCHED = "shared_matched_checks"
NOISE = ("perfect", "independent", "source_correlated")
TRIALS = 12
SEED = 20261119


def make_cases(experts, documents):
    for topic in sorted(experts, key=int):
        labels = experts[topic]
        candidate_prior = prior(experts, topic)
        candidates = sorted(labels, key=lambda e: (-candidate_prior(e), e))[:POOL_SIZE]
        source_truth = {e: documents[topic].get(e, {}) for e in candidates}
        owners = defaultdict(list)
        for expert in candidates:
            for document in source_truth[expert]:
                owners[document].append(expert)
        for trial in range(TRIALS):
            rng = random.Random(f"{SEED}:{topic}:{trial}")
            orders = {}
            for expert in candidates:
                ids = sorted(source_truth[expert])
                rng.shuffle(ids)
                orders[expert] = ids
            pair_draws = {e: {doc: rng.random() for doc in orders[e]} for e in candidates}
            source_draws = {doc: rng.random() for doc in sorted(owners)}
            yield topic, labels, candidates, source_truth, owners, orders, pair_draws, source_draws


def replay(case, method, noise, document_cap, verification_cap):
    _, labels, candidates, truth, owners, orders, pair_draws, source_draws = case
    rank = {expert: i for i, expert in enumerate(candidates)}
    observed = defaultdict(set)
    accepted = set()
    valid = set()
    read = set()
    checks = Counter()
    for _ in range(verification_cap):
        unresolved = [e for e in candidates if e not in accepted and len(observed[e]) < len(orders[e])]
        if not unresolved:
            break
        read_full = len(read) == document_cap
        proposed = {}
        for expert in unresolved:
            available = [doc for doc in orders[expert] if doc not in observed[expert]
                         and (not read_full or doc in read)]
            if not available:
                continue
            if method != "round_robin":
                available.sort(key=lambda doc: doc not in read)
            proposed[expert] = available
        if not proposed:
            break
        if method == "round_robin":
            chosen = min(proposed, key=lambda e: (checks[e], rank[e]))
            document = proposed[chosen][0]
        elif method == "cached_first" or read_full or any(docs[0] in read for docs in proposed.values()):
            chosen = min(proposed, key=lambda e: (proposed[e][0] not in read, checks[e], rank[e]))
            document = proposed[chosen][0]
        else:
            # The source mapping is available, but the per-expert judgment remains hidden.
            eligible = set(proposed)
            coverage = {
                doc: sum(1 / (1 + checks[e]) for e in owners[doc]
                         if e in eligible and doc not in observed[e])
                for docs in proposed.values() for doc in docs
            }
            _, chosen, document = min(
                ((-coverage[doc], checks[e], rank[e], di), e, doc)
                for e, docs in proposed.items() for di, doc in enumerate(docs))

        observed[chosen].add(document)
        read.add(document)
        checks[chosen] += 1
        gold = truth[chosen][document] == 2
        if noise == "perfect":
            passed = gold
        else:
            draw = pair_draws[chosen][document] if noise == "independent" else source_draws[document]
            passed = draw < (.90 if gold else .01)
        if passed:
            accepted.add(chosen)
            if gold:
                valid.add(chosen)
    paid_checks = checks.total()
    if paid_checks > verification_cap or len(read) > document_cap:
        raise ValueError("Replay exceeded a resource cap")
    return {
        **metrics(candidates, labels, accepted, valid, paid_checks),
        "unique_documents": len(read),
        "cache_hits": paid_checks - len(read),
    }


def evaluate(directory):
    experts, documents = load(directory)
    observations = defaultdict(list)
    for case in make_cases(experts, documents):
        topic = case[0]
        for read_cap, verify_cap in BUDGETS:
            for noise in NOISE:
                outcomes = {}
                for method in METHODS:
                    outcomes[method] = replay(case, method, noise, read_cap, verify_cap)
                outcomes[MATCHED] = replay(
                    case, "shared_source", noise, read_cap, outcomes["cached_first"]["checked"])
                for method, result in outcomes.items():
                    observations[topic, read_cap, verify_cap, noise, method].append(result)
    rows = []
    for (topic, reads, verifies, noise, arm), trials in observations.items():
        rows.append(dict(topic=topic, read_cap=reads, verify_cap=verifies, noise=noise, arm=arm,
                         **{name: mean(row[name] for row in trials) for name in trials[0]}))
    reported = {}
    for reads, verifies in BUDGETS:
        budget = f"{reads}-reads-{verifies}-checks"
        reported[budget] = {}
        for noise in NOISE:
            arms = {method: sorted((row for row in rows if row["read_cap"] == reads and
                                    row["noise"] == noise and row["arm"] == method),
                                   key=lambda r: int(r["topic"])) for method in (*METHODS, MATCHED)}
            if any([r["topic"] for r in outcomes] != sorted(experts, key=int) for outcomes in arms.values()):
                raise ValueError("Unpaired evaluation topics")
            fields = ("expert_p5", "supported_p5", "unsupported_promotion_p5",
                      "unique_documents", "checked", "cache_hits")
            means = {name: {field: mean(row[field] for row in outcomes) for field in fields}
                     for name, outcomes in arms.items()}
            cached = means["cached_first"]
            shared = means["shared_source"]
            saved_reads = cached["unique_documents"] - shared["unique_documents"]
            extra_checks = shared["checked"] - cached["checked"]
            reported[budget][noise] = {
                "arms": means,
                "read_to_check_cost_break_even": extra_checks / saved_reads if saved_reads > 0 else None,
                "paired_supported_p5_vs_cached": {
                    name: paired_interval([
                        a["supported_p5"] - b["supported_p5"]
                        for a, b in zip(arms[name], arms["cached_first"], strict=True)
                    ]) for name in ("shared_source", MATCHED)
                },
            }
    return rows, reported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, results = evaluate(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / "topics.json"
    data.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    summary = {
        "topics": len({row["topic"] for row in rows}),
        "trials_per_topic": TRIALS,
        "candidate_count": POOL_SIZE,
        "result_count": TOP_K,
        "seed": SEED,
        "qrels_sha256": FILES,
        "rows_sha256": sha256(data.read_bytes()).hexdigest(),
        "results": results,
        "scope": ("Fixed top ten by leave-topic-out prior on pooled candidates. All arms use an exact "
                  "document cache keyed by source ID and independent expert checks. Cache preference and "
                  "the source-sharing heuristic only observe permitted state. Source associations and "
                  "grades come from the judged pool. Orders are randomized, and simulated verifier errors "
                  "are not learned from a real model. Cost counts are not wall time or corpus inference."),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
