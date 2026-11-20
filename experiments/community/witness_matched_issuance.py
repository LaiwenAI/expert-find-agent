"""Compare verifier instructions at fixed issued citation budgets."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.community.crossvalidated_witness import write
from experiments.community.evaluate_crossvalidated_witness import connected_groups
from experiments.community.held_window_routing import EXPLICIT, REQUEST_CHARS, independent, index, load, score, work
from experiments.community.matched_witness_rescue import FIRST, TASK, digest


def owner_rank(docs, scores, owners):
    best = {}
    for doc, margin in zip(docs, scores, strict=True):
        owner = owners[doc]
        if owner not in best or margin > best[owner][1]:
            best[owner] = (doc, margin)
    return sorted(best.values(), key=lambda row: (-row[1], row[0]))[:FIRST]


def cross_site_overlap(prior, other):
    fingerprints = set()
    titles = []
    for data in prior:
        for query in data["queries"]:
            titles.append(set(tokens(query["title"])))
            for aid in query["answers"]:
                record = data["documents"][str(aid)]
                fingerprints.add(tuple(tokens(record["answer"])))
    retained = []
    for query in other["queries"]:
        title = set(tokens(query["title"]))
        sources = (other["documents"][str(aid)] for aid in query["answers"])
        if any(tuple(tokens(row["answer"])) in fingerprints for row in sources):
            continue
        if any(title and terms and len(title & terms) / len(title | terms) >= .80 for terms in titles):
            continue
        retained.append(query)
    if not retained:
        raise ValueError("No independent cross-site queries")
    return dict(other, queries=retained), len(other["queries"]) - len(retained)


def observe(data, corpus, encoder, reader):
    _, records, owners, _, _ = corpus
    outputs = defaultdict(list)
    timings = defaultdict(list)
    queries = data["queries"]
    for i, query in enumerate([queries[0], *queries]):
        text, first, added, gold = work(query, corpus, encoder)
        docs = first + added
        windows = [f"{records[doc]['question']}\n{records[doc]['answer'][:850]}" for doc in docs]
        prompts = (("original", TASK), ("explicit", EXPLICIT))
        offset = i % len(prompts)
        for name, task in prompts[offset:] + prompts[:offset]:
            start = perf_counter()
            margins, capped = score(reader, text[:REQUEST_CHARS], windows, len(first), task)
            if capped:
                raise ValueError("Short request or answer reached the packed token cap")
            ranked = owner_rank(docs, margins, owners)
            elapsed = (perf_counter() - start) * 1000
            if i:
                outputs[name].append((ranked, gold))
                timings[name].append(elapsed)
    return outputs, timings


def fixed_threshold(records, threshold):
    return [{doc for doc, margin in ranked if margin > threshold} for ranked, _ in records]


def flat_scores(records):
    return sorted(((-margin, index, doc) for index, (ranked, _) in enumerate(records)
                   for doc, margin in ranked))


def equal_issuance_cutoff(records, budget):
    ranked = flat_scores(records)
    if not 0 < budget <= len(ranked):
        raise ValueError("Invalid issuance budget")
    return float(np.nextafter(-ranked[-1][0], -np.inf) if budget == len(ranked) else -ranked[budget][0])


def exact_issuance(records, budget):
    selected = [set() for _ in records]
    for _, index, doc in flat_scores(records)[:budget]:
        selected[index].add(doc)
    return selected


def scorecard(records, selections):
    return dict(
        citations=sum(map(len, selections)),
        issued_queries=sum(bool(row) for row in selections),
        known_queries=sum(bool(keep & gold) for (_, gold), keep in zip(records, selections, strict=True)),
        known_citations=sum(len(keep & gold) for (_, gold), keep in zip(records, selections, strict=True)),
    )


def evaluate(data, observations, timing, dev_cutoff):
    original, explicit = observations["original"], observations["explicit"]
    reference = fixed_threshold(original, 0)
    budget = sum(map(len, reference))
    selections = dict(
        original=reference,
        explicit_zero=fixed_threshold(explicit, 0),
        explicit_frozen=fixed_threshold(explicit, dev_cutoff),
        explicit_exact_budget=exact_issuance(explicit, budget),
    )
    if sum(map(len, selections["explicit_exact_budget"])) != budget:
        raise ValueError("Exact budget mismatch")
    groups = connected_groups(data["queries"])
    report = {}
    for key, kept in selections.items():
        ranked = original if key == "original" else explicit
        jumps = [int(bool(variant & gold)) - int(bool(base & gold))
                 for (_, gold), variant, base in zip(ranked, kept, reference, strict=True)]
        report[key] = {
            **scorecard(ranked, kept), "gain_queries": jumps.count(1), "loss_queries": jumps.count(-1),
            "paired_delta": clustered_paired_interval(jumps, groups),
        }
    return dict(queries=len(original), arms=report, verifier_p95_ms={
        name: float(np.quantile(rows, .95)) for name, rows in timing.items()
    })


def run(args):
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or args.output.exists():
        raise ValueError("Require Slurm and a fresh result path")
    dev = load(args.dev, args.dev_manifest)
    stats, stats_excluded = independent(dev, load(args.stats, args.stats_manifest))
    math, math_excluded = cross_site_overlap((dev, stats), load(args.math, args.math_manifest))
    anchor = json.loads(args.anchor.read_bytes())
    if anchor["held_manifest"] != digest(args.stats_manifest) or len(stats["queries"]) != anchor["holdout_queries"]:
        raise ValueError("Changed Cross Validated fixture")
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    development, dev_clock = observe(dev, index(dev, encoder), encoder, reader)
    baseline_budget = sum(map(len, fixed_threshold(development["original"], 0)))
    cutoff = equal_issuance_cutoff(development["explicit"], baseline_budget)
    reports = {"development": evaluate(dev, development, dev_clock, cutoff)}
    for key, dataset in (("crossvalidated", stats), ("mathoverflow", math)):
        outputs, times = observe(dataset, index(dataset, encoder), encoder, reader)
        reports[key] = evaluate(dataset, outputs, times, cutoff)
    for name, past in (("original", "lead"), ("explicit_zero", "explicit_prompt")):
        arm = reports["crossvalidated"]["arms"][name]
        reference = anchor["reports"][past]["totals"]
        if (arm["citations"], arm["known_queries"], arm["known_citations"]) != (
                reference["citations"], reference["retained"], reference["known_citations"]):
            raise ValueError("Prior zero-threshold outcome did not reproduce")
    write(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], code_sha256=digest(Path(__file__)), reference_sha256=digest(args.anchor),
        manifests={name: digest(getattr(args, name)) for name in ("dev_manifest", "stats_manifest", "math_manifest")},
        revisions=dict(embedding=verify_checkpoint(args.embedding)["revision"], reranker=reader.provenance["revision"]),
        cutoff=cutoff, dev_citation_budget=baseline_budget, stats_excluded=stats_excluded,
        math_excluded=math_excluded, reports=reports,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev", "dev_manifest", "stats", "stats_manifest", "math", "math_manifest",
                 "anchor", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
