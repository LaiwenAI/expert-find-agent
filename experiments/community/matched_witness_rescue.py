"""Compare who receives extra evidence checks under the same budget."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.community.crossvalidated_witness import write
from experiments.community.evaluate_crossvalidated_witness import connected_groups, group_witnesses, rank

FIRST = 5
ADDITIONAL = 3
ARMS = ("same_owners", "new_owners", "dense_order", "unresolved_first")
TASK = "Find an answer whose content directly addresses the user's question."


def checks(arm, order, authors, first, margins, budget):
    first_set = set(first)
    current = {authors[doc]: score for doc, score in zip(first, margins, strict=True)}
    selected, seen = [], set()
    for doc in order:
        if len(selected) == budget:
            break
        owner = authors[doc]
        if doc in first_set or owner in seen:
            continue
        in_pool = owner in current
        if (arm == "same_owners" and not in_pool) or (arm == "new_owners" and in_pool):
            continue
        if arm == "unresolved_first" and in_pool and current[owner] > 0:
            continue
        selected.append(doc)
        seen.add(owner)
    if len(selected) != budget:
        raise ValueError("Matched realized budget unavailable")
    return selected


def evaluate(checked, scores, owners, gold):
    # Keep only the strongest verified answer per person.
    best = {}
    for doc, score in zip(checked, scores, strict=True):
        owner = owners[doc]
        if score > 0 and (owner not in best or best[owner][0] < score):
            best[owner] = (score, doc)
    ranked = sorted(best.values(), key=lambda value: (-value[0], value[1]))[:FIRST]
    cited = {doc for _, doc in ranked}
    known_owners = {owners[doc] for doc in gold}
    proposed = bool(set(checked) & gold)
    accepted = any(doc in gold and score > 0 for doc, score in zip(checked, scores, strict=True))
    retained = bool(cited & gold)
    return dict(
        proposed=int(proposed),
        retained=int(retained),
        gold_rejected_by_reader=int(proposed and not accepted),
        reader_accepted_gold_not_cited=int(accepted and not retained),
        known_citations=len(cited & gold),
        owner_hit=int(any(owners[doc] in known_owners for doc in cited)),
        citations=len(cited), checked=len(checked),
    )


def shortlist(text, corpus, owners, encoder):
    vector = encoder.encode([text], query=True)[0]
    order = rank(corpus @ vector)
    grouped = group_witnesses(order, owners)
    first = [docs[0] for _, docs in grouped[:FIRST]]
    budget = min(ADDITIONAL, sum(len(docs) > 1 for _, docs in grouped[:FIRST]))
    return order, first, budget


def query_once(query, corpus, windows, owners, index_for_id, encoder, verifier, offset):
    text = f"{query['title']}\n{query['query']}"
    start = perf_counter()
    order, first, budget = shortlist(text, corpus, owners, encoder)
    verifier_query = text[:950]
    initial, clipped = verifier.score_pairs([(verifier_query, windows[doc]) for doc in first], task=TASK)
    common_ms = (perf_counter() - start) * 1000
    if not np.isfinite(initial).all():
        raise ValueError("Invalid first-pass scores")
    gold = {index_for_id[aid] for aid in query["answers"]}
    baseline = evaluate(first, initial, owners, gold)
    original_owners = {owners[doc] for doc in first}
    result = {}
    for name in ARMS[offset:] + ARMS[:offset]:
        start = perf_counter()
        selected = checks(name, order, owners, first, initial, budget)
        extra, cropped = (verifier.score_pairs([(verifier_query, windows[doc]) for doc in selected], task=TASK)
                          if selected else ([], 0))
        if not np.isfinite(extra).all():
            raise ValueError("Invalid extra scores")
        outcome = evaluate(first + selected, initial + extra, owners, gold)
        result[name] = dict(
            **outcome, extra_gold=int(bool(set(selected) & gold)),
            new_owners=sum(owners[doc] not in original_owners for doc in selected),
            capped=clipped + cropped,
        ), common_ms + (perf_counter() - start) * 1000
    return result, baseline, int(bool(set(first) & gold)), budget


def analyze(rows, timings, baseline, first_gold, budgets, groups):
    summary = {}
    for name, cases in rows.items():
        totals = {key: sum(row[key] for row in cases) for key in cases[0]}
        total_checks = sum(row["checked"] for row in cases)
        if total_checks != FIRST * len(budgets) + sum(budgets):
            raise ValueError("Unequal check counts")
        control = rows["same_owners"]
        summary[name] = dict(
            totals=totals,
            proposed_delta_vs_same=clustered_paired_interval(
                [a["proposed"] - b["proposed"] for a, b in zip(cases, control, strict=True)], groups),
            retained_delta_vs_same=clustered_paired_interval(
                [a["retained"] - b["retained"] for a, b in zip(cases, control, strict=True)], groups),
            after_rescue_new_known_witnesses=sum(bool(row["retained"] and not before["retained"])
                                                for row, before in zip(cases, baseline, strict=True)),
            lost_initial_known_witnesses=sum(bool(before["retained"] and not row["retained"])
                                             for row, before in zip(cases, baseline, strict=True)),
            recovered_from_missing_initial_proposal=sum(bool(row["retained"] and not first)
                                                        for row, first in zip(cases, first_gold, strict=True)),
            local_p95_ms=float(np.quantile(timings[name], .95)),
        )
    return summary


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run(args):
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or args.output.exists():
        raise ValueError("Require Slurm and a fresh result path")
    manifest = json.loads(args.manifest.read_bytes())
    old = json.loads(args.reference.read_bytes())
    if manifest["snapshot_sha256"] != digest(args.snapshot) or old["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Changed snapshot")
    data = json.loads(args.snapshot.read_bytes())
    records = data["documents"]
    ids = sorted(map(int, records))
    if ids != manifest["candidate_answers"]:
        raise ValueError("Changed source IDs")
    index_for_id = {aid: i for i, aid in enumerate(ids)}
    ordered = [records[str(aid)] for aid in ids]
    owners = [doc["owner"] for doc in ordered]
    documents = [f"{doc['question']}\n{doc['question_text']}\n{doc['answer']}" for doc in ordered]
    windows = [f"{doc['question']}\n{doc['answer'][:850]}" for doc in ordered]
    model_info = verify_checkpoint(args.embedding)
    if model_info["revision"] != old["model_revision"]:
        raise ValueError("Changed embedding model")
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    verifier = Reranker(args.reranker)
    start = perf_counter()
    corpus = encoder.encode(documents)
    index_seconds = perf_counter() - start
    rows, timings = defaultdict(list), defaultdict(list)
    baseline, first_gold, budgets = [], [], []
    queries = data["queries"]
    for i, query in enumerate([queries[0], *queries]):
        result, first, gold, budget = query_once(query, corpus, windows, owners, index_for_id,
                                                encoder, verifier, i % len(ARMS))
        if i:
            baseline.append(first)
            first_gold.append(gold)
            budgets.append(budget)
            for name, (outcome, ms) in result.items():
                rows[name].append(outcome)
                timings[name].append(ms)
    if sum(first_gold) != old["metrics"]["dense"]["arms"]["one"]["queries_known_witness"]:
        raise ValueError("Initial known witness retrieval did not reproduce")
    summary = analyze(rows, timings, baseline, first_gold, budgets, connected_groups(queries))
    write(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], queries=len(queries), extra_checks=sum(budgets),
        budget_histogram={budget: budgets.count(budget) for budget in set(budgets)},
        index_seconds=index_seconds, manifest_sha256=digest(args.manifest),
        anchor_sha256=digest(args.reference), code_sha256=digest(Path(__file__)),
        revisions=dict(embedding=model_info["revision"], reranker=verifier.provenance["revision"]),
        baseline={k: sum(row[k] for row in baseline) for k in baseline[0]},
        arms=summary,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "manifest", "reference", "embedding", "reranker", "output"):
        parser.add_argument(arg, type=Path)
    run(parser.parse_args())
