"""Compare in-pool and broader rescue with identical pair caps on TREC qrels."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean

from expert_search.statistics import paired_interval
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load, prior


TRIALS = 32
SCOPES = (5, 10, 20)
CAPS = (2, 4)
VERIFIERS = {"exact": (1, 0, False), "independent": (.9, .05, False),
             "shared_document": (.9, .05, True)}
FIELDS = ("issued", "grounded", "adverse", "outside_grounded", "checked", "documents")


def source_prior(documents, held_out):
    seen, supports = Counter(), Counter()
    for topic, owners in documents.items():
        if topic == held_out:
            continue
        for items in owners.values():
            seen.update(items.keys())
            supports.update(doc for doc, grade in items.items() if grade == 2)
    total = sum(seen.values())
    prevalence = sum(supports.values()) / total
    return lambda doc: (supports[doc] + 5 * prevalence) / (seen[doc] + 5)


def measure(selected, ranks, labels, sources, accept, original):
    witnesses = {}
    for owner, doc in selected:
        if accept(owner, doc):
            witnesses.setdefault(owner, doc)
    issued = sorted(witnesses, key=ranks.__getitem__)[:TOP_K]
    grounded = [owner for owner in issued
                if labels[owner] == 2 and sources[owner][witnesses[owner]] == 2]
    return dict(
        issued=len(issued), grounded=len(grounded), adverse=len(issued) - len(grounded),
        outside_grounded=sum(ranks[owner] >= original for owner in grounded),
        checked=len(selected), documents=len({doc for _, doc in selected}),
    )


def create_case(experts, documents, topic):
    labels = experts[topic]
    source_map = documents[topic]
    ranker = prior(experts, topic)
    candidates = sorted(labels, key=lambda owner: (-ranker(owner), owner))[:max(SCOPES)]
    ranks = {owner: pos for pos, owner in enumerate(candidates)}
    source_score = source_prior(documents, topic)
    ordered = {
        owner: sorted(source_map.get(owner, ()), key=lambda doc: (-source_score(doc), doc))
        for owner in candidates
    }
    return labels, source_map, ranks, ordered, source_score


def choose_pairs(ordered, ranks, source_score, trial_key, ranking_mode):
    rng = random.Random(trial_key)
    jitter = {(owner, doc): rng.random() for owner, docs in ordered.items() for doc in docs}
    if ranking_mode != "random":
        key = lambda item: (-source_score(item[1]), jitter[item], ranks[item[0]])
    else:
        key = jitter.__getitem__
    ordered_pairs = {
        owner: sorted(((owner, doc) for doc in docs), key=key)
        for owner, docs in ordered.items()
    }
    initial = [group[0] for owner, group in ordered_pairs.items() if ranks[owner] < TOP_K and group]
    checked = set(initial)
    return initial, sorted((item for group in ordered_pairs.values() for item in group if item not in checked), key=key)


def evaluate_topic(experts, documents, topic, ranking_mode):
    labels, source_map, ranks, ordered, source_score = create_case(experts, documents, topic)
    data = {name: [] for name in VERIFIERS}
    for trial in range(TRIALS):
        seed = f"trec-abstention-v1:{topic}:{trial}"
        initial, frontier = choose_pairs(ordered, ranks, source_score, seed, ranking_mode)
        rng = random.Random(f"{seed}:error")
        pair_errors = {(owner,doc):rng.random() for owner, docs in ordered.items() for doc in docs}
        document_errors = {doc:rng.random() for doc in sorted({doc for _,doc in pair_errors})}
        for name, (sensitivity, fpr, shared) in VERIFIERS.items():
            def accept(owner, doc):
                score = document_errors[doc] if shared else pair_errors[owner, doc]
                return score < (sensitivity if source_map[owner][doc] == 2 else fpr)

            first = measure(initial, ranks, labels, source_map, accept, TOP_K)
            abstains = not first["issued"]
            opportunities = {scope: False for scope in SCOPES}
            if abstains:
                for owner, doc in frontier:
                    if labels[owner] == 2 and source_map[owner][doc] == 2:
                        for scope in SCOPES:
                            opportunities[scope] |= ranks[owner] < scope
            result = dict(initial=first, abstains=abstains, comparisons={}, opportunities=opportunities)
            for cap in CAPS:
                for scope in SCOPES:
                    if abstains:
                        eligible = [(owner, doc) for owner, doc in frontier if ranks[owner] < scope]
                        if ranking_mode == "prior_distinct":
                            owned_depth = Counter()
                            by_depth = []
                            for owner, doc in eligible:
                                by_depth.append((owned_depth[owner], (owner, doc)))
                                owned_depth[owner] += 1
                            eligible = [pair for _, pair in sorted(by_depth, key=lambda row: row[0])]
                        selected = eligible[:cap]
                        rescue = measure(selected, ranks, labels, source_map, accept, TOP_K)
                    else:
                        rescue = dict.fromkeys(FIELDS, 0)
                    result["comparisons"][cap, scope] = rescue
            data[name].append(result)
    return data


def summarize(topics, verifier):
    trials = [topic[verifier] for topic in topics]
    all_trials = [row for trials_per_topic in trials for row in trials_per_topic]
    total = len(all_trials)
    abstained = sum(row["abstains"] for row in all_trials)
    report = {}
    for cap in CAPS:
        report[cap] = {}
        for scope in SCOPES:
            values = [row["comparisons"][cap, scope] for row in all_trials]
            references = [row["comparisons"][cap, TOP_K] for row in all_trials]
            matched = [(v, b) for v,b in zip(values, references, strict=True)
                       if v["checked"] == b["checked"] == cap]
            stats = {field: sum(row[field] for row in values) for field in FIELDS}
            contrasts = {
                field: paired_interval([
                    mean((row["comparisons"][cap, scope][field] - row["comparisons"][cap, TOP_K][field]) / TOP_K
                         for row in group) for group in trials
                ]) for field in ("grounded", "adverse")
            }
            report[cap][scope] = dict(
                **stats, rescued_requests=sum(bool(row["issued"]) for row in values),
                contrast_by_all_request_slots=contrasts,
                matched_cap_trials=len(matched),
                matched_grounded_totals=[sum(a["grounded"] for a,_ in matched), sum(b["grounded"] for _,b in matched)],
                mean_checks_per_abstention=stats["checked"]/abstained,
            )
    initial = {field: sum(row["initial"][field] for row in all_trials) for field in FIELDS}
    return dict(topics=len(topics), trials=total, initial=initial, abstentions=abstained, per_cap=report,
                rescue_opportunity={scope:sum(row["opportunities"][scope] for row in all_trials)
                                    for scope in SCOPES},
                additional_scope_opportunity={scope:sum(row["opportunities"][scope] and
                    not row["opportunities"][TOP_K] for row in all_trials) for scope in SCOPES})


def run(qrels, output):
    experts, docs = load(qrels)
    report = {}
    for ranking in ("prior", "prior_distinct", "random"):
        results = [evaluate_topic(experts, docs, topic, ranking) for topic in sorted(experts, key=int)]
        report[ranking] = {verifier: summarize(results, verifier) for verifier in VERIFIERS}
    with output.open("x") as stream:
        json.dump(dict(qrels=FILES, code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                       trials_per_topic=TRIALS, scopes=SCOPES, caps=CAPS,
                       verifiers=VERIFIERS, report=report), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qrels", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
