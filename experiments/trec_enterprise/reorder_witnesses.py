"""Measure certified expert selection when the citation order is flexible."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.certified_source_dispatch import dispatch
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load
from experiments.trec_enterprise.shared_source_budget import NOISE, TRIALS, make_cases


def source_prior(documents, held_out):
    probes = Counter()
    hits = Counter()
    for topic, rows in documents.items():
        if topic == held_out:
            continue
        for sources in rows.values():
            probes.update(sources.keys())
            hits.update(doc for doc, label in sources.items() if label == 2)
    prevalence = hits.total() / probes.total()
    return {doc: (hits[doc] + 5 * prevalence) / (n + 5) - prevalence for doc, n in probes.items()}


def reorder(case, arm, prior):
    topic, labels, candidates, truth, owners, orders, pair_draws, source_draws = case
    if arm == "shared":
        key = lambda doc: -len(owners[doc])
    else:
        key = lambda doc: -prior.get(doc, 0.)
    reordered = {expert: sorted(docs, key=key) for expert, docs in orders.items()}
    return topic, labels, candidates, truth, owners, reordered, pair_draws, source_draws


def accepted(case, noise, expert, doc):
    _, _, _, truth, _, _, pair_draws, source_draws = case
    gold = truth[expert][doc] == 2
    if noise == "perfect":
        return gold
    draw = pair_draws[expert][doc] if noise == "independent" else source_draws[doc]
    return draw < (.90 if gold else .01)


def certificate(case, noise, count):
    _, _, candidates, _, _, orders, _, _ = case
    # For each expert above the result boundary, show count positives or
    # enough negatives that the remaining documents cannot reach count.
    supported = []
    checks = 0
    for expert in candidates:
        docs = orders[expert]
        if sum(accepted(case, noise, expert, doc) for doc in docs) >= count:
            supported.append(expert)
            checks += count
        else:
            checks += max(0, len(docs) - count + 1)
        if len(supported) == TOP_K:
            break
    promoted = set(supported)
    ids = (supported + [e for e in candidates if e not in promoted])[:TOP_K]
    return checks, [(e, e in promoted) for e in ids]


def measure(case, transformed, noise, count):
    floor, target = certificate(case, noise, count)
    _, labels, _, truth, _, _, _, _ = case
    reference, references = dispatch(case, noise, count, "prefix_two")
    ranking, work = dispatch(transformed, noise, count, "prefix_two")
    if work["checks"] < floor or references["checks"] < floor:
        raise AssertionError("Insufficient source checks to certify the result")
    if [(e, bool(docs)) for e, docs in ranking] != target:
        raise AssertionError("Reordering changed expert IDs, order or promotion status")
    for expert, docs in ranking:
        if docs and (len(set(docs)) != count or not all(accepted(case, noise, expert, d) for d in docs)):
            raise AssertionError("Citation verification failed")
    return work | dict(
        floor=floor,
        supported_p5=sum(labels[e] == 2 and len(docs) == count
                         and all(truth[e][doc] == 2 for doc in docs) for e, docs in ranking) / TOP_K,
        changed_citation_sets=sum(set(a) != set(b) for (_, a), (_, b) in zip(reference, ranking, strict=True)),
    )


def run(directory, output):
    if output.exists():
        raise FileExistsError(output)
    experts, documents = load(directory)
    priors = {topic: source_prior(documents, topic) for topic in experts}
    measurements = defaultdict(list)
    seen = Counter()
    for index, case in enumerate(make_cases(experts, documents)):
        topic = case[0]
        arms = {"fixed": case, **{arm: reorder(case, arm, priors[topic])
                                  for arm in ("shared", "source_prior")}}
        if index % TRIALS == 0:
            for docs in case[5].values():
                seen["associations"] += len(docs)
                seen["other_topic_source"] += sum(doc in priors[topic] for doc in docs)
        for noise in NOISE:
            for count in (1, 2):
                for arm, transformed in arms.items():
                    measurements[noise, count, arm].append((topic, measure(case, transformed, noise, count)))
    report = {}
    for (noise, count, arm), rows in measurements.items():
        control = measurements[noise, count, "fixed"]
        groups = [topic for topic, _ in rows]
        fields = tuple(rows[0][1])
        assert groups == [topic for topic, _ in control]
        key = f"{noise}/{count}/{arm}"
        report[key] = dict(cases=len(rows), means={field: mean(r[field] for _, r in rows) for field in fields})
        if arm != "fixed":
            report[key]["paired_checks_saved"] = clustered_paired_interval(
                [b["checks"] - a["checks"] for (_, b), (_, a) in zip(control, rows, strict=True)], groups
            ) | {"sampling_unit": "whole TREC topic; trial-weighted mean"}
    write_json(output, dict(inputs=FILES, script_sha256=digest(Path(__file__)),
                            topics=len(experts), trials_per_topic=TRIALS,
                            prior_coverage=seen, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
