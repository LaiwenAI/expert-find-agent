"""Audit person-specific evidence and the effect of injecting an invalid citation."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from expert_search.statistics import paired_interval
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load, prior


def shared_documents(labels, documents):
    by_document = defaultdict(dict)
    for expert, sources in documents.items():
        for doc, grade in sources.items():
            by_document[doc][expert] = grade
    counts = Counter()
    borrowed = defaultdict(dict)
    for doc, judgments in by_document.items():
        counts["documents"] += 1
        counts["shared_documents"] += int(len(judgments) > 1)
        supported = {expert for expert, grade in judgments.items() if grade == 2}
        if not supported:
            continue
        counts["documents_with_support"] += 1
        counts["documents_positive_and_negative"] += int(0 in judgments.values())
        counts["documents_positive_and_neutral"] += int(1 in judgments.values())
        all_experts = {expert for expert in judgments if labels[expert] == 2}
        counts["mixed_positive_experts"] += int(
            bool(supported & all_experts) and bool(all_experts - supported))
        for expert, grade in judgments.items():
            counts["positive_document_judged_pairs"] += 1
            if grade < 2:
                counts[f"other_owner_grade_{grade}"] += 1
                counts[f"other_owner_grade_{grade}_known_expert"] += int(labels[expert] == 2)
                borrowed[expert][doc] = grade
    return counts, borrowed


def selected_ranking(experts, topic):
    score = prior(experts, topic)
    return sorted(experts[topic], key=lambda expert: (-score(expert), expert))[:TOP_K]


def evaluate_ranking(ranking, labels, documents, borrowed):
    variants = {name: {} for name in ("reference", "append", "swap", "matched_valid", "matched_wrong")}
    for expert in ranking:
        if labels[expert] != 2:
            continue
        grades = documents.get(expert, {})
        supporting_documents = sorted(doc for doc, grade in grades.items() if grade == 2)
        if not supporting_documents:
            continue
        support = supporting_documents[0]
        replacement = min(borrowed.get(expert, {}), key=lambda doc: (grades[doc], doc), default=None)
        variants["reference"][expert] = (support,)
        variants["append"][expert] = (support, replacement) if replacement is not None else (support,)
        variants["swap"][expert] = (replacement if replacement is not None else support,)
        if replacement is not None and len(supporting_documents) > 1:
            variants["matched_valid"][expert] = tuple(supporting_documents[:2])
            variants["matched_wrong"][expert] = (support, replacement)
    evidence = {}
    for variant, citations in variants.items():
        grades = [[documents[expert][doc] for doc in sources] for expert, sources in citations.items()]
        evidence[variant] = Counter(
            supported=sum(2 in g for g in grades),
            clean=sum(all(grade == 2 for grade in g) for g in grades),
            slots_with_negative=sum(0 in g for g in grades),
            slots_with_neutral=sum(1 in g for g in grades),
            citations=sum(map(len, grades)),
            supporting_citations=sum(g.count(2) for g in grades),
        )
    return sum(labels[e] == 2 for e in ranking), evidence


def run(directory, output):
    experts, documents = load(directory)
    audits = [shared_documents(experts[t], documents[t]) for t in sorted(experts, key=int)]
    rank_rows = []
    for topic, (_, borrowed) in zip(sorted(experts, key=int), audits, strict=True):
        top = selected_ranking(experts, topic)
        rank_rows.append(evaluate_ranking(top, experts[topic], documents[topic], borrowed))
    shared = Counter()
    for counter, _ in audits:
        shared.update(counter)
    ranked = defaultdict(Counter)
    for _, variants in rank_rows:
        for variant, counts in variants.items():
            ranked[variant].update(counts)
    n = len(audits)
    expert_count = sum(expert_hits for expert_hits, _ in rank_rows)
    report = {variant: {**counts, "citation_precision": (
        counts["supporting_citations"] / counts["citations"] if counts["citations"] else None)}
        for variant, counts in ranked.items()}
    for variant in ("reference", "append", "swap"):
        counts = report[variant]
        counts["supported_p5"] = counts["supported"] / (n * TOP_K)
        counts["clean_p5"] = counts["clean"] / (n * TOP_K)
    if report["matched_valid"]["citations"] != report["matched_wrong"]["citations"]:
        raise ValueError("Changed number of citations in matched comparison")
    topic_differences = [(variants["append"]["clean"] - variants["reference"]["clean"]) / TOP_K
                         for _, variants in rank_rows]
    answer = dict(
        input_sha256=FILES, script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        topics=n, topics_with_mixed_positive_experts=sum(bool(c["mixed_positive_experts"]) for c, _ in audits),
        topics_with_counterfactual=sum(v["append"]["citations"] > v["reference"]["citations"]
                                      for _, v in rank_rows),
        shared=dict(sorted(shared.items())), expert_hits=expert_count, expert_p5=expert_count / (n * TOP_K),
        variants=report, append_clean_delta=paired_interval(topic_differences),
    )
    with output.open("x") as handle:
        json.dump(answer, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
