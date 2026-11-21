"""Prepare separate blinded scope and evidence reviews for layout disagreements."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval import build_exact_enterprise_review as evidence
from experiments.retrieval.check_joint_review import FIELDS, check
from experiments.retrieval.enterprise_project_role_rescue import profile_subset
from experiments.retrieval.enterprise_qualification_scope_review import SALT, build_packet, validate
from experiments.retrieval.enterprise_requirement_roles import ROLES, spans
from experiments.retrieval.enterprise_routed_review import digest, rows, write


def scope_cards(cases, queries):
    cards, mapping = [], {}
    for case in cases:
        qid = case["id"]
        text = queries[qid]["text"]
        coordinates = spans(text)
        if case["clauses"] != coordinates:
            raise ValueError("Clause offsets do not match the original request")
        labels = case["labels"]
        for i, anchor in enumerate(coordinates):
            predictions = {arm: ROLES[sequence[i]] for arm, sequence in labels.items()}
            if len(set(predictions.values())) == 1:
                continue
            start, end = anchor["start"], anchor["end"]
            cards.append(dict(id=qid, start=start, end=end, full_request=text, target=text[start:end],
                              chosen=predictions["whole_block"] == "prior_qualification"))
            case_id = sha256(f"{SALT}:{qid}:{start}:{end}".encode()).hexdigest()[:24]
            mapping[case_id] = predictions
    return cards, mapping


def evidence_methods(cases, corpus):
    def recommendations(outcome):
        return {(r["expert"], tuple(sorted(map(tuple, r["citations"]))))
                for r in outcome["initial"] + outcome["extra"]}

    def review_views(records):
        return [(owner, evidence.sources(corpus, owner, [
            dict(source_id=key, start=start, end=end) for key, start, end in references]))
                for owner, references in sorted(records)]

    arms = ("whole_block", "varlen", "four_rows", "agreement")
    methods = {arm: {} for arm in arms}
    for case in cases:
        variants = case["variants"]
        baseline = recommendations(next(iter(variants.values()))["interventions"]["whole"])
        records = {arm: recommendations(variants[arm]["interventions"]["variant"])
                   if arm in variants else baseline for arm in arms}
        shared = set.intersection(*records.values())
        for arm in arms:
            methods[arm][case["id"]] = review_views(records[arm] - shared)
    return methods


def export(queries, dataset, aggregate, trace, output):
    report = json.loads(aggregate.read_bytes())
    original = json.loads(trace.read_bytes())
    if digest(trace) != report["private_sha256"] or digest(queries) != original["queries_sha256"]:
        raise ValueError("The measured trace or source requests changed")
    if original["output_signature"] != report["output_signature"]:
        raise ValueError("The private trace does not match the qualified replay")
    q = evidence.index(rows(queries), lambda item: item["id"])
    cases = original["cases"]
    cards, mapping = scope_cards(cases, q)
    if len(cases) != report["audit"]["requests_with_any_layout_change"] or len(cards) != len(mapping):
        raise ValueError("The disagreement inventory changed")
    output.mkdir(mode=0o700)
    scope = output / "scope"
    build_packet(cards, [None]*len(cards), len(cards), digest(aggregate), scope)
    write(scope/"private-layout-labels.json", mapping)
    validate(scope, scope/"reviewer-a.jsonl", scope/"reviewer-b.jsonl")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    owners, corpus = (profile_subset(dataset, key, manifest) for key in ("experts", "corpus"))
    methods = evidence_methods(cases, corpus)
    selected = {item["id"]: q[item["id"]] for item in cases}
    forms, mapping, predictions, _ = evidence.packet_rows(selected, owners, methods)
    proof = output / "evidence"
    proof.mkdir(mode=0o700)
    for name in ("blind-review.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write(proof/name, forms, lines=True)
    write(proof/"private-mapping.jsonl", mapping, lines=True)
    written = ["blind-review.jsonl", "private-mapping.jsonl"]
    for arm, matches in predictions.items():
        name = f"{arm}-predictions.jsonl"
        write(proof/name, matches, lines=True)
        written.append(name)
    write(proof/"manifest.json", dict(candidate_views=len(forms), queries=len(selected),
          source_trace_sha256=digest(trace), outputs_sha256={key: digest(proof/key) for key in written}))
    checked = check(proof, proof/"reviewer-a.jsonl")
    if any(checked["counts"][field] != ({"missing": len(forms)} if forms else {}) for field in FIELDS):
        raise ValueError("Unexpected labels in the new evidence packet")
    write(output/"summary.json", dict(disputed_requests=len(cases), scope_cards=len(cards),
        changed_recommendation_views=len(forms), human_judgments=0, aggregate_sha256=digest(aggregate)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "dataset", "aggregate", "trace", "output"):
        cli.add_argument(name, type=Path)
    export(**vars(cli.parse_args()))
