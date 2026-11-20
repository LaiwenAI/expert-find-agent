"""Freeze the highest scoring deeper excerpt per policy and request for blind review."""

import argparse
import json
from pathlib import Path

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_project_role_rescue as project
from experiments.retrieval.enterprise_owned_gap_sources import ARMS
from experiments.retrieval.enterprise_routed_review import digest, write


def run(dataset, trace, aggregates, output):
    aggregate = json.loads(aggregates.read_bytes())
    private = json.loads(trace.read_bytes())
    if digest(trace) != aggregate["trace_sha256"] or len(private["requests"]) != aggregate["eligible_requests"]:
        raise ValueError("Mismatched private records and published report")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    names = ("experts", "corpus", "queries")
    experts, corpus, all_queries = (project.profile_subset(dataset, key, manifest) for key in names)
    methods = {arm: {} for arm in ("original_incomplete", *ARMS)}
    queries = {}
    for row in private["requests"]:
        prior = {}
        recovered = {arm: row["arms"][arm]["review_candidates"] for arm in ARMS}
        if not any(recovered.values()):
            continue
        key = row["id"]
        queries[key] = all_queries[key]
        for arm, matches in recovered.items():
            views = []
            for item in matches:
                owner = item["expert"]
                added = {tuple(cite) for cite in item["new_citations"]}
                original = [cite for cite in item["citations"] if tuple(cite) not in added]
                if not original or (owner in prior and prior[owner] != original):
                    raise ValueError("The preserved baseline evidence must remain identical")
                prior[owner] = original
                views.append((owner, exact_views(corpus, owner, item["citations"])))
            methods[arm][key] = views
        methods["original_incomplete"][key] = [(owner, exact_views(corpus, owner, evidence))
                                              for owner, evidence in sorted(prior.items())]
    if not queries:
        return dict(queries=0, candidate_views=0, human_judgments=0)
    forms, mapping, outputs, counts = review.packet_rows(queries, experts, methods)
    output.mkdir(mode=0o700, exist_ok=False)
    form = output/"blind-review.jsonl"
    origins = output/"private-mapping.jsonl"
    write(form, forms, lines=True)
    write(origins, mapping, lines=True)
    write(output/"private-method-outputs.json", outputs)
    receipt = dict(candidate_views=len(forms), queries=len(queries), **counts, human_judgments=0,
                   trace_sha256=digest(trace), outputs_sha256={file.name: digest(file) for file in (form, origins)})
    write(output/"manifest.json", receipt)
    for reviewer in ("a", "b"):
        copy = output/f"reviewer-{reviewer}.jsonl"
        write(copy, forms, lines=True)
        review.check(output, copy)
    return receipt


def exact_views(corpus, owner, citations):
    return review.sources(corpus, owner, [dict(source_id=key, start=a, end=b) for key, a, b in citations])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "trace", "aggregates", "output"):
        cli.add_argument(name, type=Path)
    result = run(**vars(cli.parse_args()))
    public_counts = {key: result[key] for key in ("queries", "candidate_views", "human_judgments")}
    print(json.dumps(public_counts))
