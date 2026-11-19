"""Build a private method-blind packet for changed enterprise evidence windows."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

from experiments.retrieval import build_exact_enterprise_review as base


POLICIES = ("reference", "criterion")
CONTROL_QUERIES = 4
INSTRUCTIONS = """# Review the evidence as displayed

Judge each case independently, using only the supplied request, candidate and
source excerpts. No outside search is needed. Keep case IDs, request text,
candidate details and source text unchanged.

For each of method_fit, constraint_support and personal_contribution, enter
supported, not_supported_by_view or uncertain. Method fit asks whether the
candidate's prior work addresses the task. Constraint support requires the
displayed evidence for each mandatory condition, including relevant units,
objects and operating conditions. Personal contribution asks what the person
actually did. Topical similarity, employment, authorship or an unassigned
team claim by themselves cannot establish a concrete contribution.

Use a short rationale with the displayed source label and cite the words that
support the judgment when available. Explain which evidence is missing or
ambiguous when applicable. Use your own reviewer identifier. Review the
assigned copy without seeing the other reviewer or the method mapping.
A judgment of not_supported_by_view concerns the displayed excerpts.
"""


def write(path, data, *, lines=False):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        if lines:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif isinstance(data, str):
            stream.write(data)
        else:
            stream.write(json.dumps(data, indent=2) + "\n")
    return path


def response_signature(entries):
    return tuple((item["expert"], tuple(sorted(tuple(span) for span in item["citations"])))
                 for item in entries)


def build(queries_path, corpus_path, dataset, trace, aggregate, output):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(trace.read_bytes())
    inputs = {name: base.sha(path) for name, path in (
        ("queries", queries_path), ("corpus", corpus_path), ("aggregate", aggregate))}
    if inputs != {name: meta[f"{name}_sha256"] for name in inputs}:
        raise ValueError("Changed query, source or aggregate inventory")
    queries = base.index(base.records(queries_path), lambda row: row["id"])
    sources = base.index(base.records(corpus_path), lambda row: row["id"])
    if len(queries) != 80 or any(row["split"] != "development" for row in queries.values()):
        raise ValueError("Expected development queries only")
    catalog = dataset / "experts.jsonl"
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    source_catalog = dataset / "corpus.jsonl"
    for path in (catalog, source_catalog):
        if base.sha(path) != manifest["files_sha256"][path.name]:
            raise ValueError("Changed candidate or source catalog")
    experts = base.index((row for row in base.records(catalog) if row["split"] == "development"),
                         lambda row: row["id"])
    original_sources = base.index((row for row in base.records(source_catalog) if row["split"] == "development"),
                                  lambda row: row["id"])
    by_query = base.index(meta["records"], lambda row: row["query_id"])
    if by_query.keys() != queries.keys() or any(set(r["views"]) != set(POLICIES) for r in by_query.values()):
        raise ValueError("Changed paired request inventory")

    different = set()
    matched = []
    audit = Counter()
    for query, row in by_query.items():
        ref, short = (row["views"][arm] for arm in POLICIES)
        ref_signature, short_signature = (response_signature(item) for item in (ref, short))
        for method, current in zip(POLICIES, (ref, short), strict=True):
            audit[f"issued_{method}"] += bool(current)
        ref_order = [item["expert"] for item in ref]
        short_order = [item["expert"] for item in short]
        audit["people_or_order_changes"] += ref_order != short_order
        if ref_signature != short_signature:
            different.add(query)
            audit["same_order_changed_evidence"] += ref_order == short_order
        elif ref:
            matched.append(query)
    selected_controls = sorted(matched, key=lambda q: sha256(("criterion-window-control:" + q).encode()).digest())
    chosen = different | set(selected_controls[:CONTROL_QUERIES])
    report = json.loads(aggregate.read_bytes())["metrics"]
    if (audit["people_or_order_changes"] != report["criterion"]["changed_people_or_order"]
            or any(audit[f"issued_{m}"] != report[m]["issued_queries"] for m in POLICIES)):
        raise ValueError("Trace disagrees with the execution aggregate")

    methods = {method: {} for method in POLICIES}
    for query in sorted(chosen):
        for method in POLICIES:
            methods[method][query] = pairs = []
            for item in by_query[query]["views"][method]:
                claims = [dict(source_id=doc, start=start, end=end)
                          for doc, start, end in item["citations"]]
                for claim in claims:
                    doc = claim["source_id"]
                    if any(sources[doc][field] != original_sources[doc][field] for field in ("text", "expert_id")):
                        raise ValueError("Source catalog differs from the execution projection")
                pairs.append((item["expert"], base.sources(original_sources, item["expert"], claims)))

    forms, mapping, predictions, counts = base.packet_rows(
        {query: queries[query] for query in sorted(chosen)}, experts, methods)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    form = write(output / "blind-review.jsonl", forms, lines=True)
    coordinator = write(output / "private-mapping.jsonl", mapping, lines=True)
    write(output / "private-design.json", dict(
        changed_queries=sorted(different), controls=selected_controls[:CONTROL_QUERIES], predictions=predictions))
    write(output / "review-instructions.md", INSTRUCTIONS)
    packet = write(output / "manifest.json", dict(
        candidate_views=len(forms), inputs_sha256=dict(
            inputs, trace=base.sha(trace), experts=base.sha(catalog), source_catalog=base.sha(source_catalog)),
        outputs_sha256={path.name: base.sha(path) for path in (form, coordinator)}))
    for reviewer in ("a", "b"):
        copy = write(output / f"reviewer-{reviewer}.jsonl", forms, lines=True)
        if base.check(output, copy)["cases"] != len(forms):
            raise ValueError("Reviewer copy lost cases")
    summary = dict(
        source_requests=len(queries), changed_requests=len(different), **audit,
        control_requests=len(chosen)-len(different), review_cases=len(forms),
        source_views=sum(len(row["sources"]) for row in forms), human_judgments=0, **counts)
    write(output / "summary.json", dict(summary, packet_sha256=base.sha(packet)))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("queries_path", "corpus_path", "dataset", "trace", "aggregate", "output"):
        parser.add_argument(arg, type=Path)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
