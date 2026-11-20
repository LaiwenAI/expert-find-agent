"""Freeze a two-reviewer source-only audit of the topic recheck."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval.enterprise_routed_review import digest, rows, write


def case_key(request, owner, ref):
    return sha256(json.dumps(["enterprise-topic-recheck-v1", request, owner, ref]).encode()).hexdigest()


def sample(identifier, diagnostic):
    if "control" in diagnostic:
        key, accepted = diagnostic["control"]
        yield "control", "control", key, {"0.6b": True, "4b": accepted}
        return

    original = diagnostic["original_4b_positive"]
    later_small = diagnostic["later_0_6b_positive"]
    later_large = diagnostic["later_4b_positive"]
    for group, entries in (
        ("same_passage_4b", original),
        ("later_passage_0_6b", later_small),
        ("later_passage_4b", later_large),
        ("agreed_original_negative", diagnostic["agreed_original_negative"]),
    ):
        if not entries:
            continue
        key = min(entries, key=lambda pair: case_key(identifier, *pair))
        category = "no_support" if group == "agreed_original_negative" else "added_support"
        if category == "added_support":
            votes = {"0.6b": key in later_small, "4b": key in original or key in later_large}
        else:
            votes = {"0.6b": False, "4b": False}
        yield group, category, key, votes


def run(queries, corpus, trace, original, diagnostic, report, output):
    from experiments.retrieval.enterprise_first_query import CRITERION_CHARS

    anchor = json.loads(original.read_bytes())
    aggregate = json.loads(report.read_bytes())
    if (anchor["queries_sha256"] != digest(queries) or anchor["source_sha256"] != digest(corpus)
            or anchor["trace_sha256"] != digest(trace)
            or aggregate["reference_sha256"] != digest(original)
            or aggregate["private_sha256"] != digest(diagnostic)):
        raise ValueError("Changed diagnostic or source data")

    texts = {row["id"]: row["text"] for row in rows(queries)}
    sources = {row["id"]: row for row in rows(corpus)}
    records = json.loads(diagnostic.read_bytes())
    count = aggregate["topic_recheck"]["counts"]
    if len(records) != count["requests"]:
        raise ValueError("Unexpected diagnostic panel")
    controls = sorted(
        key for key, rec in records.items() if "control" in rec["topic_recheck"]
    )
    if len(controls) != count["baseline_positive"]:
        raise ValueError("Unexpected positive control count")
    controls = set(sorted(controls, key=lambda key: case_key(key, None, None))[:5])

    cases, mapping = {}, {}
    for key, record in records.items():
        result = record["topic_recheck"]
        if "control" in result and key not in controls:
            continue
        topic = texts[key].strip().splitlines()[0][:CRITERION_CHARS]
        for group, category, (owner, ref), votes in sample(key, result):
            source, start, end = ref
            evidence = sources[source]
            if evidence["expert_id"] != owner or not 0 <= start < end <= len(evidence["text"]):
                raise ValueError("Invalid evidence coordinates")
            identifier = case_key(key, owner, ref)[:24]
            cases[identifier] = dict(case_id=identifier, criterion=topic,
                                     source_excerpt=evidence["text"][start:end],
                                     support=None, rationale=None, reviewer_id=None)
            if identifier in mapping:
                if mapping[identifier]["category"] != category or mapping[identifier]["votes"] != votes:
                    raise ValueError("Conflicting sampling strata for one excerpt")
                mapping[identifier]["strata"].append(group)
            else:
                mapping[identifier] = dict(case_id=identifier, category=category, strata=[group],
                                           request=key, candidate_owner=owner, citation=ref, votes=votes)

    output.mkdir(mode=0o700)
    blinded = [cases[key] for key in sorted(cases)]
    for name in ("blind.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write(output / name, blinded, lines=True)
    write(output / "mapping.jsonl", (mapping[key] for key in sorted(mapping)), lines=True)
    write(output / "packet.json", dict(
        cases=len(cases), strata=Counter(group for card in mapping.values() for group in card["strata"]),
        sha256={name: digest(output / name) for name in ("blind.jsonl", "mapping.jsonl")},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "trace", "original", "diagnostic", "report", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
