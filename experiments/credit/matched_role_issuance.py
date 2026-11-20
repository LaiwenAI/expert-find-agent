"""Decompose role nomination gains into matched-rank and extra-issuance terms."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.credit import elife_role_review as review
from experiments.credit import nomination_label_audit as audit
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


CONTRASTS = (
    ("person_four", "topic_four"),
    ("person_four", "union_four"),
    ("bounded_four", "topic_eight"),
)


def split_prefixes(candidate, comparator):
    shared_left, shared_right, extra_left, extra_right = [], [], [], []
    sizes = Counter()
    for left, right in zip(candidate, comparator, strict=True):
        if left["topic"] != right["topic"]:
            raise ValueError("Query alignment differs")
        a, b = left["citations"], right["citations"]
        k = min(len(a), len(b))
        sizes.update(
            candidate_issued=len(a), comparator_issued=len(b),
            matched_per_method=k, queries_with_match=bool(k),
            queries_with_more_candidate=int(len(a) > len(b)),
            queries_with_more_comparator=int(len(b) > len(a)),
            matched_lists_differ=int(a[:k] != b[:k]),
            candidate_only_requests=int(bool(a) and not b),
            comparator_only_requests=int(bool(b) and not a),
        )
        for target, citations in ((shared_left, a[:k]), (shared_right, b[:k]),
                                  (extra_left, a[k:]), (extra_right, b[k:])):
            target.append(dict(topic=left["topic"], citations=citations))
    return sizes, (shared_left, shared_right), (extra_left, extra_right)


def describe(left, right, tags):
    weights = audit.paired_weights(left, right)
    tagged = [sum(count for key, count in review.citation_counts(rows).items() if key in tags)
              for rows in (left, right)]
    tagged_delta_by_query = [
        (sum(audit.case_id(a["topic"], doi) in tags for doi in a["citations"])
         - sum(audit.case_id(b["topic"], doi) in tags for doi in b["citations"])) / pilot.TOP
        for a, b in zip(left, right, strict=True)
    ]
    return dict(
        tagged_candidate=tagged[0], tagged_comparator=tagged[1], tagged_delta=tagged[0] - tagged[1],
        paired_tagged_p5_delta_ci95=holdout.interval(tagged_delta_by_query, [r["topic"] for r in left])[
            "percentile_95"],
        support_delta_all_unjudged=review.bounds(weights, {}),
        support_delta_if_tagged_supported=review.bounds(weights, tags),
        pairs_changing_delta=len(weights),
    )


def compare(candidate, comparator, tags):
    sizes, prefix, tail = split_prefixes(candidate, comparator)
    full = describe(candidate, comparator, tags)
    shared = describe(*prefix, tags)
    extra = describe(*tail, tags)
    exclusive = tuple([
        row if not matched["citations"] else dict(row, citations=[])
        for row, matched in zip(rows, common, strict=True)
    ] for rows, common in zip(tail, prefix, strict=True))
    exclusive_report = describe(*exclusive, tags)
    if full["tagged_delta"] != shared["tagged_delta"] + extra["tagged_delta"]:
        raise ValueError("Prefix and tail decomposition does not sum to the full contrast")
    return dict(
        queries=len(candidate), fixed_output_slots=pilot.TOP * len(candidate),
        quantities=sizes, full=full, matched_prefix=shared, unmatched_tail=extra,
        tail_on_requests_served_by_only_one_method=exclusive_report,
    )


def load(packet, summary):
    if pilot.digest(packet) != json.loads(summary.read_bytes())["packet_sha256"]:
        raise ValueError("Packet differs from the released freeze")
    return json.loads(packet.read_bytes())


def summarize(witnesses, tags):
    return {
        str(coverage): {f"{arm}_vs_{base}": compare(arms[arm], arms[base], tags)
                        for arm, base in CONTRASTS}
        for coverage, arms in witnesses.items()
    }


def run(plos_packet, plos_summary, elife_packet, elife_summary, output):
    if output.exists():
        raise ValueError("Provide a fresh output path")
    plos = load(plos_packet, plos_summary)
    elife = load(elife_packet, elife_summary)
    plos_tags = {key: "relevant" for key, row in plos["cases"].items() if row["publisher_topic"]}
    elife_tags = {key: "relevant" for key, row in elife["cases"].items() if row["tag"]}
    pilot.write_json(output, dict(
        plos_packet_sha256=pilot.digest(plos_packet), elife_packet_sha256=pilot.digest(elife_packet),
        human_labels=0, panels=dict(
            plos_topics=summarize(plos["witnesses"], plos_tags),
            **{f"elife_{panel}": summarize(witnesses, elife_tags)
               for panel, witnesses in elife["witnesses"].items()},
        ),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_packet", "plos_summary", "elife_packet", "elife_summary", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
