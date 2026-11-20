"""Freeze blinded topic judgments and bound P@5 under incomplete topic tags."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import accumulate
import json
from math import ceil
from pathlib import Path

from experiments.credit import topic_role_postings as pilot


CONTRASTS = (
    ("bounded_four", "topic_four"),
    ("bounded_four", "topic_eight"),
    ("bounded_four", "bounded_union_four"),
    ("person_four", "topic_four"),
    ("person_four", "topic_eight"),
    ("bounded_four_pilot_calibrated", "topic_four"),
)
RELEVANCE = {"relevant", "irrelevant", "unclear"}


def case_id(topic, doi):
    return sha256(json.dumps([topic, doi], ensure_ascii=True).encode()).hexdigest()[:24]


def freeze(projection, run, witness_file, original):
    trace = json.loads(witness_file.read_bytes())
    if (trace["projection_sha256"], trace["results_sha256"]) != (pilot.digest(projection), pilot.digest(run)):
        raise ValueError("Trace hashes do not match the supplied artifacts")
    sources = {row["doi"]: row for row in json.loads(projection.read_bytes())["sources"]}
    replay = json.loads(run.read_bytes())
    reference = json.loads(original.read_bytes())
    for key in ("manifest_sha256", "embedding", "reranker", "pilot_thresholds", "pilot_frontier"):
        if replay[key] != reference[key]:
            raise ValueError(f"Frozen result changed: {key}")
    cases = {}
    if set(trace["witnesses"]) != set(reference["holdout"]):
        raise ValueError("Declaration coverage masks changed")
    for cover, arms in trace["witnesses"].items():
        anchor = [row["topic"] for row in arms["topic_four"]]
        if set(arms) != set(reference["holdout"][cover]):
            raise ValueError("Route set changed")
        for arm, rows in arms.items():
            if [row["topic"] for row in rows] != anchor:
                raise ValueError("Query alignment changed")
            if any(len(row["citations"]) > pilot.TOP for row in rows):
                raise ValueError("Witness count exceeds output cutoff")
            grounded, off_label = 0, 0
            for row in rows:
                topic = row["topic"]
                for doi in row["citations"]:
                    source = sources[doi]
                    identifier = case_id(topic, doi)
                    title, abstract = source["text"].split("\n", 1)
                    cases[identifier] = dict(
                        case_id=identifier, topic=topic, title=title, abstract=abstract,
                        doi=doi, url=source["url"], publisher_topic=topic in source["topics"],
                    )
                    grounded += topic in source["topics"]
                    off_label += topic not in source["topics"]
            for report in (replay, reference):
                stats = report["holdout"][cover][arm]["vs_topic_four"]
                if (grounded, off_label, len(rows)) != (
                        stats["grounded"], stats["cited_off_label"], stats["queries"]):
                    raise ValueError(f"Counts changed for {cover}, {arm}")
    return trace["witnesses"], dict(sorted(cases.items()))


def paired_weights(candidate, comparator):
    counts = Counter()
    for left, right in zip(candidate, comparator, strict=True):
        topic = left["topic"]
        if topic != right["topic"]:
            raise ValueError("Query topics differ")
        counts.update(case_id(topic, doi) for doi in left["citations"])
        counts.subtract(case_id(topic, doi) for doi in right["citations"])

    return {identifier: weight for identifier, weight in counts.items() if weight}


def contrast(candidate, comparator, labels):
    weights = paired_weights(candidate, comparator)
    observed = sum(weight for key, weight in weights.items() if labels.get(key) == "relevant")
    changed_unknown = {key: weight for key, weight in weights.items() if key not in labels}
    low = sum(min(0, weight) for weight in changed_unknown.values())
    high = sum(max(0, weight) for weight in changed_unknown.values())
    slots = pilot.TOP * len(candidate)
    return dict(
        slots=slots, known_grounded_delta=observed / slots,
        possible_grounded_delta=[(observed + low) / slots, (observed + high) / slots],
        added_unresolved_pairs=sum(weight > 0 for weight in changed_unknown.values()),
        removed_unresolved_pairs=sum(weight < 0 for weight in changed_unknown.values()),
        added_unresolved_slots=high, removed_unresolved_slots=-low,
        known_grounded_delta_slots=observed,
        possible_delta_slots=[observed + low, observed + high],
    )


def minimum_changes(weights, required_swing):
    if required_swing <= 0:
        return 0
    changes = accumulate(sorted(weights, reverse=True))
    return next((count for count, swing in enumerate(changes, 1) if swing >= required_swing), None)


def sensitivity(packet, reference, output):
    if output.exists():
        raise ValueError("Provide a fresh sensitivity output")
    if pilot.digest(packet) != json.loads(reference.read_bytes())["packet_sha256"]:
        raise ValueError("Packet differs from the published freeze")

    data = json.loads(packet.read_bytes())
    cases = data["cases"]
    witnesses = data["witnesses"]
    influential = set()
    reports = {}
    for coverage, arms in witnesses.items():
        reports[coverage] = {}
        for arm, base in CONTRASTS:
            weights = paired_weights(arms[arm], arms[base])
            influential.update(weights)
            tagged = [weight for key, weight in weights.items() if cases[key]["publisher_topic"]]
            untagged = [weight for key, weight in weights.items() if not cases[key]["publisher_topic"]]
            observed = sum(tagged)
            low = observed + sum(min(0, weight) for weight in untagged)
            high = observed + sum(max(0, weight) for weight in untagged)
            slots = pilot.TOP * len(arms[arm])
            gates = {"at_least_one_more_slot": 1}
            if base == "topic_four":
                gates["gain_at_least_two_points"] = ceil(Fraction(2 * slots, 100))
            if base == "topic_eight":
                gates["loss_at_most_one_point"] = ceil(Fraction(-slots, 100))
            reports[coverage][f"{arm}_vs_{base}"] = dict(
                slots=slots,
                influential_tagged_pairs=len(tagged),
                influential_untagged_pairs=len(untagged),
                publisher_tag_delta=observed,
                delta_if_tags_all_relevant=[low, high],
                delta_if_all_unjudged=[sum(min(0, w) for w in weights.values()),
                                       sum(max(0, w) for w in weights.values())],
                gates={
                    name: dict(
                        minimum_net_slots=minimum,
                        tagged_reversals_to_make_failure_possible=minimum_changes(
                            [w for w in tagged if w > 0], low - minimum + 1),
                        tagged_reversals_to_make_success_possible=minimum_changes(
                            [-w for w in tagged if w < 0], minimum - high),
                    )
                    for name, minimum in gates.items()
                },
            )
    pilot.write_json(output, dict(
        packet_sha256=pilot.digest(packet),
        reference_sha256=pilot.digest(reference),
        cases=len(cases),
        influential_pairs=len(influential),
        influential_tagged_pairs=sum(cases[key]["publisher_topic"] for key in influential),
        comparisons=reports,
    ))


def render_reports(witnesses, labels):
    return {
        cover: {f"{arm}_vs_{base}": contrast(arms[arm], arms[base], labels)
                for arm, base in CONTRASTS}
        for cover, arms in witnesses.items()
    }


def prepare(projection, replay, witnesses, original, output):
    arms, cases = freeze(projection, replay, witnesses, original)
    output.mkdir()
    judge_rows = [
        {key: row[key] for key in ("case_id", "topic", "title", "abstract", "doi", "url")}
        for row in cases.values()
    ]
    form = "".join(json.dumps(dict(row, relevance=None, rationale=None), ensure_ascii=False) + "\n"
                   for row in judge_rows)
    for name in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
        (output / name).write_text(form, encoding="utf-8")
    packet = output / "packet.json"
    pilot.write_json(packet, dict(cases=cases, witnesses=arms,
                                 review_prompt="Is the requested topic directly addressed by this article? "
                                               "Use relevant, irrelevant, or unclear and cite a short reason."))
    pilot.write_json(output / "summary.json", dict(
        packet_sha256=pilot.digest(packet), source_replay_sha256=pilot.digest(replay),
        cases=len(cases), off_label_cases=sum(not row["publisher_topic"] for row in cases.values()),
        human_labels=0, comparisons=render_reports(
            arms, {key: "relevant" for key, row in cases.items() if row["publisher_topic"]},
        ),
    ))


def parse_reviews(path, cases):
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["case_id"]
        titles = ("topic", "title", "abstract", "doi", "url")
        if key in results or any(row[field] != cases[key][field] for field in titles):
            raise ValueError(f"Duplicate or modified case: {key}")
        label = row["relevance"]
        if label is not None and (label not in RELEVANCE or not (row["rationale"] or "").strip()):
            raise ValueError(f"Missing rationale or invalid label: {key}")
        results[key] = label
    if results.keys() != cases.keys():
        raise ValueError("Review case set differs from the packet")
    return results


def reconcile(packet, reviewer_a, reviewer_b, output):
    data = json.loads(packet.read_bytes())
    cases = data["cases"]
    left, right = (parse_reviews(path, cases) for path in (reviewer_a, reviewer_b))
    counts = Counter()
    joint = {}
    for key in left:
        a, b = left[key], right[key]
        if a is None or b is None:
            counts["unfinished"] += 1
        elif a != b:
            counts["disagreement"] += 1
        elif a == "unclear":
            counts["agree_unclear"] += 1
        else:
            counts[f"agree_{a}"] += 1
            joint[key] = a
    reports = {}
    for cover, arms in data["witnesses"].items():
        reports[cover] = {}
        for arm, rows in arms.items():
            labels = Counter(joint.get(case_id(row["topic"], doi), "unresolved")
                             for row in rows for doi in row["citations"])
            slots = pilot.TOP * len(rows)
            reports[cover][arm] = dict(
                grounded_lower=labels["relevant"] / slots,
                grounded_upper=(labels["relevant"] + labels["unresolved"]) / slots,
                relevant=labels["relevant"], irrelevant=labels["irrelevant"],
                unresolved=labels["unresolved"], slots=slots,
            )
    if output.exists():
        raise ValueError("Provide a fresh reconciliation path")
    pilot.write_json(output, dict(
        packet_sha256=pilot.digest(packet), review_counts=dict(counts), reports=reports,
        comparisons=render_reports(data["witnesses"], joint),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("prepare")
    for name in ("projection", "replay", "witnesses", "original", "output"):
        pack.add_argument(name, type=Path)
    score = sub.add_parser("reconcile")
    for name in ("packet", "reviewer_a", "reviewer_b", "output"):
        score.add_argument(name, type=Path)
    stress = sub.add_parser("sensitivity")
    for name in ("packet", "reference", "output"):
        stress.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    command = args.pop("command")
    commands = {"prepare": prepare, "reconcile": reconcile, "sensitivity": sensitivity}
    commands[command](**args)
