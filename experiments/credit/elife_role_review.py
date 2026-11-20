"""Audit citation-label uncertainty in the frozen eLife nomination replay."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from experiments.credit import nomination_label_audit as label_audit
from experiments.credit import topic_role_postings as pilot


CONTRASTS = (
    ("person_four", "topic_four"),
    ("person_four", "union_four"),
    ("bounded_four", "topic_eight"),
)
TIMING_KEYS = {"job", "script_sha256", "p95_ms", "paired_p95_delta_ci95_ms",
               "p95_reduction_interval_below_zero"}


def quality_record(value):
    if isinstance(value, dict):
        return {k: quality_record(v) for k, v in value.items() if k not in TIMING_KEYS}
    return value


def citation_counts(rows):
    return Counter(label_audit.case_id(row["topic"], doi)
                   for row in rows for doi in row["citations"])


def bounds(weights, labels):
    known = sum(weight for key, weight in weights.items() if labels.get(key) == "relevant")
    unknown = [weight for key, weight in weights.items() if key not in labels]
    return [known + sum(min(0, weight) for weight in unknown),
            known + sum(max(0, weight) for weight in unknown)]


def compare(candidate, comparator, labels):
    counts = label_audit.paired_weights(candidate, comparator)
    left, right = citation_counts(candidate), citation_counts(comparator)
    left_size, right_size = left.total(), right.total()
    issued_weights = {
        key: Fraction(left[key], left_size) - Fraction(right[key], right_size)
        for key in left.keys() | right.keys()
    } if left_size and right_size else {}
    conditional = [float(value) for value in bounds(issued_weights, labels)] if issued_weights else (
        [0.0, 0.0] if left_size and right_size else None)
    return dict(
        output_slots=pilot.TOP * len(candidate),
        candidate_issued=left_size, comparator_issued=right_size,
        supported_slot_delta=bounds(counts, labels),
        supported_fraction_given_issuance_delta=conditional,
        influential_slot_pairs=sum(bool(w) for w in counts.values()),
        influential_conditional_pairs=sum(bool(w) for w in issued_weights.values()),
    )


def summarize(witnesses, labels):
    return {
        panel: {
            coverage: {f"{arm}_vs_{base}": compare(arms[arm], arms[base], labels)
                       for arm, base in CONTRASTS}
            for coverage, arms in coverages.items()
        }
        for panel, coverages in witnesses.items()
    }


def load_trace(projection, replay, witness_file, reference):
    trace = json.loads(witness_file.read_bytes())
    if trace["projection_sha256"] != pilot.digest(projection) or trace["replay_sha256"] != pilot.digest(replay):
        raise ValueError("Trace digests do not match the replay")
    prior = json.loads(reference.read_bytes())
    observed = json.loads(replay.read_bytes())
    if any(quality_record(observed) != quality_record(prior[run]) for run in ("run", "repeat")):
        raise ValueError("Replayed results differ from one or more frozen runs")
    articles = {row["doi"]: row for row in json.loads(projection.read_bytes())["sources"]}
    witnesses, cases = trace["witnesses"], {}
    if witnesses.keys() != {"headings", "keywords"}:
        raise ValueError("Evaluation panels differ")
    for panel, coverages in witnesses.items():
        target = observed[panel]
        if coverages.keys() != target["masks"].keys():
            raise ValueError("Declaration masks differ")
        field = "topics" if panel == "headings" else "keywords"
        for coverage, arms in coverages.items():
            if arms.keys() != target["masks"][coverage].keys():
                raise ValueError("Route set differs")
            topics = [row["topic"] for row in arms["topic_four"]]
            for arm, rows in arms.items():
                if len(rows) != target["queries"] or [r["topic"] for r in rows] != topics:
                    raise ValueError("Query alignment differs")
                tagged = issued = 0
                for row in rows:
                    topic = row["topic"]
                    if len(row["citations"]) > pilot.TOP:
                        raise ValueError("More citations than output slots")
                    for doi in row["citations"]:
                        article = articles[doi]
                        title, abstract = article["text"].split("\n", 1)
                        identifier = label_audit.case_id(topic, doi)
                        current = dict(case_id=identifier, topic=topic, title=title, abstract=abstract,
                                       doi=doi, tag=topic in article[field])
                        if identifier in cases and current != cases[identifier]:
                            raise ValueError("Inconsistent topic label for a common view")
                        cases[identifier] = current
                        issued += 1
                        tagged += current["tag"]
                metrics = target["masks"][coverage][arm]["topic_four"]
                if (issued, tagged, issued - tagged) != (
                        metrics["issued"], metrics["grounded"], metrics["cited_off_label"]):
                    raise ValueError("Citation counts differ from the replay")
    return witnesses, dict(sorted(cases.items()))


def prepare(projection, replay, witness_file, reference, output):
    witnesses, cases = load_trace(projection, replay, witness_file, reference)
    primary = witnesses["keywords"]["100"]
    priority = sorted(set().union(*(label_audit.paired_weights(primary[arm], primary[base]).keys()
                                    for arm, base in CONTRASTS)))
    output.mkdir()
    # The two forms contain only the review view, without route, labels, DOI, or priority.
    views = [dict(**{k: row[k] for k in ("case_id", "topic", "title", "abstract")},
                  relevance=None, rationale=None) for row in cases.values()]
    form = "".join(json.dumps(view, ensure_ascii=False) + "\n" for view in views)
    for filename in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
        (output / filename).write_text(form, encoding="utf-8")
    packet = output / "packet.json"
    pilot.write_json(packet, dict(
        cases=cases, witnesses=witnesses, primary_slot_priority=priority,
        instruction="Does the visible title and abstract directly address the requested topic? "
                    "Choose relevant, irrelevant, or unclear and give a short rationale.",
    ))
    positive_tags = {key: "relevant" for key, row in cases.items() if row["tag"]}
    pilot.write_json(output / "summary.json", dict(
        packet_sha256=pilot.digest(packet), reference_sha256=pilot.digest(reference),
        replay_job=json.loads(replay.read_bytes())["job"], cases=len(cases),
        tagged_cases=len(positive_tags), primary_slot_priority_cases=len(priority), human_labels=0,
        all_unjudged=summarize(witnesses, {}), tagged_positive_assumption=summarize(witnesses, positive_tags),
    ))


def read_labels(path, cases):
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["case_id"]
        if key in labels or any(row[field] != cases[key][field] for field in ("topic", "title", "abstract")):
            raise ValueError(f"Duplicate or modified case: {key}")
        label = row["relevance"]
        if label is not None and (label not in label_audit.RELEVANCE or not (row["rationale"] or "").strip()):
            raise ValueError(f"Invalid judgment or absent rationale: {key}")
        labels[key] = label
    if labels.keys() != cases.keys():
        raise ValueError("Review case set differs from the packet")
    return labels


def reconcile(packet, reviewer_a, reviewer_b, output):
    if output.exists():
        raise ValueError("Provide a fresh reconciliation output")
    data = json.loads(packet.read_bytes())
    a, b = (read_labels(path, data["cases"]) for path in (reviewer_a, reviewer_b))
    decisions = Counter()
    labels = {}
    for key in a:
        if a[key] is None or b[key] is None:
            decisions["unfinished"] += 1
        elif a[key] != b[key]:
            decisions["disagreement"] += 1
        elif a[key] == "unclear":
            decisions["both_unclear"] += 1
        else:
            decisions[f"agree_{a[key]}"] += 1
            labels[key] = a[key]
    pilot.write_json(output, dict(
        packet_sha256=pilot.digest(packet), decisions=decisions,
        comparisons=summarize(data["witnesses"], labels),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    modes = {
        "prepare": (prepare, ("projection", "replay", "witness_file", "reference", "output")),
        "reconcile": (reconcile, ("packet", "reviewer_a", "reviewer_b", "output")),
    }
    for name, (_, arguments) in modes.items():
        parser = sub.add_parser(name)
        for argument in arguments:
            parser.add_argument(argument, type=Path)
    args = vars(cli.parse_args())
    modes[args.pop("command")][0](**args)
