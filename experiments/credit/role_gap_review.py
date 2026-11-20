"""Prepare blind topical-support review and exact uncertainty bounds for gap refill."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_role_review as review
from experiments.credit import nomination_label_audit as audit
from experiments.credit import topic_role_postings as pilot


TARGET = "gap_title_four"
BASELINES = ("person_four", "role_title_four")
MASK = "100"
FIRST_BATCH = "elife:person_four:fixed_slots"
VIEW_FIELDS = ("case_id", "topic", "title", "abstract")
RAW_TIMING_FIELDS = review.TIMING_KEYS | {"title_index_encode_seconds"}


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in RAW_TIMING_FIELDS}
    return value


def input_cases(plos, elife, mesh_snapshot, replay, witnesses, reference):
    trace = json.loads(witnesses.read_bytes())
    run = json.loads(replay.read_bytes())
    frozen = json.loads(reference.read_bytes())
    if stable(run) != stable(frozen["run"]) or stable(run) != stable(frozen["repeat"]):
        raise ValueError("The audit must reproduce both frozen replays")
    if trace["replay_sha256"] != pilot.digest(replay):
        raise ValueError("Witness trace does not match its replay")
    snapshot = json.loads(mesh_snapshot.read_bytes())
    if snapshot["projection_sha256"] != pilot.digest(elife):
        raise ValueError("MEDLINE snapshot does not match the eLife catalog")
    catalog = SimpleNamespace(articles=json.loads(elife.read_bytes())["sources"], topics=defaultdict(set))
    mesh.assign_topics(catalog, snapshot["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
    broad = {topic: {catalog.articles[i]["doi"] for i in sources} for topic, sources in catalog.topics.items()}

    cases, comparisons = {}, {}
    for panel, projection in (("plos", plos), ("elife", elife)):
        if pilot.digest(projection) != trace["projections"][panel]:
            raise ValueError("Witness trace uses a different catalog")
        articles = {row["doi"]: row for row in json.loads(projection.read_bytes())["sources"]}
        routes = trace["witnesses"][panel][MASK]
        comparisons[panel] = {arm: routes[arm] for arm in (TARGET, *BASELINES)}
        topics = [row["topic"] for row in routes[TARGET]]
        for arm, rows in comparisons[panel].items():
            if [r["topic"] for r in rows] != topics:
                raise ValueError("Audit query alignment differs")
            issued = tagged = 0
            for row in rows:
                topic = row["topic"]
                for doi in row["citations"]:
                    article = articles[doi]
                    title, abstract = article["text"].split("\n", 1)
                    identifier = audit.case_id(topic, doi)
                    tag = (topic in article["topics"] if panel == "plos" else doi in broad.get(topic, ()))
                    current = dict(case_id=identifier, topic=topic, title=title, abstract=abstract,
                                   doi=doi, tag=tag, panel=panel)
                    if cases.setdefault(identifier, current) != current:
                        raise ValueError("A common review view has inconsistent provenance")
                    issued += 1
                    tagged += tag
            prior = run[panel]["metrics"][MASK][arm]["compared"]["topic_four"]
            if (len(rows), issued, tagged) != (prior["queries"], prior["issued"], prior["grounded"]):
                raise ValueError("Witness trace does not reproduce frozen issuance and labels")
    return dict(sorted(cases.items())), comparisons


def uncertainty(weights, labels, scale):
    unknown = {key: abs(value) for key, value in weights.items() if value and key not in labels}
    width = sum(unknown.values())
    lower, upper = review.bounds(weights, labels)
    impacts = sorted(unknown.items(), key=lambda item: (-item[1], item[0]))
    required = max(0, width - Fraction(1, 50) * scale)
    return dict(
        unresolved_cases=len(unknown), range_pp=[100 * float(x / scale) for x in (lower, upper)],
        most_influential_case_ids=[key for key, _ in impacts],
        width_after_resolving_pp={
            limit: 100 * float((width - sum(value for _, value in impacts[:limit])) / scale)
            for limit in (6, 12, 24)
        },
        decisive_labels_needed_for_width_at_most_2pp=audit.minimum_changes(
            list(unknown.values()), required,
        ),
    )


def evaluate(comparisons, labels):
    metrics, priority = {}, {}
    for panel, routes in comparisons.items():
        metrics[panel] = {}
        for base in BASELINES:
            candidate, comparator = routes[TARGET], routes[base]
            a, b = map(review.citation_counts, (candidate, comparator))
            if not a.total() or not b.total():
                raise ValueError("Both methods must issue at least one recommendation")
            slots = pilot.TOP * len(candidate)
            slot_weights = audit.paired_weights(candidate, comparator)
            precision_weights = {
                key: Fraction(a[key], a.total()) - Fraction(b[key], b.total())
                for key in a.keys() | b.keys()
            }
            entry = review.compare(candidate, comparator, labels)
            for name, weights, scale in (("fixed_slots", slot_weights, slots),
                                         ("conditional_support", precision_weights, 1)):
                report = uncertainty(weights, labels, scale)
                priority[f"{panel}:{base}:{name}"] = report.pop("most_influential_case_ids")
                entry[name] = report
            metrics[panel][base] = entry
    return metrics, priority


def blank_views(cases):
    return "".join(
        json.dumps(dict(**{key: case[key] for key in VIEW_FIELDS}, relevance=None, rationale=None),
                   ensure_ascii=False) + "\n"
        for case in cases.values()
    )


def prepare(plos, elife, mesh_snapshot, replay, witnesses, reference, output, summary):
    if output.exists() or summary.exists():
        raise ValueError("Provide a new review directory and aggregate output")
    cases, comparisons = input_cases(plos, elife, mesh_snapshot, replay, witnesses, reference)
    metrics, priority = evaluate(comparisons, {})
    tag_labels = {key: "relevant" for key, case in cases.items() if case["tag"]}
    tagged, _ = evaluate(comparisons, tag_labels)
    output.mkdir(mode=0o700)
    packet = output / "packet.json"
    pilot.write_json(packet, dict(cases=cases, comparisons=comparisons, priority=priority))
    lines = blank_views(cases)
    for slot in ("a", "b"):
        (output / f"reviewer-{slot}.jsonl").write_text(lines, encoding="utf-8")
    pilot.write_json(summary, dict(
        replay_job=json.loads(replay.read_bytes())["job"], replay_sha256=pilot.digest(replay),
        reference_sha256=pilot.digest(reference), packet_sha256=pilot.digest(packet),
        code_sha256=pilot.digest(Path(__file__)), cases=Counter(case["panel"] for case in cases.values()),
        tagged_cases=Counter(case["panel"] for case in cases.values() if case["tag"]),
        human_labels=0, all_unjudged=metrics, tagged_positive_assumption=tagged,
    ))


def first_batch(data):
    metrics, priority = evaluate(data["comparisons"], {})
    minimum = metrics["elife"]["person_four"]["fixed_slots"]["decisive_labels_needed_for_width_at_most_2pp"]
    return sorted(priority[FIRST_BATCH][:minimum])


def assign(packet, interface, output):
    data = json.loads(packet.read_bytes())
    ids = first_batch(data)
    if not ids:
        raise ValueError("No review cases are needed for the first batch")
    form = blank_views({key: data["cases"][key] for key in ids})
    output.mkdir(mode=0o700)
    pilot.write_json(output / "assignment.json", dict(packet_sha256=pilot.digest(packet), cases=ids))
    instructions = (
        "Unzip this folder and open offline-review.html in a browser.\n"
        "Load review.jsonl and judge the visible title and abstract against the topic.\n"
        "Choose Relevant, Irrelevant, or Unclear and write a brief rationale.\n"
        "Export an updated JSONL to save progress. Reload your own export to resume.\n"
        "Return your exported JSONL to the study organizer. Keep your judgments independent.\n"
    )
    for slot in ("a", "b"):
        path = output / f"reviewer-{slot}.zip"
        with open(path, "xb", opener=private_file) as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as bundle:
                bundle.writestr("offline-review.html", interface.read_bytes())
                bundle.writestr("review.jsonl", form)
                bundle.writestr("README.txt", instructions)


def private_file(path, flags):
    import os
    return os.open(path, flags, 0o600)


def reconcile(packet, reviewer_a, reviewer_b, output, assignment=None):
    if reviewer_a.samefile(reviewer_b):
        raise ValueError("Provide two independently completed files")
    data = json.loads(packet.read_bytes())
    cases = data["cases"]
    if assignment is not None:
        manifest = json.loads(assignment.read_bytes())
        if manifest != dict(packet_sha256=pilot.digest(packet), cases=first_batch(data)):
            raise ValueError("Assignment does not match the frozen priority batch")
        cases = {key: cases[key] for key in manifest["cases"]}
    first, second = (review.read_labels(path, cases) for path in (reviewer_a, reviewer_b))
    labels, decisions = {}, Counter()
    for key, label in first.items():
        other = second[key]
        if label is None or other is None:
            decisions["unfinished"] += 1
        elif label != other:
            decisions["disagreement"] += 1
        elif label == "unclear":
            decisions["both_unclear"] += 1
        else:
            decisions[f"agree_{label}"] += 1
            labels[key] = label
    metrics, _ = evaluate(data["comparisons"], labels)
    pilot.write_json(output, dict(packet_sha256=pilot.digest(packet), decisions=decisions, unassigned_cases=len(data["cases"]) - len(cases), comparisons=metrics))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    for command, args in (
        ("prepare", ("plos", "elife", "mesh_snapshot", "replay", "witnesses", "reference", "output", "summary")),
        ("assign", ("packet", "interface", "output")),
        ("reconcile", ("packet", "reviewer_a", "reviewer_b", "output")),
    ):
        parser = sub.add_parser(command)
        for arg in args:
            parser.add_argument(arg, type=Path)
        if command == "reconcile":
            parser.add_argument("--assignment", type=Path)
    options = vars(cli.parse_args())
    {"prepare": prepare, "assign": assign, "reconcile": reconcile}[options.pop("command")](**options)
