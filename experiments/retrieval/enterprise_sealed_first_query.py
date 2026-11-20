"""Pin a sealed enterprise cohort and validate the resulting first-query execution."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval.enterprise_hybrid_rescue import SEALED_ARMS, summary
from experiments.retrieval.enterprise_project_role_rescue import profile_subset
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.check_joint_review import FIELDS, check


MODEL_FILES = {
    "embedding": "qwen3-embedding-0.6b/checkpoint-provenance.json",
    "parser": "qwen3-4b-instruct-2507/local-provenance.json",
    "verifier": "qwen3-reranker-0.6b/checkpoint-provenance.json",
}
RUNNER = Path(__file__).with_name("enterprise_hybrid_rescue.py")


def prepare(panel, verified_summary, protocol, development, output):
    exported = json.loads(panel.read_bytes())
    previous = json.loads(verified_summary.read_bytes())
    spec = json.loads(protocol.read_bytes())
    dev = json.loads(development.read_bytes())
    entries = exported["records"]
    if (digest(panel) != previous["sealed_index_sha256"] or
            len(entries) != previous["selected_groups"] or len(entries) != spec["groups"] or
            len({r["id"] for r in entries}) != len(entries) or
            len({r["component_sha256"] for r in entries}) != len(entries) or
            spec["arms"] != list(SEALED_ARMS)):
        raise ValueError("Sealed panel registry or locked formulation differs")
    cohort = [dict(id=row["id"], group=row["component_sha256"], text_sha256=row["text_sha256"])
              for row in entries]
    output.mkdir(mode=0o700)
    frame = output / "frame.jsonl"
    write(frame, cohort, lines=True)
    write(output / "lock.json", dict(
        frame_sha256=digest(frame), panel_sha256=digest(panel),
        public_protocol_sha256=digest(protocol), builder_sha256=digest(Path(__file__)),
        runner_sha256=digest(RUNNER), development_cohort_sha256=previous["input_sha256"]["cohort"],
        development_summary_sha256=digest(development), entries=cohort,
        files={name: dev[name] for name in ("corpus_sha256", "role_index_sha256", "vectors_sha256")},
        models=dev["models"],
    ))


def materialize(lock_file, cohort_path, output):
    lock = json.loads(lock_file.read_bytes())
    if digest(cohort_path) != lock["development_cohort_sha256"]:
        raise ValueError("Changed original-language cohort")
    need = {entry["id"] for entry in lock["entries"]}
    by_id = review.index((row for row in rows(cohort_path) if row["id"] in need), lambda row: row["id"])
    query_rows = []
    for item in lock["entries"]:
        record = by_id[item["id"]]
        text = record["zh"]
        if record["split"] != "sealed" or sha256(text.encode()).hexdigest() != item["text_sha256"]:
            raise ValueError("Sealed request does not match the frozen manifest")
        query_rows.append(dict(id=item["id"], text=text, split="sealed"))
    write(output, query_rows, lines=True)


def validate_execution(lock_file, query_file, requests, corpus, roles, vectors, vector_meta, models):
    lock = json.loads(lock_file.read_bytes())
    if lock["runner_sha256"] != digest(RUNNER) or lock["builder_sha256"] != digest(Path(__file__)):
        raise ValueError("Changed frozen pipeline")
    if [(r["id"], sha256(r["text"].encode()).hexdigest()) for r in requests] != [
            (r["id"], r["text_sha256"]) for r in lock["entries"]
    ] or any(row["split"] != "sealed" for row in requests):
        raise ValueError("Requests differ from the sealed frame")
    actual = dict(corpus_sha256=digest(corpus), role_index_sha256=digest(roles), vectors_sha256=digest(vectors))
    if actual != lock["files"] or json.loads(vector_meta.read_bytes())["vectors_sha256"] != actual["vectors_sha256"]:
        raise ValueError("Changed frozen source snapshot")
    revisions = {key: json.loads((models / path).read_bytes())["revision"] for key, path in MODEL_FILES.items()}
    if revisions != lock["models"]:
        raise ValueError("Changed frozen model revisions")


def packet(lock_file, frame, queries, dataset, trace, aggregate, output):
    lock = json.loads(lock_file.read_bytes())
    if digest(frame) != lock["frame_sha256"] or rows(frame) != lock["entries"]:
        raise ValueError("Changed cohort frame")
    private = json.loads(trace.read_bytes())
    aggregate_data = json.loads(aggregate.read_bytes())
    fingerprint = private["metadata"]
    if (fingerprint["panel_lock_sha256"], fingerprint["queries_sha256"], fingerprint["code_sha256"]) != (
        digest(lock_file), digest(queries), lock["runner_sha256"]
    ) or any(aggregate_data.get(k) != v for k,v in fingerprint.items()):
        raise ValueError("Changed execution or trace")
    if any(aggregate_data[k] != v for k,v in summary(private["observations"], SEALED_ARMS).items()):
        raise ValueError("Trace and aggregate differ")
    quer = review.index(rows(queries), lambda row: row["id"])
    if {key: sha256(row["text"].encode()).hexdigest() for key,row in quer.items()} != {
            row["id"]: row["text_sha256"] for row in lock["entries"]}:
        raise ValueError("Changed query text")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    experts, corpus = (profile_subset(dataset, name, manifest) for name in ("experts", "corpus"))
    observations = review.index(private["observations"], lambda row: row["id"])
    if observations.keys() != quer.keys() or any(set(row["arms"]) != set(SEALED_ARMS)
                                                 for row in observations.values()):
        raise ValueError("Incomplete frozen predictions")
    methods = {arm: {} for arm in SEALED_ARMS}
    for query, row in observations.items():
        for arm, result in row["arms"].items():
            methods[arm][query] = [
                (item["expert"], review.sources(corpus, item["expert"], [
                    dict(source_id=source, start=start, end=end) for source,start,end in item["citations"]
                ])) for item in result["initial"] + result["extra"]
            ]
    forms, mapping, predictions, _ = review.packet_rows(quer, experts, methods)
    for form, row in zip(forms, mapping, strict=True):
        qid = row["query_id"]
        form["prior_qualifications"] = [quer[qid]["text"][a:b]
                                        for a,b in observations[qid]["arms"]["dense"]["qualifications"]]
    output.mkdir(mode=0o700)
    files = [output / name for name in ("blind-review.jsonl", "private-mapping.jsonl")]
    write(files[0], forms, lines=True)
    write(files[1], mapping, lines=True)
    for arm, predictions_for_arm in predictions.items():
        path = output / f"{arm}-predictions.jsonl"
        write(path, predictions_for_arm, lines=True)
        files.append(path)
    write(output / "manifest.json", dict(queries=len(quer), candidate_views=len(forms),
            frame_sha256=digest(frame), lock_sha256=digest(lock_file),
            outputs_sha256={path.name: digest(path) for path in files}))
    validity = check(output, files[0])
    if any(set(v) != {"missing"} for v in validity["counts"].values()):
        raise ValueError("Expected a fresh blind review")
    for rev in ("a", "b"):
        write(output / f"reviewer-{rev}.jsonl", forms, lines=True)
    print(json.dumps(dict(groups=len(quer), blinded_views=len(forms), reviewer_judgments=0)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    commands = {
        prepare: ("panel","verified_summary","protocol","development","output"),
        materialize: ("lock_file","cohort_path","output"),
        packet: ("lock_file","frame","queries","dataset","trace","aggregate","output"),
    }
    for handler, names in commands.items():
        parser = sub.add_parser(handler.__name__)
        for name in names:
            parser.add_argument(name, type=Path)
    kw = vars(cli.parse_args())
    selected = kw.pop("command")
    {handler.__name__:handler for handler in commands}[selected](**kw)
