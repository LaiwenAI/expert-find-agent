"""Build and validate source-only qualification blind-review forms."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval.enterprise_qualification_scope_audit import cases
from experiments.retrieval.enterprise_routed_review import digest, write
from experiments.retrieval.score_enterprise_requirement_role_review import load_forms, read_submission

SALT = "enterprise-qualification-scope-review-v1"


def build(private, reference, output):
    summary = json.loads(reference.read_bytes())
    if digest(private) != summary["private_sha256"]:
        raise ValueError("Changed frozen scope audit")
    audit = json.loads(private.read_bytes())
    build_packet(audit["cases"], audit["new_gate"], summary["sampled_development"]["excerpts"],
                 digest(reference), output)


def freeze(queries, trace, reference, controls, old_controls, output):
    trace_meta = json.loads(reference.read_bytes())
    if digest(trace) != trace_meta["trace_sha256"] or digest(queries) != trace_meta["queries_sha256"]:
        raise ValueError("Changed first-query source")
    _, development = cases(queries, trace, controls, old_controls)
    count = len({row["id"] for row in development})
    if count != trace_meta["multi_condition"]:
        raise ValueError("Changed source request inventory")
    build_packet(development, [None] * len(development), len(development), digest(reference), output)


def build_packet(excerpts, model_votes, expected_count, source_hash, output):
    forms, mapping = [], []
    for excerpt, decision in zip(excerpts, model_votes, strict=True):
        start, end = excerpt["start"], excerpt["end"]
        if excerpt["full_request"][start:end] != excerpt["target"]:
            raise ValueError("Mismatched request offsets")
        key = sha256(f"{SALT}:{excerpt['id']}:{start}:{end}".encode()).hexdigest()[:24]
        visible = {field: excerpt[field] for field in ("start", "end", "target", "full_request")}
        forms.append(dict(case_id=key, **visible, role=None, rationale=None, reviewer_id=None))
        mapping.append(dict(case_id=key, request=excerpt["id"], original_selected=excerpt["chosen"],
                            direct_gate=decision))
    if len(forms) != expected_count or len({r["case_id"] for r in forms}) != len(forms):
        raise ValueError("Frozen case inventory changed")
    forms.sort(key=lambda row: row["case_id"])
    mapping.sort(key=lambda row: row["case_id"])
    output.mkdir(mode=0o700)
    form = output / "review.jsonl"
    for name in ("review.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write(output / name, forms, lines=True)
    write(output / "private-method-map.jsonl", mapping, lines=True)
    (output / "instructions.md").write_text(
        "# Independent role review\n\n"
        "Read the complete request before classifying the marked excerpt. "
        "Use prior_qualification only for existing experience, credentials, or past "
        "results that the applicant explicitly needs to have. A figure promised "
        "for delivery, or measured by someone else, does not establish that role.\n\n"
        "Other labels are future_work, operating_condition, other, mixed, and uncertain. "
        "Use the inherited heading if it specifies who is required to do what. "
        "For each judgment, add a reviewer ID and a reason citing the original wording. "
        "Leave all original fields intact. Work from separate copies.\n"
    )
    write(output / "manifest.json", dict(
        cases=len(forms), original_form_sha256=digest(form), source_result_sha256=source_hash,
        private_mapping_sha256=digest(output / "private-method-map.jsonl"),
    ))


def validate(packet, first, second):
    manifest = json.loads((packet / "manifest.json").read_bytes())
    source = packet / "review.jsonl"
    if digest(source) != manifest["original_form_sha256"]:
        raise ValueError("Changed review template")
    originals = load_forms(source)
    if len(originals) != manifest["cases"]:
        raise ValueError("Changed review case count")
    first_id, a = read_submission(first, originals)
    second_id, b = read_submission(second, originals)
    if first_id is not None and first_id == second_id:
        raise ValueError("Independent reviewer IDs are required")
    shared = a.keys() & b.keys()
    unresolved = {key for key in shared if a[key] != b[key] or a[key] in ("uncertain", "mixed")}
    print(json.dumps(dict(cases=len(originals), labels_a=len(a), labels_b=len(b), scored_by_both=len(shared),
                          with_disagreement_or_uncertainty=len(unresolved),
                          awaiting_two_judgments=len(originals) - len(shared))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for command, args in ((freeze, ("queries", "trace", "reference", "controls", "old_controls", "output")),
                          (build, ("private", "reference", "output")),
                          (validate, ("packet", "first", "second"))):
        sub = commands.add_parser(command.__name__)
        for field in args:
            sub.add_argument(field, type=Path)
        sub.set_defaults(command=command)
    options = vars(parser.parse_args())
    options.pop("mode")
    options.pop("command")(**options)
