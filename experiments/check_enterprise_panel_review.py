"""Validate blind overlap judgments and summarize their remaining disagreements."""

import argparse
import json
from pathlib import Path

from experiments.freeze_enterprise_panel import digest_file, read_rows


FIELDS = {
    "intent": {"same", "different", "uncertain"},
    "organizational_link": {"linked", "distinct", "uncertain"},
}


def checked_review(template, path):
    rows = read_rows(path)
    orders = [row["case_id"] for row in rows]
    if orders != list(template):
        raise ValueError("Expected the unchanged case order and inventory")
    reviewers = set()
    for row in rows:
        frozen = template[row["case_id"]]
        if row.keys() != frozen.keys() or any(row[side] != frozen[side] for side in ("left", "right")):
            raise ValueError("Request text or reviewer fields changed")
        filled = any(row[field] is not None for field in FIELDS)
        if not filled:
            if row["reviewer"] is not None or row["rationale"] is not None:
                raise ValueError("Empty cases cannot carry reviewer metadata")
            continue
        for field, outcomes in FIELDS.items():
            if row[field] not in outcomes:
                raise ValueError("Both overlap questions require permitted judgments")
        if not all(isinstance(row[name], str) and row[name].strip() for name in ("reviewer", "rationale")):
            raise ValueError("A judged case requires an attributed rationale")
        reviewers.add(row["reviewer"].strip())
    if len(reviewers) > 1:
        raise ValueError("Use a separate review copy per reviewer")
    return rows, reviewers


def run(packet, first, second):
    summary = json.loads((packet / "summary.json").read_bytes())
    form = packet / "blind-review.jsonl"
    if digest_file(form) != summary["blind_form_sha256"]:
        raise ValueError("Blind form differs from its prospective hash")
    original = read_rows(form)
    template = {row["case_id"]: row for row in original}
    if len(template) != len(original) or len(template) != summary["blind_review_rows"]:
        raise ValueError("Repeated or changed audit case set")
    a, first_reviewers = checked_review(template, first)
    b, second_reviewers = checked_review(template, second)
    if first_reviewers & second_reviewers:
        raise ValueError("Review copies require separate reviewer identifiers")
    both = [pair for pair in zip(a, b, strict=True)
            if all(row["intent"] is not None for row in pair)]
    determinate = {
        field: [(x[field], y[field]) for x, y in both if "uncertain" not in (x[field], y[field])]
        for field in FIELDS
    }
    return dict(
        audit_cases=len(template),
        both_reviewed=len(both),
        requiring_adjudication_or_review=len(template) - sum(
            all(x[field] == y[field] and x[field] != "uncertain" for field in FIELDS) for x, y in both
        ),
        agreed_intent_duplicates=sum(x == y == "same" for x, y in determinate["intent"]),
        agreed_organizational_links=sum(x == y == "linked" for x, y in determinate["organizational_link"]),
        fields={field: dict(both_determinate=len(pairs), agreements=sum(a == b for a, b in pairs))
                for field, pairs in determinate.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "first", "second"):
        parser.add_argument(name, type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
