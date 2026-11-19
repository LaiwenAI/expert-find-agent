"""Score independently reviewed requirement roles on the frozen development packet."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from experiments.retrieval.check_enterprise_adaptive_review import digest, indexed
from experiments.retrieval.enterprise_requirement_roles import ROLES
from experiments.retrieval.export_enterprise_requirement_role_review import SEED, stratum


ARMS = ("full", "local", "cascade")
LABELS = set(ROLES) | {"mixed", "uncertain"}
MUTABLE = {"role", "rationale", "reviewer_id"}
QUALIFICATION = "prior_qualification"


def load_forms(path):
    return indexed(json.loads(line) for line in path.read_text().splitlines())


def require_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Completed forms require a reviewer ID and rationale")
    return value.strip()


def read_submission(path, originals):
    if path is None:
        return None, {}
    rows = load_forms(path)
    if rows.keys() != originals.keys():
        raise ValueError("Submission changed the frozen case set")
    labels = {}
    reviewers = set()
    for key, row in rows.items():
        source = originals[key]
        if row.keys() != source.keys() or any(row[field] != source[field] for field in source.keys() - MUTABLE):
            raise ValueError("Submission changed frozen source text or offsets")
        role = row["role"]
        if role is None:
            continue
        if not isinstance(role, str) or role not in LABELS:
            raise ValueError("Invalid role label")
        require_text(row["rationale"])
        reviewer = unicodedata.normalize("NFKC", require_text(row["reviewer_id"])).casefold()
        reviewers.add(reviewer)
        labels[key] = role
    if len(reviewers) > 1:
        raise ValueError("Use one reviewer per file")
    return next(iter(reviewers), None), labels


def frozen_predictions(packet, trace):
    baseline = packet / "review.jsonl"
    manifest = json.loads((packet / "summary.json").read_text())
    if digest(baseline) != manifest["review_sha256"] or digest(trace) != manifest["trace_sha256"]:
        raise ValueError("Frozen form or source trace hash differs")
    originals = load_forms(baseline)
    if not originals or len(originals) != manifest["cases"]:
        raise ValueError("Frozen case count differs")
    raw = json.loads(trace.read_bytes())
    if raw["development_sha256"] != manifest["development_sha256"]:
        raise ValueError("Trace and packet reference different development data")
    predictions = {}
    strata = {}
    for request in raw["traces"]["development"]:
        for index, span in enumerate(request["spans"]):
            key = sha256(f"{SEED}:{request['id']}:{span['start']}:{span['end']}".encode()).hexdigest()[:24]
            if key not in originals:
                continue
            if key in predictions:
                raise ValueError("Duplicate trace match")
            source = originals[key]
            start, end = span["start"], span["end"]
            if (source["start"], source["end"], source["target"]) != (start, end, source["full_request"][start:end]):
                raise ValueError("Trace offsets do not match the review excerpt")
            labels = {arm: request["arms"][arm]["labels"][str(index)] for arm in ARMS}
            if not set(labels.values()) <= set(ROLES):
                raise ValueError("Invalid model role in trace")
            predictions[key] = labels
            strata[key] = stratum(labels["full"], labels["cascade"])
    if originals.keys() != predictions.keys() or Counter(strata.values()) != manifest["selected_counts"]:
        raise ValueError("Trace does not match packet selection")
    return originals, predictions, strata


def summarize(predictions, resolved):
    def bounds(compute):
        endpoints = []
        for key, labels in predictions.items():
            gold = (resolved[key],) if key in resolved else ROLES
            values = [compute(labels, role) for role in gold]
            endpoints.append((min(values), max(values)))
        return [sum(low for low, _ in endpoints), sum(high for _, high in endpoints)]

    return {
        "cases": len(predictions),
        "resolved": len(resolved),
        "arms": {
            arm: {
                "correct_role_count": bounds(lambda labels, gold: labels[arm] == gold),
                "false_qualification_count": bounds(
                    lambda labels, gold: labels[arm] == QUALIFICATION and gold != QUALIFICATION),
                "missed_qualification_count": bounds(
                    lambda labels, gold: labels[arm] != QUALIFICATION and gold == QUALIFICATION),
            }
            for arm in ARMS
        },
        "paired_correct_role_difference_vs_full": {
            arm: bounds(lambda labels, gold: int(labels[arm] == gold) - int(labels["full"] == gold))
            for arm in ARMS[1:]
        },
    }


def score(packet, trace, review_a=None, review_b=None, adjudication=None):
    originals, predictions, strata = frozen_predictions(packet, trace)
    reviewer_a, first = read_submission(review_a, originals)
    reviewer_b, second = read_submission(review_b, originals)
    adjudicator, third = read_submission(adjudication, originals)
    reviewers = [person for person in (reviewer_a, reviewer_b, adjudicator) if person is not None]
    if len(set(reviewers)) != len(reviewers):
        raise ValueError("File reviewers and adjudicator must have distinct IDs")

    paired = first.keys() & second.keys()
    disputed = {key for key in paired if first[key] != second[key] or first[key] not in ROLES}
    if not third.keys() <= disputed:
        raise ValueError("Adjudication requires two submitted judgments and an unresolved role")
    resolved = {key: first[key] for key in paired - disputed}
    resolved.update({key: role for key, role in third.items() if role in ROLES})
    by_stratum = {}
    for name in sorted(set(strata.values())):
        selected = {key: predictions[key] for key in predictions if strata[key] == name}
        by_stratum[name] = summarize(selected, {key: resolved[key] for key in selected if key in resolved})
    return {
        "reviewed_by_both": len(paired),
        "exact_human_agreement": len(paired) - sum(first[key] != second[key] for key in paired),
        "needs_adjudication_or_uncertain": len(disputed - resolved.keys()),
        "packet": summarize(predictions, resolved),
        "strata": by_stratum,
        "scope": ("Descriptive count bounds on a purposive development packet, allowing unresolved cases to take any "
                  "of the four scored roles. Mixed and uncertain remain unresolved. Use the paired difference to "
                  "compare arms on the same cases. No population estimates or confidence intervals."),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("trace", type=Path)
    for name in ("review-a", "review-b", "adjudication"):
        parser.add_argument(f"--{name}", type=Path)
    print(json.dumps(score(**vars(parser.parse_args())), indent=2))
