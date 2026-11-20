"""Count independent judgments on the enterprise evidence-extent audit."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from experiments.retrieval.check_enterprise_adaptive_review import indexed


LABELS = {"supported", "not_supported_by_excerpt", "uncertain"}
EDITABLE = {"support", "rationale", "reviewer_id"}
CATEGORIES = ("added_support", "no_support")


def records(path):
    with path.open(encoding="utf-8") as stream:
        return indexed(json.loads(line) for line in stream)


def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Completed judgments require a reviewer ID and rationale")
    return value.strip()


def load_submission(path, originals):
    if path is None:
        return None, {}
    rows = records(path)
    if rows.keys() != originals.keys():
        raise ValueError("Submission changed the case inventory")
    judges, labels = set(), {}
    for key, row in rows.items():
        original = originals[key]
        if row.keys() != original.keys() or any(row[f] != original[f] for f in row.keys() - EDITABLE):
            raise ValueError("Submission changed blinded source material")
        label = row["support"]
        if label is None:
            continue
        if label not in LABELS:
            raise ValueError("Unrecognized support label")
        nonempty(row["rationale"])
        judges.add(unicodedata.normalize("NFKC", nonempty(row["reviewer_id"])).casefold())
        labels[key] = label
    if len(judges) > 1:
        raise ValueError("Use one reviewer per submission")
    return next(iter(judges), None), labels


def score(packet, review_a, review_b, adjudication=None, categories=CATEGORIES):
    meta = json.loads((packet / "packet.json").read_text())
    for name in ("blind.jsonl", "mapping.jsonl"):
        observed = sha256((packet / name).read_bytes()).hexdigest()
        if meta["sha256"][name] != observed:
            raise ValueError("Original review packet changed")
    originals = records(packet / "blind.jsonl")
    mapping = records(packet / "mapping.jsonl")
    if originals.keys() != mapping.keys() or len(originals) != meta["cases"]:
        raise ValueError("The private mapping does not align with the reviewer packet")
    a, first = load_submission(review_a, originals)
    b, second = load_submission(review_b, originals)
    c, third = load_submission(adjudication, originals)
    judges = [person for person in (a, b, c) if person is not None]
    if len(judges) != len(set(judges)):
        raise ValueError("Use distinct reviewer and adjudicator IDs")
    totals = defaultdict(Counter)
    for key, row in mapping.items():
        category = row["category"]
        if category not in categories:
            raise ValueError("Unrecognized frozen sampling category")
        counts = totals[category]
        counts["cases"] += 1
        left, right = first.get(key), second.get(key)
        if key in third and (None in (left, right) or (left == right and left != "uncertain")):
            raise ValueError("Adjudication requires two disputed or uncertain judgments")
        if left is None or right is None:
            counts["missing_review"] += 1
            continue
        counts["jointly_reviewed"] += 1
        counts["raw_agreement"] += left == right
        resolved = third.get(key, left if left == right else None)
        if resolved is None:
            counts["unresolved_dispute"] += 1
        else:
            counts[resolved] += 1
    return {
        name: dict(totals[name]) | {
            "possible_supported_count": [
                totals[name]["supported"], totals[name]["cases"] - totals[name]["not_supported_by_excerpt"]
            ]
        } for name in categories
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "review_a", "review_b"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--category", dest="categories", action="append", default=None,
                        help="Expected private sampling category (repeat for each category)")
    args = parser.parse_args()
    print(json.dumps(score(args.packet, args.review_a, args.review_b, args.adjudication,
                           args.categories or CATEGORIES), indent=2))
