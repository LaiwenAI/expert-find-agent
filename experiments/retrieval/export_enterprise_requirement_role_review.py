"""Create a local model-blind review queue for full-request clause roles."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.followir.core17_reranking import digest


SEED = "enterprise-requirement-role-review-v1"
LIMITS = {"cascade_only_prerequisite": 12, "full_only_prerequisite": 12,
          "other_role_disagreement": 12, "both_prerequisite": 12, "agreement_control": 12}


def stratum(full, cascade):
    prerequisite = "prior_qualification"
    if full == cascade:
        return "both_prerequisite" if full == prerequisite else "agreement_control"
    if cascade == prerequisite:
        return "cascade_only_prerequisite"
    if full == prerequisite:
        return "full_only_prerequisite"
    return "other_role_disagreement"


def export(development, trace, output):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(trace.read_bytes())
    if data["development_sha256"] != digest(development):
        raise ValueError("Changed development input")
    source = {row["id"]: row for row in map(json.loads, development.read_bytes().splitlines())}
    rows = data["traces"]["development"]
    if (len(source) != 80 or any(row["split"] != "development" for row in source.values())
            or {row["id"] for row in rows} != source.keys()):
        raise ValueError("Review queue requires only the paired 80-request development trace")

    strata = defaultdict(list)
    for row in rows:
        original = source[row["id"]]["text"]
        for i, span in enumerate(row["spans"]):
            full, cascade = (row["arms"][name]["labels"][str(i)] for name in ("full", "cascade"))
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(original) or not original[start:end].strip():
                raise ValueError("Invalid source excerpt")
            key = sha256(f"{SEED}:{row['id']}:{start}:{end}".encode()).hexdigest()
            strata[stratum(full, cascade)].append((key, row["id"], span, full, cascade))
    forms = []
    mapping = {}
    selected = Counter()
    for name, candidates in strata.items():
        seen = set()
        for key, identifier, span, full, cascade in sorted(candidates):
            if identifier in seen:
                continue
            seen.add(identifier)
            original = source[identifier]["text"]
            start, end = span["start"], span["end"]
            forms.append(dict(case_id=key[:24], start=start, end=end, target=original[start:end],
                              full_request=original, role=None, rationale=None, reviewer_id=None))
            mapping[key[:24]] = dict(request_id=identifier, stratum=name, full=full, cascade=cascade)
            selected[name] += 1
            if selected[name] >= LIMITS[name]:
                break
    if len(mapping) != len(forms):
        raise ValueError("Review identifiers collided")
    forms.sort(key=lambda row: sha256(f"{SEED}:order:{row['case_id']}".encode()).digest())
    output.mkdir(parents=True, exist_ok=False)
    review = output / "review.jsonl"
    review.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in forms))
    (output / "private-method-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (output / "instructions.md").write_text(
        "# Blinded request-clause review\n\n"
        "Read the entire requirement, then label the supplied excerpt by its role. "
        "Use other, future_work, prior_qualification, operating_condition, mixed, or uncertain. "
        "Use prior_qualification only when the text explicitly makes a past achievement, credential, "
        "or demonstrated experience a requirement of the applicant. Historical claims about other groups "
        "and targets for the proposed project need their own labels. Use operating_condition for an "
        "existing interface, operating environment, or system the work must accommodate. "
        "For independently completed copies, keep the source and offsets fixed and enter a rationale "
        "and reviewer identifier. Confirm disagreements with a separate adjudicator. Keep the queue local.\n"
    )
    summary = dict(cases=len(forms), source_requests=len({r["request_id"] for r in mapping.values()}),
                   selected_counts=dict(selected), available_counts={k:len(v) for k,v in strata.items()},
                   development_sha256=digest(development), trace_sha256=digest(trace),
                   review_sha256=digest(review), human_labels=0,
                   scope="Purposive model-disagreement and agreement audit for development. No population estimate.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "trace", "output"):
        parser.add_argument(field, type=Path)
    export(**vars(parser.parse_args()))
