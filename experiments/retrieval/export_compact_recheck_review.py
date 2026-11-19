"""Prepare a locally blinded review of clause-role disagreements."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from experiments.followir.core17_reranking import digest
from experiments.retrieval.enterprise_requirement_roles import ROLES, spans


def export(queries, aggregate, disagreements, output):
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(aggregate.read_bytes())
    if digest(disagreements) != result["private_disagreements_sha256"]:
        raise ValueError("Private disagreements differ from their measured aggregate")
    rows = [{k: row[k] for k in ("id", "text", "split")}
            for row in map(json.loads, queries.read_text(encoding="utf-8").splitlines())
            if row["split"] == "development"]
    expected = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    if sha256(expected).hexdigest() != result["development_sha256"]:
        raise ValueError("Changed development projection")
    by_id = {row["id"]: row["text"] for row in rows}
    valid = {identifier: {(span["start"], span["end"]) for span in spans(text)}
             for identifier, text in by_id.items()}
    forms, mapping = [], {}
    for item in json.loads(disagreements.read_bytes()):
        identifier, start, end, labels = (item[k] for k in ("query_id", "start", "end", "labels"))
        if ((start, end) not in valid.get(identifier, ())
                or set(labels) != set(result["report"]["development"]["arms"])
                or len(set(labels.values())) < 2 or not set(labels.values()) <= set(ROLES)):
            raise ValueError("Mismatch in measured source anchors or roles")
        case = sha256(f"compact-review-v1:{identifier}:{start}:{end}".encode()).hexdigest()[:24]
        if case in mapping:
            raise ValueError("Repeated source anchor")
        text = by_id[identifier]
        forms.append(dict(case_id=case, start=start, end=end, target=text[start:end],
                          full_request=text, role=None, rationale=None, reviewer_id=None))
        mapping[case] = dict(request_id=identifier, labels=labels)
    forms.sort(key=lambda row: row["case_id"])
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    def write(name, text):
        path = output / name
        with path.open("x", encoding="utf-8") as handle:
            path.chmod(0o600)
            handle.write(text)
        return path

    review = write("review.jsonl", "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in forms))
    write("private-method-map.json", json.dumps(mapping, indent=2) + "\n")
    write("instructions.md",
          "# Blinded clause-role review\n\n"
          "Read the full request and label the highlighted target as other, future_work, prior_qualification, "
          "operating_condition, mixed, or uncertain. A prior qualification is an explicit condition about the "
          "applicant's existing credentials, experience, or past performance. A goal the applicant will work "
          "towards belongs under future_work. An existing interface, environment or system belongs under "
          "operating_condition. Keep source text and offsets fixed. Record a reviewer ID and rationale for each "
          "judgment. Use separate reviewer copies, resolve disagreements independently, and keep all files local.\n")
    summary = dict(
        cases=len(forms), requests=len({row["request_id"] for row in mapping.values()}), human_judgments=0,
        source_aggregate_sha256=digest(aggregate), source_disagreements_sha256=digest(disagreements),
        form_sha256=digest(review), scope="Complete measured disagreement queue on reused development data",
    )
    write("summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("queries", "aggregate", "disagreements", "output"):
        parser.add_argument(arg, type=Path)
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))
