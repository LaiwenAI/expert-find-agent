"""Propose issuer-alias pairs for local organizational review."""

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path

from experiments.audit_enterprise_panel_neighbors import checked_records, make_packet, write_private
from experiments.freeze_enterprise_panel import digest_file, issuer_key


def close(left, right):
    if min(len(left), len(right)) < 6:
        return 0.
    if left in right or right in left:
        return (1. + min(len(left), len(right)) / max(len(left), len(right))) / 2
    if abs(len(left) - len(right)) > 2:
        return 0.
    if len(set(left) & set(right)) < min(len(set(left)), len(set(right))) - 1:
        return 0.
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def run(workbook, cohort, legacy, registry, panel, output, exposure):
    if output.exists():
        raise FileExistsError(output)
    data, representatives, prior, anchor = checked_records(workbook, cohort, legacy, registry, panel, exposure)
    rows = [data[key] for key in representatives + prior]
    names = [issuer_key(row["company"]) for row in rows]
    n = len(representatives)
    scopes = {"within_panel": [], "prior_use": []}
    for i in range(n):
        prior_scores = [close(names[i], previous) for previous in names[n:]]
        candidate = max(range(len(prior_scores)), key=prior_scores.__getitem__)
        scopes["prior_use"].append((i, n + candidate, prior_scores[candidate]))
        panel_scores = [close(names[i], name) if i != j else 0. for j, name in enumerate(names[:n])]
        candidate = max(range(n), key=panel_scores.__getitem__)
        scopes["within_panel"].append((i, candidate, panel_scores[candidate]))
    forms, mapping, report = make_packet(rows, scopes)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_private(output / "blind-review.jsonl", forms)
    write_private(output / "private-mapping.jsonl", mapping)
    summary = dict(
        audited_index_sha256=anchor, script_sha256=digest_file(Path(__file__)),
        selected_groups=n, prior_use_requests=len(prior), diagnostic=report,
        blind_review_rows=len(forms), human_reviews=0,
        blind_form_sha256=digest_file(output / "blind-review.jsonl"),
        private_mapping_sha256=digest_file(output / "private-mapping.jsonl"),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Local blind issuer-alias review pairs:", len(forms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "cohort", "legacy", "registry", "panel", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--exposure", type=Path, action="append", required=True)
    run(**vars(parser.parse_args()))
