"""Build local, arm-blind DOI reference sheets from the provenance cache."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import quote

from experiments.retrieval.cof_public_provenance_audit import read_packet, titles_for


def markdown(text):
    return re.sub(r"([\\`*_[\]<>#|])", r"\\\1", " ".join(text.split()))


def lookup_cache(path):
    raw = path.read_bytes()
    # Capture appends one complete line at a time. An in-flight final line
    # belongs to a later snapshot.
    if not raw.endswith(b"\n"):
        raw = raw[:raw.rfind(b"\n") + 1]
    entries = {}
    for line in raw.splitlines():
        row = json.loads(line)
        status = row["result"]["status"]
        if status != "request_failed" or row["title"] not in entries:
            entries[row["title"]] = row["result"]
    return entries, sha256(raw).hexdigest()


def reference(title, entries):
    outcome = entries.get(title, {"status": "not_checked"})
    if outcome["status"] != "exact":
        states = dict(not_checked="lookup pending", no_exact_title="no exact title found",
                      request_failed="lookup failed", ambiguous="multiple exact records")
        return f"{markdown(title)} · {states[outcome['status']]}", outcome["status"]
    url = "https://doi.org/" + quote(outcome["doi"], safe="/")
    return f"{markdown(title)} · [DOI record]({url})", "exact"


def render(rows, entries):
    lines = [
        "# Publisher references for blind citation review", "",
        "Match a case code to the immutable review form. Links open public publication records. "
        "Profile papers can help identify a candidate. Contribution must be checked in the cited publication.", "",
    ]
    counts = Counter()
    for row in rows:
        headings = ("Citation", *["Profile paper"] * len(row["profile_titles"]))
        lines.extend([f"## {row['case']}", ""])
        hits = 0
        *profile, citation = titles_for(row)
        for heading, title in zip(headings, (citation, *profile), strict=True):
            source, status = reference(title, entries)
            counts[f"{heading.lower().replace(' ', '_')}_{status}"] += 1
            hits += heading == "Profile paper" and status == "exact"
            lines.append(f"- **{heading}** · {source}")
        counts["cases_with_two_exact_profile_links"] += hits >= 2
        lines.append("")
    return "\n".join(lines), dict(counts)


def run(packet, cache, output):
    panels = read_packet(packet)
    entries, snapshot = lookup_cache(cache / "crossref.jsonl")
    if not entries:
        raise ValueError("The bibliographic cache has no completed records")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    summaries = {}
    for panel, rows in panels.items():
        layout, counts = render(rows, entries)
        (output / f"{panel}.md").write_text(layout)
        summaries[panel] = dict(cases=len(rows), counts=counts)
    titles = {title for rows in panels.values() for row in rows for title in titles_for(row)}
    result = dict(cache_snapshot_sha256=snapshot, distinct_titles=len(titles),
                  lookup_records_available=len(titles & entries.keys()), panels=summaries)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "cache", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
