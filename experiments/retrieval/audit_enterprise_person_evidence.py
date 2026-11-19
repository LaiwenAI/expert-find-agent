"""Count visible identity and contribution markers in private evidence views."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

from experiments.retrieval.enterprise_routed_review import digest, write


PREFIX_LENGTH = 800
ROLE_HEADER = re.compile(
    r"\b(?:author contributions?|credit authorship contribution statement)\s*[:\uff1a]"
    r"|(?:\u4f5c\u8005\u8d21\u732e|\u4f5c\u8005\u5206\u5de5)\s*[:\uff1a]",
    re.IGNORECASE,
)


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def inspect(text, name):
    return bool(name and name in normalized(text)), bool(ROLE_HEADER.search(text))


def catalog(dataset):
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    paths = [dataset / f"{name}.jsonl" for name in ("experts", "corpus")]
    for path in paths:
        if digest(path) != manifest["files_sha256"][path.name]:
            raise ValueError("Changed catalog snapshot")
    with paths[0].open(encoding="utf-8") as stream:
        people = {row["id"]: normalized(row["name"]) for line in stream
                  if (row := json.loads(line))["split"] == "development"}
    counts = defaultdict(Counter)
    with paths[1].open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "development":
                continue
            person = people[row["expert_id"]]
            prefix_name, prefix_role = inspect(row["text"][:PREFIX_LENGTH], person)
            full_name, full_role = inspect(row["text"], person)
            counts[row["kind"]].update({
                "sources": 1, "name_in_prefix": prefix_name, "name_in_full": full_name,
                "role_header_in_prefix": prefix_role, "role_header_in_full": full_role,
            })
    return dict(sorted(counts.items()))


def paper_metadata(dataset, sidecar):
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    papers = sidecar / "papers.jsonl"
    if digest(papers) != manifest["metadata_sha256"]:
        raise ValueError("Changed paper metadata sidecar")
    with (dataset / "corpus.jsonl").open(encoding="utf-8") as stream:
        inventory = {row["id"]: row["expert_id"] for line in stream if
                     (row := json.loads(line))["split"] == "development" and row["kind"] == "paper"}
    counts, matched, doi_values = Counter(), set(), set()
    with papers.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            source = row["source_id"]
            if source not in inventory:
                continue
            if source in matched or inventory[source] != row["expert_id"]:
                raise ValueError("Duplicate or inconsistent catalog link")
            matched.add(source)
            doi = row["doi"]
            if doi:
                doi_values.add(doi)
            counts.update({
                "with_doi": bool(doi), "with_author_list": bool(row["author_list_text"]),
                "with_doi_and_author_list": bool(doi and row["author_list_text"]),
                "with_verified_role": bool(row["contribution"]["verified_roles"]),
            })
    return dict(paper_sources=len(inventory), linked_sources=len(matched), distinct_dois=len(doi_values), **counts)


def packet(packet_directory):
    from experiments.retrieval.check_joint_review import check

    path = packet_directory / "blind-review.jsonl"
    verified = check(packet_directory, path)
    counts = defaultdict(Counter)
    cases = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            hits = []
            for passage in row["sources"]:
                name, header = inspect(passage["text"], normalized(row["candidate"]["name"]))
                hits.append(name)
                counts[passage["kind"]].update({
                    "view_occurrences": 1, "name_in_view": name, "role_header_in_view": header,
                })
            cases["cases"] += 1
            cases["any_name"] += any(hits)
            cases["all_name"] += all(hits)
    if cases["cases"] != verified["cases"]:
        raise ValueError("Incomplete review packet audit")
    return dict(cases=cases, source_view_occurrences=dict(sorted(counts.items())))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--packet", type=Path)
    cli.add_argument("--paper-metadata", type=Path)
    args = cli.parse_args()
    report = dict(prefix_characters=PREFIX_LENGTH, development_catalog=catalog(args.dataset))
    if args.packet:
        report["review_packet"] = packet(args.packet)
    if args.paper_metadata:
        report["development_paper_metadata"] = paper_metadata(args.dataset, args.paper_metadata)
    write(args.output, report)
