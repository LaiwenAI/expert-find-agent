"""Freeze source- and identity-disjoint eLife contribution proof panels."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

from experiments.credit.linked_witness_dispatch import stable
from experiments.credit.multisource_proof_filter import owners_for
from experiments.followir.core17_reranking import digest, write_json


def publication_rows(path, manifest):
    expected = json.loads(manifest.read_bytes())
    if digest(path) != expected["projection_sha256"]:
        raise ValueError("Changed publisher projection")
    files = {item["filename"].split("-")[1]: item
             for item in expected["source_inventory"] if item["exclusion"] is None}
    result = {}
    for item in json.loads(path.read_bytes())["sources"]:
        if not item["declaration_complete"]:
            continue
        article_id = item["doi"].split(".")[-1]
        result[item["doi"]] = (
            {author["person"]: set(author["roles"]) for author in item["authors"] if author["person"]},
            files[article_id],
        )
    return result


def freeze(rows, limit):
    records = {doi: roles for doi, (roles, _) in rows.items()}
    by_owner = defaultdict(list)
    for article, authors in records.items():
        for person in authors:
            by_owner[person].append(article)
    candidates = {p for mentions in by_owner.values() for p in combinations(sorted(mentions), 2)}
    selected = []
    articles, people = set(), set()
    for pair in sorted(candidates, key=lambda pair: stable("conjunction-calibration", pair)):
        left, right = (records[paper] for paper in pair)
        identities = set(left) | set(right)
        if articles.intersection(pair) or people.intersection(identities):
            continue
        available = [set.union(*article.values()) for article in (left, right)]
        common = owners_for(records, pair)
        constructions = {(a, b) for owner in common for a in left[owner] for b in right[owner]
                         if a != b and b in available[0] and a in available[1]
                         and not any(b in left[p] and a in right[p] for p in common)}
        if constructions:
            options = sorted(constructions, key=lambda t: stable(pair, t))[:3]
            selected.append({"papers": pair, "bindings": options})
            articles.update(pair)
            people.update(identities)
        if len(selected) == limit:
            break
    return selected, {doi: rows[doi][1] for doi in sorted(articles)}, people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "confirmation_manifest", "extended", "extended_manifest", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cal = publication_rows(args.confirmation, args.confirmation_manifest)
    test = publication_rows(args.extended, args.extended_manifest)
    panels = {}
    calibration, files, people = freeze(cal, 30)
    expected = {doi: item for doi, item in test.items()
                if doi not in files and not people.intersection(item[0])}
    for name, (panel, sources, _) in (
        ("calibration", (calibration, files, people)), ("held", freeze(expected, 30))
    ):
        if not panel:
            raise ValueError(f"No source-disjoint pairs in {name}")
        panels[name] = dict(pairs=panel, articles=sources)
    write_json(args.output, dict(panels=panels, manifests=[digest(args.confirmation_manifest), digest(args.extended_manifest)]))
    print({name: (len(x["pairs"]), sum(len(p["bindings"]) for p in x["pairs"])) for name,x in panels.items()})


if __name__ == "__main__":
    main()
