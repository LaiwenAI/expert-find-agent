"""Build ownership-sensitive, cross-publication requirement edits from PLOS XML."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

from experiments.credit.counterfactual_role_edits import write_json
from experiments.credit.cross_source_identity import load


VERSION = "plos-credit-compositional-edits-v1"
CAP = 6
VIEWS = ("present", "left_withheld", "right_withheld", "different_author")
STATUSES = ("supported", "rejected", "unresolved")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def stable(*values):
    return sha256(json.dumps([VERSION, *values], ensure_ascii=True).encode()).hexdigest()


def people(sources):
    links = defaultdict(set)
    for doi, record in sources.items():
        for author in record["authors"]:
            if author["person"]:
                links[author["person"]].add(doi)
    return links


def components(sources, links):
    groups = {doi: {doi} for doi in sources}
    for members in links.values():
        merged = set().union(*(groups[doi] for doi in members))
        for doi in merged:
            groups[doi] = merged
    return {doi: min(group) for doi, group in groups.items()}


def eligible(sources, doi):
    rows = [a for a in sources[doi]["authors"] if a["roles"] and a["person"]]
    lookup = {a["person"]: a for a in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate contributor identifier in {doi}")
    return lookup


def generate(sources):
    links = people(sources)
    groups = components(sources, links)
    candidates = {pair for docs in links.values() for pair in combinations(sorted(docs), 2)}
    staged = []
    for pair in sorted(candidates):
        proposals = []
        for docs in (pair, pair[::-1]):
            left, right = (eligible(sources, doi) for doi in docs)
            common = sorted(left.keys() & right.keys())
            for person in common:
                a, b = left[person], right[person]
                # Require an identified coauthor who really holds the edited-in role.
                others = [row for key, row in right.items() if key != person]
                swaps = sorted({role for row in others for role in row["roles"]} - set(b["roles"]))
                for first, second, before, after in product(a["roles"], b["roles"], b["roles"], swaps):
                    if len({first, second, before, after}) != 4:
                        continue
                    def supports(key, role):
                        return first in left[key]["roles"] and {second, role}.issubset(right[key]["roles"])
                    if sum(supports(key, before) != supports(key, after) for key in common) != 1:
                        continue
                    comparator = next((key for key in common if key != person and supports(key, before)), None)
                    alternatives = [row for row in others if after in row["roles"]]
                    foil = min(alternatives, key=lambda row: (second not in row["roles"], row["ordinal"]))
                    group = stable(docs, a["ordinal"], b["ordinal"], first, second, before, after)
                    proposals.append(dict(
                        id=group, component=groups[docs[0]], sources=docs,
                        author_ordinals=[a["ordinal"], b["ordinal"]],
                        stable_ordinals=None if comparator is None else
                        [left[comparator]["ordinal"], right[comparator]["ordinal"]],
                        distractor_ordinal=foil["ordinal"],
                        requirements_before=[[0, first], [1, second], [1, before]],
                        requirements_after=[[0, first], [1, second], [1, after]],
                    ))
        proposals.sort(key=lambda row: (row["stable_ordinals"] is None, row["id"]))
        staged.extend(proposals[:CAP])
    blocks = sorted({row["component"] for row in staged}, key=stable)
    if len(blocks) < 3:
        raise ValueError("Need a publication component in each split")
    split = {key: ("development", "calibration", "test", "test", "test")[i % 5]
             for i, key in enumerate(blocks)}
    for row in staged:
        row["split"] = split[row["component"]]
    return staged


def make_queries(groups):
    rows = []
    for group in groups:
        for edit in ("before", "after"):
            requirements = group[f"requirements_{edit}"]
            for view in VIEWS:
                ordinals = list(group["author_ordinals"])
                if view == "different_author":
                    ordinals[1] = group["distractor_ordinal"]
                if view == "different_author" or (edit == "after" and view != "right_withheld"):
                    reference = "rejected"
                elif view == "present":
                    reference = "supported"
                else:
                    reference = "unresolved"
                rows.append(dict(group=group["id"], split=group["split"], track="flip", edit=edit,
                                 view=view, sources=group["sources"], ordinals=ordinals,
                                 requirements=requirements, reference=reference))
            if group["stable_ordinals"] is not None:
                rows.append(dict(group=group["id"], split=group["split"], track="stable", edit=edit,
                                 view="present", sources=group["sources"], ordinals=group["stable_ordinals"],
                                 requirements=requirements, reference="supported"))
    return rows


def decide(sources, query, method):
    records = [sources[doi]["authors"] for doi in query["sources"]]
    hidden = {"left_withheld": 0, "right_withheld": 1}.get(query["view"])
    authors = [record[ordinal-1] for record, ordinal in zip(records, query["ordinals"], strict=True)]
    identity = None if hidden is not None else authors[0]["person"] == authors[1]["person"]
    atoms = list(query["requirements"])
    # Edited role first produces an ablation for short-circuiting on a verified failure.
    if method == "edited_first":
        atoms.reverse()
    checks = []
    observations = []
    for slot, role in atoms:
        if slot == hidden:
            observations.append(None)
            continue
        checks.append(slot)
        available = records[slot] if method == "paper_pool" else [authors[slot]]
        present = any(role in a["roles"] for a in available)
        observations.append(present)
        if method == "edited_first" and not present:
            return "rejected", len(checks), len(set(checks))
    if identity is False or False in observations:
        status = "rejected"
    elif identity is None or None in observations:
        status = "unresolved"
    else:
        status = "supported"
    return status, len(checks), len(set(checks))


def report(sources, queries):
    rows = {}
    for method in ("paper_pool", "author_scoped", "edited_first"):
        counts = Counter()
        for query in queries:
            grade, checks, reads = decide(sources, query, method)
            expected = query["reference"]
            counts["queries"] += 1
            counts["correct"] += grade == expected
            counts["unsupported_promotions"] += grade == "supported" and grade != expected
            counts["missing_as_rejected"] += grade == "rejected" and expected == "unresolved"
            counts["role_checks"] += checks
            counts["role_sources_checked"] += reads
        rows[method] = dict(counts)
    return rows


def main(args):
    if args.replay:
        folder = args.replay
        manifest = json.loads((folder / "manifest.json").read_bytes())
        if manifest["version"] != VERSION or any(digest(folder / name) != sig for name, sig in manifest["files"].items()):
            raise ValueError("Changed benchmark fixture")
        sources = json.loads((folder / "sources.json").read_bytes())
        queries = json.loads((folder / "queries.json").read_bytes())
        with args.output.open("x") as stream:
            json.dump(report(sources, queries), stream, indent=2)
        return
    sources = load(args.sources)
    groups = generate(sources)
    queries = make_queries(groups)
    keep = set(doi for row in groups for doi in row["sources"])
    vocabulary = {key: source for key, source in sources.items() if key in keep}
    # Verify every generated label using the parsed declaration and identity semantics.
    for query in queries:
        if decide(vocabulary, query, "author_scoped")[0] != query["reference"]:
            raise AssertionError(f"Inconsistent constructed label in {query['group']}")
    args.output.mkdir(exist_ok=False, parents=True)
    payloads = {"sources.json": vocabulary, "groups.json": groups, "queries.json": queries}
    for name, payload in payloads.items():
        write_json(args.output / name, payload)
    manifest = dict(version=VERSION, input_sources=[
        {key: source[key] for key in ("doi", "url", "source_sha256")}
        for _, source in sorted(sources.items())
    ], files={name: digest(args.output / name) for name in payloads},
        groups=dict(Counter(g["split"] for g in groups)),
        queries=dict(Counter(q["split"] for q in queries)))
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="*")
    parser.add_argument("--replay", type=Path)
    main(parser.parse_args())
