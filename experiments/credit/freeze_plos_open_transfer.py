"""Freeze a local PLOS ONE catalog and disjoint exact-proof groups."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
from xml.etree import ElementTree as ET

from experiments.credit import held_open_proof_search as base
from experiments.credit.cross_source_identity import orcid_id, xml_sources
from experiments.credit.plos_selective_review_holdout import authors
from experiments.credit.role_first_proof_decoding import read

MAX_GROUPS = 60


def text(node):
    return "" if node is None else " ".join("".join(node.itertext()).split())


def recorded(metadata):
    return {person for author in metadata.findall("contrib-group/contrib[@contrib-type='author']")
            for node in author.findall("contrib-id[@contrib-id-type='orcid']")
            if (person := orcid_id(node.text or ""))}


def load(folders, exclude_people, permitted=None):
    projections, conflicts, checksums, skips = {}, set(), {}, Counter()
    for folder in sorted(folders):
        for file in xml_sources(folder):
            try:
                source = file.read_bytes()
                meta = ET.fromstring(source).find("./front/article-meta")
                doi = meta.findtext("article-id[@pub-id-type='doi']") if meta is not None else None
                if not doi or not doi.startswith("10.1371/journal.pone."):
                    continue
                digest = base.digest(file)
                if permitted is not None and permitted.get(doi) != digest:
                    continue
                if doi in conflicts:
                    continue
                if checksums.setdefault(doi, digest) != digest:
                    conflicts.add(doi)
                    projections.pop(doi, None)
                    skips["conflicting_xml"] += 1
                    continue
                if doi in projections:
                    continue
                identities = recorded(meta)
                if any(sha256(f"elife-role-search:{person}".encode()).hexdigest()[:24] in exclude_people
                       for person in identities):
                    skips["shares_prior_elife_identity"] += 1
                    continue
                declared = authors(source, doi, meta)
                title = text(meta.find("title-group/article-title"))
                body = text(meta.find("abstract"))
                indexed, clue = base.partition(f"{title}\n{body}")
            except (ValueError, ET.ParseError) as error:
                skips[str(error).split(":", 1)[0]] += 1
                continue
            # Papers without resolved owners still act as search distractors.
            projections[doi] = dict(indexed=indexed, clue=clue, roles={p: v["roles"] for p, v in declared.items()},
                                    recorded=sorted(identities), digest=digest)
    return projections, skips


def freeze(catalog, earlier):
    used_papers = set(earlier)
    used_people = {person for paper in used_papers if paper in catalog for person in catalog[paper]["recorded"]}
    authors_by_paper = defaultdict(list)
    for paper, data in sorted(catalog.items()):
        if paper in used_papers or used_people.intersection(data["recorded"]):
            continue
        for person in sorted(data["roles"]):
            authors_by_paper[person].append(paper)
    candidates = {pair for papers in authors_by_paper.values() for pair in combinations(papers, 2)}
    groups = []
    for pair in sorted(candidates, key=lambda pair: base.stable("plos-open-transfer-v1", pair)):
        people = set(catalog[pair[0]]["recorded"]) | set(catalog[pair[1]]["recorded"])
        if used_people.intersection(people) or used_papers.intersection(pair):
            continue
        left, right = (catalog[doc]["roles"] for doc in pair)
        witnesses = {roles for person in left.keys() & right.keys() for roles in product(left[person], right[person])}
        universe = [set(label for labels in side.values() for label in labels) for side in (left, right)]
        contrast = {(roles, roles[::-1]) for roles in witnesses if roles[0] != roles[1]
                    and roles[::-1] not in witnesses and all(role in labels for role, labels in
                                                           zip(roles[::-1], universe, strict=True))}
        if not contrast:
            # Also allow single-role replacements with a genuine declared
            # rival role and no linked contributor satisfying that edit.
            contrast = {(roles, rival) for roles in witnesses for rival in
                        ({(label, roles[1]) for label in universe[0]} |
                         {(roles[0], label) for label in universe[1]}) if rival not in witnesses}
        if not contrast:
            continue
        chosen = sorted(contrast, key=lambda row: base.stable(pair, row))[:3]
        groups.append(dict(papers=list(pair), contrasts=[[list(yes), list(no)] for yes, no in chosen]))
        used_people.update(people)
        used_papers.update(pair)
        if len(groups) == MAX_GROUPS:
            break
    return groups, len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("extended", type=Path)
    parser.add_argument("earlier_edits", type=Path)
    parser.add_argument("earlier_swaps", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("projection", type=Path)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.manifest.exists() or args.projection.exists():
        raise ValueError("Use fresh manifest and projection paths")
    elife = (args.confirmation, args.extended)
    prior_hashes = tuple(base.digest(path) for path in elife)
    if prior_hashes != base.INPUTS:
        raise ValueError("Changed eLife sources")
    earlier = set()
    for path in (args.earlier_edits, args.earlier_swaps):
        earlier.update(read(path)["panels"]["plos"]["articles"])
    exclude_people = {row["person"] for path in elife for article in read(path)["sources"]
                      for row in article["authors"] if row["person"]}
    restore = read(args.restore) if args.restore else None
    allowed = None if restore is None else restore["source_sha256"]
    catalog, skips = load(args.folders, exclude_people, allowed)
    groups, n = freeze(catalog, earlier)
    if not groups:
        raise ValueError("No new groups in this archive")
    projection = {doi: {field: item[field] for field in ("indexed", "clue", "roles")}
                  for doi, item in sorted(catalog.items())}
    base.write_json(args.projection, projection)
    manifest = dict(pairs=groups, initial_candidate_pairs=n, prior_papers_present=len(earlier & catalog.keys()),
                    source_sha256={doi: catalog[doi]["digest"] for doi in sorted(catalog)},
                    projection_sha256=base.digest(args.projection), elife_sha256=prior_hashes,
                    prior_plos_sha256=[base.digest(args.earlier_edits), base.digest(args.earlier_swaps)],
                    filter_counts=skips)
    if restore:
        for key in ("pairs", "source_sha256", "projection_sha256", "prior_plos_sha256"):
            if manifest[key] != restore[key]:
                raise ValueError(f"Frozen archive did not reproduce: {key}")
        manifest = restore
    base.write_json(args.manifest, manifest)


if __name__ == "__main__":
    main()
