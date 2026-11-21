"""Inventory the effective proof space and withhold roles within known rows."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.credit.open_topic_role_bundles import annotation_postings, publisher_topics, query_panel
from experiments.credit.cross_source_identity import load


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def inventory(ownership, tags):
    people = defaultdict(set)
    for doi, paper in ownership.items():
        for author in paper["authors"]:
            if author["person"]:
                people[author["person"]].add(doi)
    edges = {pair for docs in people.values() for pair in combinations(sorted(docs), 2)}
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    unseen = set(graph)
    components = 0
    while unseen:
        queue = [unseen.pop()]
        while queue:
            neighbors = graph[queue.pop()] & unseen
            unseen -= neighbors
            queue.extend(neighbors)
        components += 1

    gold = annotation_postings(SimpleNamespace(ownership=ownership), tags)
    panel = query_panel(gold)
    supported_sets = []
    linked_load = Counter()
    tagged_pairs = set()
    for left, right in panel:
        witnesses = {(owner, a, b) for owner in gold[left].keys() & gold[right].keys()
                     for a, b in product(gold[left][owner], gold[right][owner]) if a != b}
        supported_sets.append(witnesses)
        linked_load.update({owner for owner, _, _ in witnesses})
        tagged_pairs.update(tuple(sorted((a, b))) for _, a, b in witnesses)
    if any(not witnesses for witnesses in supported_sets):
        raise ValueError("Constructed request is missing its structural reference")

    role_rows = {(a["person"], doi): tuple(sorted(a["roles"]))
                 for doi, paper in ownership.items() for a in paper["authors"] if a["person"]}
    stress = {}
    for mask in (50, 100):
        deleted = {}
        for (person, doc), roles in role_rows.items():
            code = sha256(f"{doc}:{person}".encode()).digest()
            if len(roles) >= 2 and int.from_bytes(code[:4], "big") % 100 < mask:
                deleted[person, doc] = roles[int.from_bytes(code[4:8], "big") % len(roles)]
        retained = sum(any(deleted.get((owner, a)) != left[1] and deleted.get((owner, b)) != right[1]
                           for owner, a, b in witnesses)
                       for (left, right), witnesses in zip(panel, supported_sets, strict=True))
        stress[mask] = dict(masked_nonempty_rows=len(deleted), trusted_retains=retained,
                            partial_unknown_retains=len(panel))
    return dict(sources=len(ownership), people_with_verified_ids=len(people),
                cross_source_people=sum(len(docs) >= 2 for docs in people.values()),
                linked_sources=len(graph), linked_components=components, byline_pairs=len(edges),
                topic_pairs=len({(a[0], b[0]) for a, b in panel}), requests=len(panel),
                tagged_witness_pairs=len(tagged_pairs), requests_per_eligible_person=sorted(linked_load.values()),
                within_row_mask=stress)


def run(plos, fixture, elife, reference, output):
    plos_rows = load([plos])
    recorded = json.loads((fixture / "manifest.json").read_bytes())["input_sources"]
    if {r["doi"]: r["source_sha256"] for r in recorded} != {key: r["source_sha256"] for key, r in plos_rows.items()}:
        raise ValueError("PLOS source snapshot changed")
    archive = json.loads(reference.read_bytes())["runs"][0]
    if digest(elife) != archive["elife_projection_sha256"]:
        raise ValueError("eLife projection changed")
    elife_rows = {r["doi"]: r for r in json.loads(elife.read_bytes())["sources"]}
    elife_tags = {doc: {" ".join(topic.casefold().split()) for topic in row["topics"]}
                  for doc, row in elife_rows.items()}
    stats = {"plos": inventory(plos_rows, publisher_topics([plos])), "elife": inventory(elife_rows, elife_tags)}
    if [x["requests"] for x in stats.values()] != [160, 51]:
        raise ValueError("Different query inventory")
    with output.open("x") as handle:
        json.dump(dict(panels=stats, reference_sha256=digest(reference), script_sha256=digest(Path(__file__))),
                  handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plos", "fixture", "elife", "reference", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
