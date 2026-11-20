"""Constrain same-author evidence assignments by the requested source separation."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_source_request as live
from experiments.credit.live_joint_role_postings import RolePostings
from experiments.credit.mixed_publisher_joint import merge, raw_identifiers
from experiments.credit.role_column_index import encode
from experiments.credit.role_pair_placement import DeclarationIndex
from experiments.followir.core17_reranking import digest, write_json


SHAPES = {"distinct_three": ((0, 1), (0, 2), (1, 2)), "one_distinct_pair": ((0, 1),)}
ARMS = ("byline", "role_exhaustive", "role_projected", "role_bounded")
QUERY_CAP = 12


def distinct(docs, conflicts):
    return all(docs[a] != docs[b] for a, b in conflicts)


def witness_queries(witnesses):
    atoms_by_person = defaultdict(list)
    for atom, people in witnesses.items():
        for person in people:
            atoms_by_person[person].append(atom)
    cases = set()
    for person, atoms in atoms_by_person.items():
        if len(set().union(*(witnesses[atom][person] for atom in atoms))) < 3:
            continue
        eligible = []
        for group in combinations(sorted(atoms), 3):
            if len({atom[0] for atom in group}) < 3 or len({atom[1] for atom in group}) < 3:
                continue
            choices = (witnesses[atom][person] for atom in group)
            if any(distinct(docs, SHAPES["distinct_three"]) for docs in product(*choices)):
                key = sha256(json.dumps(group).encode()).hexdigest()
                eligible.append((key, group))
        cases.update(group for _, group in sorted(eligible)[:QUERY_CAP])
    return sorted(cases)


def source_postings(postings, ranks, atoms, *, byline=False, limit=None):
    people = defaultdict(lambda: [[] for _ in atoms])
    for side, (rank, (_, role)) in enumerate(zip(ranks, atoms, strict=True)):
        for doc in rank:
            for owner, roles in postings[doc].items():
                if not byline and role not in roles:
                    continue
                earlier = people[owner][side]
                if limit is None or len(earlier) < limit[side]:
                    earlier.append(doc)
    return people


def proof_key(owner, docs, ranks):
    positions = [rank[doc] for rank, doc in zip(ranks, docs, strict=True)]
    return max(positions), sum(positions), owner, docs


def choose(people, ranks, conflicts):
    best = None
    combinations_seen = 0
    for owner, lists in people.items():
        for docs in product(*lists):
            combinations_seen += 1
            if not distinct(docs, conflicts):
                continue
            key = proof_key(owner, docs, ranks)
            if best is None or key < best:
                best = key
    return (None if best is None else (best[-2], best[-1])), combinations_seen


def verified(index, plan, ranks, atoms, conflicts):
    files = index.open_sources(plan)
    observed = {}
    for doc, contents in files.items():
        people = defaultdict(set)
        for author in contents["authors"]:
            if author["person"]:
                people[author["person"]].update(author["roles"])
        observed[doc] = people
    scoped = tuple({doc: score for doc, score in facet.items() if doc in observed} for facet in ranks)
    owners = source_postings(observed, scoped, atoms)
    return choose(owners, ranks, conflicts)[0]


def topical_fill(ranks):
    pooled = sorted(set().union(*ranks), key=lambda doc: (min(r.get(doc, float("inf")) for r in ranks), doc))
    return pooled[:len(ranks)]


def audit_known(atoms, ranks, witnesses, conflicts):
    options = []
    for atom, rank in zip(atoms, ranks, strict=True):
        options.append({person: sources.intersection(rank) for person, sources in witnesses[atom].items()})
    for person in options[0].keys() & options[1].keys() & options[2].keys():
        if any(distinct(t, conflicts) for t in product(*(facet[person] for facet in options))):
            return True
    return False


def evaluate(index, panels, depth, offset):
    views = RolePostings(index.catalogue)
    degree = Counter(owner for owners in views.byline.values() for owner in owners)
    columns = DeclarationIndex(views)
    total = len(encode(columns.ids, columns.authors, columns.labels, columns.masks))
    catalogue_sizes = {}
    projections = {}
    for name, conflicts in SHAPES.items():
        floor = 3 if len(conflicts) == 3 else 2
        projected = {
            doc: {owner: roles for owner, roles in owners.items() if degree[owner] >= floor}
            for doc, owners in views.personal.items()
        }
        authors = {doc: sorted(people) for doc, people in projected.items()}
        masks = {}
        for doc, people in projected.items():
            mask = 0
            for owner in people:
                mask |= columns.labels[owner, doc]
            masks[doc] = mask
        payload = encode(columns.ids, authors, columns.labels, masks)
        projections[name] = projected
        catalogue_sizes[name] = dict(eligible_sources=sum(bool(mask) for mask in masks.values()),
                                     projected_role_bytes=len(payload))
    results = {}
    for publisher, subjects, witness, queries in panels:
        results[publisher] = {}
        if not queries:
            continue
        for name, conflicts in SHAPES.items():
            limits = tuple(1 + sum(i in edge for edge in conflicts) for i in range(3))
            projected = projections[name]
            rank_keys = ("plan_ms", "request_ms")
            summaries = {arm: Counter() for arm in ARMS}
            timings = {arm: defaultdict(list) for arm in ARMS}
            known_reachable = 0
            for q, atoms in enumerate([queries[0], *queries]):
                initial = perf_counter()
                request = json.loads(live.request(atoms))
                ranks = index.rank([item["topic"] for item in request["requirements"]], depth=depth)
                rank_ms = (perf_counter() - initial) * 1000
                expected = {}
                shift = (q + offset) % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    clock = perf_counter()
                    if arm == "byline":
                        # The same exact limit also benefits the byline baseline.
                        candidates = source_postings(views.personal, ranks, atoms, byline=True, limit=limits)
                    elif arm == "role_exhaustive":
                        candidates = source_postings(views.personal, ranks, atoms)
                    else:
                        cap = limits if arm == "role_bounded" else None
                        candidates = source_postings(projected, ranks, atoms, limit=cap)
                    nominated, work = choose(candidates, ranks, conflicts)
                    if nominated is not None:
                        plan = list(dict.fromkeys(nominated[1]))
                    elif arm == "byline":
                        plan = topical_fill(ranks)
                    else:
                        plan = []
                    planned_ms = (perf_counter() - clock) * 1000
                    proof = verified(index, plan, ranks, atoms, conflicts) if plan else None
                    json.dumps({"matches": [proof] if proof else []}, separators=(",", ":"))
                    request_ms = (perf_counter() - clock) * 1000 + rank_ms
                    expected[arm] = (nominated, plan, proof)
                    if arm != "byline" and proof != nominated:
                        raise ValueError("The original declarations did not confirm the role index's proof")
                    if q:
                        known = bool(proof) and all(topic in subjects.get(doc, ())
                                                    for (topic, _), doc in zip(atoms, proof[1], strict=True))
                        summaries[arm].update(
                            known=known, issued=bool(proof), reads=len(plan), enumerated=work,
                            postings=sum(len(p) for v in candidates.values() for p in v),
                        )
                        for key, elapsed in zip(rank_keys, (planned_ms, request_ms), strict=True):
                            timings[arm][key].append(elapsed)
                if any(expected[arm] != expected["role_exhaustive"] for arm in ("role_projected", "role_bounded")):
                    raise ValueError("Projection changed the selected proof, fetch plan or output")
                if q:
                    known_reachable += audit_known(atoms, ranks, witness, conflicts)
            results[publisher][name] = dict(
                queries=len(queries), known_reachable=known_reachable, facet_limits=limits,
                metrics={arm: dict(**summaries[arm], **{
                    key: np.quantile(ms, [.5, .95]).tolist() for key, ms in timings[arm].items()
                }) for arm in ARMS},
            )
    return dict(full_role_bytes=total, projections=catalogue_sizes, panels=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--depth", type=int, default=128)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = live.verify_embedding(args.embedding)
    base = live.BASE
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if (checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.elife_manifest) != expected["elife_manifest_sha256"]
        or raw_identifiers(args.plos) & raw_identifiers(args.elife)):
        raise ValueError("Unexpected frozen inputs")
    plos_subjects = base.publisher_topics([args.plos])
    elife_subjects, elife_witness, _ = live.elife.build_requests(elife)["topics"]
    plos_witness = base.annotation_postings(plos, plos_subjects)
    panels = tuple((name, subjects, witness, witness_queries(witness))
                   for name, subjects, witness in (("plos", plos_subjects, plos_witness),
                                                    ("elife", elife_subjects, elife_witness)))
    print({name: len(panel) for name, _, _, panel in panels}, flush=True)
    if not any(queries for _, _, _, queries in panels):
        raise ValueError("No eligible three-condition queries")
    with TemporaryDirectory(dir=args.output.parent) as root:
        index = live.LocalIndex(merge(plos, elife), QwenTorchEmbedding(args.embedding), Path(root) / "mixed")
        results = evaluate(index, panels, args.depth, args.order_offset)
    write_json(args.output, dict(
        depth=args.depth, order_offset=args.order_offset, **results, gpu=torch.cuda.get_device_name(),
        embedding_checkpoint=checkpoint, elife_projection_sha256=projection,
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)),
        query_counts={name: len(panel) for name, _, _, panel in panels},
    ))


if __name__ == "__main__":
    main()
