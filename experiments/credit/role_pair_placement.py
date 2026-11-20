"""Evaluate byte-limited placement of same-person evidence across two sources."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import live_joint_role_postings as typed
from experiments.credit import owner_evidence_bound as replay
from experiments.credit import role_column_index as columns
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("hash_order", "identity_bridges", "query_frequency", "witness_frequency", "joint_coverage")
REFERENCE = "witness_frequency"
FOLDS = 5
BUDGETS = (15, 30, 50)
MAX_PAIRS_PER_QUERY = 8


class DeclarationIndex:
    def __init__(self, postings, *, bridge_only=False):
        self.docs = sorted(postings.byline)
        self.ids = {doi: i for i, doi in enumerate(self.docs)}
        occurrences = Counter(person for people in postings.byline.values() for person in people)
        self.authors = {
            doi: sorted(person for person in people if not bridge_only or occurrences[person] > 1)
            for doi, people in postings.byline.items()
        }
        self.eligible = {doi: set(people) for doi, people in self.authors.items()}
        self.role_bits = {role: 1 << i for i, role in enumerate(columns.ROLES)}
        self.labels = {}
        self.masks = {}
        for doi, people in self.authors.items():
            mask = 0
            for person in people:
                flags = sum(self.role_bits[role] for role in postings.personal[doi][person])
                self.labels[person, doi] = flags
                mask |= flags
            self.masks[doi] = mask
        self.costs = {
            doi: columns.HEADER.size + ((len(self.authors[doi]) + 7) // 8) * mask.bit_count()
            for doi, mask in self.masks.items() if mask
        }

    def materialize(self, selected, budget):
        raw = columns.encode(self.ids, self.authors, self.labels, {doi: self.masks[doi] for doi in selected})
        if len(raw) != sum(self.costs[doi] for doi in selected) or len(raw) > budget:
            raise AssertionError("Packed declarations violate the equal-byte cap")
        view = columns.decode(raw, self.docs, self.authors, self.masks)
        if any(bits != self.labels[person, doi] for doi, (_, authors) in view.items()
               for person, bits in authors.items()):
            raise AssertionError("The packed index changed the declared roles")
        return view, len(raw)

    def facet(self, rank, role, indexed):
        bit = self.role_bits[role]
        result = {}
        for doi in rank:
            if doi in indexed:
                _, authors = indexed[doi]
                result[doi] = ({person for person, flags in authors.items() if flags & bit},
                               {person for person, flags in authors.items() if not flags})
            else:
                result[doi] = (set(), self.eligible[doi])
        return result


def candidate_pairs(atoms, ranks, postings):
    """Use retrieval rank and original declared roles. Publisher topic labels are held out."""
    sides = []
    for (_, role), facet in zip(atoms, ranks, strict=True):
        candidates = defaultdict(list)
        for doi in facet:
            for person, roles in postings.personal[doi].items():
                if role in roles:
                    candidates[person].append(doi)
        sides.append(candidates)
    left, right = sides
    unique = {}
    for person in left.keys() & right.keys():
        for a, b in product(left[person], right[person]):
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            key = replay.proof_key((person, a, b), ranks)
            if pair not in unique or key < unique[pair]:
                unique[pair] = key
    return sorted(unique, key=unique.__getitem__)[:MAX_PAIRS_PER_QUERY]


def workload(rows, postings):
    occurrence = Counter()
    witnesses = Counter()
    bundles = defaultdict(int)
    for i, (atoms, ranks, _) in enumerate(rows):
        scores = {}
        for ranking in ranks:
            for doi, rank in ranking.items():
                scores[doi] = max(scores.get(doi, 0), 1 / rank)
        occurrence.update(scores)
        pairs = candidate_pairs(atoms, ranks, postings)
        witnesses.update({doi for pair in pairs for doi in pair})
        for pair in pairs:
            bundles[pair] |= 1 << i
    return occurrence, witnesses, bundles


def fill(order, costs, budget, selected):
    for doi in order:
        if doi not in selected and costs[doi] <= budget:
            selected.add(doi)
            budget -= costs[doi]
    return selected


def joined_placement(costs, bundles, order, budget):
    """Greedily buy the missing members of a pair that covers requests per byte."""
    selected = set()
    covered = 0
    while budget:
        best = None
        for pair, claims in bundles.items():
            cost = sum(costs[doi] for doi in pair if doi not in selected)
            marginal = (claims & ~covered).bit_count()
            if not marginal or cost > budget or not cost:
                continue
            key = (-marginal / cost, cost, pair)
            if best is None or key < best:
                best = key
        if best is None:
            break
        _, cost, pair = best
        selected.update(pair)
        budget -= cost
        for (a, b), claims in bundles.items():
            if a in selected and b in selected:
                covered |= claims
    return fill(order, costs, budget, selected)


def arms(costs, postings, occurrence, witnesses, bundles, budget):
    degree = Counter(person for people in postings.byline.values() for person in people)
    bridges = {doi: sum(degree[person] - 1 for person in postings.byline[doi]) for doi in costs}

    def by_weight(counts):
        return sorted(costs, key=lambda doi: (-counts.get(doi, 0) / costs[doi], costs[doi], doi))

    witness_order = by_weight(witnesses)
    orders = {
        "hash_order": sorted(costs, key=postings.order.__getitem__),
        "identity_bridges": by_weight(bridges),
        "query_frequency": by_weight(occurrence),
        "witness_frequency": witness_order,
    }
    result = {arm: fill(order, costs, budget, set()) for arm, order in orders.items()}
    result["joint_coverage"] = joined_placement(costs, bundles, witness_order, budget)
    return result


def topic_folds(atoms):
    return {int.from_bytes(sha256(topic.encode()).digest()[:4], "big") % FOLDS for topic, _ in atoms}


def report(outcomes, rows):
    result = {}
    for publisher in ("all", "plos", "elife"):
        ids = [i for i, (_, _, (group, _, _)) in enumerate(rows) if publisher == "all" or publisher == group]
        clusters = [json.dumps([topic for topic, _ in rows[i][0]]) for i in ids]
        reference = np.array([outcomes[REFERENCE][i]["known"] for i in ids], dtype=int)
        group = {}
        for arm, records in outcomes.items():
            chosen = [records[i] for i in ids]
            counts = Counter()
            for record in chosen:
                counts.update({key: value for key, value in record.items() if key != "ms"})
            differences = np.array([x["known"] for x in chosen], dtype=int) - reference
            interval = clustered_paired_interval(differences, clusters)
            interval["sampling_unit"] = "ordered topic pair; query-weighted mean"
            group[arm] = dict(counts, queries=len(ids), p95_ms=float(np.percentile([x["ms"] for x in chosen], 95)),
                              known_delta_vs_witness_frequency=interval)
        result[publisher] = group
    return result



def pick_sources(encoder, ranks, atoms, indexed, depth):
    facets = tuple(encoder.facet(rank, role, indexed) for (_, role), rank in zip(atoms, ranks, strict=True))
    pair = typed.first_pair(ranks, facets)
    return [entry[2] for entry in pair] if pair is not None else replay.live.simple_order(ranks, depth=depth)[0]


def exact_projection(index, rows, postings, depth, encoder, rotation):
    """Check which source IDs survive exact removal of single-source identities."""
    variants = {"full": encoder, "repeated_author_only": DeclarationIndex(postings, bridge_only=True)}
    views = {name: view.materialize(view.costs, sum(view.costs.values())) for name, view in variants.items()}
    report = {name: dict(bytes=size, documents=len(indexed), selection_ms=[], known=Counter())
              for name, (indexed, size) in views.items()}
    order = tuple(variants)
    for i, (atoms, ranks, (publisher, subjects, _)) in enumerate(rows):
        selections = {}
        for phase in range(2):
            name = order[(i + rotation + phase) % 2]
            start = perf_counter()
            selections[name] = pick_sources(variants[name], ranks, atoms, views[name][0], depth)
            report[name]["selection_ms"].append((perf_counter() - start) * 1000)
        if selections["full"] != selections["repeated_author_only"]:
            raise AssertionError("Pruning single-source authors changed the ordered read plan")
        plan = selections["full"]
        proof = replay.verified(plan, index.open_sources(plan), atoms, ranks)[1]
        known = bool(proof) and all(topic in subjects.get(doi, ())
                                   for (topic, _), doi in zip(atoms, proof[1:], strict=True))
        for entry in report.values():
            entry["known"][publisher] += known
    for entry in report.values():
        entry["selection_p95_ms"] = float(np.percentile(entry.pop("selection_ms"), 95))
    return dict(exact_plan_checks=len(rows), arms=report)


def evaluate(index, panels, depth, rotation):
    postings = typed.RolePostings(index.catalogue)
    encoder = DeclarationIndex(postings)
    rows = []
    for publisher, subjects, queries in panels:
        cache = replay.prepare(index, queries, depth)
        rows.extend((atoms, ranks, (publisher, subjects, ms))
                    for atoms, (ranks, ms) in zip(queries, cache, strict=True))
    masks = [topic_folds(atoms) for atoms, _, _ in rows]
    results = {}
    total_bytes = sum(encoder.costs.values())
    for percent in BUDGETS:
        budget = percent * total_bytes // 100
        outcomes = {arm: {} for arm in ARMS}
        allocations = []
        for fold in range(FOLDS):
            train = [row for row, mask in zip(rows, masks, strict=True) if fold not in mask]
            test = [i for i, mask in enumerate(masks) if fold == min(mask)]
            occurrence, witnesses, bundles = workload(train, postings)
            start = perf_counter()
            designs = arms(encoder.costs, postings, occurrence, witnesses, bundles, budget)
            allocation_ms = (perf_counter() - start) * 1000
            packed = {arm: encoder.materialize(selected, budget) for arm, selected in designs.items()}
            allocated = {}
            for arm, (view, payload_bytes) in packed.items():
                complete = 0
                for (a, b), claims in bundles.items():
                    if a in view and b in view:
                        complete |= claims
                allocated[arm] = dict(bytes=payload_bytes, indexed_sources=len(view),
                                      training_covered=complete.bit_count())
            allocations.append(dict(fold=fold, training_queries=len(train), test_queries=len(test),
                                    allocation_ms=allocation_ms, arms=allocated))
            for i in test:
                atoms, ranks, (_, subjects, base_ms) = rows[i]
                phase = (i + rotation) % len(ARMS)
                for arm in ARMS[phase:] + ARMS[:phase]:
                    indexed, _ = packed[arm]
                    start = perf_counter()
                    first = pick_sources(encoder, ranks, atoms, indexed, depth)
                    if len(set(first)) != 2:
                        raise AssertionError("Expected two distinct articles")
                    proof = replay.verified(first, index.open_sources(first), atoms, ranks)[1]
                    json.dumps({"matches": [proof] if proof else []})
                    ms = (perf_counter() - start) * 1000 + base_ms
                    known = bool(proof) and all(topic in subjects.get(doi, ())
                                                for (topic, _), doi in zip(atoms, proof[1:], strict=True))
                    outcomes[arm][i] = dict(known=known, issued=bool(proof), reads=len(first), ms=ms)
        if any(set(records) != set(range(len(rows))) for records in outcomes.values()):
            raise AssertionError("Every request needs exactly one held-out prediction")
        results[str(percent)] = dict(byte_cap=budget, allocations=allocations, results=report(outcomes, rows))
    exact = exact_projection(index, rows, postings, depth, encoder, rotation)
    return dict(full_index_bytes=total_bytes, measurements=results, exact_projection=exact)


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or args.output.exists():
        raise ValueError("Use a Slurm compute node and a fresh result file")
    live = replay.live
    base = replay.BASE
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]:
        raise ValueError("Changed embedding or eLife projection")
    if (
        digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.elife_manifest) != expected["elife_manifest_sha256"]
        or replay.raw_identifiers(args.plos) & replay.raw_identifiers(args.elife)
    ):
        raise ValueError("Changed publisher snapshots")
    plos_subjects = base.publisher_topics([args.plos])
    elife_subjects, _, elife_queries = live.elife.build_requests(elife)["topics"]
    panels = (
        ("plos", plos_subjects, base.query_panel(base.annotation_postings(plos, plos_subjects))),
        ("elife", elife_subjects, elife_queries),
    )
    with TemporaryDirectory(dir=args.output.resolve().parent) as directory:
        index = live.LocalIndex(replay.merge(plos, elife), QwenTorchEmbedding(args.embedding),
                                Path(directory) / "mixed")
        index.rank([topic for topic, _ in panels[0][2][0]], depth=args.depth)
        results = evaluate(index, panels, args.depth, args.order_offset)
    write_json(args.output, dict(depth=args.depth, order_offset=args.order_offset,
                                 embedding_checkpoint=checkpoint, fixture_projection=projection,
                                 input_manifest_sha256=expected["plos_manifest_sha256"],
                                 max_training_pairs_per_query=MAX_PAIRS_PER_QUERY, **results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--order-offset", type=int, default=0)
    run(parser.parse_args())
