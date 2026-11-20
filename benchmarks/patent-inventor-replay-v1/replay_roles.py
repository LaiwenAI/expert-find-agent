"""Replay two-condition inventor search with partial grant-roster availability."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
from pathlib import Path


def stable_number(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class Facet:
    documents: list[dict]
    visibility: int
    excluded: frozenset[str] = frozenset()
    next_hidden: int = 0
    support: dict[str, int] = field(default_factory=dict)
    hidden: list[tuple[int, dict]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for rank, document in enumerate(self.documents, start=1):
            if stable_number("availability-v1", document["patent_number"]) % 100 < self.visibility:
                self.observe(rank, document)
            elif document["patent_number"] not in self.excluded:
                self.hidden.append((rank, document))

    def observe(self, rank: int, document: dict) -> None:
        for inventor in document["inventors"]:
            self.support[inventor] = min(rank, self.support.get(inventor, rank))

    def next_rank(self) -> int:
        if self.next_hidden == len(self.hidden):
            return len(self.documents) + 1
        return self.hidden[self.next_hidden][0]

    def can_read(self) -> bool:
        return self.next_hidden < len(self.hidden)

    def read(self) -> None:
        rank, document = self.hidden[self.next_hidden]
        self.next_hidden += 1
        self.observe(rank, document)

    def score(self, rank: int) -> int:
        if rank > len(self.documents):
            return 0
        return 1_000_000 // (10 + rank)

    def optimistic(self, inventor: str) -> int:
        return self.score(min(self.next_rank(), self.support.get(inventor, len(self.documents) + 1)))


def confirmed(a: Facet, b: Facet) -> list[str]:
    joint = a.support.keys() & b.support.keys()
    return sorted(joint, key=lambda actor: (-a.score(a.support[actor]) - b.score(b.support[actor]), actor))


def certified(a: Facet, b: Facet, winners: list[str], k: int) -> bool:
    if not a.can_read() and not b.can_read():
        return True
    # Each winning citation must already be the highest-ranked possible citation.
    if any(a.support[actor] > a.next_rank() or b.support[actor] > b.next_rank() for actor in winners):
        return False
    cutoff = 0
    if len(winners) == k:
        last = winners[-1]
        cutoff = a.score(a.support[last]) + b.score(b.support[last])
    known = (a.support.keys() | b.support.keys()) - set(winners)
    for actor in known:
        upper_a, upper_b = a.optimistic(actor), b.optimistic(actor)
        if upper_a and upper_b and upper_a + upper_b >= cutoff:
            return False
    unseen_a, unseen_b = a.score(a.next_rank()), b.score(b.next_rank())
    return not (unseen_a and unseen_b and unseen_a + unseen_b >= cutoff)


def action_value(current: Facet, other: Facet, rank: int, cutoff: int) -> int:
    """Estimate how much a read could improve known cross-facet candidates."""
    next_score = current.score(rank)
    value = next_score  # Keep exploring when only completely unseen candidates remain.
    for actor, other_rank in other.support.items():
        old_score = current.score(current.support.get(actor, len(current.documents) + 1))
        combined = next_score + other.score(other_rank)
        if next_score > old_score and combined >= cutoff:
            value += next_score - old_score
    return value


def replay(documents: tuple[list[dict], list[dict]], visible: int, budget: int,
           policy: str, k: int = 5,
           excluded: tuple[frozenset[str], frozenset[str]] = (frozenset(), frozenset()),
           ) -> tuple[tuple[tuple[str, str, str], ...], int]:
    facets = (Facet(documents[0], visible, excluded[0]), Facet(documents[1], visible, excluded[1]))
    a, b = facets
    used = 0
    if policy in {"static", "static_certified"}:
        static_scores = [
            {
                rank: action_value(facet, facets[1 - index], rank, 0)
                for rank, _ in facet.hidden[:budget]
            }
            for index, facet in enumerate(facets)
        ]
    while used < budget:
        available = [index for index, facet in enumerate(facets) if facet.can_read()]
        if not available:
            break
        winners = confirmed(a, b)[:k]
        if policy in {"static_certified", "adaptive_certified"} and certified(a, b, winners, k):
            break
        if policy == "round_robin":
            selected = min(available, key=lambda index: (facets[index].next_hidden, index))
        elif policy in {"static", "static_certified"}:
            selected = max(available, key=lambda index: (
                static_scores[index][facets[index].next_rank()],
                -facets[index].next_rank(), -index,
            ))
        else:
            cutoff = 0
            if len(winners) == k:
                last = winners[-1]
                cutoff = a.score(a.support[last]) + b.score(b.support[last])
            selected = max(available, key=lambda index: (
                action_value(facets[index], facets[1 - index], facets[index].next_rank(), cutoff),
                -facets[index].next_rank(), -index,
            ))
        facets[selected].read()
        used += 1

    result = tuple(
        (actor, a.documents[a.support[actor] - 1]["patent_number"], b.documents[b.support[actor] - 1]["patent_number"])
        for actor in confirmed(a, b)[:k]
    )
    return result, used


def exhaustive(documents: tuple[list[dict], list[dict]]) -> tuple[tuple[str, str, str], ...]:
    populations: tuple[dict, dict] = ({}, {})
    for pool, facet in zip(populations, documents):
        for rank, row in enumerate(facet, start=1):
            for person in row["inventors"]:
                pool.setdefault(person, (1_000_000 // (10 + rank), row["patent_number"]))
    left, right = populations
    joint = set(left).intersection(right)
    ordered = sorted(joint, key=lambda person: (-(left[person][0] + right[person][0]), person))
    return tuple((person, left[person][1], right[person][1]) for person in ordered[:5])


def cohorts(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    groups = {"development": defaultdict(list), "held_out": defaultdict(list)}
    for row in rows:
        label = row["main_cpc_label"][:4]
        date = row["patent_issue_date"]
        if not label or not date or not row["inventors"]:
            raise ValueError(f"Missing experimental field on {row['patent_number']}")
        period = "development" if date < "2019-01-01" else "held_out"
        groups[period][label].append(row)
    for group in groups.values():
        for documents in group.values():
            documents.sort(key=lambda doc: (doc["patent_issue_date"], doc["patent_number"]), reverse=True)
    return groups


def fingerprints(groups: dict[str, list[dict]], bits: int) -> tuple[dict[str, set[int]], dict[str, set[int]], int]:
    grant_sketches = {}
    subclass_sketches = {}
    mask = (1 << bits) - 1
    for label, rows in groups.items():
        subclass = set()
        for row in rows:
            sketch = {stable_number("actor-v1", actor) & mask for actor in row["inventors"]}
            grant_sketches[row["patent_number"]] = sketch
            subclass.update(sketch)
        subclass_sketches[label] = subclass
    payload_bits = bits * sum(map(len, grant_sketches.values()))
    return grant_sketches, subclass_sketches, (payload_bits + 7) // 8


def measure(queries: list, visibility: int, budget: int, policy: str) -> dict[str, int]:
    hits = exact = exact_citations = reads = 0
    for documents, reference, excluded in queries:
        predicted, used = replay(documents, visibility, budget, policy, excluded=excluded)
        ref_ids = {row[0] for row in reference}
        predicted_ids = {row[0] for row in predicted}
        hits += len(predicted_ids & ref_ids)
        exact += int([row[0] for row in predicted] == [row[0] for row in reference])
        exact_citations += int(predicted == reference)
        reads += used
    return {"top5_hits": hits, "same_rank": exact, "same_rank_and_witness": exact_citations, "reads": reads}


def evaluate(rows: list[dict]) -> dict:
    result = {}
    policies = ("round_robin", "static", "static_certified", "adaptive", "adaptive_certified")
    for split, groups in cohorts(rows).items():
        labels = sorted(label for label, grants in groups.items() if len(grants) >= 12)
        pairs = [(a, b) for a, b in combinations(labels, 2) if a[:3] == b[:3]]
        cases = {}
        payloads = {}
        for bits in (0, 8, 12, 16):
            queries = []
            if bits:
                grant_sketches, subclass_sketches, payloads[str(bits)] = fingerprints(groups, bits)
            for first, second in pairs:
                documents = (groups[first], groups[second])
                excluded = (frozenset(), frozenset())
                if bits:
                    excluded = tuple(
                        frozenset(row["patent_number"] for row in facet
                                  if not grant_sketches[row["patent_number"]] & other_sketch)
                        for facet, other_sketch in zip(
                            documents, (subclass_sketches[second], subclass_sketches[first])
                        )
                    )
                queries.append((documents, exhaustive(documents), excluded))
            cases[str(bits)] = queries
        all_queries = cases["0"]
        panel = {}
        sketch_panel = {}
        for visible in (40, 70, 90):
            for budget in (0, 4, 12, 30):
                panel[f"{visible}pct_{budget}reads"] = {
                    policy: measure(all_queries, visible, budget, policy) for policy in policies
                }
                if budget:
                    sketch_panel[f"{visible}pct_{budget}reads"] = {
                        bits: {policy: measure(queries, visible, budget, policy) for policy in ("static", "adaptive")}
                        for bits, queries in cases.items() if bits != "0"
                    }
            for bits, queries in cases.items():
                exact_reads = {"static_certified": 0, "adaptive_certified": 0, "full_reads": 0}
                for documents, reference, excluded in queries:
                    cap = sum(map(len, documents))
                    baseline_reads = sum(
                        stable_number("availability-v1", row["patent_number"]) % 100 >= visible
                        for facet in documents for row in facet
                    )
                    exact_reads["full_reads"] += baseline_reads
                    for policy in ("static_certified", "adaptive_certified"):
                        prediction, used = replay(documents, visible, cap, policy, excluded=excluded)
                        if prediction != reference or used > baseline_reads:
                            raise AssertionError(f"Invalid termination: {split}, {visible}%, {bits}, {policy}")
                        exact_reads[policy] += used
                if bits == "0":
                    panel[f"{visible}pct_exact"] = exact_reads
                else:
                    sketch_panel.setdefault(f"{visible}pct_exact", {})[bits] = exact_reads
        result[split] = {
            "grants": sum(map(len, groups.values())),
            "queries": len(all_queries),
            "queries_with_reference_inventors": sum(bool(ref) for _, ref, _ in all_queries),
            "reference_top5_slots": sum(len(ref) for _, ref, _ in all_queries),
            "by_budget": panel,
            "sketch_by_budget": sketch_panel,
            "packed_sketch_payload_bytes": payloads,
            "packed_dense_id_payload_bytes": 2 * sum(len(row["inventors"]) for docs in groups.values() for row in docs),
        }
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("rosters", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    rows = [json.loads(line) for line in args.rosters.read_text().splitlines()]
    report = evaluate(rows)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({split: {key: value for key, value in results.items() if key in {"grants", "queries", "queries_with_reference_inventors", "reference_top5_slots"}} for split, results in report.items()}))
