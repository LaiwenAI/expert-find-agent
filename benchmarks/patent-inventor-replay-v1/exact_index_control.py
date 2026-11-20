"""Measure a compact exact-role index with optimum top-five witness reads."""

import argparse
from itertools import combinations
import json
from pathlib import Path

from replay_roles import cohorts, exhaustive, stable_number


def best_bundle(required: list[set[str]], budget: int) -> tuple[int, int]:
    """Maximize confirmed top-five slots and then minimize unique witness reads."""
    for count in range(len(required), -1, -1):
        feasible_costs = []
        for bundle in combinations(required, count):
            cost = len(set().union(*bundle))
            if cost <= budget:
                feasible_costs.append(cost)
        if feasible_costs:
            return count, min(feasible_costs)
    raise AssertionError("The empty bundle must always be feasible")


def evaluate(rows: list[dict]) -> dict:
    result = {}
    for split, groups in cohorts(rows).items():
        labels = sorted(label for label, docs in groups.items() if len(docs) >= 12)
        pairs = [(a, b) for a, b in combinations(labels, 2) if a[:3] == b[:3]]
        references = [exhaustive((groups[a], groups[b])) for a, b in pairs]
        actor_count = len({person for docs in groups.values() for row in docs for person in row["inventors"]})
        if actor_count > 65536:
            raise ValueError(f"Inventor set requires more than 16 bits in {split}")
        panel = {}
        for visible in (40, 70, 90):
            for budget in (0, 4, 12, 30):
                hits = exact = reads = 0
                for reference in references:
                    missing_witnesses = [
                        {grant for grant in (left, right)
                         if stable_number("availability-v1", grant) % 100 >= visible}
                        for _, left, right in reference
                    ]
                    accepted, used = best_bundle(missing_witnesses, budget)
                    hits += accepted
                    exact += int(accepted == len(reference))
                    reads += used
                panel[f"{visible}pct_{budget}reads"] = {
                    "top5_hits": hits, "same_rank_and_witness": exact, "reads": reads,
                }
        result[split] = {"distinct_dense_ids": actor_count, "by_budget": panel}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rosters", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rosters.read_text().splitlines()]
    args.output.write_text(json.dumps(evaluate(rows), indent=2) + "\n")
