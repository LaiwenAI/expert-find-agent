"""Bound the attainable gain from choosing among frozen first-query routes."""

import argparse
import json
from pathlib import Path

from experiments.credit import role_first_query_factorial as factorial
from experiments.credit import role_first_query_router as router
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json

BASE = "topic_fixed"
LOCAL = (BASE, "person_fixed")
ACTIONS = (*LOCAL, "person_global_fixed")


def pointwise(rows, arms, constraint):
    metrics = dict(grounded=0, off_label=0, issued=0)
    switched = 0
    for row in rows:
        eligible = [arm for arm in arms if constraint(row[arm], row[BASE])]
        best = max(eligible, key=lambda arm: (row[arm]["grounded"], -row[arm]["off_label"]))
        switched += best != BASE
        for key in metrics:
            metrics[key] += row[best][key]
    return dict(metrics, switched=switched)


def pooled(rows, arms, off_label_cap):
    # State at one total off-label cost: the best attainable tag-backed count.
    frontier = {0: 0}
    for row in rows:
        following = {}
        for prior_cost, prior_gain in frontier.items():
            for arm in arms:
                cost = prior_cost + row[arm]["off_label"]
                if cost <= off_label_cap:
                    gain = prior_gain + row[arm]["grounded"]
                    following[cost] = max(gain, following.get(cost, -1))
        frontier = following
    cost = max(frontier, key=lambda c: (frontier[c], -c))
    return dict(grounded=frontier[cost], off_label=cost)


def compare(rows):
    totals = {
        arm: {key: sum(row[arm][key] for row in rows)
              for key in ("grounded", "off_label", "issued", "reads")}
        for arm in ACTIONS
    }
    baseline = totals[BASE]
    fixed = {}
    for universe, choices in (("local", LOCAL), ("including_global", ACTIONS)):
        only_better = lambda trial, ref: trial["off_label"] <= ref["off_label"]
        within_issuance = lambda trial, ref: trial["issued"] <= ref["issued"]
        fixed[universe] = dict(
            unconstrained=pointwise(rows, choices, lambda trial, ref: True),
            per_query_off_label=pointwise(rows, choices, only_better),
            per_query_issuance=pointwise(rows, choices, within_issuance),
            aggregate_off_label=pooled(rows, choices, baseline["off_label"]),
            queries_with_label_gain=sum(any(row[arm]["grounded"] > row[BASE]["grounded"]
                                            for arm in choices) for row in rows),
            queries_with_gain_at_matched_issuance=sum(any(
                row[arm]["grounded"] > row[BASE]["grounded"] and within_issuance(row[arm], row[BASE])
                for arm in choices
            ) for row in rows),
        )
    return dict(queries=len(rows), topical_and_role_routes=totals, hindsight=fixed)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("elife", "plos", "plos_manifest", "reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    prior = json.loads(args.reference.read_bytes())
    plos_pin = json.loads(args.plos_manifest.read_bytes())
    if digest(args.elife) != prior["projection_sha256"] or digest(args.plos) != plos_pin["projection_sha256"]:
        raise ValueError("Unexpected publisher projection")
    for directory, key in ((args.embedding, "embedding_sha256"),
                           (args.reranker / factorial.Verifier.name, "reranker_sha256")):
        if {name: digest(directory / name) for name in prior[key]} != prior[key]:
            raise ValueError("Unexpected model checkpoint")

    encoder, verifier = factorial.Encoder(args.embedding), factorial.Verifier(args.reranker)
    reports = {}
    for publisher in ("elife", "plos"):
        projection = getattr(args, publisher)
        catalog = pilot.Catalog(json.loads(projection.read_bytes())["sources"], encoder, verifier)
        _, cells = router.run_panels(catalog, publisher, prior["cutoff"], arms=ACTIONS)
        reports[publisher] = {name: compare(rows) for name, rows in cells.items()}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), replay_sha256=digest(Path(router.__file__)),
        reference_sha256=digest(args.reference), elife_sha256=digest(args.elife),
        plos_sha256=digest(args.plos), reports=reports,
    ))


if __name__ == "__main__":
    main()
