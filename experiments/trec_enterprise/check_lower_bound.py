"""Audit the minimum expert-document checks needed to certify judged TREC outputs."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.certified_source_dispatch import dispatch
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load
from experiments.trec_enterprise.shared_source_budget import TRIALS, make_cases


ARMS = ("rank_first", "prefix_two", "prefix_four")


def lower_bound(candidates, orders, truth, count):
    """Minimum checks for exact ordered IDs and first qualifying citations.

    The oracle uses gold answers only to obtain a lower bound. Each label is
    an independent possible value until checked, even for shared documents.
    """
    confirmed = {}
    prefix = {}
    for expert in candidates:
        citations = []
        for i, doc in enumerate(orders[expert], 1):
            if truth[expert][doc] == 2:
                citations.append(doc)
                if len(citations) == count:
                    confirmed[expert] = tuple(citations)
                    prefix[expert] = i
                    break
        if len(confirmed) == TOP_K:
            break
    if len(confirmed) == TOP_K:
        boundary = candidates.index(next(reversed(confirmed))) + 1
    else:
        boundary = len(candidates)
    # A confirmed expert requires its canonical prefix. To eliminate an
    # unconfirmed expert, show there are fewer than count possibly good rows.
    checks = sum(prefix[e] if e in prefix else max(0, len(orders[e]) - count + 1)
                 for e in candidates[:boundary])
    ranking = list(confirmed.items())
    ranking.extend((e, ()) for e in candidates if e not in confirmed)
    return checks, ranking[:TOP_K]


def run(directory, output):
    if output.exists():
        raise FileExistsError(output)
    experts, documents = load(directory)
    result = defaultdict(list)
    for case in make_cases(experts, documents):
        _, labels, candidates, truth, _, orders, _, _ = case
        for count in (1, 2):
            floor, expected = lower_bound(candidates, orders, truth, count)
            observations = {}
            for method in ARMS:
                ranking, work = dispatch(case, "perfect", count, method)
                if ranking != expected or work["checks"] < floor:
                    raise AssertionError("Invalid ranked output or check lower bound")
                observations[method] = work
            supported = sum(labels[e] == 2 and bool(sources) for e, sources in expected) / TOP_K
            result[count].append((floor, supported, observations))

    report = {
        count: dict(
            cases=len(rows), minimum_checks_mean=mean(floor for floor, _, _ in rows),
            minimum_checks_p95=float(np.quantile([floor for floor, _, _ in rows], .95)),
            supported_p5=mean(label for _, label, _ in rows),
            arms={
                method: dict(
                    mean_checks=mean(arms[method]["checks"] for _, _, arms in rows),
                    mean_unique_reads=mean(arms[method]["reads"] for _, _, arms in rows),
                    mean_waves=mean(arms[method]["waves"] for _, _, arms in rows),
                    mean_excess_checks=mean(arms[method]["checks"] - floor for floor, _, arms in rows),
                    reaches_lower_bound=sum(arms[method]["checks"] == floor for floor, _, arms in rows),
                ) for method in ARMS
            },
        ) for count, rows in result.items()
    }
    write_json(output, dict(inputs=FILES, script_sha256=digest(Path(__file__)),
                            topics=len(experts), trials_per_topic=TRIALS, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
