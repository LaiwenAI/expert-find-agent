"""Assess one-wave exact pruning when indexed and optimistic winners differ."""

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
import torch

from experiments.credit import pre_read_proof_bound as bound
from experiments.credit import pre_read_proof_http as transport


def source_frontier(case):
    """Keep both sources of every pair that can beat the known incumbent."""
    facets = case["facets"]
    ranks = case["ranks"]
    incumbent = bound.best_bound(case["plan"], ranks, facets, False)
    ceiling = bound.proof_key(incumbent, ranks) if incumbent else None
    eligible = [{doc: known | unknown for doc, (known, unknown) in facet.items()} for facet in facets]
    selected = set()
    for left, possible_left in eligible[0].items():
        for right, possible_right in eligible[1].items():
            if left == right:
                continue
            common = possible_left & possible_right
            if common and (ceiling is None or bound.proof_key((min(common), left, right), ranks) <= ceiling):
                selected.update((left, right))
    return [doc for doc in case["plan"] if doc in selected]


def check(cases, files, reverse):
    outcomes, timings, signatures = Counter(), {"pre_read": [], "frontier": []}, {}
    ordered = list(enumerate(cases))
    for j, case in (reversed(ordered) if reverse else ordered):
        if bool(j % 2) == reverse:
            start = perf_counter_ns()
            prior, _ = transport.choose(case, "pre_read")
            timings["pre_read"].append(perf_counter_ns() - start)
            start = perf_counter_ns()
            selected = source_frontier(case)
            timings["frontier"].append(perf_counter_ns() - start)
        else:
            start = perf_counter_ns()
            selected = source_frontier(case)
            timings["frontier"].append(perf_counter_ns() - start)
            start = perf_counter_ns()
            prior, _ = transport.choose(case, "pre_read")
            timings["pre_read"].append(perf_counter_ns() - start)
        if len(selected) > len(prior):
            raise ValueError("The source frontier used more reads than the prior endpoint policy")
        loaded = {doc: {"authors": json.loads(files[case["urls"][doc]].read_bytes())}
                  for doc in selected}
        match, _ = bound.verify(loaded, case["atoms"], case["ranks"])
        if transport.serialize(match) != case["reference"]:
            raise ValueError("The source frontier changed the verified result")
        signatures[j] = (selected, prior, match)
        outcomes.update(
            requests=1, reference_reads=len(case["plan"]), fixed_two_reads=2, prior_reads=len(prior),
            frontier_reads=len(selected), additionally_pruned=int(len(selected) < len(prior)),
            zero_read_requests=int(not selected), issued=int(match is not None),
        )
    return dict(counts=outcomes,
                dispatch_p95_us={key: float(np.quantile(values, .95)/1e3) for key, values in timings.items()},
                decision_sha256=sha256(json.dumps(sorted(signatures.items()), sort_keys=True).encode()).hexdigest())


async def resolve(port, case, arm):
    if arm != "frontier":
        return await transport.resolve(port, case, arm)
    selected = source_frontier(case)
    if not selected:
        return transport.serialize(None), 0, 0
    loaded = await transport.fetch_docs(port, case, selected)
    match, _ = bound.verify(loaded, case["atoms"], case["ranks"])
    return transport.serialize(match), len(selected), 1


async def http_replay(paths, cases, expected, reverse):
    order = list(cases)
    Random(20261121).shuffle(order)
    arms = ("two_only", "pre_read", "frontier")
    rates = (165, 205)
    if reverse:
        order.reverse()
        arms = arms[::-1]
        rates = rates[::-1]
    results = {}
    for rate in rates:
        results[rate] = {}
        for arm in arms:
            row = await transport.trial(paths, order, rate, arm, resolver=resolve)
            if row["reads"] != expected[arm]:
                raise ValueError("HTTP source count disagrees with the local frontier comparison")
            results[rate][arm] = row
            print(f"{rate}/s {arm}: {row['reads']} reads, P95 {row['p95_ms']:.2f} ms", flush=True)
    return results


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require a fresh output on a Slurm GPU allocation")
    linked = bound.linked
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = linked.previous.live.verify_embedding(args.embedding)
    if checkpoint != reference["embedding"]:
        raise ValueError("Changed embedding checkpoint")
    outcomes, http = {}, None
    if args.http:
        transport.raise_file_descriptor_limit()
    with TemporaryDirectory(dir=args.output.parent) as directory:
        source_paths, partial_cases = {}, []
        for name in linked.PANELS:
            catalogue = linked.load(getattr(args, name), getattr(args, name + "_manifest"))
            paths, cases = transport.prepare(catalogue, args.embedding, Path(directory) / name, name)
            if args.http:
                source_paths.update(paths)
                partial_cases.extend(cases[50])
            outcomes[name] = {visibility: check(panel, paths, args.reverse) for visibility, panel in cases.items()}
            for visibility, result in outcomes[name].items():
                before = reference["reports"][f"{name}_{transport.DEPTH}"]["panels"][str(visibility)]
                counts = result["counts"]
                if (counts["prior_reads"], counts["reference_reads"], counts["issued"]) != (
                        before["pre_read"]["reads"], before["full"]["reads"], before["full"]["issued"]):
                    raise ValueError("Reference counts did not reproduce")
        if args.http:
            columns = {"two_only": "fixed_two_reads", "pre_read": "prior_reads", "frontier": "frontier_reads"}
            expected = {arm: sum(panel[50]["counts"][column] for panel in outcomes.values())
                        for arm, column in columns.items()}
            http = asyncio.run(http_replay(source_paths, partial_cases, expected, args.reverse))
    linked.prior.write_json(args.output, dict(
        reports=outcomes, http=http, depth=transport.DEPTH, reverse=args.reverse, embedding=checkpoint,
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=linked.prior.digest(Path(__file__)), reference_sha256=linked.prior.digest(args.reference),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("extended", "extended_manifest", "confirmation", "confirmation_manifest", "embedding",
                  "reference", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--http", action="store_true", help="Also replay the partial-role frontier on the source service.")
    run(parser.parse_args())
