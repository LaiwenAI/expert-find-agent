"""Compare source-prefetch schedules with live, attributed CRediT judgments."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.constraint_verification import resolve_constraints
from expert_search.statistics import clustered_paired_interval
from experiments.credit.constraint_certificate import VIEWS, candidates
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("typed", "exhaustive", "shortest_first", "refutation_first", "prefetch_four")
TOP = 3
MAX_FULL_BATCH = 16
SCOPE = "either_publication"
QUESTION = "Does the target author's declaration explicitly list the requested contributor role?"
TASK = "Verify exact role membership in this author's published declaration. Do not infer undeclared roles."


def declaration_map(sources, query, owners):
    visible = {(expert, slot) for expert, requirements in owners
               for witnesses in requirements for slot in witnesses}
    return {
        (author["person"], slot): author["roles"]
        for slot, doi in enumerate(query["sources"])
        for author in sources[doi]["authors"]
        if (author["person"], slot) in visible
    }


def assessment_order(owners):
    order = {}
    for expert, requirements in owners:
        share = Counter(source for witnesses in requirements for source in witnesses)
        criteria = sorted(enumerate(requirements),
                          key=lambda item: (len(item[1]), -share[item[1][0]] if item[1] else 0))
        for criterion, witnesses in criteria:
            for slot in witnesses:
                order.setdefault((expert, slot), []).append(criterion)
    return order


def evaluate(arm, reader, query, owners, lookup, declarations, role_in_query):
    start = perf_counter()
    calls = scores = 0
    observed = {}
    false_positives = false_negatives = negative = positive = 0
    if arm == "typed":
        answer = resolve_constraints(owners, lookup, top=TOP, strategy="shortest_first")
        return dict(ranking=answer["ranking"], judged=answer["reads"], used=answer["reads"], scores=0, calls=0,
                    false_positives=0, false_negatives=0, negative=0, positive=0,
                    ms=(perf_counter() - start) * 1000)

    tasks = assessment_order(owners)
    queue = iter(tasks)

    def read_many(batch):
        nonlocal calls, scores
        jobs = [(key, criterion) for key in batch for criterion in tasks[key]]
        inputs = []
        for key, criterion in jobs:
            claim = f"Requested contributor role: {query['requirements'][criterion][1]}"
            evidence = f"Target author's complete declaration: {', '.join(declarations[key])}"
            pair = (f"{QUESTION}\n{claim}", evidence) if role_in_query else (QUESTION, f"{claim}\n{evidence}")
            inputs.append(pair)
        margins, clipped = reader.score_pairs(inputs, task=TASK)
        if clipped or not all(math.isfinite(margin) for margin in margins):
            raise ValueError("Invalid short source assessment")
        for (key, criterion), margin in zip(jobs, margins, strict=True):
            observed.setdefault(key, {})[criterion] = margin > 0
        calls += 1
        scores += len(jobs)

    def read(expert, slot):
        key = (expert, slot)
        if key not in observed:
            batch = [key]
            if arm == "prefetch_four":
                for pending in queue:
                    if pending != key and pending not in observed:
                        batch.append(pending)
                        if len(batch) == 4:
                            break
            read_many(batch)
        return observed[key]

    if arm == "exhaustive":
        keys = list(tasks)
        for start_at in range(0, len(keys), MAX_FULL_BATCH):
            read_many(keys[start_at:start_at + MAX_FULL_BATCH])
    strategy = "refutation_first" if arm == "refutation_first" else "shortest_first"
    result = resolve_constraints(owners, read, top=TOP, strategy=strategy)
    elapsed = (perf_counter() - start) * 1000
    for key, values in observed.items():
        gold = lookup(*key)
        for criterion, prediction in values.items():
            expected = gold[criterion]
            false_positives += prediction and not expected
            false_negatives += not prediction and expected
            negative += not expected
            positive += expected
    return dict(ranking=result["ranking"], judged=len(observed), used=result["reads"],
                scores=scores, calls=calls, ms=elapsed, false_positives=false_positives,
                false_negatives=false_negatives, positive=positive, negative=negative)


def summarize(rows):
    result = {}
    for arm in ARMS:
        trials = [row["arms"][arm] for row in rows]
        result[arm] = dict(
            mean_reads=mean(arm["judged"] for arm in trials),
            mean_committed_reads=mean(arm["used"] for arm in trials),
            mean_calls=mean(arm["calls"] for arm in trials),
            total_scores=sum(arm["scores"] for arm in trials),
            p50_ms=float(np.median([arm["ms"] for arm in trials])),
            p95_ms=float(np.quantile([arm["ms"] for arm in trials], .95)),
            exact_ranked_citations=sum(
                result["ranking"] == row["arms"]["typed"]["ranking"] for row, result in zip(rows, trials, strict=True)
            ),
            matches_batched_model=sum(
                result["ranking"] == row["arms"]["exhaustive"]["ranking"] for row, result in zip(rows, trials, strict=True)
            ),
            returned_experts=sum(len(result["ranking"]) for result in trials),
            unsupported_ranked_citations=sum(result["unsupported"] for result in trials),
            criterion_counts={field: sum(result[field] for result in trials)
                              for field in ("false_positives", "false_negatives", "positive", "negative")},
        )
    return result


def run(fixture, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a fresh output path and a Slurm GPU")
    manifest = json.loads((fixture / "manifest.json").read_bytes())
    files = [fixture / f"{name}.json" for name in ("sources", "queries", "groups")]
    if any(digest(path) != manifest["files"][path.name] for path in files):
        raise ValueError("Changed source-scoped input")
    sources, queries, groups = [json.loads(path.read_bytes()) for path in files]
    components = {row["id"]: row["component"] for row in groups}
    cases = [row for row in queries if row["split"] == "test" and row["track"] == "flip" and row["view"] in VIEWS]
    reader = Reranker(checkpoint)
    if (reader.provenance["repository"] != "Qwen/Qwen3-Reranker-0.6B"
            or reader.provenance["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473"):
        raise ValueError("Unexpected frozen reranker")
    rows = {False: [], True: []}
    warmup = next(row for row in cases if row["view"] == "present")
    owners, read = candidates(sources, warmup, warmup["view"], SCOPE)
    declarations = declaration_map(sources, warmup, owners)
    for role_in_query in rows:
        for arm in ARMS:
            evaluate(arm, reader, warmup, owners, read, declarations, role_in_query)
    for i, query in enumerate(cases):
        owners, read = candidates(sources, query, query["view"], SCOPE)
        declarations = declaration_map(sources, query, owners)
        offset = i % len(ARMS)
        for mode in (bool(i % 2), not bool(i % 2)):
            results = {arm: evaluate(arm, reader, query, owners, read, declarations, mode)
                       for arm in ARMS[offset:] + ARMS[:offset]}
            for result in results.values():
                result["unsupported"] = sum(
                    len(witnesses) != len(query["requirements"])
                    or not all(read(expert, source).get(criterion) is True
                               for criterion, source in enumerate(witnesses))
                    for expert, witnesses in result["ranking"]
                )
            rows[mode].append(dict(component=components[query["group"]], view=query["view"], arms=results))
        if (i + 1) % 24 == 0:
            print(i + 1, "of", len(cases), "live public cases", flush=True)
    report = {}
    for mode, measurements in rows.items():
        paired = {
            arm: clustered_paired_interval(
                [row["arms"][arm]["ms"] - row["arms"]["shortest_first"]["ms"] for row in measurements],
                [row["component"] for row in measurements],
            ) | {"sampling_unit": "publisher-linked component"}
            for arm in ("exhaustive", "refutation_first", "prefetch_four")
        }
        report["role_in_query" if mode else "role_in_document"] = dict(
            arms=summarize(measurements), latency_minus_shortest_ms=paired,
            views={view: summarize([row for row in measurements if row["view"] == view]) for view in VIEWS},
        )
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), reader=reader.provenance,
        script_sha256=digest(Path(__file__)), fixture_sha256=digest(fixture / "manifest.json"),
        scheduler_sha256=digest(Path(__file__).parents[2] / "src/expert_search/constraint_verification.py"),
        ranker_sha256=digest(Path(__file__).parents[1] / "followir/core17_verifier.py"),
        cases=len(cases), scope=SCOPE, top=TOP, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("fixture", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
