"""Test source confirmation against LitSearch targets with and without masking."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
from heapq import nsmallest
import json
from pathlib import Path
import re
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from experiments.credit.conjunctive_source_retrieval import sentences
from experiments.credit.source_confirmation import accepts, measure
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
QUERY_SHA256 = "38cdbe4a6b7a7f5776d08055bfbf6e4511000aa2a523866592b4bb273388b914"
TASK = "Given a scientific literature search question, retrieve papers that directly address it."
POOL = 6
POLICY_BUDGETS = (.05, .10)


def normalize(text):
    return re.findall(r"\w+", text.casefold())


def split(questions):
    test = [row for row in questions if row["query_set"].startswith("manual_")]
    target_ids = {doi for row in test for doi in row["corpusids"]}
    normalized = {tuple(normalize(r["query"])) for r in test}
    available = [row for row in questions if row["query_set"].startswith("inline_")
                 and not target_ids.intersection(row["corpusids"])
                 and tuple(normalize(row["query"])) not in normalized]
    ranked = sorted(available, key=lambda row: sha256(row["query"].encode()).hexdigest())
    if len(test) != 246 or len(ranked) < 96:
        raise ValueError("Unexpected manual/calibration inventory")
    return ranked[:96], test, len(available)


def calibrate(rows, key, budget):
    visible = [row["views"]["present"][key] for row in rows]
    absent = [row["views"]["withheld"][key] for row in rows]
    obs = [o for o in visible + absent if o[0] is not None]
    score_grid = np.unique(np.quantile([o[1] for o in obs], np.linspace(0, 1, 31))).tolist()
    gap_grid = [0.] + np.unique(np.quantile([o[2] for o in obs], np.linspace(0, 1, 16))).tolist()
    max_errors = int(len(rows) * budget)
    best = None
    for threshold in score_grid:
        for gap in gap_grid:
            policy = dict(score=threshold, margin=gap)
            withhold_errors = sum(accepts(o, policy) for o in absent)
            if withhold_errors > max_errors:
                continue
            yes = [(o[0] in row["targets"]) for row, o in zip(rows, visible, strict=True) if accepts(o, policy)]
            if len(yes) - sum(yes) > max_errors:
                continue
            result = (sum(yes), -withhold_errors, -(len(yes) - sum(yes)), threshold, gap)
            if any(yes) and (best is None or result > best):
                best = result
    if best is None:
        return dict(policy=None, selected_targets=0, present_unjudged=0, withheld_accepts=0)
    return dict(policy={"score": best[3], "margin": best[4]}, selected_targets=best[0],
                present_unjudged=-best[2], withheld_accepts=-best[1])


class Endpoint:
    def __init__(self, papers, index, reranker):
        self.ids = [row["corpusid"] for row in papers]
        self.docs = {row["corpusid"]: f"{row['title'] or ''}\n{row['abstract'] or ''}" for row in papers}
        self.index = index
        self.reranker = reranker
        self.segments = {}

    def segments_for(self, doi):
        if doi not in self.segments:
            title, abstract = self.docs[doi].split("\n", 1)
            self.segments[doi] = [normalize(t) for t in (title, *sentences(abstract))]
        return self.segments[doi]

    def run(self, question, withheld):
        start = perf_counter()
        text, targets = question["query"], set(question["corpusids"])
        hidden = targets if withheld else set()
        scores = self.index.document_scores(text)
        ranking = nsmallest(POOL, (j for j in scores if self.ids[int(j)] not in hidden),
                           key=lambda j: (-scores[j], self.ids[int(j)]))
        pool = [self.ids[int(index)] for index in ranking]
        lexical_ms = (perf_counter() - start) * 1000
        if not pool:
            return dict(pool=[], coverage=(None,0.,0.), reranker=(None,0.,0.),
                        bm25_ms=lexical_ms, coverage_ms=lexical_ms, reranker_ms=lexical_ms, truncated=0)
        start = perf_counter()
        terms = normalize(text)
        coverage = measure(terms, pool[:4], {doi: self.segments_for(doi) for doi in pool[:4]}, "coverage")
        coverage_ms = lexical_ms + (perf_counter() - start) * 1000
        start = perf_counter()
        margins, truncations = self.reranker.score(text, self.docs, pool, task=TASK)
        order = sorted(range(len(pool)), key=lambda i: (-margins[i], i))
        best = order[0]
        margin = margins[best] - margins[order[1]] if len(order) > 1 else 0.
        reranked = (pool[best], margins[best], margin)
        reranker_ms = lexical_ms + (perf_counter() - start) * 1000
        return dict(pool=pool, coverage=coverage, reranker=reranked, bm25_ms=lexical_ms,
                    coverage_ms=coverage_ms, reranker_ms=reranker_ms, truncated=truncations)


def evaluate(endpoint, questions):
    rows = []
    for i, question in enumerate(questions):
        views = {}
        order = ("present", "withheld")
        for view in order[i % 2:] + order[:i % 2]:
            views[view] = endpoint.run(question, view == "withheld")
        rows.append(dict(id=sha256(question["query"].encode()).hexdigest(), category=question["query_set"],
                         targets=question["corpusids"], terms=len(normalize(question["query"])), views=views))
    return rows


def summarize(rows, policies):
    summary = {}
    for name in ("bm25", "reranker", "plos_coverage", *policies):
        scores = {}
        for view in ("present", "withheld"):
            counts = Counter()
            times = []
            for row in rows:
                record = row["views"][view]
                key = "coverage" if "coverage" in name else "reranker" if "reranker" in name else "bm25"
                if name == "bm25":
                    result = record["pool"][0] if record["pool"] else None
                else:
                    observation = record[key]
                    accepted = name == "reranker"
                    if name == "plos_coverage":
                        accepted = row["terms"] > 8 and accepts(observation, {"score": .75, "margin": .30})
                    if name in policies:
                        accepted = accepts(observation, policies[name]["policy"])
                    result = observation[0] if accepted else None
                counts["target_in_bm25_pool"] += bool(set(row["targets"]).intersection(record["pool"]))
                counts["accepted"] += result is not None
                counts["target_returned"] += result in row["targets"]
                counts["unjudged_returned"] += result is not None and result not in row["targets"]
                times.append(record[f"{key}_ms"])
            scores[view] = dict(counts, requests=len(rows), mean_ms=mean(times),
                                p95_ms=float(np.quantile(times, .95)))
        summary[name] = scores
    return summary


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with gzip.open(args.projection, "rb") as stream:
        data = json.load(stream)
    if data["revision"] != REVISION or data["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch revision")
    validation, test, eligible = split(data["questions"])
    papers = data["papers"]
    mapping = {}
    for i, row in enumerate(papers):
        text = f"{row['title'] or ''}\n{row['abstract'] or ''}"
        while text in mapping:
            text += "\n"
        mapping[text] = i
    start = perf_counter()
    with TemporaryDirectory(dir=args.projection.parent) as temporary:
        index = build_lexical(mapping, Path(temporary))
    index_seconds = perf_counter() - start
    print("Built lexical index", len(papers), "papers in", round(index_seconds, 2), "seconds", flush=True)
    reranker = Reranker(args.checkpoint)
    endpoint = Endpoint(papers, index, reranker)
    endpoint.run(validation[0], False)
    validation_rows = evaluate(endpoint, validation)
    policies = {f"{key}_{round(budget*100)}": calibrate(validation_rows, key, budget)
                for key in ("coverage","reranker") for budget in POLICY_BUDGETS}
    print("Frozen policies", policies, flush=True)
    test_rows = evaluate(endpoint, test)
    if any(set(row["views"]["withheld"]["pool"]) & set(row["targets"]) for row in test_rows):
        raise AssertionError("Withheld source leaked into the test ranking")
    summaries = {"manual_all": summarize(test_rows, policies)}
    for category in ("manual_acl", "manual_iclr"):
        summaries[category] = summarize([row for row in test_rows if row["category"] == category], policies)
    write_json(args.output, dict(
        revision=REVISION, projection_sha256=digest(args.projection), script_sha256=digest(Path(__file__)),
        corpus=len(papers), calibration_eligible=eligible, categories=dict(Counter(row["category"] for row in test_rows)),
        policies=policies, validation=validation_rows, test=test_rows, summary=summaries,
        checkpoint=reranker.provenance, gpu=torch.cuda.get_device_name(), index_seconds=index_seconds,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
