"""Compare single-candidate confirmation with multi-candidate reranking."""

import argparse
import gzip
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.litsearch.dense_retrieval import target_groups
from experiments.litsearch.learned_confirmation import Endpoint, evaluate, observation
from experiments.litsearch.source_withholding import QUERY_SHA256, REVISION, TASK, calibrate, split


DEPTHS = {"dense": 0, "single": 1, "two": 2, "six": 6}
METHODS = (*DEPTHS, "single_absolute")
BUDGETS = (.02, .05, .10, .15)
STATES = ("present", "withheld")


class DenseConfirmation(Endpoint):
    def run(self, question, view, method, policy=None):
        start = perf_counter()
        pool, _, dense = self.retrieve(question, view)
        depth = DEPTHS[method]
        if depth:
            scores, truncated = self.reranker.score(question["query"], self.docs, pool[:depth], task=TASK)
            scored, _ = observation(pool[:depth], scores)
            if depth == 1:
                scored = scored[0], scored[1], float(dense[0] - dense[1])
        else:
            truncated = 0
            scored, _ = observation(pool, dense)
        return dict(pool=pool, scored=scored, ms=(perf_counter() - start) * 1000,
                    truncated=truncated)


def paired_rows(runs):
    for records in zip(*(runs[method] for method in DEPTHS), strict=True):
        first = records[0]
        if any(row["id"] != first["id"] or row["targets"] != first["targets"] for row in records):
            raise ValueError("Sources or request order changed across methods")
        views = {}
        for state in STATES:
            result = {method: row["views"][state] for method, row in zip(DEPTHS, records, strict=True)}
            if any(arm["pool"] != result["dense"]["pool"] for arm in result.values()):
                raise ValueError("Dense candidate pools differ across methods")
            doi, score, _ = result["single"]["scored"]
            result["single_absolute"] = dict(result["single"], scored=(doi, score, 0.))
            views[state] = result
        yield dict(id=first["id"], targets=first["targets"], views=views)


def fit_and_score(validation, test):
    projected = [dict(targets=row["targets"], views={
        state: {method: arm["scored"] for method, arm in row["views"][state].items()}
        for state in STATES
    }) for row in validation]
    groups = target_groups(test)
    report = {}
    for budget in BUDGETS:
        fitted = {method: calibrate(projected, method, budget) for method in METHODS}
        results = {}
        for method in METHODS:
            policy = fitted[method]["policy"]
            outcomes = {}
            for state in STATES:
                arms = [row["views"][state][method] for row in test]
                accepted = [arm["scored"][0] if accepts(arm["scored"], policy) else None for arm in arms]
                known = np.array([doi in row["targets"] for doi, row in zip(accepted, test, strict=True)])
                unjudged = np.array([doi is not None for doi in accepted]) & ~known
                outcomes[state] = dict(
                    known=int(known.sum()), unjudged=int(unjudged.sum()), requests=len(test),
                    top_known=sum(arm["scored"][0] in row["targets"]
                                  for arm, row in zip(arms, test, strict=True)),
                    truncated=sum(arm["truncated"] for arm in arms),
                    p95_ms=float(np.quantile([arm["ms"] for arm in arms], .95)),
                )
                if state == "present":
                    outcomes["known_mask"] = known
                else:
                    outcomes["withheld_mask"] = unjudged
            results[method] = outcomes
        differences = {}
        comparisons = (("single", "single_absolute"), ("single", "two"), ("single", "six"))
        for left, right in comparisons:
            differences[f"{left}_minus_{right}"] = {
                label: clustered_paired_interval(
                    results[left][label].astype(int) - results[right][label].astype(int), groups
                )
                for label in ("known_mask", "withheld_mask")
            }
        for outcomes in results.values():
            del outcomes["known_mask"], outcomes["withheld_mask"]
        report[f"{budget:.2f}"] = dict(calibration=fitted, test=results, paired=differences)
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    with gzip.open(args.projection, "rb") as stream:
        data = json.load(stream)
    if data["revision"] != REVISION or data["query_sha256"] != QUERY_SHA256:
        raise ValueError("Unexpected LitSearch projection")
    validation, test, _ = split(data["questions"])
    papers = data["papers"]
    provenance = verify_checkpoint(args.embedding)
    projection_hash = digest(args.projection)
    with np.load(args.cache, allow_pickle=False) as cache:
        if (str(cache["projection_sha256"]) != projection_hash
                or json.loads(str(cache["provenance"])) != provenance
                or cache["corpus_ids"].tolist() != [paper["corpusid"] for paper in papers]):
            raise ValueError("Vector cache mismatch")
        vectors = torch.from_numpy(cache["vectors"]).cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    endpoint = DenseConfirmation(papers, vectors, encoder, None, reader)
    for method in DEPTHS:
        endpoint.run(validation[0], "present", method)
    validation_rows = list(paired_rows(evaluate(endpoint, validation, tuple(DEPTHS))))
    test_rows = list(paired_rows(evaluate(endpoint, test, tuple(DEPTHS))))
    report = fit_and_score(validation_rows, test_rows)
    write_json(args.output, dict(
        projection_sha256=projection_hash, cache_sha256=digest(args.cache),
        script_sha256=digest(Path(__file__)), corpus=len(papers), validation_count=len(validation),
        validation=validation_rows, test=test_rows, budgets=report, embedding=provenance,
        reranker=reader.provenance, gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
    ))
    for budget, values in report.items():
        print(budget, {method: (arm["present"]["known"], arm["withheld"]["unjudged"])
                       for method, arm in values["test"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "cache", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
