"""Test whether calibration on the rescue route transfers across collections."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir import constraint_rescue_scope as rescue
from experiments.followir import core17_reranking as followir


CAPS = (5, 10, 20)
ARMS = ("topical_cutoff", "rescue_cutoff", "relative_rescue_cutoff")


def negatives(rows, pairs):
    return [score for index, scores in enumerate(rows) for doc, score in scores.items()
            if pairs[index]["qrels"][1].get(doc) == 0]


def source_cutoff(scores, cap):
    return rescue.operating_threshold(np.asarray(scores), cap)


def topical_scores(fixture, embedding, reader):
    data, matrix, positions = fixture
    return [
        rescue.simulate(pair, data, (matrix, positions), embedding, reader, "flat_six", 0)["margins"]
        for pair in data["pairs"]
    ]


def route(fixture, embedding, reader, thresholds):
    data, matrix, positions = fixture
    output = []
    for index, (pair, threshold) in enumerate(zip(data["pairs"], thresholds, strict=True)):
        result = rescue.simulate(pair, data, (matrix, positions), embedding, reader,
                                 "constraint_full_pool", threshold)
        initial_max = max(result["margins"][doc] for doc in result["selected"])
        for doc in result["rescued"]:
            margin = result["margins"][doc]
            output.append(dict(topic=index, margin=margin, relative=margin-initial_max,
                               label=pair["qrels"][1].get(doc), invalidated=doc in pair["invalidated"]))
    return output


def evaluate(calibration, target, topic_negative):
    report = {}
    for cap in CAPS:
        rescue_neg = [row for row in calibration if row["label"] == 0]
        variants = {
            "topical_cutoff": ("margin", topic_negative),
            "rescue_cutoff": ("margin", [row["margin"] for row in rescue_neg]),
            "relative_rescue_cutoff": ("relative", [row["relative"] for row in rescue_neg]),
        }
        base_cutoff = source_cutoff(topic_negative, cap)
        original = [row["margin"] > base_cutoff for row in target]
        groups = sorted({row["topic"] for row in target})
        report[cap] = {}
        for arm in ARMS:
            field, scores = variants[arm]
            cutoff = source_cutoff(scores, cap)
            negative_cal = sum(value > cutoff for value in scores)
            evaluated = [row[field] > cutoff for row in target]
            positives = [row["label"] is not None and row["label"] > 0 for row in target]
            delta = {}
            for row, accept, old, positive in zip(target, evaluated, original, positives, strict=True):
                if positive:
                    key = row["topic"]
                    delta[key] = delta.get(key, 0)+int(accept)-int(old)
            report[cap][arm] = dict(
                threshold=cutoff, calibration_negative_accepted=negative_cal, calibration_negatives=len(scores),
                accepted_positive=sum(keep and positive for keep,positive in zip(evaluated, positives, strict=True)),
                accepted_negative=sum(keep and row["label"] == 0 for keep,row in zip(evaluated,target,strict=True)),
                accepted_unjudged=sum(keep and row["label"] is None for keep,row in zip(evaluated,target,strict=True)),
                accepted_invalidated=sum(keep and row["invalidated"] for keep,row in zip(evaluated,target,strict=True)),
                paired_positive_count_delta_vs_topical=paired_interval([delta.get(i, 0) for i in groups]),
            )
    return dict(
        calibration_queries=len({row["topic"] for row in calibration}),
        calibration_rescue_candidates=len(calibration),
        calibration_rescue_negatives=sum(row["label"] == 0 for row in calibration),
        test_rescue_queries=len({row["topic"] for row in target}),
        test_candidates=len(target), test_positives=sum((row["label"] or 0) > 0 for row in target),
        test_negatives=sum(row["label"] == 0 for row in target), cutoffs=report,
    )


def run(core, core_cache, news, news_cache, embedding, reranker, reference, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = json.loads(reference.read_bytes())
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    datasets = {name: rescue.load(records, cache, provenance) for name,records,cache in (
        ("core",core,core_cache),("news",news,news_cache))}
    if (frozen["dataset_hashes"] != [followir.digest(path) for path in (core, news)]
            or frozen["cache_hashes"] != [followir.digest(path) for path in (core_cache, news_cache)]
            or frozen["embedding_revision"] != provenance["revision"]
            or frozen["reranker_revision"] != reader.provenance["revision"]):
        raise ValueError("Changed upstream replay fixture")
    start = perf_counter()
    reports = {}
    for source, target in (("core","news"),("news","core")):
        fit, test = datasets[source], datasets[target]
        domain = f"{source}_to_{target}"
        scores = topical_scores(fit, encoder, reader)
        pairs = fit[0]["pairs"]
        negative = negatives(scores,pairs)
        first = source_cutoff(negative, 10)
        if first != frozen["calibration"][source]["threshold"]:
            raise ValueError("First-pass calibration did not reproduce")
        cross_fitted = [source_cutoff(negatives(
            scores[:i]+scores[i+1:], pairs[:i]+pairs[i+1:]), 10) for i in range(len(pairs))]
        calibration = route(fit, encoder, reader, cross_fitted)
        target_data = route(test, encoder, reader, [first]*len(test[0]["pairs"]))
        anchor = frozen["transfer"][domain]["arms"]["constraint_full_pool"]
        if (len(target_data) != rescue.RESCUE * anchor["rescued_queries"] or
                sum((row["label"] or 0) > 0 for row in target_data) != anchor["rescue_proposed_positive"]):
            raise ValueError("Fixed rescue candidates did not reproduce")
        reports[domain] = evaluate(calibration, target_data, negative)
    followir.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=followir.digest(Path(__file__)), reference_sha256=followir.digest(reference),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        diagnostic_runtime_seconds=perf_counter()-start, transfer=reports,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("core","core_cache","news","news_cache","embedding","reranker","reference","output"):
        cli.add_argument(field,type=Path)
    run(**vars(cli.parse_args()))
