"""Cross the evidence window and reader on disjoint Robust04 rescue candidates."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir import constraint_rescue_scope as rescue
from experiments.followir import robust04_rescue_risk as study
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.rescue_reader_diagnosis import InstructionReader, collect

CAP = 10
REFERENCE = "0.6b_lead"


def select(fixture, encoder, reader, cutoffs):
    data, matrix, positions = fixture
    return [
        (pair, rescue.simulate(pair, data, (matrix, positions), encoder, reader,
                               "constraint_full_pool", cutoff)["rescued"])
        for pair, cutoff in zip(data["pairs"], cutoffs, strict=True)
    ]


def labelled(candidates, scores):
    return [
        dict(topic=index, label=grade, invalidated=doc in candidates[index][0]["invalidated"], margin=score)
        for index, doc, grade, score in scores
    ]


def thresholds(pools):
    per_collection = {name: study.calibrated_cutoff(rows, CAP, study.judged_negative)
                      for name, rows in pools.items()}
    hard = study.calibrated_cutoff([row for rows in pools.values() for row in rows], CAP, study.invalidated)
    return dict(cutoff=max(*per_collection.values(), hard), domains=per_collection, hard=hard)


def report(test, timings, calibrated, n_topics):
    arms = {}
    ref_cutoff = calibrated[REFERENCE]["cutoff"]
    for arm, rows in test.items():
        cutoff = calibrated[arm]["cutoff"]
        changes = [0] * n_topics
        if [row["label"] for row in rows] != [row["label"] for row in test[REFERENCE]]:
            raise ValueError("The methods did not score the same candidates")
        for row, anchor in zip(rows, test[REFERENCE], strict=True):
            if (row["label"] or 0) > 0:
                changes[row["topic"]] += int(row["margin"] > cutoff) - int(anchor["margin"] > ref_cutoff)
        measurement = timings[arm]
        arms[arm] = dict(
            decision=study.disposition(rows, cutoff),
            retrospective_ceiling=study.holdout_diagnostic(rows, CAP),
            positive_delta_vs_lead=paired_interval(changes),
            verification_p95_ms=float(np.quantile([value["ms"] for value in measurement], .95)),
            capped=sum(value["clipped"] for value in measurement),
        )
    return arms


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    old = json.loads(args.anchor.read_bytes())
    base = json.loads(args.base.read_bytes())
    sources = (args.core, args.news)
    caches = (args.core_cache, args.news_cache)
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    chat = InstructionReader(args.chat)
    if (base["dataset_hashes"] != [digest(path) for path in sources]
            or base["cache_hashes"] != [digest(path) for path in caches]
            or old["prepared_sha256"] != digest(args.robust)
            or old["cache_sha256"] != digest(args.robust_cache)
            or old["reference_sha256"] != digest(args.base)
            or base["embedding_revision"] != provenance["revision"]
            or base["reranker_revision"] != reader.provenance["revision"]):
        raise ValueError("Changed frozen inputs")
    dev = {name: rescue.load(record, cache, provenance) for name, record, cache in
           zip(("core", "news"), sources, caches, strict=True)}
    negatives, cutoffs = study.development_thresholds(dev, encoder, reader)
    initial = study.source_cutoff(negatives, CAP)
    if initial != old["initial_threshold"]:
        raise ValueError("Initial cutoff did not reproduce")
    development = {}
    for name, fixture in dev.items():
        candidates = select(fixture, encoder, reader, cutoffs[name])
        raw, _ = collect(fixture[0], candidates, reader, chat)
        development[name] = {arm: labelled(candidates, values) for arm, values in raw.items()}
    calibrated = {}
    for arm in development["core"]:
        pools = {name: values[arm] for name, values in development.items()}
        calibrated[arm] = thresholds(pools)
        cutoff = calibrated[arm]["cutoff"]
        calibrated[arm]["development"] = {
            name: study.disposition(rows, cutoff) for name, rows in pools.items()
        }
    key = "worst_domain_with_hard_negative_cap"
    previous = old["comparisons"][str(CAP)]["policies"][key]
    if calibrated[REFERENCE]["cutoff"] != previous["threshold"]:
        raise ValueError("Original reader cutoff did not reproduce")

    data, matrix, positions = rescue.load(args.robust, args.robust_cache, provenance,
                                         additional_sources={study.REVISION: study.FILES})
    held = dict(data, pairs=study.independent_topics(dev, data))
    n_topics = len(held["pairs"])
    candidates = select((held, matrix, positions), encoder, reader, [initial] * n_topics)
    if n_topics != old["holdout_topics"] or sum(len(ids) for _, ids in candidates) != old["robust04"]["candidates"]:
        raise ValueError("The frozen test workload changed")
    raw, timings = collect(held, candidates, reader, chat)
    signature = [row[:3] for row in raw[REFERENCE]]
    if any([row[:3] for row in values] != signature for values in raw.values()):
        raise ValueError("Candidate order differs between verification arms")
    test = {arm: labelled(candidates, values) for arm, values in raw.items()}
    evaluation = report(test, timings, calibrated, n_topics)
    if evaluation[REFERENCE]["decision"] != previous["holdout"]:
        raise ValueError("Baseline held-out decisions did not reproduce")
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), anchor_sha256=digest(args.anchor),
        revisions=dict(embedding=provenance["revision"], reader=reader.provenance["revision"], chat=chat.revision),
        initial_threshold=initial, holdout_topics=n_topics, calibrated=calibrated, evaluation=evaluation,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "core_cache", "news", "news_cache", "robust", "robust_cache",
                 "embedding", "reranker", "chat", "base", "anchor", "output"):
        cli.add_argument(name, type=Path)
    run(cli.parse_args())
