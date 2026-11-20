"""Evaluate real rescue calibration on a third held-out FollowIR collection."""

import argparse
import json
import os
from pathlib import Path

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, prepare, write_json

REVISION = "0a3efedfcac0a7f859c46cff3a0fac0f8747b28f"
FILES = {
    "corpus": "c65afaa6ed66fd6470fd5c7cdb0854e74d3835559c7f40f5977f0dddf7337439",
    "default": "f9742258107bc2b6b3c5cdbeda1fb2e7c052864e4ac75f78dc3b2fc11adde293",
    "instruction": "d9af40670221c25bc99b640613b26e5932bafeabd17958ccaafd948f09c26390",
    "qrel_diff": "220905cb6c1fc2e86087feab0e6ad4c7deeece14fe9c287a3d41e796cc9f5400",
    "queries": "4204778e94b8f0c85a9091282c2416a43a22e8081bb04c78d5ae352cac567f13",
    "top_ranked": "58edd5a63db81b1644a95b76ad6e0d20dc7ece2d7963284b013a3458160b92eb",
}
CAPS = (5, 10, 20)


def source_cutoff(scores, cap):
    from experiments.followir.rescue_threshold_transfer import source_cutoff as threshold

    return threshold(scores, cap)


def calibrated_cutoff(rows, cap, predicate):
    return source_cutoff([row["margin"] for row in rows if predicate(row)], cap)


def judged_negative(row):
    return row["label"] == 0


def invalidated(row):
    return row["invalidated"]


def fitted_policies(calibration, cap, first_negative):
    pooled = [row for rows in calibration.values() for row in rows]
    per_domain = {name: calibrated_cutoff(rows, cap, judged_negative) for name, rows in calibration.items()}
    hard = calibrated_cutoff(pooled, cap, invalidated)
    return {
        "first_pass": source_cutoff(first_negative, cap),
        "core_rescue": per_domain["core"],
        "news_rescue": per_domain["news"],
        "pooled_rescue": calibrated_cutoff(pooled, cap, judged_negative),
        "worst_domain": max(per_domain.values()),
        "worst_domain_with_hard_negative_cap": max(*per_domain.values(), hard),
    }


def disposition(rows, threshold):
    accepted = [row for row in rows if row["margin"] > threshold]
    return dict(
        positives=sum((row["label"] or 0) > 0 for row in accepted),
        negatives=sum(judged_negative(row) for row in accepted),
        invalidated=sum(invalidated(row) for row in accepted),
        unjudged=sum(row["label"] is None for row in accepted),
        issued_queries=len({row["topic"] for row in accepted}),
    )


def holdout_diagnostic(rows, cap):
    # Label-informed ceiling for this score family, unavailable at deployment.
    bounds = [calibrated_cutoff(rows, cap, predicate) for predicate in (judged_negative, invalidated)
              if any(predicate(row) for row in rows)]
    if not bounds:
        raise ValueError("No held-out hard-negative diagnostic")
    threshold = max(bounds)
    return dict(threshold=threshold, observed=disposition(rows, threshold))


def report(calibration, test, initial_negative, topics, cap):
    cutoffs = fitted_policies(calibration, cap, initial_negative)
    result = {}
    for name, threshold in cutoffs.items():
        changes = [0] * topics
        for row in test:
            if (row["label"] or 0) > 0:
                changes[row["topic"]] += int(row["margin"] > threshold) - int(row["margin"] > cutoffs["first_pass"])
        result[name] = dict(
            threshold=threshold, calibration={key: disposition(rows, threshold)
                                              for key, rows in calibration.items()},
            holdout=disposition(test, threshold), per_topic_positive_delta=paired_interval(changes),
        )
    return dict(policies=result, retrospective_score_ceiling=holdout_diagnostic(test, cap))


def development_thresholds(dev, encoder, reader):
    from experiments.followir.rescue_threshold_transfer import negatives, topical_scores

    topic_scores = {name: topical_scores(fixture, encoder, reader) for name, fixture in dev.items()}
    source_negatives = {
        name: [negatives([scores], [pair]) for scores, pair in zip(topic_scores[name], data["pairs"], strict=True)]
        for name, (data, _, _) in dev.items()
    }
    negative = [score for collection in source_negatives.values() for topic in collection for score in topic]
    thresholds = {}
    for name, groups in source_negatives.items():
        other = [score for key, rows in source_negatives.items() if key != name for row in rows for score in row]
        thresholds[name] = [
            source_cutoff(other + [score for j, row in enumerate(groups) if j != i for score in row], 10)
            for i in range(len(groups))
        ]
    return negative, thresholds


def independent_topics(dev, test):
    from experiments.followir.conditional_decision import overlaps, shingles, words

    prior = [pair for collection, _, _ in dev.values() for pair in collection["pairs"]]
    terms = [set(words(pair["query"])) for pair in prior]
    narratives = [shingles(text, 5) for pair in prior for text in pair["instructions"]]
    return [pair for pair in test["pairs"]
                if not any(overlaps(set(words(pair["query"])), seen, .75) for seen in terms)
                and not any(overlaps(shingles(text, 5), seen, .70)
                            for text in pair["instructions"] for seen in narratives)]


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir import constraint_rescue_scope as rescue
    from experiments.followir.core17_verifier import Reranker
    from experiments.followir.rescue_threshold_transfer import route
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    anchor = json.loads(args.reference.read_bytes())
    sources = (args.core, args.news)
    caches = (args.core_cache, args.news_cache)
    if (anchor["dataset_hashes"] != [digest(path) for path in sources]
            or anchor["cache_hashes"] != [digest(path) for path in caches]
            or anchor["embedding_revision"] != provenance["revision"]
            or anchor["reranker_revision"] != reader.provenance["revision"]):
        raise ValueError("Changed source fixtures or checkpoints")

    dev = {name: rescue.load(record, cache, provenance) for name, record, cache in
           zip(("core", "news"), sources, caches, strict=True)}
    for name, (data, _, _) in dev.items():
        if len(data["pairs"]) == 0:
            raise ValueError(f"Missing {name} topics")
    negative, thresholds = development_thresholds(dev, encoder, reader)
    calibration = {name: route(fixture, encoder, reader, thresholds[name])
                   for name, fixture in dev.items()}
    frozen_first = source_cutoff(negative, 10)
    robust = rescue.load(args.robust, args.robust_cache, provenance, additional_sources={REVISION: FILES})
    if robust[0]["revision"] != REVISION:
        raise ValueError("Unexpected holdout collection")
    robust_data, matrix, positions = robust
    eligible = independent_topics(dev, robust_data)
    topical_count = len(eligible)
    final = route((dict(robust_data, pairs=eligible), matrix, positions),
                  encoder, reader, [frozen_first] * topical_count)
    def totals(rows):
        return dict(candidates=len(rows), queries=len({row["topic"] for row in rows}),
                    **disposition(rows, float("-inf")))

    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        prepared_sha256=digest(args.robust), cache_sha256=digest(args.robust_cache),
        reference_sha256=digest(args.reference), sources=FILES,
        initial_threshold=frozen_first, first_batch_negatives=len(negative), holdout_topics=topical_count,
        overlapping_topics_excluded=len(robust_data["pairs"]) - topical_count,
        development={name: totals(rows) for name, rows in calibration.items()},
        robust04=totals(final),
        comparisons={cap: report(calibration, final, negative, topical_count, cap) for cap in CAPS},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("parquet", type=Path)
    setup.add_argument("output", type=Path)
    replay = sub.add_parser("run")
    for name in ("core", "core_cache", "news", "news_cache", "robust", "robust_cache",
                 "embedding", "reranker", "reference", "output"):
        replay.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.parquet, args.output, revision=REVISION, files=FILES, topics=52, changed_pairs=722)
    else:
        run(args)
