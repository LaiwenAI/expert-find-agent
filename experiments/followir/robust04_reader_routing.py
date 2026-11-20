"""Test live 4B short-window routing on a fixed rescue cohort."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir import constraint_rescue_scope as rescue
from experiments.followir import robust04_rescue_risk as study
from experiments.followir.constraint_window_verification import contexts
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.rescue_contrast_head import lexical
from experiments.followir.rescue_reader_diagnosis import InstructionReader
from experiments.followir.robust04_reader_factorial import select

ARMS = ("always_4b", "lexical", "small_model", "both")
REPEATS = 3


def prompt(pair):
    return f"{pair['query']}\nRelevance criteria: {pair['instructions'][1]}"


def lexical_max(pair, snippets):
    return max(lexical(pair["query"], pair["instructions"][1], doc)[1] for doc in snippets)


def model_max(reader, pair, snippets):
    scores, clipped = reader.score_pairs([(prompt(pair), text) for text in snippets])
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid 0.6B verification")
    return max(scores)


def verify(reader, pair, snippets, threshold):
    scores, clipped = reader.score(prompt(pair), snippets)
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid 4B verification")
    return tuple(index for index, score in enumerate(scores) if score > threshold)


def route(arm, pair, snippets, gates, threshold, small, large):
    # Keep the incoming batch intact if it is sent to the larger reader.
    if arm in ("lexical", "both") and lexical_max(pair, snippets) < gates["lexical"]:
        return (), False
    if arm in ("small_model", "both") and model_max(small, pair, snippets) < gates["small_model"]:
        return (), False
    return verify(large, pair, snippets, threshold), True


def calibrate(dev, candidates, small, large, previous, threshold):
    rows_by_domain = {}
    accept_features = []
    for name, batches in candidates.items():
        data = dev[name][0]["docs"]
        accepted = []
        features = []
        for topic, (pair, ids) in enumerate(batches):
            if not ids:
                continue
            snippets, _ = contexts(data, ids, {}, "lead")
            choice = verify(large, pair, snippets, threshold)
            feature = dict(lexical=lexical_max(pair, snippets),
                           small_model=model_max(small, pair, snippets))
            features.append(feature)
            if choice:
                accept_features.append(feature)
            for i in choice:
                doc = ids[i]
                accepted.append(dict(topic=topic, label=pair["qrels"][1].get(doc),
                                     invalidated=doc in pair["invalidated"], margin=1))
        if study.disposition(accepted, 0) != previous[name]:
            raise ValueError("4B development decisions did not reproduce")
        rows_by_domain[name] = features
    if not accept_features:
        raise ValueError("4B issued no development responses")
    thresholds = {gate: min(row[gate] for row in accept_features) for gate in ("lexical", "small_model")}
    dev_routes = {}
    for name, features in rows_by_domain.items():
        dev_routes[name] = dict(
            batches=len(features), lexical=sum(r["lexical"] >= thresholds["lexical"] for r in features),
            small_model=sum(r["small_model"] >= thresholds["small_model"] for r in features),
            both=sum(all(r[k] >= value for k, value in thresholds.items()) for r in features),
        )
    return thresholds, dev_routes


def replay(data, candidates, gates, threshold, small, large):
    live = {name: [] for name in ARMS}
    calls = {name: 0 for name in ARMS}
    outputs = {name: {} for name in ARMS}
    queries = [i for i, (_, ids) in enumerate(candidates) if ids]
    for turn, index in enumerate([queries[0], *queries]):
        pair, ids = candidates[index]
        for repeat in range(REPEATS):
            offset = (turn + repeat) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                start = perf_counter()
                snippets, _ = contexts(data["docs"], ids, {}, "lead")
                output, invoked = route(arm, pair, snippets, gates, threshold, small, large)
                elapsed = (perf_counter() - start) * 1000
                if turn:
                    if index in outputs[arm] and output != outputs[arm][index]:
                        raise ValueError("One policy changed decisions between repetitions")
                    outputs[arm][index] = output
                    live[arm].append(elapsed)
                    if not repeat:
                        calls[arm] += int(invoked)
    reports = {}
    anchor = outputs["always_4b"]
    for arm in ARMS:
        counts = Counter()
        per_topic_lost = [0] * len(candidates)
        for topic, output in outputs[arm].items():
            if output and output != anchor[topic]:
                raise ValueError("One routed batch did not reproduce the 4B result")
            pair, ids = candidates[topic]
            accepted = []
            for i in output:
                doc = ids[i]
                accepted.append(dict(topic=topic, label=pair["qrels"][1].get(doc),
                                     invalidated=doc in pair["invalidated"], margin=1))
            counts.update(study.disposition(accepted, 0))
            per_topic_lost[topic] = sum((pair["qrels"][1].get(ids[i], 0) or 0) > 0
                                        for i in set(anchor[topic]) - set(output))
        reports[arm] = dict(
            decision=dict(counts), batches_4b=calls[arm],
            p50_ms=float(np.quantile(live[arm], .5)), p95_ms=float(np.quantile(live[arm], .95)),
            mean_ms=float(np.mean(live[arm])),
            exact_batches=sum(value == anchor[topic] for topic, value in outputs[arm].items()),
            lost_positive_per_topic=paired_interval(per_topic_lost),
        )
    return reports


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    previous = json.loads(args.factorial.read_bytes())
    reference = json.loads(args.anchor.read_bytes())
    base = json.loads(args.base.read_bytes())
    sources = (args.core, args.news)
    caches = (args.core_cache, args.news_cache)
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    small = Reranker(args.reranker)
    large = InstructionReader(args.chat)
    revisions = dict(embedding=provenance["revision"], reader=small.provenance["revision"], chat=large.revision)
    if (base["dataset_hashes"] != [digest(path) for path in sources]
            or base["cache_hashes"] != [digest(path) for path in caches]
            or reference["prepared_sha256"] != digest(args.robust)
            or reference["cache_sha256"] != digest(args.robust_cache)
            or reference["reference_sha256"] != digest(args.base)
            or previous["anchor_sha256"] != digest(args.anchor)
            or revisions != previous["revisions"]):
        raise ValueError("Changed frozen fixtures")

    dev = {name: rescue.load(record, cache, provenance) for name, record, cache in
           zip(("core", "news"), sources, caches, strict=True)}
    scores, cutoffs = study.development_thresholds(dev, encoder, small)
    initial = study.source_cutoff(scores, 10)
    if initial != previous["initial_threshold"]:
        raise ValueError("First-pass cutoff changed")
    dev_candidates = {name: select(fixture, encoder, small, cutoffs[name])
                      for name, fixture in dev.items()}
    target = previous["calibrated"]["4b_lead"]
    threshold = target["cutoff"]
    gates, development = calibrate(dev, dev_candidates, small, large, target["development"], threshold)

    data, matrix, positions = rescue.load(args.robust, args.robust_cache, provenance,
                                         additional_sources={study.REVISION: study.FILES})
    held = dict(data, pairs=study.independent_topics(dev, data))
    candidates = select((held, matrix, positions), encoder, small, [initial] * len(held["pairs"]))
    if (len(candidates) != reference["holdout_topics"]
            or sum(len(ids) for _, ids in candidates) != reference["robust04"]["candidates"]):
        raise ValueError("Frozen holdout candidate count changed")
    report = replay(held, candidates, gates, threshold, small, large)
    if report["always_4b"]["decision"] != previous["evaluation"]["4b_lead"]["decision"]:
        raise ValueError("Always-4B holdout result did not reproduce")
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=digest(Path(__file__)), reference_sha256=digest(args.factorial),
        n_topics=len(candidates), rescued_requests=sum(bool(ids) for _, ids in candidates),
        gates=gates, threshold_4b=threshold, repeats=REPEATS, development=development, report=report,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "core_cache", "news", "news_cache", "robust", "robust_cache",
                 "embedding", "reranker", "chat", "base", "anchor", "factorial", "output"):
        cli.add_argument(name, type=Path)
    run(cli.parse_args())
