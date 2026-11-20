"""Transfer synthetic contrast calibration to frozen FollowIR rescue candidates."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir import conditional_decision as training
from experiments.followir import constraint_rescue_scope as rescue
from experiments.followir import core17_reranking as followir
from experiments.followir import rescue_threshold_transfer as reference
from experiments.followir.constraint_window_verification import contexts

CAP = 10
BATCH = 12
FEATURES = {
    "synthetic_lexical": (0, 1, 2),
    "synthetic_contrast": (0, 3),
    "synthetic_contrast_matched": (0, 3),
}
STOP = set("a an and are as be by for from has have in is it of on or that the their they this to was were will with would".split())


def lexical(query, instruction, doc):
    topic = set(training.words(query)) - STOP
    criteria = set(training.words(instruction)) - STOP - topic
    available = set(training.words(doc))
    return len(topic & available) / max(1, len(topic)), len(criteria & available) / max(1, len(criteria))


def full_prompt(query, instruction):
    return f"{query}\nRelevance criteria: {instruction}"


def margins(reader, pairs):
    scores, clipped = reader.score_pairs(pairs)
    if clipped or not np.isfinite(scores).all():
        raise ValueError("Invalid score input")
    return scores


def synthetic_features(rows, reader):
    output = []
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        documents = [row["document"][:850] for row in batch]
        full = margins(reader, [(full_prompt(row["query"], row["instruction"]), doc)
                                for row, doc in zip(batch, documents, strict=True)])
        topical = margins(reader, [(row["query"], doc) for row, doc in zip(batch, documents, strict=True)])
        output.extend([joint, *lexical(row["query"], row["instruction"], doc), joint - base]
                      for row, doc, joint, base in zip(batch, documents, full, topical, strict=True))
    return np.asarray(output)


def calibration(train_features, rows):
    import torch
    from torch.nn import functional as F

    fit, dev, matches = training.groups_for_fit(rows)
    negative = [i for i in dev if rows[i]["label"] == "false"]
    labels = torch.tensor([row["label"] == "true" for row in rows], dtype=torch.float64)
    pos, neg = matches
    reports = {}
    for name, columns in FEATURES.items():
        features = train_features[:, columns]
        center, spread = features[fit].mean(axis=0), features[fit].std(axis=0)
        if not np.all(spread > 0):
            raise ValueError("Constant training feature")
        x = torch.from_numpy((features - center) / spread)
        coef = torch.zeros(len(columns), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([coef, bias], max_iter=50, line_search_fn="strong_wolfe")

        def objective():
            optimizer.zero_grad(set_to_none=True)
            logits = x @ coef + bias
            loss = F.binary_cross_entropy_with_logits(logits[fit], labels[fit]) + 0.01 * coef.square().sum()
            if name == "synthetic_contrast_matched":
                loss = loss + 0.5 * F.softplus(logits[neg] - logits[pos]).mean()
            loss.backward()
            return loss

        optimizer.step(objective)
        dev_scores = (x @ coef + bias).detach().numpy()
        cutoff = reference.source_cutoff(dev_scores[negative], CAP)
        reports[name] = dict(
            center=center.tolist(), scale=spread.tolist(), weights=coef.detach().tolist(),
            bias=float(bias.detach().item()), cutoff=cutoff, negatives=len(negative),
            accepted_negative=int((dev_scores[negative] > cutoff).sum()),
            accepted_positive=int(sum(dev_scores[i] > cutoff for i in dev if rows[i]["label"] == "true")),
        )
    raw = reference.source_cutoff(train_features[negative, 0], CAP)
    return reports, raw, dict(fit=len(fit), calibration=len(dev), calibration_negatives=len(negative),
                              calibration_positives=len(dev)-len(negative), matched_pairs=len(pos))


def collect(fixture, embedding, reader, first):
    data, matrix, positions = fixture
    rows, initial, clean_times = [], [], []
    augmented_times, lexical_times = [], []
    for topic, pair in enumerate(data["pairs"]):
        replay = rescue.simulate(pair, data, (matrix, positions), embedding, reader, "constraint_full_pool", first)
        clean_times.append(replay["milliseconds"])
        initial.append(replay)
        candidates = replay["rescued"]
        if not candidates:
            continue
        started = perf_counter()
        snippets, _ = contexts(data["docs"], candidates, {}, "lead")
        overlaps = [lexical(pair["query"], pair["instructions"][1], doc) for doc in snippets]
        lexical_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        topic_only = margins(reader, [(pair["query"], doc) for doc in snippets])
        extra_ms = (perf_counter() - started) * 1000
        lexical_times.append(replay["milliseconds"] + lexical_ms)
        augmented_times.append(replay["milliseconds"] + extra_ms)
        for doc, (topic_share, criteria_share), topical in zip(candidates, overlaps, topic_only, strict=True):
            joint = replay["margins"][doc]
            rows.append(dict(topic=topic, label=pair["qrels"][1].get(doc),
                             invalidated=doc in pair["invalidated"],
                             features=[joint, topic_share, criteria_share, joint - topical]))
    return rows, initial, dict(
        base_p95_ms=float(np.quantile(clean_times, .95)),
        rescued_base_p95_ms=float(np.quantile([r["milliseconds"] for r in initial if r["rescued"]], .95)),
        rescued_lexical_p95_ms=float(np.quantile(lexical_times, .95)),
        rescued_augmented_p95_ms=float(np.quantile(augmented_times, .95)),
        rescued_reader_pairs=2 * len(rows), baseline_rescue_reader_pairs=len(rows),
    )


def evaluate(rows, reports, thresholds, n_topics):
    x = np.asarray([row["features"] for row in rows])
    accepted = {name: x[:, 0] > cutoff for name, cutoff in thresholds.items()}
    for name, model in reports.items():
        scaled = (x[:, FEATURES[name]] - model["center"]) / model["scale"]
        scores = scaled @ model["weights"] + model["bias"]
        accepted[name] = scores > model["cutoff"]
    masks = {
        "positive": np.asarray([(row["label"] or 0) > 0 for row in rows]),
        "negative": np.asarray([row["label"] == 0 for row in rows]),
        "unjudged": np.asarray([row["label"] is None for row in rows]),
        "invalidated": np.asarray([row["invalidated"] for row in rows]),
    }
    topic = np.asarray([row["topic"] for row in rows])
    summary = {}
    for name, keep in accepted.items():
        counts = {key: int((keep & mask).sum()) for key, mask in masks.items()}
        contrasts = {key: paired_interval(np.bincount(
            topic, weights=(keep.astype(int) - accepted["cross_collection_raw"].astype(int)) * mask,
            minlength=n_topics,
        )) for key, mask in masks.items()}
        summary[name] = dict(counts=counts, paired_per_query_delta=contrasts)
    return dict(labels={k: int(v.sum()) for k, v in masks.items()}, arms=summary)


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = json.loads(args.base.read_bytes())
    previous = json.loads(args.thresholds.read_bytes())
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    splits = {
        "core": rescue.load(args.core, args.core_cache, provenance),
        "news": rescue.load(args.news, args.news_cache, provenance),
    }
    if (frozen["dataset_hashes"] != [followir.digest(path) for path in (args.core, args.news)]
            or frozen["cache_hashes"] != [followir.digest(path) for path in (args.core_cache, args.news_cache)]
            or frozen["embedding_revision"] != provenance["revision"]
            or frozen["reranker_revision"] != reader.provenance["revision"]
            or previous["reference_sha256"] != followir.digest(args.base)):
        raise ValueError("Changed pinned replay")
    source_rows, excluded = training.training_data(args.training, [split[0] for split in splits.values()])
    started = perf_counter()
    train_features = synthetic_features(source_rows, reader)
    models, raw_cutoff, stats = calibration(train_features, source_rows)
    fit_seconds = perf_counter() - started
    outputs = {}
    for fit, held in (("core", "news"), ("news", "core")):
        transfer = f"{fit}_to_{held}"
        first = frozen["calibration"][fit]["threshold"]
        rows, initial, timing = collect(splits[held], encoder, reader, first)
        known = frozen["transfer"][transfer]["arms"]["constraint_full_pool"]
        if (len(rows) != rescue.RESCUE * known["rescued_queries"] or
                sum((row["label"] or 0) > 0 for row in rows) != known["rescue_proposed_positive"]):
            raise ValueError("Frozen rescue workload changed")
        raw = previous["transfer"][transfer]["cutoffs"][str(CAP)]["rescue_cutoff"]
        comparison = evaluate(rows, models, dict(
            first_pass_raw=first, synthetic_raw=raw_cutoff, cross_collection_raw=raw["threshold"],
        ), len(initial))
        counts = comparison["arms"]["cross_collection_raw"]["counts"]
        if any(counts[key] != raw[f"accepted_{key}"] for key in counts):
            raise ValueError("Cross-collection baseline did not reproduce")
        outputs[transfer] = dict(**comparison, timings=timing, first_cutoff=first,
                                 cross_collection_cutoff=raw["threshold"], queries=len(initial))
    followir.write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        code_sha256=followir.digest(Path(__file__)), train_sha256=followir.digest(args.training),
        prior_sha256=followir.digest(args.thresholds), overlap_removed=excluded,
        fit_seconds=fit_seconds, calibration=stats, synthetic_cutoff=raw_cutoff,
        models=models, transfer=outputs,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "core", "core_cache", "news", "news_cache",
                 "embedding", "reranker", "base", "thresholds", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
