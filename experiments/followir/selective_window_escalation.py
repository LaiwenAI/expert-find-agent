"""Calibrate short-to-full verification at a matched read budget."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir import constraint_window_verification as window
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.scifact.minilm_sufficiency_operating_points import operating_threshold
from experiments.scifact.minilm_sufficient_facts import bootstrap_delta


CAP = 10
MAX_BLOCKS = 2
BATCH = window.BATCH
METHODS = ("short", "full", "rank_first", "margin_route")


def score(reader, prompt, documents, blocks):
    result, elapsed = {}, {}
    for block in blocks:
        selected = documents[block * BATCH:(block + 1) * BATCH]
        started = perf_counter()
        margins, _ = reader.score_pairs([(prompt, text) for text in selected])
        if not np.isfinite(margins).all():
            raise ValueError("Unexpected verifier margin")
        elapsed[block] = (perf_counter() - started) * 1000
        result[block] = np.asarray(margins)
    return result, elapsed


def corpus(dataset_path, cache_path, provenance, encoder, reader):
    import torch

    data = json.loads(dataset_path.read_text())
    allowed = {window.followir.REVISION: window.followir.FILES, window.NEWS_REVISION: window.NEWS_FILES}
    if data["upstream_sha256"] != allowed.get(data["revision"]):
        raise ValueError("Unexpected FollowIR fixture")
    with np.load(cache_path, allow_pickle=False) as cached:
        ids = cached["doc_ids"].tolist()
        if (ids != sorted(data["docs"]) or str(cached["input_sha256"]) != window.followir.digest(dataset_path)
                or json.loads(str(cached["checkpoint_provenance"])) != provenance):
            raise ValueError("Frozen vectors do not match this corpus")
        vectors = torch.from_numpy(cached["vectors"]).cuda()
    positions = {doc: i for i, doc in enumerate(ids)}
    rows = []
    for i, pair in enumerate([data["pairs"][0], *data["pairs"]]):
        selected = window.encode_candidate_pool(pair, encoder, vectors, positions)
        full = [data["docs"][doc] for doc in selected]
        small = window.contexts(data["docs"], selected, {}, "lead")[0]
        prompt = f'{pair["query"]}\nRelevance criteria: {pair["instructions"][1]}'
        blocks = range(len(full) // BATCH)
        if not full or len(full) != len(blocks) * BATCH:
            raise ValueError("Expected complete fixed-size blocks")
        observed = {}
        for context in ((small, full) if i % 2 else (full, small)):
            observed["short" if context is small else "full"] = score(reader, prompt, context, blocks)
        if i:
            grades = pair["qrels"][1]
            labels = np.asarray([1 if grades[doc] > 0 else 0 if grades[doc] == 0 else -1
                                 for doc in selected if doc in grades], dtype=int)
            all_labels = np.full(len(selected), -1, dtype=int)
            all_labels[[j for j, doc in enumerate(selected) if doc in grades]] = labels
            rows.append(dict(topic=pair["topic"], prompt=prompt, short_docs=small, full_docs=full, labels=all_labels,
                             invalidated=np.asarray([doc in pair["invalidated"] for doc in selected]),
                             **{key: np.stack(list(m.values())) for key, (m, _) in observed.items()}))
    return rows


def selected_blocks(short, threshold, radius, method):
    if method == "short":
        return []
    if method == "full":
        return list(range(len(short)))
    if method == "rank_first":
        return list(range(MAX_BLOCKS))
    distance = np.abs(short - threshold).min(axis=1)
    return [int(i) for i in np.argsort(distance, kind="stable")[:MAX_BLOCKS] if distance[i] <= radius]


def decision(row, method, short_cut, full_cut, radius):
    chosen = selected_blocks(row["short"], short_cut, radius, method)
    accepted = row["short"] > short_cut
    if chosen:
        accepted[chosen] = row["full"][chosen] > full_cut
    return accepted.flatten(), chosen


def calibrate(rows):
    short_cut, full_cut = (
        operating_threshold(np.concatenate([row[arm].flatten()[row["labels"] == 0] for row in rows]), CAP)
        for arm in ("short", "full")
    )
    distances = np.concatenate([abs(row["short"] - short_cut).min(axis=1) for row in rows])
    radii = [-1.0, *np.unique(np.quantile(distances, np.linspace(0, 1, 21))).tolist()]
    truth = np.concatenate([item["labels"] for item in rows])
    negatives = int(np.sum(truth == 0))
    output = {}
    for method in METHODS:
        allowed = radii if method == "margin_route" else [-1.0]
        candidates = []
        for radius in allowed:
            chosen = [decision(row, method, short_cut, full_cut, radius) for row in rows]
            accepts = np.concatenate([item[0] for item in chosen])
            visits = sum(len(item[1]) for item in chosen)
            support = int(np.sum(accepts & (truth == 1)))
            false = int(np.sum(accepts & (truth == 0)))
            if method != "margin_route" or false <= negatives * CAP // 100:
                candidates.append((support, -visits, -false, -radius))
        winner = max(candidates)
        output[method] = dict(radius=-winner[3], positive_accepted=winner[0],
                              negative_accepted=-winner[2], full_blocks=-winner[1], negative_count=negatives)
    return dict(short_cutoff=short_cut, full_cutoff=full_cut, calibration=output)


def replay_test(rows, policy, reader):
    observed = defaultdict(list)
    labels = np.concatenate([row["labels"] for row in rows])
    invalidated = np.concatenate([row["invalidated"] for row in rows])
    topics = np.repeat([row["topic"] for row in rows], len(rows[0]["labels"]))
    positive, negative = labels == 1, labels == 0
    for i, row in enumerate(rows):
        for method in METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]:
            short_cut, full_cut = policy["short_cutoff"], policy["full_cutoff"]
            radius = policy["calibration"][method]["radius"]
            expected, expected_blocks = decision(row, method, short_cut, full_cut, radius)
            started = perf_counter()
            if method == "full":
                chosen = list(range(len(row["full"])))
                points, _ = score(reader, row["prompt"], row["full_docs"], chosen)
                decision_array = np.concatenate([m > full_cut for m in points.values()])
            else:
                values, _ = score(reader, row["prompt"], row["short_docs"], range(len(row["short"])))
                short = np.stack(list(values.values()))
                chosen = selected_blocks(short, short_cut, radius, method)
                full, _ = score(reader, row["prompt"], row["full_docs"], chosen)
                for block, value in full.items():
                    values[block] = value
                decision_array = np.concatenate([value > (full_cut if block in full else short_cut)
                                                 for block, value in values.items()])
            elapsed = (perf_counter() - started) * 1000
            if chosen != expected_blocks or not np.array_equal(expected, decision_array):
                raise ValueError("Selected fixed-block replay did not reproduce calibration trace decisions")
            observed[method].append((decision_array, len(chosen), elapsed))
    reference = np.concatenate([item[0] for item in observed["full"]])
    report = {}
    for method, items in observed.items():
        accepts = np.concatenate([item[0] for item in items])
        delta = accepts.astype(float) - reference
        report[method] = dict(positive_accepted=int(np.sum(accepts & positive)), negative_accepted=int(np.sum(accepts & negative)),
                              invalidated_accepted=int(np.sum(accepts & invalidated)), unjudged_accepted=int(np.sum(accepts & (labels < 0))),
                              full_blocks=sum(item[1] for item in items),
                              p95_ms=float(np.quantile([item[2] for item in items], .95)),
                              positives_delta_ci95=bootstrap_delta(delta[positive], topics[positive]),
                              negatives_delta_ci95=bootstrap_delta(delta[negative], topics[negative]))
    return dict(positives=int(positive.sum()), negatives=int(negative.sum()), unknown=int(np.sum(labels < 0)),
                topics=len(rows), report=report)


def run(core, core_cache, news, news_cache, embedding, reranker, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and a Slurm GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    reader = window.Reranker(reranker)
    datasets = {
        "core17": corpus(core, core_cache, provenance, encoder, reader),
        "news21": corpus(news, news_cache, provenance, encoder, reader),
    }
    results = {}
    for train, test in (("core17", "news21"), ("news21", "core17")):
        policy = calibrate(datasets[train])
        results[f"{train}_to_{test}"] = dict(policy=policy, evaluation=replay_test(datasets[test], policy, reader))
        print(train, "to", test, "complete", flush=True)
    window.followir.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                            script_sha256=window.followir.digest(Path(__file__)),
                                            inputs=[window.followir.digest(p) for p in (core, core_cache, news, news_cache)],
                                            embedding_revision=provenance["revision"],
                                            reranker_revision=reader.provenance["revision"], report=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("core", "core_cache", "news", "news_cache", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
