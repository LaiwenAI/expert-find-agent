"""Compare literal evidence windows and full documents on judged constraint changes."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from experiments.followir import core17_reranking as followir
from experiments.followir.core17_verifier import Reranker
from experiments.followir.decision_features import NEWS_FILES, REVISION as NEWS_REVISION
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.scifact.minilm_sufficiency_operating_points import operating_threshold
from experiments.scifact.minilm_sufficient_facts import bootstrap_delta, drops_auc


METHODS = ("full", "lead", "topic_window", "criterion_window")
WINDOW_CHARS = 850
STRIDE = 425
BATCH = 4
CAPS = (5, 10, 20)


def locations(body):
    end = max(0, len(body) - WINDOW_CHARS)
    return sorted({*range(0, end + 1, STRIDE), end})


def select_window(body, keywords, weights):
    candidates = locations(body)
    start = max(candidates, key=lambda offset: (
        sum(weights[word] for word in set(tokens(body[offset:offset + WINDOW_CHARS])) & keywords), -offset
    ))
    end = min(start + WINDOW_CHARS, len(body))
    return body[start:end], end - start


def encode_candidate_pool(pair, model, vectors, positions):
    import torch

    topic = followir.texts("query_only", pair["query"], "")
    joint = followir.texts("joint", pair["query"], pair["instructions"][1])
    embedding = model.encode(topic + joint)
    pool = pair["pool"]
    similarities = (vectors[[positions[doc] for doc in pool]] @ torch.from_numpy(embedding.T.copy()).cuda()).cpu().numpy()
    ranking = np.argsort(-similarities[:, 0], kind="stable")
    return [pool[j] for j in ranking[4:24]]


def contexts(docs, pool, weights, arm):
    if arm == "full":
        return [docs[doc] for doc in pool], 0
    views = []
    chars = 0
    for doc in pool:
        title, _, body = docs[doc].partition("\n")
        if arm == "lead":
            selected, count = body[:WINDOW_CHARS], min(WINDOW_CHARS, len(body))
        else:
            selected, count = select_window(body, weights[arm]["terms"], weights[arm]["idf"])
        chars += count
        views.append(f"{title}\n{selected}")
    return views, chars


def collect(dataset_path, cache_path, encoder, reader, provenance):
    import torch

    data = json.loads(dataset_path.read_text())
    allowed = {followir.REVISION: followir.FILES, NEWS_REVISION: NEWS_FILES}
    if data["upstream_sha256"] != allowed.get(data["revision"]):
        raise ValueError("Unexpected FollowIR fixture")
    with np.load(cache_path, allow_pickle=False) as cached:
        ids = cached["doc_ids"].tolist()
        if (ids != sorted(data["docs"]) or str(cached["input_sha256"]) != followir.digest(dataset_path)
                or json.loads(str(cached["checkpoint_provenance"])) != provenance):
            raise ValueError("Frozen vectors do not match this corpus")
        vectors = torch.from_numpy(cached["vectors"]).cuda()
    positions = {doc: i for i, doc in enumerate(ids)}
    df = Counter(word for text in data["docs"].values() for word in set(tokens(text)))
    total = len(data["docs"])
    records, timing, audited = [], {method: [] for method in METHODS}, {method: Counter() for method in METHODS}
    for i, pair in enumerate([data["pairs"][0], *data["pairs"]]):
        pool = encode_candidate_pool(pair, encoder, vectors, positions)
        after = pair["qrels"][1]
        invalidated = set(pair["invalidated"])
        topic = pair["query"]
        criterion = pair["instructions"][1]
        prompt = f"{topic}\nRelevance criteria: {criterion}"
        topic_words = set(tokens(topic))
        criterion_words = set(tokens(criterion)) - topic_words
        idf = {w: float(np.log((total + 1) / (df[w] + 1))) for w in topic_words | criterion_words}
        weights = {arm: dict(terms=words, idf=idf) for arm, words in (
            ("topic_window", topic_words), ("criterion_window", criterion_words)
        )}
        batch_scores = {}
        for arm in METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]:
            started = perf_counter()
            documents, chars = contexts(data["docs"], pool, weights, arm)
            margins, clipped = [], 0
            for first in range(0, len(documents), BATCH):
                batch = documents[first:first+BATCH]
                scored, clipped_now = reader.score_pairs([(prompt, text) for text in batch])
                margins.extend(scored)
                clipped += clipped_now
            elapsed = (perf_counter() - started) * 1000
            if len(margins) != len(pool) or not np.isfinite(margins).all():
                raise ValueError("Unexpected number of verifier scores")
            batch_scores[arm] = margins
            if i:
                audited[arm].update(dict(clipped_inputs=clipped, window_body_chars=chars,
                                         prediction_pairs=len(margins)))
                timing[arm].append(elapsed)
        if i:
            for j, doc in enumerate(pool):
                records.append(dict(topic=pair["topic"], grade=after.get(doc), invalidated=doc in invalidated,
                                    **{arm: batch_scores[arm][j] for arm in METHODS}))
    return records, {arm: dict(audited[arm], component_p95_ms=float(np.quantile(timing[arm], .95))) for arm in METHODS}


def summarize(calibration, held):
    judged = np.asarray([r["grade"] is not None for r in held])
    positive = np.asarray([r["grade"] is not None and r["grade"] > 0 for r in held])
    negative = np.asarray([r["grade"] == 0 for r in held])
    invalidated = np.asarray([r["invalidated"] for r in held])
    topics = np.asarray([r["topic"] for r in held])
    report, accepted = {}, {}
    for arm in METHODS:
        values = np.asarray([r[arm] for r in held])
        cal_neg = np.asarray([r[arm] for r in calibration if r["grade"] == 0])
        report[arm] = dict(auc=drops_auc(values[positive], values[negative]),
                           auc_retained_vs_invalidated=drops_auc(values[positive], values[invalidated]), gates={})
        for cap in CAPS:
            threshold = operating_threshold(cal_neg, cap)
            passes = values > threshold
            accepted[arm, cap] = passes
            gate = dict(threshold=threshold, calibration_negatives=len(cal_neg),
                        calibration_accepted=int(np.sum(cal_neg > threshold)),
                        positives=int(np.sum(passes & positive)), negatives=int(np.sum(passes & negative)),
                        invalidated=int(np.sum(passes & invalidated)), unjudged=int(np.sum(passes & ~judged)))
            if arm != "full":
                delta = passes.astype(float) - accepted["full", cap]
                gate["positive_delta_ci95"] = bootstrap_delta(delta[positive], topics[positive])
                gate["negative_delta_ci95"] = bootstrap_delta(delta[negative], topics[negative])
            report[arm]["gates"][cap] = gate
    return dict(positives=int(positive.sum()), negatives=int(negative.sum()), unjudged=int((~judged).sum()),
                invalidated=int(invalidated.sum()), arms=report)


def run(core, core_cache, news, news_cache, embedding, reranker, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    core_rows, core_ms = collect(core, core_cache, encoder, reader, provenance)
    print("Core17 complete", flush=True)
    news_rows, news_ms = collect(news, news_cache, encoder, reader, provenance)
    print("News21 complete", flush=True)
    followir.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                     script_sha256=followir.digest(Path(__file__)), caps=CAPS, window_chars=WINDOW_CHARS,
                                     inputs=[followir.digest(p) for p in (core, core_cache, news, news_cache)],
                                     embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
                                     core17_ms=core_ms, news21_ms=news_ms,
                                     core17_to_news21=summarize(core_rows, news_rows),
                                     news21_to_core17=summarize(news_rows, core_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("core", "core_cache", "news", "news_cache", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
