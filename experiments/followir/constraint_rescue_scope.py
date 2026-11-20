"""Measure abstention-triggered constraint reranking with a real verifier."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir import core17_reranking as followir
from experiments.followir.constraint_window_verification import contexts
from experiments.followir.decision_features import NEWS_FILES, REVISION as NEWS_REVISION
from experiments.scifact.minilm_sufficiency_operating_points import operating_threshold


FIRST, RESCUE, LOCAL = 4, 2, 24
ARMS = ("flat_six", "depth_on_abstention", "constraint_local", "constraint_full_pool", "constraint_flat_six")


def load(path, cache_path, provenance):
    import torch

    dataset = json.loads(path.read_bytes())
    if dataset["upstream_sha256"] != {followir.REVISION: followir.FILES,
                                        NEWS_REVISION: NEWS_FILES}.get(dataset["revision"]):
        raise ValueError("Unexpected FollowIR fixture")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["doc_ids"].tolist()
        if (ids != sorted(dataset["docs"]) or str(cache["input_sha256"]) != followir.digest(path)
                or json.loads(str(cache["checkpoint_provenance"])) != provenance):
            raise ValueError("Embedding cache differs from the frozen corpus")
        matrix = torch.from_numpy(cache["vectors"]).cuda()
    return dataset, matrix, {key: i for i, key in enumerate(ids)}


def order(pair, encoder, vector_rows, permitted, joint=False):
    import torch

    name = "joint" if joint else "query_only"
    texts = followir.texts(name, pair["query"], pair["instructions"][1] if joint else "")
    vector = torch.as_tensor(encoder.encode(texts)[0], device="cuda")
    scores = torch.mv(vector_rows[permitted], vector).cpu().numpy()
    return [permitted[j] for j in np.argsort(-scores, kind="stable")]


def score(pair, source_ids, dataset, reader):
    docs, _ = contexts(dataset["docs"], source_ids, {}, "lead")
    prompt = f"{pair['query']}\nRelevance criteria: {pair['instructions'][1]}"
    values, truncated = reader.score_pairs([(prompt, doc) for doc in docs])
    if truncated or not np.isfinite(values).all():
        raise ValueError("Invalid short-view verifier inputs")
    return dict(zip(source_ids, values, strict=True))


def simulate(pair, dataset, matrices, embedding, reader, arm, cutoff):
    started = perf_counter()
    matrix, positions = matrices
    pool = pair["pool"]
    vectors = matrix[[positions[doc] for doc in pool]]
    topical = order(pair, embedding, vectors, list(range(len(pool))),
                    joint=arm == "constraint_flat_six")
    top_ids = [pool[j] for j in topical[:FIRST]]
    if arm in ("flat_six", "constraint_flat_six"):
        selected = [pool[j] for j in topical[:FIRST+RESCUE]]
        margins = score(pair, selected, dataset, reader)
        rescued = []
    else:
        selected = top_ids
        margins = score(pair, selected, dataset, reader)
        rescued = []
        if all(value <= cutoff for value in margins.values()):
            tail = topical[FIRST:]
            if arm != "depth_on_abstention":
                permitted = tail[:LOCAL-FIRST] if arm == "constraint_local" else tail
                tail = order(pair, embedding, vectors, permitted, joint=True)
            rescued = [pool[j] for j in tail[:RESCUE]]
            margins.update(score(pair, rescued, dataset, reader))
    ms = (perf_counter()-started)*1000
    returned = [doc for doc, value in margins.items() if value > cutoff]
    gold = pair["qrels"][1]
    positive = sum(gold.get(doc, 0) > 0 for doc in returned)
    return dict(
        selected=selected, rescued=rescued, returned=returned, margins=margins, milliseconds=ms,
        grounded=positive, issued=len(returned), negative=sum(gold.get(doc) == 0 for doc in returned),
        unjudged=sum(doc not in gold for doc in returned), invalidated=sum(doc in pair["invalidated"]
                                                                           for doc in returned),
        checks=len(margins), rescue_proposed_positive=sum(gold.get(doc, 0) > 0 for doc in rescued),
        rescue_accepted_positive=sum(doc in returned and gold.get(doc, 0) > 0 for doc in rescued),
        rescue_proposed_negative=sum(gold.get(doc) == 0 for doc in rescued),
        rescue_accepted_negative=sum(doc in returned and gold.get(doc) == 0 for doc in rescued),
        rescue_beyond_local=sum(doc in rescued and j >= LOCAL
                                                   for j, doc in enumerate([pool[k] for k in topical])),
    )


def calibrate(fixture, embedding, reader):
    data, matrix, positions = fixture
    previous = [simulate(pair, data, (matrix, positions), embedding, reader,
                         "flat_six", float("inf")) for pair in data["pairs"]]
    negatives = [margin for pair,row in zip(data["pairs"], previous, strict=True)
                 for doc,margin in row["margins"].items() if pair["qrels"][1].get(doc) == 0]
    threshold = operating_threshold(np.asarray(negatives), 10)
    return dict(threshold=threshold, negatives=len(negatives),
                accepted=sum(value > threshold for value in negatives))


def evaluate(fixture, embedding, reader, cutoff):
    data, matrix, positions = fixture
    results = {arm: [] for arm in ARMS}
    for i, pair in enumerate([data["pairs"][0], *data["pairs"]]):
        offset = i % len(ARMS)
        records = {arm: simulate(pair, data, (matrix, positions), embedding, reader, arm, cutoff)
                   for arm in ARMS[offset:] + ARMS[:offset]}
        anchor = records["depth_on_abstention"]
        for arm in ("constraint_local", "constraint_full_pool"):
            candidate = records[arm]
            if candidate["selected"] != anchor["selected"] or bool(candidate["rescued"]) != bool(
                    anchor["rescued"]) or candidate["checks"] != anchor["checks"]:
                raise ValueError("Adaptive policies diverged before the rescue decision")
            if any(candidate["margins"][doc] != anchor["margins"][doc] for doc in anchor["selected"]):
                raise ValueError("Initial scores differ across adaptive routes")
        if i:
            for arm, result in records.items():
                results[arm].append(result)
    reference = results["depth_on_abstention"]
    output = {}
    for arm, rows in results.items():
        paired = list(zip(rows, reference, strict=True))
        counts = {key: sum(row[key] for row in rows) for key in (
            "grounded", "issued", "negative", "unjudged", "invalidated", "checks", "rescue_beyond_local",
            "rescue_proposed_positive", "rescue_accepted_positive",
            "rescue_proposed_negative", "rescue_accepted_negative")}
        output[arm] = dict(counts, rescued_queries=sum(bool(row["rescued"]) for row in rows),
                           queries_with_grounded=sum(row["grounded"] > 0 for row in rows),
                           p95_ms=float(np.quantile([row["milliseconds"] for row in rows],.95)),
                           paired_gain={field: paired_interval(
                               [(a[field]-b[field])/(FIRST+RESCUE) for a,b in paired])
                               for field in ("grounded","negative","unjudged","invalidated")})
    return output


def run(core, core_cache, news, news_cache, embedding, reranker, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    packed = {"core": load(core, core_cache, provenance), "news": load(news, news_cache, provenance)}
    margins = {name: calibrate(fixture, encoder, reader) for name, fixture in packed.items()}
    diagnostics = {f"{source}_to_{target}": dict(
        topics=len(packed[target][0]["pairs"]),
        arms=evaluate(packed[target], encoder, reader, margins[source]["threshold"]))
        for source,target in (("core","news"),("news","core"))}
    followir.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=followir.digest(Path(__file__)), embedding_revision=provenance["revision"],
        reranker_revision=reader.provenance["revision"],
        dataset_hashes=[followir.digest(p) for p in (core,news)],
        cache_hashes=[followir.digest(p) for p in (core_cache,news_cache)],
        calibration=margins, first=FIRST, rescue=RESCUE, local=LOCAL, transfer=diagnostics,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "core_cache", "news", "news_cache", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
