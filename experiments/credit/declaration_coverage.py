"""Replay expert nomination under different locations of missing role declarations."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.credit.legacy_contribution_text import stable


COVERAGE_PERCENT = (0, 25, 50, 75, 100)
REFERENCE_ARMS = ("topic_four", "topic_eight", "person_four", "bounded_four")
MASKS = ("hash_0", "hash_1", "hash_2", "front_visible", "front_hidden")


def make_masks(catalog, ranked, count):
    complete = catalog.visible[100]
    choices = [article for article in ranked if article in complete]
    random_orders = [
        sorted(complete, key=lambda i: stable("coverage-replay", seed, catalog.articles[i]["doi"]))
        for seed in range(3)
    ]
    orders = (*random_orders, choices, list(reversed(choices)))
    return {name: set(order[:count]) for name, order in zip(MASKS, orders, strict=True)}


def select(catalog, ranked, pair, visible, arm):
    if arm == "topic_four":
        return ranked[:4]
    if arm == "topic_eight":
        return ranked[:8]
    available = ranked[:8] if arm == "bounded_four" else ranked
    bindings = catalog.person[pair]
    return [i for i in available if i in bindings or i not in visible][:4]


def collect(catalog):
    queries = batches.queries(catalog)
    complete = len(catalog.visible[100])
    counts = {pct: (pct * complete + 50) // 100 for pct in COVERAGE_PERCENT}
    result = {pct: {mask: {arm: Counter() for arm in REFERENCE_ARMS} for mask in MASKS}
              for pct in counts}
    anchors = {pct: {arm: Counter() for arm in REFERENCE_ARMS} for pct in pilot.CACHE}
    scored = Counter()
    for topic, _, pair in queries:
        ranked = catalog.rank(topic)
        per_query_scores = {}

        def measure(visible, totals):
            for arm, total in totals.items():
                selected = tuple(select(catalog, ranked, pair, visible, arm))
                if selected not in per_query_scores:
                    per_query_scores[selected] = catalog.verify(topic, selected)
                row = nomination.evaluate(
                    catalog, topic, pair, selected, per_query_scores[selected], holdout.CUTOFF)
                total.update({metric: row[metric] for metric in ("grounded", "cited_off_label", "issued")})
                total.update(
                    reads=len(selected),
                    no_tagged_role_source=int(not any(
                        i in catalog.person[pair] and i in catalog.topics[topic] for i in selected)),
                    front_four_visible=len(visible.intersection(ranked[:4])),
                )

        for pct, count in counts.items():
            for name, visible in make_masks(catalog, ranked, count).items():
                measure(visible, result[pct][name])
        for pct, totals in anchors.items():
            measure(catalog.visible[pct], totals)
        scored.update(unique_queries_batches=len(per_query_scores),
                      model_batches=sum(bool(batch) for batch in per_query_scores))
    return dict(queries=len(queries), topics=len({topic for topic, _, _ in queries}),
                indexed=counts, anchor_indexed={pct: len(v) for pct, v in catalog.visible.items()},
                score_batch_counts=scored, anchors=anchors, coverage=result)


def run(projection, manifest, embedding, reranker, reference, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    if pilot.digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Projection differs from the frozen catalog")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    expected = json.loads(reference.read_bytes())
    if (encoder_info["revision"], reader.provenance["revision"], pilot.digest(manifest)) != (
            expected["embedding"], expected["reranker"], expected["manifest_sha256"]):
        raise ValueError("Model or manifest differs from the reference")
    catalog = pilot.Catalog(json.loads(projection.read_bytes())["sources"], encoder, reader)
    topic, _, _ = batches.queries(catalog)[0]
    catalog.verify(topic, catalog.rank(topic)[:8])
    observed = collect(catalog)
    for pct, arms in observed["anchors"].items():
        for arm, metrics in arms.items():
            original = expected["holdout"][str(pct)][arm]["vs_topic_four"]
            for name in ("reads", "grounded", "cited_off_label", "issued"):
                if metrics[name] != original[name]:
                    raise ValueError(f"Reference replay differs for {pct}, {arm}, {name}")
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)),
        projection_sha256=pilot.digest(projection), manifest_sha256=pilot.digest(manifest),
        reference_sha256=pilot.digest(reference), embedding=encoder_info["revision"],
        reranker=reader.provenance["revision"], cutoff=holdout.CUTOFF, **observed,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "embedding", "reranker", "reference", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
