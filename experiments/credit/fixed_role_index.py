"""Choose a single declaration index from past topic ranks, then replay expert search."""

import argparse
from collections import Counter
import json
from math import fsum, log1p
import os
from pathlib import Path

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import declaration_coverage as coverage
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.credit.legacy_contribution_text import stable


BUDGETS = (20, 39)
HASH_DRAWS = 12
METHODS = ("topic_four", "topic_eight", "person_four", "bounded_four")


def allocation(doi, histories, maximum):
    """Use topic-rank exposure without topic tags or contributor roles from the target corpus."""
    exposures = [{} for _ in doi]
    for topic_index, ranked in enumerate(histories):
        for rank, article in enumerate(ranked[:8], 1):
            exposures[article][topic_index] = 1 / rank
    by_exposure = sorted(range(len(doi)), key=lambda i: (-fsum(exposures[i].values()), doi[i]))
    covered = [0.] * len(histories)
    balanced = []
    remaining = set(range(len(doi)))
    for _ in range(maximum):
        def priority(article):
            gain = fsum(log1p(weight / (1 + covered[topic])) for topic, weight in exposures[article].items())
            return -gain, doi[article]

        article = min(remaining, key=priority)
        remaining.remove(article)
        balanced.append(article)
        for topic, weight in exposures[article].items():
            covered[topic] += weight
    return {
        "exposure": by_exposure,
        "balanced": balanced,
        **{
            f"hash_{seed:02}": sorted(range(len(doi)), key=lambda i: stable("fixed-role-index", seed, doi[i]))
            for seed in range(HASH_DRAWS)
        },
    }


def replay(catalog, masks, past):
    queries = batches.queries(catalog)
    estimates = {}
    for budget, policies in masks.items():
        estimates[budget] = {
            policy: {panel: {arm: Counter() for arm in METHODS} for panel in ("novel", "seen")}
            for policy in policies
        }
    batches_scored = 0
    for topic, _, pair in queries:
        ranked = catalog.rank(topic)
        panel = "seen" if topic in past else "novel"
        scores = {}
        for budget, policies in masks.items():
            for policy, visible in policies.items():
                for arm, total in estimates[budget][policy][panel].items():
                    selected = tuple(coverage.select(catalog, ranked, pair, visible, arm))
                    if selected not in scores:
                        scores[selected] = catalog.verify(topic, selected)
                    row = nomination.evaluate(
                        catalog, topic, pair, selected, scores[selected], holdout.CUTOFF)
                    total.update(
                        queries=1, reads=len(selected), grounded=row["grounded"],
                        cited_off_label=row["cited_off_label"], issued=row["issued"],
                        front_four_visible=len(visible.intersection(ranked[:4])),
                        no_tagged_role_source=int(not any(
                            i in catalog.person[pair] and i in catalog.topics[topic] for i in selected)),
                    )
        batches_scored += sum(bool(batch) for batch in scores)
    return estimates, batches_scored


def run(pilot_projection, projection, manifest, embedding, reranker, xml_bytes, reference, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    if (pilot.digest(pilot_projection), pilot.digest(projection)) != (
            frozen["original_projection_sha256"], frozen["projection_sha256"]):
        raise ValueError("Projection differs from the frozen catalog")
    expected = json.loads(reference.read_bytes())
    costs = json.loads(xml_bytes.read_bytes())
    articles = json.loads(projection.read_bytes())["sources"]
    if set(costs) != {row["source_sha256"] for row in articles} or any(
            type(value) is not int or value <= 0 for value in costs.values()):
        raise ValueError("Provide positive byte counts for every pinned XML snapshot")
    past = Counter(topic for row in json.loads(pilot_projection.read_bytes())["sources"]
                   for topic in row["topics"])
    history = sorted(topic for topic, count in past.items() if count >= 4)
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"], pilot.digest(manifest)) != (
            expected["embedding"], expected["reranker"], expected["manifest_sha256"]):
        raise ValueError("Model or manifest differs from the reference")
    catalog = pilot.Catalog(articles, encoder, reader)
    if len(catalog.visible[100]) != len(articles):
        raise ValueError("This replay requires a catalog with complete declarations for masking")
    rankings = [catalog.rank(topic) for topic in history]
    ordered = allocation([row["doi"] for row in articles], rankings, max(BUDGETS))
    masks = {budget: {policy: set(order[:budget]) for policy, order in ordered.items()}
             for budget in BUDGETS}
    topic, _, _ = batches.queries(catalog)[0]
    catalog.verify(topic, catalog.rank(topic)[:8])
    results, scored = replay(catalog, masks, set(history))
    for budget, policies in masks.items():
        for policy in policies:
            for arm in ("topic_four", "topic_eight"):
                for metric in ("reads", "grounded", "cited_off_label", "issued"):
                    observed = sum(results[budget][policy][panel][arm][metric] for panel in ("novel", "seen"))
                    if observed != expected["holdout"]["100"][arm]["vs_topic_four"][metric]:
                        raise ValueError(f"Reference mismatch for {arm}, {metric}")
    xml_loads = {
        budget: {policy: sum(costs[articles[i]["source_sha256"]] for i in visible)
                 for policy, visible in policies.items()}
        for budget, policies in masks.items()
    }
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), projection_sha256=pilot.digest(projection),
        pilot_projection_sha256=pilot.digest(pilot_projection), manifest_sha256=pilot.digest(manifest),
        xml_bytes_sha256=pilot.digest(xml_bytes), reference_sha256=pilot.digest(reference),
        embedding=encoder_info["revision"], reranker=reader.provenance["revision"],
        historical_topics=len(history), model_batches=scored, xml_bytes=xml_loads, results=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot_projection", "projection", "manifest", "embedding", "reranker",
                 "xml_bytes", "reference", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
