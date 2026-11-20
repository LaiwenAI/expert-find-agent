"""Open a fixed-budget role search only when the original role evidence is thin."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import elife_role_review as review
from experiments.credit import nomination_label_audit as audit
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = ("topic_four", "topic_eight", "person_four", "empty_prefix", "one_witness")


def route(catalog, ranked, pair, pct, arm):
    prefix = ranked[:4]
    if arm in ("topic_four", "topic_eight", "person_four"):
        return holdout.select(catalog, ranked, pair, pct, arm), arm == "person_four"
    visible = catalog.visible[pct]
    available_witnesses = catalog.person[pair]
    maximum = {"empty_prefix": 0, "one_witness": 1}[arm]
    expand = all(article in visible for article in prefix) and (
        sum(article in available_witnesses for article in prefix) <= maximum)
    return (holdout.select(catalog, ranked, pair, pct, "person_four") if expand else prefix), expand


def measure(catalog, topic, pair, ranked, pct, arm, rank_ms):
    started = perf_counter()
    selected, expand = route(catalog, ranked, pair, pct, arm)
    values = catalog.verify(topic, selected)
    result = nomination.evaluate(catalog, topic, pair, selected, values, holdout.CUTOFF)
    return dict(
        result, topic=topic, reads=len(selected), expanded=expand,
        ms=rank_ms + (perf_counter() - started) * 1000,
        characters=sum(len(catalog.texts[source]) for source in selected),
    )


def published_control(reference, panel, pct, arm):
    if panel == "plos_topics":
        return reference["holdout"][str(pct)][arm]["vs_topic_four"]
    return reference["run"][panel]["masks"][str(pct)][arm]["topic_four"]


def validate_and_summarize(catalog, panel, by_mask, reference):
    out = {}
    for pct, arms in by_mask.items():
        for arm in ("topic_four", "topic_eight", "person_four"):
            published = published_control(reference, panel, pct, arm)
            for metric in ("reads", "grounded", "cited_off_label", "issued"):
                if sum(row[metric] for row in arms[arm]) != published[metric]:
                    raise ValueError(f"Published control differs: {panel}, {pct}, {arm}, {metric}")
        for new, base, expanded in zip(arms["empty_prefix"], arms["topic_four"],
                                       arms["person_four"], strict=True):
            expected = expanded if new["expanded"] else base
            if new["witnesses"] != expected["witnesses"] or (new["expanded"] and base["issued"]):
                raise ValueError("The empty-prefix route did not preserve the existing recommendation")
        for new, base, expanded in zip(arms["one_witness"], arms["topic_four"],
                                       arms["person_four"], strict=True):
            expected = expanded if new["expanded"] else base
            if new["witnesses"] != expected["witnesses"]:
                raise ValueError("The one-witness route differs from its selected control")
        base = arms["topic_four"]
        records = {}
        for arm, rows in arms.items():
            targets = [dict(topic=row["topic"],
                            citations=[catalog.articles[i]["doi"] for _, i in row["witnesses"]])
                       for row in rows]
            baseline = [dict(topic=row["topic"],
                             citations=[catalog.articles[i]["doi"] for _, i in row["witnesses"]])
                        for row in base]
            labels = {
                audit.case_id(row["topic"], catalog.articles[i]["doi"]): "relevant"
                for row in (*rows, *base) for _, i in row["witnesses"] if i in catalog.topics[row["topic"]]
            }
            comparisons = {
                control: holdout.analyze(rows, arms[control]) for control in ("topic_four", "topic_eight")
            }
            records[arm] = dict(
                compared=comparisons, expanded=sum(row["expanded"] for row in rows),
                requests_answered=sum(row["issued"] > 0 for row in rows),
                requests_newly_answered=sum(a["issued"] > 0 and b["issued"] == 0
                                            for a, b in zip(rows, base, strict=True)),
                bounds_all_unjudged=review.compare(targets, baseline, {}),
                bounds_tagged_supported=review.compare(targets, baseline, labels),
            )
        out[pct] = records
    return out


def evaluate(catalog, topics, panel, reference):
    queries = list(pilot.query_panel(catalog, topics))
    results = {pct: {arm: [] for arm in ARMS} for pct in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        started = perf_counter()
        ranked = catalog.rank(topic)
        rank_ms = (perf_counter() - started) * 1000
        shift = index % len(ARMS)
        mask_order = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for pct in mask_order:
            for arm in ARMS[shift:] + ARMS[:shift]:
                results[pct][arm].append(measure(catalog, topic, pair, ranked, pct, arm, rank_ms))
    outcome = validate_and_summarize(catalog, panel, results, reference)
    return dict(
        queries=len(queries), topics=len({topic for topic, _, _ in queries}),
        indexed={pct: len(ids) for pct, ids in catalog.visible.items()}, masks=outcome,
    )


def run(plos, plos_manifest, plos_reference, elife, elife_manifest, elife_reference,
        embedding, reranking, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    for path, manifest in ((plos, plos_manifest), (elife, elife_manifest)):
        if pilot.digest(path) != json.loads(manifest.read_bytes())["projection_sha256"]:
            raise ValueError(f"Projection differs: {path}")
    plos_scores = json.loads(plos_reference.read_bytes())
    elife_scores = json.loads(elife_reference.read_bytes())
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranking)
    for reference in (plos_scores, elife_scores["run"]):
        if (encoder_info["revision"], reader.provenance["revision"]) != (
                reference["embedding"], reference["reranker"]):
            raise ValueError("Expected the frozen checkpoints")

    by_panel = {}
    for name, path, reference in (("plos", plos, plos_scores), ("elife", elife, elife_scores)):
        articles = json.loads(path.read_bytes())["sources"]
        catalog = pilot.Catalog(articles, encoder, reader)
        limit = 4 if name == "plos" else 8
        topics = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= limit)
        if name == "elife":
            complete = {i for i, article in enumerate(articles) if article["declaration_complete"]}
            catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
        catalog.verify(topics[0], catalog.rank(topics[0])[:8])
        panel = "plos_topics" if name == "plos" else "headings"
        by_panel[panel] = evaluate(catalog, topics, panel, reference)
        if name == "elife":
            catalog.topics.clear()
            for source, article in enumerate(articles):
                for topic in article["keywords"]:
                    catalog.topics[topic].add(source)
            keywords = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= 4)
            by_panel["keywords"] = evaluate(catalog, keywords, "keywords", reference)

    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), cutoff=holdout.CUTOFF,
        embedding=encoder_info["revision"], reranker=reader.provenance["revision"],
        input_digests={name: pilot.digest(path) for name, path in (
            ("plos_projection", plos), ("plos_manifest", plos_manifest), ("plos_reference", plos_reference),
            ("elife_projection", elife), ("elife_manifest", elife_manifest), ("elife_reference", elife_reference),
        )}, panels=by_panel,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "plos_manifest", "plos_reference", "elife", "elife_manifest", "elife_reference",
                 "embedding", "reranking", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
