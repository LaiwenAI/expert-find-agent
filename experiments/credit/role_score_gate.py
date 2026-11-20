"""Calibrate a topic-similarity floor for injecting new role sources."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import role_empty_prefix as previous
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


CONTROL = ("topic_four", "topic_eight", "person_four", "bounded_four")
MARGINS = (-0.10, -0.05, -0.025, -0.01, -0.005, 0.0, 0.01, 0.025, 0.05, 0.10)
BASELINE_DELTA_CEILING = .01


def choose_sources(catalog, ranked, scores, pair, pct, arm, margin=None):
    if arm == "bounded_four":
        return holdout.select(catalog, ranked[:8], pair, pct, "person_four")
    if arm in CONTROL:
        return holdout.select(catalog, ranked, pair, pct, arm)
    first = ranked[:4]
    if margin is None:
        return first
    floor = min(scores[i] for i in first) - margin
    complete, postings = catalog.visible[pct], catalog.person[pair]
    selected = {
        i for i in first if i not in complete or i in postings
    }
    for i in ranked[4:]:
        if len(selected) == 4:
            break
        if scores[i] >= floor and (i not in complete or i in postings):
            selected.add(i)
    for i in first:
        if len(selected) == 4:
            break
        selected.add(i)
    return [i for i in ranked if i in selected]


def collect(catalog, minimum_topic_size, arms, cache_scores):
    topics = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= minimum_topic_size)
    queries = list(pilot.query_panel(catalog, topics))
    result = {pct: {arm: [] for arm, _ in arms} for pct in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        start = perf_counter()
        ranked, scores = catalog.rank(topic, return_dense_scores=True)
        rank_ms = (perf_counter() - start) * 1000
        shift = index % len(arms)
        repeats = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        verified = {}
        for pct in repeats:
            for arm, margin in arms[shift:] + arms[:shift]:
                start = perf_counter()
                chosen = choose_sources(catalog, ranked, scores, pair, pct, arm, margin)
                selected = tuple(chosen)
                if not cache_scores:
                    values = catalog.verify(topic, selected)
                else:
                    if selected not in verified:
                        verified[selected] = catalog.verify(topic, selected)
                    values = verified[selected]
                evaluated = nomination.evaluate(catalog, topic, pair, selected, values, holdout.CUTOFF)
                result[pct][arm].append(dict(
                    evaluated, topic=topic, reads=len(selected), injected=len(set(selected) - set(ranked[:4])),
                    ms=rank_ms + (perf_counter() - start) * 1000,
                    characters=sum(len(catalog.texts[i]) for i in selected),
                ))
    return result


def locked_margin(results):
    def totals(rows):
        return {k: sum(row[k] for row in rows) for k in ("grounded", "cited_off_label", "injected")}
    frontier = []
    for margin in (None, *MARGINS):
        key = f"gate_{margin}"
        by_mask = {pct: totals(arms[key]) for pct, arms in results.items()}
        eligible = all(
            by_mask[pct]["cited_off_label"] - totals(arms["topic_four"])["cited_off_label"]
            <= BASELINE_DELTA_CEILING * pilot.TOP * len(arms["topic_four"])
            for pct, arms in results.items()
        )
        frontier.append(dict(margin=margin, masks=by_mask, eligible=eligible))
    best = max((x for x in frontier if x["eligible"]), key=lambda row: (
        sum(m["grounded"] for m in row["masks"].values()),
        -sum(m["cited_off_label"] for m in row["masks"].values()),
        -sum(m["injected"] for m in row["masks"].values()),
    ))
    return best["margin"], frontier


def evaluate(catalog, panel, threshold, minimum_size, published):
    controls = tuple((arm, None) for arm in CONTROL)
    result = collect(catalog, minimum_size, (*controls, ("gated_four", threshold)), cache_scores=False)
    summaries = {}
    for pct, arms in result.items():
        for arm in CONTROL:
            if arm == "bounded_four" and panel != "plos_topics":
                control = published["run"][panel]["masks"][str(pct)][arm]["topic_four"]
            else:
                control = previous.published_control(published, panel, pct, arm)
            for metric in ("grounded", "cited_off_label", "issued", "reads"):
                if sum(r[metric] for r in arms[arm]) != control[metric]:
                    raise ValueError(f"Frozen counts differ for {panel}, {pct}, {arm}, {metric}")
        summaries[pct] = {
            arm: dict(
                injected=sum(r["injected"] for r in rows),
                requests_answered=sum(bool(r["issued"]) for r in rows),
                compared={base: holdout.analyze(rows, arms[base]) for base in ("topic_four", "topic_eight")},
            )
            for arm, rows in arms.items()
        }
    return summaries


def run(pilot_projection, plos, plos_manifest, plos_reference, elife, elife_manifest,
        elife_reference, embedding, reranking, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    manifest = json.loads(plos_manifest.read_bytes())
    for path, expected in (
        (pilot_projection, manifest["original_projection_sha256"]),
        (plos, manifest["projection_sha256"]),
        (elife, json.loads(elife_manifest.read_bytes())["projection_sha256"]),
    ):
        if pilot.digest(path) != expected:
            raise ValueError(f"Unexpected projection {path}")
    references = [json.loads(path.read_bytes()) for path in (plos_reference, elife_reference)]
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranking)
    for reference in (references[0], references[1]["run"]):
        if (encoder_info["revision"], reader.provenance["revision"]) != (
                reference["embedding"], reference["reranker"]):
            raise ValueError("Expected the frozen checkpoints")

    def catalog_at(path):
        return pilot.Catalog(json.loads(path.read_bytes())["sources"], encoder, reader)

    development = catalog_at(pilot_projection)
    warm_topic = sorted(development.topics)[0]
    development.verify(warm_topic, development.rank(warm_topic)[:8])
    calibrators = (("topic_four", None),) + tuple((f"gate_{m}", m) for m in (None, *MARGINS))
    threshold, frontier = locked_margin(collect(development, 4, calibrators, cache_scores=True))
    frozen_plos = catalog_at(plos)
    plos_result = evaluate(frozen_plos, "plos_topics", threshold, 4, references[0])
    frozen_elife = catalog_at(elife)
    complete = {i for i, article in enumerate(frozen_elife.articles) if article["declaration_complete"]}
    frozen_elife.visible = {pct: ids & complete for pct, ids in frozen_elife.visible.items()}
    heading_result = evaluate(frozen_elife, "headings", threshold, 8, references[1])
    frozen_elife.topics.clear()
    for source, article in enumerate(frozen_elife.articles):
        for topic in article["keywords"]:
            frozen_elife.topics[topic].add(source)
    keyword_result = evaluate(frozen_elife, "keywords", threshold, 4, references[1])
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), cutoff=holdout.CUTOFF,
        ranker_sha256=pilot.digest(Path(pilot.__file__)),
        plos_manifest_sha256=pilot.digest(plos_manifest), elife_manifest_sha256=pilot.digest(elife_manifest),
        embedding=encoder_info["revision"], reranker=reader.provenance["revision"],
        selected_margin=threshold, calibration=frontier, panels=dict(
            plos_topics=plos_result, elife_headings=heading_result, elife_keywords=keyword_result),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot_projection", "plos", "plos_manifest", "plos_reference", "elife",
                 "elife_manifest", "elife_reference", "embedding", "reranking", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
