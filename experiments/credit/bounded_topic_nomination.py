"""Test whether topical bounds and pilot-only gating improve role nomination."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ROLE_ARMS = ("union_four", "person_four", "bounded_union_four", "bounded_four")
ARMS = ("topic_four", "topic_eight", *ROLE_ARMS)
CUTOFFS = (holdout.CUTOFF, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def evaluate(catalog, topic, pair, selected, values, cutoff):
    witnesses = {}
    for article, score in sorted(zip(selected, values, strict=True), key=lambda row: -row[1]):
        if score <= cutoff:
            continue
        for owner in catalog.bindings.get((article, pair), ()):
            witnesses.setdefault(owner, article)
            if len(witnesses) == pilot.TOP:
                break
        if len(witnesses) == pilot.TOP:
            break
    grounded = sum(article in catalog.topics[topic] for article in witnesses.values())
    return dict(grounded=grounded, cited_off_label=len(witnesses) - grounded,
                issued=len(witnesses), witnesses=tuple(witnesses.items()))


def measure(catalog, topic, pair, ranked, cover, arm, cutoffs, rank_ms):
    started = perf_counter()
    bounded = {"bounded_four": "person_four", "bounded_union_four": "union_four"}
    selected = holdout.select(catalog, ranked[:8] if arm in bounded else ranked,
                              pair, cover, bounded.get(arm, arm))
    values = catalog.verify(topic, selected) if selected else []
    inference_ms = (perf_counter() - started) * 1000
    characters = sum(len(catalog.texts[index]) for index in selected)
    observations = {}
    for cutoff in cutoffs:
        started = perf_counter()
        result = evaluate(catalog, topic, pair, selected, values, cutoff)
        elapsed = rank_ms + inference_ms + (perf_counter() - started) * 1000
        observations[cutoff] = dict(result, topic=topic, reads=len(selected), characters=characters,
                                    ms=elapsed, beyond_eight=len(set(selected) - set(ranked[:8])))
    return observations


def collect(catalog, thresholds):
    queries = batches.queries(catalog)
    result = {cover: {arm: {cutoff: [] for cutoff in thresholds[arm]} for arm in ARMS}
              for cover in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        started = perf_counter()
        ranked = catalog.rank(topic)
        rank_ms = (perf_counter() - started) * 1000
        shift = index % len(ARMS)
        covers = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for cover in covers:
            for arm in ARMS[shift:] + ARMS[:shift]:
                measured = measure(catalog, topic, pair, ranked, cover, arm, thresholds[arm], rank_ms)
                for cutoff, row in measured.items():
                    result[cover][arm][cutoff].append(row)
    return result


def calibrate(development):
    """Allow at most the pilot off-label count of topic_four under either mask."""
    chosen = {}
    frontier = {}
    ceilings = {cover: sum(r["cited_off_label"] for r in arms["topic_four"][holdout.CUTOFF])
                for cover, arms in development.items()}
    for arm in ROLE_ARMS:
        candidates = []
        for cutoff in CUTOFFS:
            metrics = {cover: {key: sum(row[key] for row in arms[arm][cutoff])
                               for key in ("grounded", "cited_off_label")}
                       for cover, arms in development.items()}
            candidates.append(dict(cutoff=cutoff, masks=metrics,
                                   eligible=all(metrics[c]["cited_off_label"] <= ceilings[c] for c in metrics)))
        feasible = [row for row in candidates if row["eligible"]]
        if feasible:
            chosen[arm] = max(feasible, key=lambda row: (
                sum(m["grounded"] for m in row["masks"].values()),
                -sum(m["cited_off_label"] for m in row["masks"].values()),
                row["cutoff"],
            ))["cutoff"]
        frontier[arm] = candidates
    return chosen, ceilings, frontier


def run(pilot_projection, projection, manifest, embedding, reranking, output, witness_output=None):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if (output.exists() or (witness_output is not None and witness_output.exists())
            or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    if (pilot.digest(pilot_projection), pilot.digest(projection)) != (
            frozen["original_projection_sha256"], frozen["projection_sha256"]):
        raise ValueError("Unexpected topic projections")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reranker = Reranker(reranking)
    development = pilot.Catalog(json.loads(pilot_projection.read_bytes())["sources"], encoder, reranker)
    articles = json.loads(projection.read_bytes())["sources"]
    target = pilot.Catalog(articles, encoder, reranker)
    initial = {arm: CUTOFFS if arm in ROLE_ARMS else (holdout.CUTOFF,)
               for arm in ARMS}
    topic, _, _ = batches.queries(development)[0]
    development.verify(topic, development.rank(topic)[:8])
    chosen, ceilings, frontier = calibrate(collect(development, initial))
    frozen_cutoffs = {arm: sorted({holdout.CUTOFF, chosen.get(arm, holdout.CUTOFF)}) for arm in ARMS}
    tested = collect(target, frozen_cutoffs)
    result = {}
    witnesses = {}
    for cover, arms in tested.items():
        result[cover] = {}
        witnesses[cover] = {}
        for arm, by_cutoff in arms.items():
            for cutoff, rows in by_cutoff.items():
                name = arm if cutoff == holdout.CUTOFF else f"{arm}_pilot_calibrated"
                result[cover][name] = dict(cutoff=cutoff, beyond_eight=sum(r["beyond_eight"] for r in rows))
                if witness_output is not None:
                    witnesses[cover][name] = [
                        dict(topic=row["topic"], citations=[articles[source]["doi"] for _, source in row["witnesses"]])
                        for row in rows
                    ]
                for baseline in ("topic_four", "topic_eight"):
                    result[cover][name][f"vs_{baseline}"] = holdout.analyze(rows, arms[baseline][holdout.CUTOFF])
                if arm in ("person_four", "bounded_four"):
                    control = "union_four" if arm == "person_four" else "bounded_union_four"
                    control_cutoff = chosen.get(control, holdout.CUTOFF) if cutoff != holdout.CUTOFF else holdout.CUTOFF
                    result[cover][name]["vs_matched_union"] = holdout.analyze(rows, arms[control][control_cutoff])
    pilot.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 script_sha256=pilot.digest(Path(__file__)), manifest_sha256=pilot.digest(manifest),
                                 embedding=encoder_info["revision"], reranker=reranker.provenance["revision"],
                                 pilot_off_label_ceilings=ceilings, pilot_thresholds=chosen,
                                 pilot_frontier=frontier, holdout=result))
    if witness_output is not None:
        pilot.write_json(witness_output, dict(projection_sha256=pilot.digest(projection),
                                             results_sha256=pilot.digest(output), witnesses=witnesses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot_projection", "projection", "manifest", "embedding", "reranking", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--witness-output", type=Path)
    run(**vars(parser.parse_args()))
