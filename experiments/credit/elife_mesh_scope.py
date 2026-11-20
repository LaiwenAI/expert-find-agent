"""Compare broad and major MeSH assignment on identical expert-search responses."""

import argparse
import json
import os
from pathlib import Path

from experiments.credit import elife_mesh as mesh
from experiments.credit import role_score_gate as gate
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


MINIMUM_SUPPORT = 8
MAXIMUM_SUPPORT = 40


def major_scores(rows, major):
    """Measure major-topic hits on the same recommendations used for ordinary descriptors."""
    return [
        dict(row, grounded=sum(source in major.get(row["topic"], ()) for _, source in row["witnesses"]))
        for row in rows
    ]


def summarize(measurements, major, topics):
    results = {}
    for pct, arms in measurements.items():
        results[pct] = {}
        for arm, all_rows in arms.items():
            rows = [r for r in all_rows if r["topic"] in topics]
            if not rows:
                continue
            by_base = {}
            for base in ("topic_four", "topic_eight"):
                baseline = [r for r in arms[base] if r["topic"] in topics]
                major_arm = major_scores(rows, major)
                major_base = major_scores(baseline, major)
                summary = holdout.analyze(rows, baseline)
                summary["major_hits"] = sum(row["grounded"] for row in major_arm)
                summary["major_delta"] = holdout.interval(
                    [(a["grounded"] - b["grounded"]) / pilot.TOP
                     for a, b in zip(major_arm, major_base, strict=True)],
                    [row["topic"] for row in rows],
                )
                by_base[base] = summary
            results[pct][arm] = dict(
                requests_answered=sum(bool(row["issued"]) for row in rows),
                injected=sum(row["injected"] for row in rows), compared=by_base,
            )
    return results


def run(projection, snapshot, mesh_reference, gate_reference, embedding, reranker, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    frozen = json.loads(snapshot.read_bytes())
    prior = json.loads(mesh_reference.read_bytes())["run"]
    calibrated = json.loads(gate_reference.read_bytes())["expanded_grid"]
    if (pilot.digest(projection), pilot.digest(snapshot), pilot.digest(gate_reference)) != (
            prior["projection_sha256"], prior["snapshot_sha256"], prior["gate_reference_sha256"]):
        raise ValueError("Mismatch with the prior major-descriptor study")
    if pilot.digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Unexpected DOI projection")
    margin = calibrated["selected_margin"]
    if margin != prior["locked_margin"] or calibrated["cutoff"] != holdout.CUTOFF:
        raise ValueError("The PLOS gate is frozen")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (
            calibrated["embedding"], calibrated["reranker"]):
        raise ValueError("Expected the frozen checkpoints")
    articles = json.loads(projection.read_bytes())["sources"]
    catalog = pilot.Catalog(articles, encoder, reader)
    complete = {i for i, article in enumerate(articles) if article["declaration_complete"]}
    catalog.visible = {pct: known & complete for pct, known in catalog.visible.items()}
    excluded = mesh.ROUTINE_DESCRIPTORS

    coverage = mesh.assign_topics(catalog, frozen["records"], excluded)
    major = dict(catalog.topics)
    mesh.assign_topics(catalog, frozen["records"], excluded, major_only=False)
    for term in tuple(catalog.topics):
        if not MINIMUM_SUPPORT <= len(catalog.topics[term]) <= MAXIMUM_SUPPORT:
            del catalog.topics[term]
    topic_choices = sorted(catalog.topics)
    warm_topic = topic_choices[0]
    catalog.verify(warm_topic, catalog.rank(warm_topic)[:8])
    panel = list(pilot.query_panel(catalog, topic_choices))
    topics = {topic for topic, _, _ in panel}
    major_reachable = sum(bool(major.get(topic, set()) & catalog.person[pair]) for topic, _, pair in panel)
    previous_terms = {term.casefold() for article in articles for field in ("topics", "keywords")
                      for term in article[field]}
    novel = {name for name in topics if name.casefold() not in previous_terms}
    controls = tuple((arm, None) for arm in gate.CONTROL)
    rows = gate.collect(catalog, MINIMUM_SUPPORT, (*controls, ("gated_four", margin)), cache_scores=False)
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), snapshot_sha256=pilot.digest(snapshot),
        projection_sha256=pilot.digest(projection), mesh_reference_sha256=pilot.digest(mesh_reference),
        gate_reference_sha256=pilot.digest(gate_reference),
        locked_margin=margin, evidence_cutoff=holdout.CUTOFF, coverage=coverage,
        eligible_topics=len(topic_choices), evaluated_topics=len(topics), queries=len(panel),
        major_reachable_queries=major_reachable, topics_absent_from_publisher_terms=len(novel),
        declaration_indexed={pct: len(ids) for pct, ids in catalog.visible.items()},
        all_requests=summarize(rows, major, topics),
        terms_absent_from_publisher_metadata=summarize(rows, major, novel),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "snapshot", "mesh_reference", "gate_reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
