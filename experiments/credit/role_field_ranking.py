"""Reserve verification budget for title-focused nomination using one query encoding."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_mesh_scope as scope
from experiments.credit import mesh_stage_diagnostic as diagnostic
from experiments.credit import role_empty_prefix as controls
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


BASELINES = ("topic_four", "topic_eight", "person_four")
ARMS = (*BASELINES, "topic_title_four", "role_title_four", "title_role_four")


def field_candidates(ranked, title_ranked, complete, posting, *, full_reserved, use_roles):
    def eligible(source):
        return not use_roles or source in posting or source not in complete

    chosen = []
    for ordering, limit in ((ranked, full_reserved), (title_ranked, 4), (ranked, 4)):
        for source in ordering:
            if len(chosen) >= limit:
                break
            if source not in chosen and eligible(source):
                chosen.append(source)
    for source in ranked:
        if len(chosen) >= 4:
            break
        if source not in chosen:
            chosen.append(source)
    return chosen


def collect(catalog, title_vectors, minimum_support):
    topics = sorted(topic for topic, ids in catalog.topics.items() if len(ids) >= minimum_support)
    queries = list(pilot.query_panel(catalog, topics))
    results = {pct: {arm: [] for arm in ARMS} for pct in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        started = perf_counter()
        vector = catalog.encoder.encode([f"Research about {topic}"], query=True)[0]
        ranked = catalog.rank_for_vector(topic, vector)
        rank_ms = (perf_counter() - started) * 1000
        shift = index % len(ARMS)
        mask_order = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for pct in mask_order:
            for arm in ARMS[shift:] + ARMS[:shift]:
                started = perf_counter()
                if arm in BASELINES:
                    selected = holdout.select(catalog, ranked, pair, pct, arm)
                else:
                    title_ranked = np.argsort(-(title_vectors @ vector), kind="stable").tolist()
                    selected = field_candidates(
                        ranked, title_ranked, catalog.visible[pct], catalog.person[pair],
                        full_reserved=0 if arm == "title_role_four" else 2,
                        use_roles=arm != "topic_title_four",
                    )
                values = catalog.verify(topic, selected)
                row = nomination.evaluate(catalog, topic, pair, selected, values, holdout.CUTOFF)
                row.update(
                    topic=topic, role_pair=pair, reads=len(selected), topic_order=ranked,
                    injected=len(set(selected) - set(ranked[:4])),
                    characters=sum(len(catalog.texts[i]) for i in selected),
                    selected_sources=selected,
                    passing_sources=tuple(i for i, value in zip(selected, values, strict=True)
                                          if value > holdout.CUTOFF),
                    ms=rank_ms + (perf_counter() - started) * 1000,
                )
                results[pct][arm].append(row)
    return results


def data_model(path, encoder, reader):
    articles = json.loads(path.read_bytes())["sources"]
    catalog = pilot.Catalog(articles, encoder, reader)
    started = perf_counter()
    titles = encoder.encode([text.split("\n", 1)[0] for text in catalog.texts])
    return catalog, titles, dict(
        base_vector_bytes=catalog.vectors.nbytes, title_vector_bytes=titles.nbytes,
        title_index_encode_seconds=perf_counter() - started,
    )


def verify_baselines(results, reference, label):
    for pct, arms in results.items():
        for arm in BASELINES:
            summary = (controls.published_control(reference, label, pct, arm) if label == "plos_topics"
                       else reference["run"]["all_requests"][str(pct)][arm]["compared"]["topic_four"])
            for metric in ("grounded", "cited_off_label", "issued", "reads"):
                if sum(row[metric] for row in arms[arm]) != summary[metric]:
                    raise ValueError(f"Frozen results differ: {label}, {pct}, {arm}, {metric}")


def run(plos, plos_manifest, plos_reference, elife, mesh_snapshot, mesh_reference,
        embedding, reranker, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    plos_dev = json.loads(plos_manifest.read_bytes())
    mesh_data = json.loads(mesh_snapshot.read_bytes())
    refs = [json.loads(path.read_bytes()) for path in (plos_reference, mesh_reference)]
    if (pilot.digest(plos), pilot.digest(elife), pilot.digest(mesh_snapshot)) != (
            plos_dev["projection_sha256"], mesh_data["projection_sha256"], refs[1]["run"]["snapshot_sha256"]):
        raise ValueError("Expected the frozen catalogs and independent topic assignments")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (refs[0]["embedding"], refs[0]["reranker"]):
        raise ValueError("Expected frozen models")

    plos_catalog, plos_titles, plos_index = data_model(plos, encoder, reader)
    warm = sorted(plos_catalog.topics)[0]
    plos_catalog.verify(warm, plos_catalog.rank(warm)[:8])
    plos_rows = collect(plos_catalog, plos_titles, 4)
    verify_baselines(plos_rows, refs[0], "plos_topics")
    plos_results = {
        pct: {arm: dict(
            injected=sum(row["injected"] for row in rows),
            compared={base: holdout.analyze(rows, arms[base]) for base in ("topic_four", "topic_eight")},
        ) for arm, rows in arms.items()}
        for pct, arms in plos_rows.items()
    }

    catalog, titles, elife_index = data_model(elife, encoder, reader)
    complete = {i for i, a in enumerate(catalog.articles) if a["declaration_complete"]}
    catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
    mesh.assign_topics(catalog, mesh_data["records"], mesh.ROUTINE_DESCRIPTORS)
    major = dict(catalog.topics)
    mesh.assign_topics(catalog, mesh_data["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
    for term in tuple(catalog.topics):
        if not scope.MINIMUM_SUPPORT <= len(catalog.topics[term]) <= scope.MAXIMUM_SUPPORT:
            del catalog.topics[term]
    elife_rows = collect(catalog, titles, scope.MINIMUM_SUPPORT)
    verify_baselines(elife_rows, refs[1], "elife")
    elife_results = scope.summarize(elife_rows, major, set(catalog.topics))
    stages = diagnostic.summarize(elife_rows, catalog, major)
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), cutoff=holdout.CUTOFF,
        ranking_sha256=pilot.digest(Path(pilot.__file__)),
        plos_reference_sha256=pilot.digest(plos_reference), mesh_reference_sha256=pilot.digest(mesh_reference),
        embedding=encoder_info["revision"], reranker=reader.provenance["revision"],
        plos_index=plos_index, elife_index=elife_index, plos_results=plos_results,
        elife_results=elife_results, elife_stage_diagnostic=stages,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "plos_manifest", "plos_reference", "elife", "mesh_snapshot",
                 "mesh_reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
