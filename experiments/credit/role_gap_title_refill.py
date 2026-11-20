"""Refill structurally unusable slots in the topic prefix from a second index."""

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
from experiments.credit import role_field_ranking as field
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = (*field.BASELINES, "role_title_four", "gap_title_four")
BUDGET = 4


def refill(ranked, title_order, excluded):
    prefix = ranked[:BUDGET]
    chosen = [source for source in prefix if source not in excluded]
    if len(chosen) == BUDGET:
        return chosen
    for source in title_order():
        if source not in excluded and source not in chosen:
            chosen.append(source)
            if len(chosen) == BUDGET:
                return chosen
    # Match the reading budget even when too few usable sources exist.
    return chosen + [source for source in prefix if source in excluded][:BUDGET - len(chosen)]


def collect(catalog, titles, minimum_support):
    topics = sorted(topic for topic, sources in catalog.topics.items() if len(sources) >= minimum_support)
    queries = list(pilot.query_panel(catalog, topics))
    result = {pct: {arm: [] for arm in ARMS} for pct in pilot.CACHE}
    for index, (topic, _, pair) in enumerate(queries):
        start = perf_counter()
        vector = catalog.encoder.encode([f"Research about {topic}"], query=True)[0]
        ranked = catalog.rank_for_vector(topic, vector)
        shared_ms = (perf_counter() - start) * 1000
        shift = index % len(ARMS)

        def title_order():
            return np.argsort(-(titles @ vector), kind="stable").tolist()

        masks = pilot.CACHE if index % 2 == 0 else tuple(reversed(pilot.CACHE))
        for pct in masks:
            for arm in ARMS[shift:] + ARMS[:shift]:
                start = perf_counter()
                known, posting = catalog.visible[pct], catalog.person[pair]
                excluded = known - posting
                title_scanned = arm == "role_title_four" or (
                    arm == "gap_title_four" and bool(excluded.intersection(ranked[:BUDGET])))
                if arm == "gap_title_four":
                    selected = refill(ranked, title_order, excluded)
                elif arm == "role_title_four":
                    selected = field.field_candidates(
                        ranked, title_order(), known, posting, full_reserved=2, use_roles=True,
                    )
                else:
                    selected = holdout.select(catalog, ranked, pair, pct, arm)
                if arm != "topic_eight" and (len(selected) != BUDGET or len(set(selected)) != BUDGET):
                    raise ValueError("Expected exactly four unique sources")
                if arm == "gap_title_four" and set(ranked[:BUDGET]) - excluded - set(selected):
                    raise ValueError("A potentially usable original candidate was lost")
                values = catalog.verify(topic, selected)
                row = nomination.evaluate(catalog, topic, pair, selected, values, holdout.CUTOFF)
                row.update(
                    topic=topic, role_pair=pair, reads=len(selected), topic_order=ranked,
                    selected_sources=selected, title_scan=title_scanned,
                    injected=len(set(selected) - set(ranked[:BUDGET])),
                    characters=sum(len(catalog.texts[source]) for source in selected),
                    passing_sources=tuple(i for i, value in zip(selected, values, strict=True)
                                          if value > holdout.CUTOFF),
                    ms=shared_ms + (perf_counter() - start) * 1000,
                )
                result[pct][arm].append(row)
    return result


def summarize(measurements, catalog, major=None):
    score = scope.summarize(measurements, major or {}, set(catalog.topics))
    for pct, arms in measurements.items():
        gap, person = arms["gap_title_four"], arms["person_four"]
        comparison = holdout.analyze(gap, person)
        if major:
            major_gap = scope.major_scores(gap, major)
            major_person = scope.major_scores(person, major)
            comparison["major_delta"] = holdout.interval(
                [(a["grounded"] - b["grounded"]) / pilot.TOP for a, b in zip(major_gap, major_person, strict=True)],
                [row["topic"] for row in gap],
            )
        score[pct]["gap_title_four"]["vs_person_four"] = comparison
        for arm, rows in arms.items():
            base = arms["topic_four"]
            score[pct][arm].update(
                title_scans=sum(row["title_scan"] for row in rows),
                initial_witnesses_lost=sum(
                    len(set(baseline["witnesses"]) - set(row["witnesses"]))
                    for row, baseline in zip(rows, base, strict=True)
                ),
                initial_lists_with_a_loss=sum(
                    bool(set(baseline["witnesses"]) - set(row["witnesses"]))
                    for row, baseline in zip(rows, base, strict=True)
                ),
            )
    return dict(metrics=score, major_stage=diagnostic.summarize(measurements, catalog, major) if major else None)


def citation_trace(rows, catalog):
    return {
        mask: {
            arm: [dict(topic=row["topic"],
                       citations=[catalog.articles[i]["doi"] for _, i in row["witnesses"]])
                  for row in measured]
            for arm, measured in arms.items()
        }
        for mask, arms in rows.items()
    }


def verify_mixed(rows, frozen):
    for pct, arms in rows.items():
        target = frozen[str(pct)]["role_title_four"]["compared"]["topic_four"]
        for key in ("grounded", "cited_off_label", "issued", "reads"):
            if sum(row[key] for row in arms["role_title_four"]) != target[key]:
                raise ValueError(f"Frozen mixed control differs: {pct}, {key}")


def run(plos, plos_manifest, plos_reference, elife, mesh_snapshot, mesh_reference,
        mixed_reference, embedding, reranker, output, witness_output=None):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or (witness_output is not None and witness_output.exists()) or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    plos_spec = json.loads(plos_manifest.read_bytes())
    mesh_data = json.loads(mesh_snapshot.read_bytes())
    plos_prior, elife_prior, mixed_prior = (
        json.loads(path.read_bytes()) for path in (plos_reference, mesh_reference, mixed_reference)
    )
    if (pilot.digest(plos), pilot.digest(elife), pilot.digest(mesh_snapshot)) != (
            plos_spec["projection_sha256"], mesh_data["projection_sha256"], elife_prior["run"]["snapshot_sha256"]):
        raise ValueError("Expected the frozen projections and topic labels")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (plos_prior["embedding"], plos_prior["reranker"]):
        raise ValueError("Expected the frozen checkpoints")

    def replay(catalog, titles, cutoff):
        warm = sorted(catalog.topics)[0]
        catalog.verify(warm, catalog.rank(warm)[:8])
        return collect(catalog, titles, cutoff)

    catalog, titles, plos_index = field.data_model(plos, encoder, reader)
    plos_rows = replay(catalog, titles, 4)
    field.verify_baselines(plos_rows, plos_prior, "plos_topics")
    verify_mixed(plos_rows, mixed_prior["run"]["plos_results"])
    plos_results = summarize(plos_rows, catalog)
    if witness_output is not None:
        plos_witnesses = citation_trace(plos_rows, catalog)

    catalog, titles, elife_index = field.data_model(elife, encoder, reader)
    complete = {i for i, article in enumerate(catalog.articles) if article["declaration_complete"]}
    catalog.visible = {pct: ids & complete for pct, ids in catalog.visible.items()}
    mesh.assign_topics(catalog, mesh_data["records"], mesh.ROUTINE_DESCRIPTORS)
    major = dict(catalog.topics)
    mesh.assign_topics(catalog, mesh_data["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
    for term in tuple(catalog.topics):
        if not scope.MINIMUM_SUPPORT <= len(catalog.topics[term]) <= scope.MAXIMUM_SUPPORT:
            del catalog.topics[term]
    elife_rows = replay(catalog, titles, scope.MINIMUM_SUPPORT)
    field.verify_baselines(elife_rows, elife_prior, "elife")
    verify_mixed(elife_rows, mixed_prior["run"]["elife_results"])
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=pilot.digest(Path(__file__)),
        mixed_reference_sha256=pilot.digest(mixed_reference), plos_reference_sha256=pilot.digest(plos_reference),
        mesh_reference_sha256=pilot.digest(mesh_reference), embedding=encoder_info["revision"],
        reranker=reader.provenance["revision"], plos_index=plos_index, elife_index=elife_index,
        plos=plos_results, elife=summarize(elife_rows, catalog, major),
    ))
    if witness_output is not None:
        pilot.write_json(witness_output, dict(
            replay_sha256=pilot.digest(output),
            projections={"plos": pilot.digest(plos), "elife": pilot.digest(elife)},
            witnesses={"plos": plos_witnesses, "elife": citation_trace(elife_rows, catalog)},
        ))



if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "plos_manifest", "plos_reference", "elife", "mesh_snapshot", "mesh_reference",
                 "mixed_reference", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--witness-output", type=Path)
    run(**vars(cli.parse_args()))
