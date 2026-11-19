"""Isolate the effect of reserving source slots across SE-PEF communities."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json


METHODS = ("max_answer", "expert_centroid")
METRICS = ("hit5", "mrr10")


def join(anchor, reserved):
    base_owners, base_ids, base_texts, base_positions, base_meta = anchor
    alt_owners, alt_ids, alt_texts, alt_positions, alt_meta = reserved
    if base_owners != alt_owners or not np.array_equal(np.bincount(base_positions), np.bincount(alt_positions)):
        raise ValueError("Expert populations or per-expert answer budgets differ")
    for name in ("val.json", "test.json"):
        if base_meta["derived_sha256"][name] != alt_meta["derived_sha256"][name]:
            raise ValueError("Query inputs do not match")
    if alt_meta["selection"] != "reserve_active_sites" or len(set(alt_ids) - set(base_ids)) != alt_meta["answers_replaced"]:
        raise ValueError("Unexpected source selection")
    documents = {}
    for ids, texts, positions in ((base_ids, base_texts, base_positions), (alt_ids, alt_texts, alt_positions)):
        for identifier, text, position in zip(ids, texts, positions, strict=True):
            item = (int(position), text)
            if documents.setdefault(identifier, item) != item:
                raise ValueError("A shared answer changed owner or text")
    ordered = sorted(documents, key=lambda source: (documents[source][0], source))
    text = [documents[identifier][1] for identifier in ordered]
    positions = np.array([documents[identifier][0] for identifier in ordered])
    return base_owners, ordered, text, positions, {"recent": set(base_ids), "reserved": set(alt_ids)}


def banks(matrix, source_ids, positions, selection, bank_type):
    groups = defaultdict(list)
    for i, source in enumerate(source_ids):
        if source in selection:
            groups[source.split("_", 1)[0]].append(i)
    return {site: bank_type(matrix, np.array(docs), positions) for site, docs in groups.items()}


def score(ranking, labels, positions):
    if any(owner != positions[source] for owner, source in ranking):
        raise ValueError("Returned source ownership differs from its expert")
    rank = next((i for i, (owner, _) in enumerate(ranking, 1) if owner in labels), None)
    return dict(hit5=bool(rank and rank <= 5), mrr10=1 / rank if rank else 0.0)


def means(rows):
    return {
        method: {panel: {metric: float(np.mean([r["scores"][method][panel][metric] for r in rows]))
                         for metric in METRICS}
                 for panel in ("recent", "reserved")}
        for method in METHODS
    }


def summarize(rows):
    groups = [r["community"] for r in rows]
    intervals = {
        method: {metric: clustered_paired_interval(
            [float(r["scores"][method]["reserved"][metric]) - float(r["scores"][method]["recent"][metric])
             for r in rows], groups,
        ) | {"sampling_unit": "whole community; query-weighted mean"}
            for metric in METRICS}
        for method in METHODS
    }
    strata = defaultdict(list)
    for row in rows:
        strata[row["reach"]].append(row)
    return dict(
        queries=len(rows), arms=means(rows), reserved_minus_recent=intervals,
        availability={name: dict(queries=len(selected), arms=means(selected))
                      for name, selected in sorted(strata.items())},
    )


def run(recent, reserved, checkpoint, output):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint
    from experiments.retrieval import sepef_first_query as baseline

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Require a fresh output and Slurm CUDA")
    anchor_hash = digest(recent / "manifest.json")
    alternate = baseline.payload(reserved)
    if alternate[-1]["anchor_manifest_sha256"] != anchor_hash:
        raise ValueError("Reservation was built from a different source panel")
    owners, ids, texts, positions, panels = join(baseline.payload(recent), alternate)
    owner_numbers = {owner: i for i, owner in enumerate(owners)}
    model = verify_checkpoint(checkpoint)
    if model["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise ValueError("Unexpected frozen embedding checkpoint")
    encoder = QwenTorchEmbedding(checkpoint, max_length=256, batch_size=64)
    encoder.model.config.use_cache = False
    encoder.encode(["Warm-up query."])
    started = perf_counter()
    matrix = torch.as_tensor(encoder.encode(texts), device="cuda")
    torch.cuda.synchronize()
    encode_seconds = perf_counter() - started
    indexes = {name: banks(matrix, ids, positions, selected, baseline.SourceBank)
               for name, selected in panels.items()}
    report = {}
    for split, boundary in (("val", baseline.SNAPSHOT), ("test", baseline.TEST_START)):
        queries = json.loads((recent / f"{split}.json").read_bytes())
        rows = []
        for row in queries:
            if row["timestamp"] < boundary:
                raise ValueError("An evaluation query precedes its cutoff")
            site = row["community"]
            text = baseline.clean(row["text"])
            targets = {owner_numbers[owner] for owner in row["expert_ids"]}
            query = torch.as_tensor(encoder.encode([f"Instruct: {baseline.TASK}\nQuery: {text}"])[0], device="cuda")
            old = targets.intersection(indexes["recent"][site].owners)
            new = targets.intersection(indexes["reserved"][site].owners)
            reach = ("both" if old and new else "recent_only" if old else "reserved_only" if new else "neither")
            scores = {
                method: {name: score(index[site].dense(query, centroid=method == "expert_centroid"), targets, positions)
                         for name, index in indexes.items()}
                for method in METHODS
            }
            rows.append(dict(community=site, reach=reach, scores=scores))
        report[split] = summarize(rows)
        print(split, len(rows), "paired queries", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), model=model,
        script_sha256=digest(Path(__file__)), scorer_sha256=digest(Path(baseline.__file__)),
        recent_manifest_sha256=anchor_hash, reservation_manifest_sha256=digest(reserved / "manifest.json"),
        experts=len(owners), source_budgets={key: len(panel) for key, panel in panels.items()},
        distinct_sources=len(ids), offline_union_encoding_seconds=encode_seconds, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recent", "reserved", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
