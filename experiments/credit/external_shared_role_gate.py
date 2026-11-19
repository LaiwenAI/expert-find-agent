"""Transfer a frozen shared-embedding role gate to matched publisher edits."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from experiments.credit import linked_witness_dispatch as replay
from experiments.credit import shared_embedding_witness_gate as shared
from experiments.credit.compositional_edits import stable
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_prototypes import encode_tasks, normalized, train_banks
from experiments.scifact.minilm_sufficient_facts import bootstrap_delta


METHODS = ("shared_definition", "shared_ridge", "dedicated_ridge", "paper_pool_definition")


def held_articles(fixture, projection):
    manifest = json.loads((fixture / "manifest.json").read_text())
    audit = json.loads((fixture / "source-audit.json").read_text())
    pair_file = fixture / "pairs.jsonl"
    if (audit["fixtures"]["transfer"] != replay.digest(fixture / "manifest.json")
            or manifest["pairs_sha256"] != replay.digest(pair_file)):
        raise ValueError("Changed external matched edits")
    items, _ = replay.panel(projection)
    used = {d for row in items for d in row[0]}
    known = {owner for doi in used for owner in projection[doi]["identities"]}
    excluded = Counter()
    documents = {}
    groups = [json.loads(line) for line in pair_file.read_text().splitlines()]
    for source in manifest["sources"]:
        doi = source["doi"]
        if doi not in audit["allowed"]:
            raise ValueError("Expected the audited external inventory")
        record = projection[doi]
        if source["source_sha256"] != record["sha256"]:
            raise ValueError("Changed publisher declaration")
        if doi in used:
            excluded["doi"] += 1
        elif known.intersection(record["identities"]):
            excluded["publisher_id"] += 1
        else:
            documents[doi] = source["roles_by_author"]
    retained = [group for group in groups if group["doi"] in documents]
    if len(documents) < 8 or not retained:
        raise ValueError("Insufficient publisher-disjoint external evaluation")
    return retained, documents, dict(excluded), len(used)


def evaluate_group(group, declarations, phrases, encoder, banks, cutoffs, method):
    start = perf_counter()
    query = encode_tasks(encoder, phrases) if method == "dedicated_ridge" else encoder.encode(phrases, query=True)
    role_scores = query @ banks[method].T
    changed = declarations[group["changed_author"] - 1]
    stable_author = declarations[group["stable_author"] - 1]
    if method == "paper_pool_definition":
        union = set().union(*declarations)
        changed = stable_author = union
    predictions = []
    for roles in (changed, stable_author):
        matches = [shared.membership_margin(row, roles) > cutoffs[method] for row in role_scores]
        predictions.append([all(matches[:2]), all(matches[2:])])
    elapsed_ms = (perf_counter() - start) * 1000
    return predictions, elapsed_ms


def run(projection_file, fixture, glossary_file, training, embedding, prior_file, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh result and Slurm CUDA")
    projection_hash = replay.digest(projection_file)
    glossary_hash = replay.digest(glossary_file)
    if projection_hash != replay.PROJECTION or glossary_hash != replay.EXAMPLES:
        raise ValueError("Changed projection or glossary")
    prior = json.loads(prior_file.read_text())
    if (prior["projection_sha256"] != projection_hash or prior["glossary_sha256"] != glossary_hash
            or prior["training_sha256"] != replay.digest(training)
            or prior["script_sha256"] != replay.digest(Path(shared.__file__))):
        raise ValueError("Expected the same frozen shared role gates")
    projection = json.loads(projection_file.read_text())["documents"]
    groups, documents, excluded, original_used = held_articles(fixture, projection)
    glossary = json.loads(glossary_file.read_text())
    boundary_path = fixture / "boundary-phrases.json"
    boundary = json.loads(boundary_path.read_text())
    phrases = [p for role in ROLES for p in boundary[role]]
    known = {normalized(row["text"]) for row in glossary["examples"]}
    known.update(normalized(p) for ps in json.loads(training.read_text()).values() for p in ps)
    if len({normalized(p) for p in phrases}) != 28 or known.intersection(map(normalized, phrases)):
        raise ValueError("Expected 28 distinct previously unseen boundary phrases")
    provenance = replay.verify_checkpoint(embedding)
    if prior["embedding_revision"] != provenance["revision"]:
        raise ValueError("Changed embedding")
    encoder = replay.QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    shared_banks, _ = shared.build_banks(encoder, glossary, training)
    dedicated, _ = train_banks(training, SimpleNamespace(embedder=encoder, vectors=shared_banks["definition"]),
                               glossary["examples"])
    banks = dict(shared_definition=shared_banks["definition"], shared_ridge=shared_banks["ridge"],
                 dedicated_ridge=dedicated["ridge"], paper_pool_definition=shared_banks["definition"])
    cutoffs = {f"shared_{name}": row["caps"]["10"]["threshold"] for name, row in prior["thresholds"].items()}
    cutoffs.update(dedicated_ridge=0.0, paper_pool_definition=cutoffs["shared_definition"])
    results, times = defaultdict(list), defaultdict(list)
    for i, group in enumerate([groups[0], *groups]):
        roles = group["before"] + group["after"]
        words = [min(boundary[role], key=lambda text: stable(group["id"], role, text)) for role in roles]
        expected = [[x == "declared" for x in group[key]] for key in ("changed_labels", "stable_labels")]
        paper = documents[group["doi"]]
        for ordinal, key in ((group["changed_author"], "changed_labels"), (group["stable_author"], "stable_labels")):
            computed = [all(r in paper[ordinal - 1] for r in group[version]) for version in ("before", "after")]
            if computed != [x == "declared" for x in group[key]]:
                raise ValueError("Published roles must reproduce paired labels")
        shift = i % len(METHODS)
        for method in METHODS[shift:] + METHODS[:shift]:
            predictions, elapsed_ms = evaluate_group(group, paper, words, encoder, banks, cutoffs, method)
            if i:
                results[method].append((predictions, expected))
                times[method].append(elapsed_ms)
    report = {}
    base = np.asarray([p == t for p, t in results["dedicated_ridge"]])
    for method, pairs in results.items():
        correct = np.asarray([p == t for p, t in pairs])
        counts = Counter()
        for predicted, expected in pairs:
            counts.update(dict(exact_groups=predicted == expected, target_pair=predicted[0] == expected[0],
                               stable_pair=predicted[1] == expected[1],
                               false_promotions=sum(p and not t for a, b in zip(predicted, expected, strict=True)
                                                    for p, t in zip(a, b, strict=True)),
                               lost_supports=sum(t and not p for a, b in zip(predicted, expected, strict=True)
                                                 for p, t in zip(a, b, strict=True))))
        report[method] = dict(counts, component_p95_ms=float(np.quantile(times[method], .95)))
        if method != "dedicated_ridge":
            delta = correct.astype(float) - base
            report[method]["delta_vs_dedicated"] = dict(point=float(delta.mean()), doi_ci95=bootstrap_delta(
                delta, [group["doi"] for group in groups]))
    replay.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                    script_sha256=replay.digest(Path(__file__)), prior_sha256=replay.digest(prior_file),
                                    fixture_sha256=replay.digest(fixture / "manifest.json"),
                                    boundary_sha256=replay.digest(boundary_path), embedding_revision=provenance["revision"],
                                    excluded=excluded, prior_used_sources=original_used, groups=len(groups),
                                    publications=len(documents), cutoffs=cutoffs, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection_file", "fixture", "glossary_file", "training", "embedding", "prior_file", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
