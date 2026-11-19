"""Reuse the retrieval embedding to verify publisher role membership."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import calibrated_witness_search as live
from experiments.credit import linked_witness_dispatch as replay
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_prototypes import normalized
from experiments.scifact.minilm_sufficiency_operating_points import operating_threshold


CAPS = (10, 20, 30)
BANKS = ("definition", "ridge")
ARMS = ("round_robin", "owner_mean", "adaptive")
ROLE_INDEX = {role: idx for idx, role in enumerate(ROLES)}


def build_banks(encoder, glossary, training_path):
    records = json.loads(training_path.read_text())
    if set(records) != set(ROLES) or {len(rows) for rows in records.values()} != {6}:
        raise ValueError("Expected six independent training phrases per role")
    phrases = [phrase for role in ROLES for phrase in records[role]]
    norm = {normalized(phrase) for phrase in phrases}
    if len(norm) != len(phrases) or norm & {normalized(row["text"]) for row in glossary["examples"]}:
        raise ValueError("Training phrase overlap")
    start = perf_counter()
    definitions = encoder.encode([glossary["definitions"][role] for role in ROLES])
    features = encoder.encode(phrases, query=True)
    identity = np.eye(len(features), dtype=features.dtype)
    labels = np.repeat(np.eye(len(ROLES), dtype=features.dtype), 6, axis=0)
    ridge = labels.T @ np.linalg.solve(features @ features.T + .1 * identity, features)
    return {"definition": definitions, "ridge": ridge}, (perf_counter() - start) * 1000


def membership_margin(scores, declared):
    """Compare the best declared role with the best undeclared role."""
    mask = np.asarray([role in declared for role in ROLES], dtype=bool)
    # Sentinels represent exact empty or complete inventories during calibration.
    if not mask.any():
        return -1e6
    if mask.all():
        return 1e6
    return float(scores[mask].max() - scores[~mask].max())


def fit(calibration, indexed, phrases, encoder, banks):
    evidence = {name: {} for name in banks}
    for item in calibration:
        dois, _, _, requested, _ = item
        rows, _, _ = indexed[dois]
        words = live.request_words(item, phrases)
        query = encoder.encode(words, query=True)
        for name, bank in banks.items():
            scores = query @ bank.T
            for index, (slot, _, declared) in enumerate(rows):
                for j, (side, role) in enumerate(requested):
                    if side == slot:
                        key = dois, index, role, words[j]
                        evidence[name][key] = role in declared, membership_margin(scores[j], declared)
    thresholds = {}
    for name, values in evidence.items():
        gold, scores = map(np.asarray, zip(*values.values(), strict=True))
        caps = {}
        for cap in CAPS:
            threshold = max(0., operating_threshold(scores[~gold], cap))
            caps[str(cap)] = dict(threshold=threshold, false_accept=int(np.sum((scores > threshold) & ~gold)),
                                  true_accept=int(np.sum((scores > threshold) & gold)))
        thresholds[name] = dict(positive=int(gold.sum()), negative=int((~gold).sum()), caps=caps)
    return thresholds


def verify(rows, scores, role_scores, scopes, budget, arm, threshold):
    task_ids = tuple(range(len(scopes)))
    tasks = tuple(zip(scopes, task_ids, strict=True))
    observed = [(side, owner, set()) for side, owner, _ in rows]
    opened, checked = [], []
    ranking = live.round_robin(rows, scores, scopes) if arm == "round_robin" else None
    for step in range(min(budget, len(rows))):
        choice = (ranking[step] if ranking is not None else
                  replay.choose_read(scores, observed, scopes, task_ids, arm, opened))
        if choice is None:
            break
        slot, _, declared = rows[choice]
        for side, task in tasks:
            if side == slot:
                accepted = membership_margin(role_scores[task], declared) > threshold
                checked.append((choice, task, accepted))
                if accepted:
                    observed[choice][2].add(task)
        opened.append(choice)
    return replay.output(observed, opened, tasks), opened, checked


def run(projection, glossary_file, training, embedding, prior_file, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh results and Slurm CUDA")
    if replay.digest(projection) != replay.PROJECTION or replay.digest(glossary_file) != replay.EXAMPLES:
        raise ValueError("Unexpected inputs")
    previous = json.loads(prior_file.read_text())
    if (previous["projection_sha256"] != replay.PROJECTION or previous["glossary_sha256"] != replay.EXAMPLES
            or previous["script_sha256"] != replay.digest(Path(live.__file__))):
        raise ValueError("Expected the live MiniLM cohort and schedule reference")
    provenance = replay.verify_checkpoint(embedding)
    if provenance["revision"] != previous["embedding_revision"]:
        raise ValueError("Changed embedding revision")
    encoder = replay.QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    glossary = json.loads(glossary_file.read_text())
    documents = json.loads(projection.read_text())["documents"]
    cal, test, indexed, phrases, _, _ = live.panels(documents, glossary, encoder)
    if len(test) != previous["split"]["test_requests"] or len(cal) != previous["split"]["calibration_requests"]:
        raise ValueError("Changed cohort")
    banks, offline_ms = build_banks(encoder, glossary, training)
    thresholds = fit(cal, indexed, phrases, encoder, banks)
    seen_phrases = {word for row in cal for word in live.request_words(row, phrases)}
    counts, times, matched, atomic = defaultdict(Counter), defaultdict(list), defaultdict(lambda: defaultdict(list)), {}
    for j, item in enumerate([test[0], *test]):
        dois, family, _, requested, truth = item
        rows, _, vectors = indexed[dois]
        words = live.request_words(item, phrases)
        started = perf_counter()
        query = encoder.encode(words, query=True)
        ranking = vectors @ query.T
        shared_ms = (perf_counter() - started) * 1000
        scopes, role_labels = tuple(zip(*requested, strict=True))
        for name, bank in banks.items():
            started = perf_counter()
            role_scores = query @ bank.T
            mapping_ms = (perf_counter() - started) * 1000
            if j:
                for k, (_, role) in enumerate(requested):
                    atomic[name, words[k], role] = ROLE_INDEX[role] == int(role_scores[k].argmax())
            for budget in live.BUDGETS:
                for arm in ARMS:
                    for cap_str, gate in thresholds[name]["caps"].items():
                        started = perf_counter()
                        found, reads, checks = verify(rows, ranking, role_scores, scopes, budget, arm, gate["threshold"])
                        elapsed = shared_ms + mapping_ms + (perf_counter() - started) * 1000
                        if not j:
                            continue
                        key = name, cap_str, budget, arm
                        c = counts[key]
                        roles_correct = [(role_labels[task] in rows[row][2], accept) for row, task, accept in checks]
                        c.update(dict(queries=1, reads=len(reads), positive_queries=bool(truth),
                                      recovered_positive=bool(found & truth), unsupported_queries=bool(found - truth),
                                      negative_promotions=bool(found) and not truth, exact=found == truth,
                                      false_role_checks=sum(accept and not gold for gold, accept in roles_correct),
                                      negative_role_checks=sum(not gold for gold, _ in roles_correct)))
                        times[key].append(elapsed)
                        matched[key][family].append(found == truth)
    report = {}
    for name in BANKS:
        report[name] = {}
        for cap in CAPS:
            report[name][cap] = {}
            for budget in live.BUDGETS:
                rows = {}
                for arm in ARMS:
                    key = name, str(cap), budget, arm
                    rows[arm] = dict(counts[key], p95_ms=float(np.quantile(times[key], .95)),
                                     both_variants_exact=sum(len(v) == 2 and all(v) for v in matched[key].values()))
                report[name][cap][budget] = rows
    atomic_report = {}
    for name in BANKS:
        scores = [(phrase in seen_phrases, correct) for (bank, phrase, _), correct in atomic.items() if bank == name]
        atomic_report[name] = dict(correct=sum(correct for _, correct in scores), total=len(scores),
                                   new_correct=sum(correct and not seen for seen, correct in scores),
                                   new_total=sum(not seen for seen, _ in scores))
    replay.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                    script_sha256=replay.digest(Path(__file__)), training_sha256=replay.digest(training),
                                    reference_sha256=replay.digest(prior_file), embedding_revision=provenance["revision"],
                                    projection_sha256=replay.PROJECTION, glossary_sha256=replay.EXAMPLES,
                                    offline_bank_ms=offline_ms, thresholds=thresholds, atomic=atomic_report, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "glossary_file", "training", "embedding", "prior_file", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
