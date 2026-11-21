"""Compare role definition, prototype, and ridge projections on held wording."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.engineering_all_author_gate import (
    NISO_SHA256, VIEWS, calibrate, compile_article, evaluate,
)
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("definition", "prototype", "ridge", "agreement")
TASK = "Match the research task to a contributor role."


def queries(examples, training):
    if digest(examples) != NISO_SHA256:
        raise ValueError("Changed NISO vocabulary and example snapshot")
    source = json.loads(examples.read_bytes())
    bank = json.loads(training.read_bytes())
    if set(bank) != set(ROLES) or any(len(bank[role]) != 6 for role in ROLES):
        raise ValueError("Expected six previous training phrases per role")
    definitions = [f"{role}: {source['definitions'][role]}" for role in ROLES]
    phrases = [phrase for role in ROLES for phrase in bank[role]]
    held = {name: [(i, i, phrase) for i, phrase in enumerate(rows)]
            for name, rows in VIEWS.items()}
    held["niso"] = [(ROLES.index(row["role"]), row["ordinal"], row["text"]) for row in source["examples"]]
    fit_names = {normalize(name) for name in phrases}
    if len(fit_names) != len(phrases) or any(normalize(phrase) in fit_names for rows in held.values() for _, _, phrase in rows):
        raise ValueError("Training and evaluation phrases overlap")
    return definitions, phrases, held


def encode(encoder, rows):
    return encoder.encode([f"Instruct: {TASK}\nQuery:{row}" for row in rows])


def projections(encoder, definitions, phrases):
    dictionary = encoder.encode(definitions)
    vectors = encode(encoder, phrases)
    prototypes = vectors.reshape(len(ROLES), 6, -1).mean(axis=1)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    targets = np.repeat(np.eye(len(ROLES), dtype=np.float32), 6, axis=0)
    kernel = vectors @ vectors.T
    kernel.flat[::len(vectors) + 1] += .1
    ridge = np.linalg.solve(kernel, targets).T @ vectors
    return dict(definition=dictionary, prototype=prototypes, ridge=ridge)


def project(vector, banks, membership):
    logits = {name: values @ vector for name, values in banks.items()}
    scores = {}
    relative = []
    for name, weights in logits.items():
        yes = np.where(membership, weights, -np.inf).max(axis=1)
        no = np.where(~membership, weights, -np.inf).max(axis=1)
        scores[name] = np.clip(yes - no, -100, 100)
        relative.append((yes - no) / max(.01, float(weights.std())))
    scores["agreement"] = np.clip(np.min(relative, axis=0), -100, 100)
    return logits, scores


def collect(panel, encoder, banks, wording, offset):
    articles = [(pmc, row["journal"], compile_article(row)) for pmc, row in sorted(panel.items())]
    if offset:
        articles.reverse()
    results = {}
    recognition = {}
    for name, phrases in wording.items():
        by_source = {}
        for pmc, venue, article in articles:
            observations = by_source.setdefault(venue, [])
            for i, ordinal, text in phrases:
                start = perf_counter()
                vector = encode(encoder, [text])[0]
                roles, strengths = project(vector, banks, article["memberships"])
                packed = {key: values.tolist() for key, values in strengths.items()}
                ranks = {key: np.argsort(-scores, kind="stable").tolist() for key, scores in roles.items()}
                json.dumps(packed)
                elapsed = (perf_counter() - start) * 1000
                signature = name, i, ordinal
                if signature in recognition and recognition[signature] != ranks:
                    raise ValueError("Stable query text changed its role ordering")
                recognition[signature] = ranks
                observations.append(dict(
                    article=pmc, role=ROLES[i], ordinal=ordinal, labels=article["labels"][i],
                    routes={key: dict(scores=values, ms=elapsed, reader_pairs=0, clipped=0)
                            for key, values in packed.items()},
                ))
        results[name] = {key: sorted(rows, key=lambda row: (row["article"], row["role"], row["ordinal"]))
                         for key, rows in by_source.items()}
    return results, recognition


def source_cutoff(rows, arm):
    # Reuse the same nonnegative membership cut and request-level calibration
    # procedure as the previous dense glossary experiment.
    source = [dict(labels=row["labels"], routes={"dense": row["routes"][arm]}) for row in rows]
    return calibrate(source, "dense")


def summarize(rows):
    results = {}
    for held, development in (("frontiers", "sensors"), ("sensors", "frontiers")):
        dev = rows["original"][development]
        thresholds = {arm: source_cutoff(dev, arm) for arm in ARMS}
        results[held] = {}
        for name, publishers in rows.items():
            test = publishers[held]
            reference = evaluate(test, "definition", thresholds["definition"])[1]
            data = {}
            for arm, threshold in thresholds.items():
                measures, clean, _ = evaluate(test, arm, threshold)
                changes = [a - b for a, b in zip(clean, reference, strict=True)]
                data[arm] = dict(
                    threshold=threshold, source=evaluate(dev, arm, threshold)[0], held=measures,
                    article_paired_vs_definition=clustered_paired_interval(changes, [r["article"] for r in test]),
                    role_paired_vs_definition=clustered_paired_interval(changes, [r["role"] for r in test]),
                    evidence=[dict(article=r["article"], role=r["role"], ordinal=r["ordinal"], labels=r["labels"],
                                   scores=r["routes"][arm]["scores"]) for r in test],
                )
            results[held][name] = dict(
                results=data, shared_p95_ms=float(np.quantile([r["routes"]["definition"]["ms"] for r in test], .95)),
            )
    return results


def recognition_metrics(ranks):
    report = {}
    for (wording, truth, ordinal), arms in sorted(ranks.items()):
        for name, ordering in arms.items():
            row = report.setdefault(wording, {}).setdefault(name, Counter())
            row.update(queries=1, top1=int(ordering[0] == truth), top3=int(truth in ordering[:3]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "examples", "training", "embedding", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    definitions, phrases, wording = queries(args.examples, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, phrases)
    rows, ranks = collect(json.loads(args.panel.read_bytes()), encoder, banks, wording, args.offset)
    write_json(args.output, dict(
        panel_sha256=digest(args.panel), examples_sha256=digest(args.examples), training_sha256=digest(args.training),
        script_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(), offset=args.offset,
        embedding_revision=checkpoint["revision"], recognition=recognition_metrics(ranks),
        results=summarize(rows),
        role_orders=[dict(wording=w, role=ROLES[r], ordinal=i, order=order)
                     for (w, r, i), order in sorted(ranks.items())],
    ))
