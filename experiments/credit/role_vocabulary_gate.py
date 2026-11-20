"""Calibrate a role-selection gate on a separate publisher catalog."""

import argparse
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import fresh_role_search as search
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json

MODELS = ("single_clause_cross_entropy", "near_role_swap_cross_entropy", "reader")
RISK_CAP = 0.10
MIN_COVERAGE = 0.50


def clean(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def lexical(phrase):
    text = f" {clean(phrase)} "
    hits = [i for i, role in enumerate(ROLES) if f" {clean(role)} " in text]
    return hits[0] if len(hits) == 1 else None


def top_role(scores):
    top, runner_up = np.argsort(-np.asarray(scores), kind="stable")[:2]
    return int(top), float(scores[top] - scores[runner_up])


def issue(shortlist, declarations, role):
    if role is None:
        return []
    scores = [int(i == role) for i in range(len(ROLES))]
    return search.nominate(shortlist, declarations, scores, 0, True)


def summarize(rows):
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    covered = sum(row["emitted"] > 0 for row in rows)
    return dict(
        counts=totals, issued=covered, groups=len(rows),
        coverage=covered / len(rows), grounded_p5=totals["supported"] / (search.TOP_K * len(rows)),
        conditional_wrong_role=totals["wrong_role"] / totals["emitted"] if totals["emitted"] else None,
    )


def at_cutoff(rows, cutoff):
    blank = dict.fromkeys(rows[0][1], 0)
    return [row if margin > cutoff else blank for margin, row in rows]


def calibrate(rows):
    best = None
    # Iterate operating points induced by the development scores, including margin > 0.
    for cutoff in sorted({0.0, *(margin for margin, _ in rows)}):
        summary = summarize(at_cutoff(rows, cutoff))
        risk = summary["conditional_wrong_role"]
        if (risk is None or risk > RISK_CAP or summary["coverage"] < MIN_COVERAGE):
            continue
        key = (summary["grounded_p5"], -risk, summary["coverage"], cutoff)
        if best is None or key > best:
            best = key
    return best[-1] if best is not None else None


def build_panel(folder, manifest_path, wording_path):
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["wording_sha256"] != digest(wording_path):
        raise ValueError("Changed wording")
    wording = json.loads(wording_path.read_bytes())
    dois = sorted(manifest["sources"])
    texts, declarations = zip(*(search.article(folder, doi, manifest["sources"][doi]) for doi in dois), strict=True)
    queries = []
    for doi in manifest["queries"]:
        target = dois.index(doi)
        topic = wording["topics"][doi.removeprefix(search.PREFIX)]
        for role, phrases in wording["activities"].items():
            count = sum(role in labels for labels in declarations[target])
            if 0 < count < len(declarations[target]):
                for phrase in (phrases if isinstance(phrases, list) else [phrases]):
                    queries.append((target, topic, role, phrase))
    if not queries:
        raise ValueError("Missing evaluation queries")
    ids, names = set(), set()
    for doi in dois:
        metadata = search.plos.meta(search.xml_path(folder, doi).read_bytes())
        authors = search.plos.authors(metadata)
        ids.update(identity for identity, _, _ in search.plos.joined(authors))
        names.update(search.plos.normalized_name(author) for author in authors)
    return texts, declarations, queries, (set(dois), ids, names)


def collect(panel, encoder, score_roles, original_cutoffs):
    texts, declarations, queries, _ = panel
    index = encoder.encode(list(texts))
    observations = {model: dict(locked=[], top=[]) for model in MODELS}
    lexical_rows, groups = [], []
    topic_times, model_times = [], {model: [] for model in MODELS}
    hits = 0
    for i, (target, topic, role, phrase) in enumerate([queries[0], *queries]):
        started = perf_counter()
        vector = encoder.encode([topic], query=True)[0]
        shortlist = np.argsort(-(index @ vector), kind="stable")[:search.DEPTH].tolist()
        topic_ms = (perf_counter() - started) * 1000
        for model in MODELS[i % len(MODELS):] + MODELS[:i % len(MODELS)]:
            started = perf_counter()
            scores = score_roles(model, phrase)
            top, margin = top_role(scores)
            selected = issue(shortlist, declarations, top)
            model_ms = (perf_counter() - started) * 1000
            original = search.nominate(shortlist, declarations, scores, original_cutoffs[model], True)
            if i:
                observations[model]["locked"].append(search.metrics(original, target, role))
                observations[model]["top"].append((margin, search.metrics(selected, target, role)))
                model_times[model].append(topic_ms + model_ms)
        if i:
            topic_times.append(topic_ms)
            lexical_rows.append(search.metrics(issue(shortlist, declarations, lexical(phrase)), target, role))
            groups.append(target)
            hits += int(target in shortlist)
    return observations, lexical_rows, groups, hits, topic_times, model_times


def report(measurement, thresholds):
    observations, lexical_rows, groups, hits, topic_times, model_times = measurement
    results = dict(lexical=summarize(lexical_rows))
    decisions = {}
    for model, entries in observations.items():
        limit = thresholds[model]
        gated = at_cutoff(entries["top"], limit if limit is not None else math.inf)
        for variant, rows in (("locked", entries["locked"]), ("top_one", at_cutoff(entries["top"], 0)),
                              ("calibrated", gated)):
            arm = f"{model}_{variant}"
            decisions[arm] = rows
            results[arm] = summarize(rows)
    comparisons = {}
    single = f"{MODELS[0]}_calibrated"
    for arm in (f"{MODELS[0]}_locked", f"{MODELS[1]}_calibrated", f"{MODELS[2]}_calibrated"):
        comparisons[f"{single}_minus_{arm}"] = {
            key: clustered_paired_interval(
                [(a[key] - b[key]) / search.TOP_K
                 for a, b in zip(decisions[single], decisions[arm], strict=True)], groups,
            )
            for key in ("supported", "wrong_role")
        }
    return dict(papers=len(set(groups)), shortlist_hits=hits, arms=results, contrasts=comparisons,
                topic_p95_ms=float(np.quantile(topic_times, .95)),
                top_one_path_p95_ms={model: float(np.quantile(times, .95))
                                     for model, times in model_times.items()})


def run(args):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.factored_article_role_verification import ROLE_TASK
    from experiments.credit.factorial_role_training import fit
    from experiments.credit.train_role_decision import encode
    from experiments.followir.core17_verifier import Reranker
    from experiments.litsearch.dense_cache import verify_checkpoint

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a new output")
    reference = json.loads(args.reference.read_bytes())
    if digest(args.training) != reference["training_sha256"]:
        raise ValueError("Changed training descriptions")
    encoder = QwenTorchEmbedding(args.embedding)
    reader = Reranker(args.reranker)
    if reference["checkpoints"] != dict(embedding=verify_checkpoint(args.embedding)["revision"],
                                        reader=reader.provenance["revision"]):
        raise ValueError("Changed checkpoints")
    weights, _ = fit(encoder, args.training)

    def score_roles(model, phrase):
        if model == "reader":
            scores, clipped = reader.score_pairs([(phrase, role) for role in ROLES], task=ROLE_TASK)
            if clipped or not np.isfinite(scores).all():
                raise ValueError("Invalid role scores")
            return scores
        with torch.inference_mode():
            return (weights[model] @ encode(encoder, [phrase])[0]).cpu().tolist()

    dev = build_panel(args.dev_xml, args.dev_manifest, args.dev_wording)
    held = build_panel(args.test_xml, args.test_manifest, args.test_wording)
    if any(a & b for a, b in zip(dev[3], held[3], strict=True)):
        raise ValueError("Overlapping source DOI, ORCID or normalized byline")
    if {clean(row[3]) for row in dev[2]} & {clean(row[3]) for row in held[2]}:
        raise ValueError("Activity wording overlaps")
    development = collect(dev, encoder, score_roles, reference["thresholds"])
    limits = {model: calibrate(rows["top"]) for model, rows in development[0].items()}
    locked = json.dumps(limits, sort_keys=True)
    held_output = collect(held, encoder, score_roles, reference["thresholds"])
    if json.dumps(limits, sort_keys=True) != locked:
        raise RuntimeError("Calibration changed during evaluation")
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        fixtures={name: digest(getattr(args, name)) for name in
                  ("dev_manifest", "dev_wording", "test_manifest", "test_wording", "reference")},
        risk_cap=RISK_CAP, coverage_floor=MIN_COVERAGE, limits=limits,
        development=report(development, limits), holdout=report(held_output, limits),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev_xml", "dev_manifest", "dev_wording", "test_xml", "test_manifest", "test_wording",
                 "training", "reference", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
