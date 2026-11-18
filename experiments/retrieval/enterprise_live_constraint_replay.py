"""Replay hard-constraint schedules on local open-model scores for frozen evidence pools."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.constraint_verification import resolve_constraints
from expert_search.qwen_reranker import pack_tokens, QwenReranker
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


METHODS = ("exhaustive", "short_circuit", "bundled_first", "certificate")
TASK = (
    "Judge whether the document explicitly records the requested concrete technical condition "
    "in the stated setting. General topic overlap, goals, and indirect capability are insufficient. "
    "Check the stated thresholds, units, and operating conditions if supplied."
)
THRESHOLDS = (.90, .50)
MODEL_BATCH = 8
TOP = 3


def bigrams(value):
    letters = "".join(re.findall(r"\w", value.casefold()))
    return set(zip(letters, letters[1:]))


def pool(job, conditions):
    if any(not isinstance(c, str) or not c or c not in job["demand"] for c in conditions):
        raise ValueError("Condition is not an anchored substring of the original demand")
    if len(conditions) != len(set(conditions)) or len(conditions) < 2:
        raise ValueError("Expected multiple distinct hard conditions")
    candidates = job["candidates"]
    order = list(dict.fromkeys([*job["dense_order"], *candidates]))
    targets = [bigrams(value) for value in conditions]
    owners = []
    for expert in order:
        sources = candidates[expert]
        if any(source["expert_id"] != expert or source["id"] != doc for doc, source in sources.items()):
            raise ValueError("Source and candidate ownership differ from the frozen pool")
        source_terms = {doc: bigrams(row["text"]) for doc, row in sources.items()}
        witnesses = [tuple(sorted(sources, key=lambda doc: (-len(terms & source_terms[doc]), doc)))
                     for terms in targets]
        owners.append((expert, witnesses))
    return owners


def score_sources(model, job, conditions, owners):
    title = job["demand"].splitlines()[0]
    queued = []
    truncated = 0
    for expert, witnesses in owners:
        for doc in dict.fromkeys(source for group in witnesses for source in group):
            source = job["candidates"][expert][doc]["text"]
            for i, condition in enumerate(conditions):
                body = model.tokenizer.encode(
                    f"<Instruct>: {TASK}\n<Query>: Project: {title}\nRequired condition: {condition}\n<Document>: {source}",
                    add_special_tokens=False,
                )
                sequence, raw = pack_tokens(model.prefix, body, model.suffix, model.max_length)
                truncated += raw > len(sequence)
                queued.append((expert, doc, i, sequence))
    queued.sort(key=lambda entry: len(entry[3]))
    observed = {}
    for start in range(0, len(queued), MODEL_BATCH):
        batch = queued[start:start+MODEL_BATCH]
        probabilities = model._batch([sequence for _, _, _, sequence in batch])
        for (expert, doc, i, _), score in zip(batch, probabilities, strict=True):
            if not np.isfinite(score) or not 0. <= score <= 1.:
                raise ValueError("Invalid source score")
            observed.setdefault((expert, doc), {})[i] = score
    return observed, truncated


def summarize(rows):
    report = {}
    for method in METHODS:
        arms = [r["arms"][method] for r in rows]
        report[method] = dict(
            requests=len(arms), exact=sum(a["ranking"] == r["arms"]["exhaustive"]["ranking"]
                                         for r, a in zip(rows, arms, strict=True)),
            returned=sum(len(a["ranking"]) for a in arms), nonempty=sum(bool(a["ranking"]) for a in arms),
            mean_source_reads=mean(a["reads"] for a in arms),
            mean_criterion_scores=mean(a["checks"] for a in arms),
        )
    return report


def run(jobs_path, requirements_path, checkpoint, output, review=None):
    if output.exists() or (review is not None and review.exists()):
        raise FileExistsError("The result or review path already exists")
    review_rows = []
    demands = {}
    for line in jobs_path.read_bytes().splitlines():
        job = json.loads(line)
        if job["split"] == "development":
            if job["id"] in demands:
                raise ValueError("Duplicate development query")
            demands[job["id"]] = job
    requirements = json.loads(requirements_path.read_bytes())
    if not requirements.keys() <= demands.keys():
        raise ValueError("Use only the development split")
    provenance = verify_checkpoint(checkpoint)
    model = QwenReranker(checkpoint, max_length=768)
    rows = []
    if not requirements:
        raise ValueError("The development sample is empty")
    for i, (key, conditions) in enumerate(requirements.items()):
        job = demands[key]
        owners = pool(job, conditions)
        start = perf_counter()
        scores, truncated = score_sources(model, job, conditions, owners)
        trace_ms = (perf_counter() - start) * 1000
        if review is not None:
            for criterion, condition in enumerate(conditions):
                order = sorted(scores, key=lambda pair: (-scores[pair][criterion], pair))
                # Two highest scores and one low-score diagnostic. Scores and rank are hidden.
                for expert, source in dict.fromkeys(order[:2] + order[-1:]):
                    stable = sha256(repr((key, criterion, expert, source)).encode()).hexdigest()[:20]
                    review_rows.append(dict(
                        case_id=stable, demand_title=job["demand"].splitlines()[0],
                        condition=condition, evidence=job["candidates"][expert][source]["text"],
                        support=None, personal_role=None, quote=None, rationale=None,
                    ))
        variants = {}
        coverage = {}
        for threshold in THRESHOLDS:
            def read(expert, source):
                return {criterion: score >= threshold
                        for criterion, score in scores[expert, source].items()}

            arms = {method: resolve_constraints(owners, read, top=TOP, strategy=method)
                    for method in METHODS}
            if any(arm["ranking"] != arms["exhaustive"]["ranking"] for arm in arms.values()):
                raise AssertionError("Schedule changed final citations")
            key_threshold = f"{threshold:.2f}"
            variants[key_threshold] = arms
            supports = {expert: set() for expert, _ in owners}
            for (expert, _), values in scores.items():
                supports[expert].update(i for i, score in values.items() if score >= threshold)
            coverage[key_threshold] = dict(
                criterion_slots=len(conditions) * len(supports),
                supported_slots=sum(len(value) for value in supports.values()),
                candidates_with_any=sum(bool(value) for value in supports.values()),
                candidates_with_all=sum(len(value) == len(conditions) for value in supports.values()),
            )
        rows.append(dict(id=key, candidate_experts=len(job["candidates"]), full_sources=len(scores),
                         truncated=truncated, model_trace_ms=trace_ms, variants=variants, coverage=coverage))
        print(f"Completed development query {i+1}/{len(requirements)}", flush=True)
    report = {f"{threshold:.2f}": summarize([
        dict(arms=row["variants"][f"{threshold:.2f}"]) for row in rows
    ]) for threshold in THRESHOLDS}
    counts = {threshold: {field: sum(row["coverage"][threshold][field] for row in rows)
                         for field in rows[0]["coverage"][threshold]} for threshold in report}
    if review is not None:
        shuffled = sorted(review_rows, key=lambda row: sha256(row["case_id"].encode()).digest())
        with review.open("x") as destination:
            for row in shuffled:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output, dict(
        inputs={"jobs": digest(jobs_path), "requirements": digest(requirements_path)},
        script_sha256=digest(Path(__file__)), verifier=provenance, top=TOP,
        total_candidates=sum(r["candidate_experts"] for r in rows),
        total_sources=sum(r["full_sources"] for r in rows), total_truncated=sum(r["truncated"] for r in rows),
        report=report, coverage=counts, review_cases=len(review_rows), rows=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("jobs", "requirements", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    run(args.jobs, args.requirements, args.checkpoint, args.output, args.review)
