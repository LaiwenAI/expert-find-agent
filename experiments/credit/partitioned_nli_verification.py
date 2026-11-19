"""Evaluate author-scoped lightweight verification with shared qualification rules."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from experiments.credit.compositional_edits import VERSION, stable
from experiments.credit.role_language_router import NISO_SHA256
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.minilm_sufficient_facts import SEEDS
from experiments.scifact.minilm_transfer import apply_weights


METHODS = ("paper_pooled_joint", "author_scoped_joint", "author_scoped_by_source", "author_scoped_by_task")
HIDDEN = {"left_withheld": 0, "right_withheld": 1}


def request_inputs(sources, row, phrases, method):
    hidden = HIDDEN.get(row["view"])
    records = [sources[doi]["authors"] for doi in row["sources"]]
    authors = [record[index - 1] for record, index in zip(records, row["ordinals"], strict=True)]
    if hidden is None:
        if any(author["person"] is None for author in authors):
            raise ValueError("Expected identity information in complete records")
        if authors[0]["person"] != authors[1]["person"]:
            return "different", []
    texts = {}
    for slot, author in enumerate(authors):
        if slot == hidden:
            continue
        roles = author["roles"]
        if method == "paper_pooled_joint":
            roles = sorted({role for record in records[slot] for role in record["roles"]})
        article = "AB"[slot]
        texts[slot] = f"The complete contribution statement in publication {article} credits the author with: " + ", ".join(roles) + "."
    tasks = [(slot, f"In publication {'AB'[slot]}: {phrase}")
             for (slot, _), phrase in zip(row["requirements"], phrases, strict=True) if slot in texts]
    if method.endswith("_joint"):
        groups = [tasks]
    elif method.endswith("_source"):
        groups = [[task for task in tasks if task[0] == slot] for slot in texts]
    else:
        groups = [[task] for task in tasks]
    pairs = [
        dict(source=" ".join(texts[slot] for slot in sorted({slot for slot, _ in group})),
             criterion="The author is credited with all the following work. " + " ".join(phrase for _, phrase in group))
        for group in groups if group
    ]
    return ("missing" if hidden is not None else "same"), pairs


def predict(observation, reject_below, support_above):
    identity, margin = observation
    if identity == "different" or margin < reject_below:
        return "rejected"
    if identity == "same" and margin > support_above:
        return "supported"
    return "unresolved"


def fit(rows, observations, *, eligible_only):
    scores = np.unique([obs[1] for obs in observations if obs[0] != "different"])
    thresholds = [-1e6, *scores, 1e6]
    unsupported = [obs[1] for row, obs in zip(rows, observations, strict=True)
                   if obs[0] == "same" and row["reference"] != "supported"]
    non_support_count = len(unsupported) if eligible_only else sum(row["reference"] != "supported" for row in rows)
    support = next(t for t in thresholds if sum(m > t for m in unsupported) <= non_support_count // 20)
    would_reject = [obs[1] for row, obs in zip(rows, observations, strict=True) if row["reference"] != "rejected"]
    rejection = next(t for t in reversed(thresholds)
                     if t <= support and sum(m < t for m in would_reject) <= len(would_reject) // 20)
    return float(rejection), float(support)


def summarize(rows, observations, times, cutoffs):
    suites = {"all": list(range(len(rows)))}
    suites.update({v: [i for i, row in enumerate(rows) if row["view"] == v]
                   for v in sorted({row["view"] for row in rows})})
    predictions = [predict(obs, *cutoffs) for obs in observations]
    out = {}
    for name, subset in suites.items():
        confusion = Counter((rows[i]["reference"], predictions[i]) for i in subset)
        out[name] = dict(cases=len(subset), correct=sum(n for (a, b), n in confusion.items() if a == b),
                         confusion=[dict(gold=a, predicted=b, count=n) for (a, b), n in sorted(confusion.items())],
                         p95_ms=float(np.quantile([sum(times[i]) for i in subset], .95)),
                         forwards=sum(len(times[i]) for i in subset))
    grouped = defaultdict(list)
    for row, predicted in zip(rows, predictions, strict=True):
        if row["view"] == "present":
            grouped[row["group"], row["track"]].append(row["reference"] == predicted)
    for name in ("flip", "stable"):
        pairs = [items for (_, track), items in grouped.items() if track == name]
        if any(len(items) != 2 for items in pairs):
            raise ValueError("Unexpected matched-pair shape")
        out[f"{name}_pairs"] = dict(total=len(pairs), both_correct=sum(all(items) for items in pairs))
    return out


def run(fixture, examples, training, weights, checkpoint, output):
    import torch
    from experiments.scifact.small_nli_verifier import load, measure

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    manifest_path = fixture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["version"] != VERSION or any(digest(fixture / n) != h for n, h in manifest["files"].items()):
        raise ValueError("Changed benchmark fixture")
    if digest(examples) != NISO_SHA256:
        raise ValueError("Changed request phrasings")
    sources = json.loads((fixture / "sources.json").read_text())
    queries = json.loads((fixture / "queries.json").read_text())
    by_role = defaultdict(list)
    for entry in json.loads(examples.read_text())["examples"]:
        by_role[entry["role"]].append(entry["text"])
    panels = {split: [r for r in queries if r["split"] == split] for split in ("calibration", "test")}
    rendered = {
        split: {method: [request_inputs(sources, row, [
            min(by_role[role], key=lambda phrase: stable(row["group"], role, phrase))
            for _, role in row["requirements"]
        ], method) for row in rows] for method in METHODS}
        for split, rows in panels.items()
    }
    trained = json.loads(training.read_text())
    results = {}
    for arm in ("frozen", *(f"rationale_pair_{seed}" for seed in SEEDS)):
        tokenizer, reader, positions, provenance = load(checkpoint, "minilm")
        if provenance["revision"] != trained["model_revision"]:
            raise ValueError("Changed model revision")
        if arm != "frozen":
            apply_weights(reader, weights / f"{arm}.safetensors", trained["report"][arm]["training"]["weight_sha256"])
        measure(tokenizer, reader, positions, [{"source": "A complete record.", "criterion": "A role."}], batch_size=1)
        results[arm] = {}
        for method in METHODS:
            traces = {}
            for split, inputs in rendered.items():
                flat = [item for _, items in inputs[method] for item in items]
                scores, times = measure(tokenizer, reader, positions, flat, batch_size=1)
                if any(score is None for score in scores):
                    raise ValueError("Attribution comparisons must fit the reader context")
                observations, request_times = [], []
                offset = 0
                for identity, pairs in inputs[method]:
                    end = offset + len(pairs)
                    margin = min((item[1] for item in scores[offset:end]), default=0.0)
                    observations.append((identity, margin))
                    request_times.append(times[offset:end])
                    offset = end
                traces[split] = observations, request_times
            results[arm][method] = {}
            for policy, eligible_only in (("global", False), ("eligible_only", True)):
                cutoffs = fit(panels["calibration"], traces["calibration"][0], eligible_only=eligible_only)
                results[arm][method][policy] = dict(
                    reject_below=cutoffs[0], support_above=cutoffs[1],
                    panels={split: summarize(rows, *traces[split], cutoffs) for split, rows in panels.items()},
                )
        del reader
        print(arm, "complete", flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            script_sha256=digest(Path(__file__)), fixture_sha256=digest(manifest_path),
                            examples_sha256=digest(examples), training_sha256=digest(training),
                            model_revision=provenance["revision"], report=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "examples", "training", "weights", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
