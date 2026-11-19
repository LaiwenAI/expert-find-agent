"""Calibrate an enterprise evidence verifier on human SciFact labels."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.retrieval.enterprise_first_query import TASK as ENTERPRISE_TASK


ARCHIVE = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
TASKS = {
    "existing": ENTERPRISE_TASK,
    "explicit": (
        "Does the document explicitly establish the complete claim as an observed finding? "
        "Require the same entities, conditions, measurement, and direction. "
        "A related finding, proposed result, or contrary result does not support the claim."
    ),
    "refute": (
        "Does the document explicitly contradict the complete claim under its stated conditions? "
        "A related finding or lack of information does not contradict the claim."
    ),
}
ARMS = ("existing", "explicit", "contradiction_adjusted")
LABELS = ("support", "contradict", "unknown")
CAP = 0.05
MAX_CHARACTERS = 2500


def read_jsonl(archive, name):
    with archive.extractfile(f"data/{name}.jsonl") as stream:
        return [json.loads(line) for line in stream]


def row_key(claim, doc):
    return sha256(f"criterion-calibration-v1:{claim}:{doc}".encode()).digest()


def prepare(claims, documents, excluded_docs):
    eligible = []
    for row in claims:
        claim = row["claim"]
        choices = []
        if not row["evidence"]:
            choices = [(str(doc), "unknown") for doc in row["cited_doc_ids"]]
        else:
            for doc, labels in row["evidence"].items():
                unique = {label["label"].lower() for label in labels}
                if len(unique) == 1:
                    choices.append((doc, next(iter(unique))))
        if len({label for _, label in choices}) != 1:
            continue
        for doc, label in choices:
            if doc not in excluded_docs and doc in documents and len(documents[doc]) <= MAX_CHARACTERS:
                eligible.append((row_key(row["id"], doc), row["id"], doc, label, claim, documents[doc]))
    used_claims, used_docs, result = set(), set(), []
    for _, case, doc, label, claim, document in sorted(eligible):
        if case not in used_claims and doc not in used_docs:
            used_claims.add(case)
            used_docs.add(doc)
            result.append(dict(case=case, doc=doc, label=label, criterion=claim, source=document))
    if set(LABELS) != {row["label"] for row in result}:
        raise ValueError("Missing a reference verdict")
    return result


def measure(reader, data, batch_size=16):
    results = []
    timings = {name: [] for name in TASKS}
    arms = tuple(TASKS)
    for start_at in range(0, len(data), batch_size):
        rows = data[start_at:start_at + batch_size]
        pairs = [(row["criterion"], row["source"]) for row in rows]
        offset = (start_at // batch_size) % len(arms)
        measured = {}
        for arm in arms[offset:] + arms[:offset]:
            start = perf_counter()
            scores, truncated = reader.score_pairs(pairs, task=TASKS[arm])
            timings[arm].append((perf_counter() - start) * 1000)
            if truncated or not all(math.isfinite(score) for score in scores):
                raise ValueError("Unexpected truncated or nonfinite verification input")
            measured[arm] = scores
        for i in range(len(rows)):
            existing, explicit, refute = (measured[arm][i] for arm in arms)
            results.append(dict(existing=existing, explicit=explicit, refute=refute,
                                contradiction_adjusted=explicit - max(0., refute)))
    return results, timings


def counts(positive, gold):
    return {label: {"cases": sum(g == label for g in gold),
                    "predicted_support": sum(g == label and pred for g, pred in zip(gold, positive, strict=True))}
            for label in LABELS}


def calibrate(margins, gold):
    """Maximize true supports under separate contradiction and unknown false-support caps."""
    choices = []
    for t in sorted(set(margins) | {0.}):
        report = counts([value > t for value in margins], gold)
        false = {label: report[label]["predicted_support"] for label in LABELS[1:]}
        if all(false[label] <= CAP * report[label]["cases"] for label in false):
            choices.append((report["support"]["predicted_support"], -sum(false.values()), t, report))
    if not choices:
        raise AssertionError("The strictest threshold must pass the calibration cap")
    _, _, threshold, report = max(choices, key=lambda item: item[:3])
    return threshold, report


def evaluation(samples, scores, thresholds):
    labels = [row["label"] for row in samples]
    baseline = [row["existing"] > 0 for row in scores]
    methods = {
        "existing_zero": baseline,
        "explicit_zero": [row["explicit"] > 0 for row in scores],
        **{arm + "_calibrated": [row[arm] > thresholds[arm] for row in scores] for arm in ARMS},
    }
    return {arm: {
        "labels": counts(positive, labels),
        "binary_accuracy_delta_vs_existing_zero": paired_interval([
            int(pred == (gold == "support")) - int(base == (gold == "support"))
            for pred, base, gold in zip(positive, baseline, labels, strict=True)
        ]),
    } for arm, positive in methods.items()}


def run(archive_path, checkpoint, output, private_input=None, private_output=None):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or (private_input is None) != (private_output is None)
            or (private_output is not None and private_output.exists())):
        raise ValueError("Require new results paths and a Slurm GPU")
    if digest(archive_path) != ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        docs = {str(row["doc_id"]): " ".join(row["abstract"]) for row in read_jsonl(archive, "corpus")}
        train = read_jsonl(archive, "claims_train")
        dev = read_jsonl(archive, "claims_dev")
    blocked = {str(doc) for c in dev for doc in (*c["cited_doc_ids"], *c["evidence"])}
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("Overlapping claim identities")
    fit, test = prepare(train, docs, blocked), prepare(dev, docs, set())
    reader = Reranker(checkpoint)
    if reader.provenance["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473":
        raise ValueError("Require the enterprise verifier checkpoint")
    measure(reader, fit[:16])
    fitted, _ = measure(reader, fit)
    calibrated = {arm: calibrate([row[arm] for row in fitted], [row["label"] for row in fit]) for arm in ARMS}
    cutoffs = {arm: value[0] for arm, value in calibrated.items()}
    evaluated, times = measure(reader, test)
    if private_input is not None:
        samples = [json.loads(line) for line in private_input.read_text().splitlines()]
        if len({row["case_id"] for row in samples}) != len(samples):
            raise ValueError("Review packet has duplicate case IDs")
        values, _ = measure(reader, samples)
        with private_output.open("x", encoding="utf-8") as stream:
            private_output.chmod(0o600)
            for row, score in zip(samples, values, strict=True):
                record = {"case_id": row["case_id"],
                          "decisions": {arm: score[arm] > cutoffs[arm] for arm in ARMS}
                          | {"existing_zero": score["existing"] > 0, "explicit_zero": score["explicit"] > 0}}
                stream.write(json.dumps(record) + "\n")
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        archive_sha256=digest(archive_path), script_sha256=digest(Path(__file__)),
        verifier=reader.provenance["revision"], threshold_cap=CAP, maximum_abstract_characters=MAX_CHARACTERS,
        calibration_cases=len(fit), evaluation_cases=len(test),
        calibration={arm: dict(threshold=threshold, labels=report)
                     for arm, (threshold, report) in calibrated.items()},
        evaluation=evaluation(test, evaluated, cutoffs),
        evaluation_batch_p50_p95_ms={arm: np.quantile(values, [.5, .95]).tolist() for arm, values in times.items()},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive-path", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("private-input", "private-output"):
        parser.add_argument(f"--{name}", type=Path)
    run(**vars(parser.parse_args()))
