"""Separate evidence-window recall from lightweight NLI judgment errors."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import torch

from expert_search.retrieval import tokens
from experiments.followir.core17_reranking import digest, write_json
from experiments.scifact.criterion_verifier_calibration import ARCHIVE, prepare, read_jsonl
from experiments.scifact.small_nli_verifier import BATCH, fit, load, measure, summarize


ARMS = ("full", "lead_three", "lexical_three", "lexical_two_windows", "annotated_rationale")
DOC_SENTENCES = 3


def choose_windows(claim, sentences):
    wanted = set(tokens(claim))
    sentence_terms = [set(tokens(sent)) & wanted for sent in sentences]
    width = min(DOC_SENTENCES, len(sentences))
    windows = [tuple(range(i, i + width)) for i in range(len(sentences) - width + 1)]
    words = [set().union(*(sentence_terms[i] for i in block)) for block in windows]
    first = max(range(len(windows)), key=lambda i: (len(words[i]), -i))
    allowed = [i for i, block in enumerate(windows)
               if not set(block).intersection(windows[first]) and words[i] - words[first]]
    if not allowed:
        return windows[first], windows[first]
    second = max(allowed, key=lambda i: (len(words[i] - words[first]), -i))
    return windows[first], tuple(sorted((*windows[first], *windows[second])))


def views(row, case, sentences):
    lexical, two = choose_windows(row["criterion"], sentences)
    rationale = [tuple(sorted(set(group["sentences"]))) for group in case["evidence"].get(row["doc"], [])
                 if group["label"].lower() == row["label"]]
    if any(not group or min(group) < 0 or max(group) >= len(sentences) for group in rationale):
        raise ValueError("Invalid annotated rationale location")
    indices = {
        "full": tuple(range(len(sentences))),
        "lead_three": tuple(range(min(DOC_SENTENCES, len(sentences)))),
        "lexical_three": lexical,
        "lexical_two_windows": two,
        "annotated_rationale": min(rationale, key=lambda seq: (len(seq), seq)) if rationale else lexical,
    }
    return indices, rationale


def evaluate(tokenizer, reader, positions, rows, cases, abstracts):
    variants = {arm: [] for arm in ARMS}
    selection_ms = []
    available = Counter()
    for offset in range(0, len(rows), BATCH):
        started = perf_counter()
        for row in rows[offset:offset+BATCH]:
            abstract = abstracts[row["doc"]]
            indices, rationale = views(row, cases[row["case"]], abstract)
            for arm, chosen in indices.items():
                evidence = " ".join(abstract[i] for i in chosen)
                variants[arm].append(dict(source=evidence, criterion=row["criterion"]))
                available[arm] += bool(rationale and any(set(group) <= set(chosen) for group in rationale))
            available["with_annotations"] += bool(rationale)
        selection_ms.append((perf_counter() - started) * 1000)
    results = {}
    for arm, batch in variants.items():
        scores, times = measure(tokenizer, reader, positions, batch)
        results[arm] = dict(scores=scores, times=times)
    return results, dict(available), selection_ms


def run(archive_path, checkpoint, output):
    if output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output path on a Slurm GPU")
    if digest(archive_path) != ARCHIVE:
        raise ValueError("Changed SciFact archive")
    with tarfile.open(archive_path, "r:gz") as archive:
        abstracts = {str(row["doc_id"]): row["abstract"] for row in read_jsonl(archive, "corpus")}
        train = read_jsonl(archive, "claims_train")
        dev = read_jsonl(archive, "claims_dev")
    docs = {key: " ".join(sentences) for key, sentences in abstracts.items() if sentences}
    blocked = {str(doc) for c in dev for doc in (*c["cited_doc_ids"], *c["evidence"])}
    calibration = prepare(train, docs, blocked)
    test = prepare(dev, docs, set())
    if (len(calibration), len(test)) != (309, 226):
        raise ValueError("Changed public cohorts")
    cases = {row["id"]: row for row in (*train, *dev)}
    tokenizer, reader, positions, provenance = load(checkpoint, "mdeberta")
    measure(tokenizer, reader, positions, calibration[:BATCH])
    fitted, _, _ = evaluate(tokenizer, reader, positions, calibration, cases, abstracts)
    tested, availability, selection = evaluate(tokenizer, reader, positions, test, cases, abstracts)
    gold_calibration = [row["label"] for row in calibration]
    gold_test = [row["label"] for row in test]
    report = {}
    for arm in ARMS:
        threshold, control = fit(fitted[arm]["scores"], gold_calibration)
        observations = tested[arm]
        report[arm] = dict(
            threshold=threshold, calibration=control, evaluation=summarize(observations["scores"], gold_test, threshold),
            verifier_batch_p50_p95_ms=np.quantile(observations["times"], [.5, .95]).tolist(),
        )
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), archive_sha256=ARCHIVE, checkpoint=provenance["revision"],
        calibration_cases=len(calibration), evaluation_cases=len(test),
        rationale_containment=availability,
        all_views_selection_batch_p50_p95_ms=np.quantile(selection, [.5, .95]).tolist(),
        results=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive-path", "checkpoint", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    run(**vars(parser.parse_args()))
