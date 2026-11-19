"""Test live topical ranking and person-specific attribution after one known duty."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit.compile_legacy_notes import char_grams, compile_note, gram_norm, grams_score
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("top2_topic", "top2_person", "all_person", "top1_then_scope", "top2_scope", "scope_chars")
TOPICAL = "Find the contribution statement that directly describes the requested activity."
PERSONAL = ("Find a contribution statement that explicitly credits the named author with the requested activity. "
            "Credit to other authors alone is insufficient.")
MAX_CALIBRATION_WRONG = .10


def make_case(row, variant):
    doi, text, duties, correct, distractor = row
    clauses = compile_note(text)
    roles = [legacy.LABELS.get(clause.action.casefold()) for clause in clauses]
    shared = [role for role in duties
              if any(item == role and {correct, distractor} <= clause.owners
                     for item, clause in zip(roles, clauses, strict=True))]
    if len(shared) != 1:
        return None
    missing = next(role for role in duties if role != shared[0])
    remainder = [(clause, role) for clause, role in zip(clauses, roles, strict=True) if role != shared[0]]
    labels = [role == missing and correct in clause.owners for clause, role in remainder]
    if not any(labels) or any(role == missing and distractor in clause.owners for clause, role in remainder):
        return None
    return dict(
        doi=doi, owners=(correct, distractor), missing=missing, phrase=legacy.PHRASES[missing][variant],
        docs=[text[clause.start:clause.end] for clause, _ in remainder],
        scope=[clause.owners for clause, _ in remainder],
        roles=[role for _, role in remainder],
        grams=[(counts, gram_norm(counts)) for counts in [char_grams(clause.action) for clause, _ in remainder]],
    )


def propose(reader, case, arm):
    start = perf_counter()
    scopes = case["scope"]
    task = case["phrase"]
    docs = case["docs"]
    if arm == "all_person":
        ranking = list(range(len(docs)))
    else:
        raw = grams_score(char_grams(task), case["grams"])
        ranking = np.argsort(-raw, kind="stable").tolist()[:2]
    if arm == "top1_then_scope":
        ranking = ranking[:1]
    personalized = arm in ("top2_person", "all_person")
    prefilter = arm in ("top2_scope", "scope_chars")
    cells = []
    for owner in case["owners"]:
        eligible = [idx for idx in ranking if not prefilter or owner in scopes[idx]]
        for idx in eligible:
            cells.append((owner, idx))
    if personalized:
        inputs = [(f"Author initials: {owner}. Requested activity: {task}", docs[idx]) for owner, idx in cells]
        positions = cells
    else:
        indices = list(dict.fromkeys(idx for _, idx in cells))
        inputs = [(task, docs[idx]) for idx in indices]
        positions = [(None, idx) for idx in indices]
    if arm == "scope_chars":
        values, clipped = [float(raw[idx]) for _, idx in positions], 0
        inputs = []
    else:
        values, clipped = reader.score_pairs(inputs, task=PERSONAL if personalized else TOPICAL) if inputs else ([], 0)
    scores = dict(zip(positions, values, strict=True))
    outputs = []
    for owner in case["owners"]:
        eligible = [
            (scores[owner if personalized else None, idx], idx)
            for who, idx in cells if who == owner
        ]
        if not eligible:
            outputs.append(None)
            continue
        score, best = max(eligible, key=lambda pair: (pair[0], -pair[1]))
        if arm == "top1_then_scope" and owner not in scopes[best]:
            outputs.append(None)
        else:
            outputs.append((score, case["roles"][best] == case["missing"] and owner in scopes[best]))
    return dict(outputs=outputs, pairs=len(inputs), clipped=clipped, ms=(perf_counter() - start) * 1000)


def collect(rows, groups, reader, variant):
    cases = [(case, group) for row, group in zip(rows, groups, strict=True)
             if (case := make_case(row, variant)) is not None]
    if not cases:
        raise ValueError("No independent anchor contrasts")
    report = []
    for i, (case, group) in enumerate(cases):
        offset = i % len(ARMS)
        arms = ARMS[offset:] + ARMS[:offset]
        report.append((group, {arm: propose(reader, case, arm) for arm in arms}))
    return report


def calibrate(outputs):
    n = len(outputs)
    thresholds = {threshold for _, row in outputs for cell in row["outputs"]
                  if cell is not None for threshold in (float(np.nextafter(cell[0], -np.inf)), cell[0])}
    thresholds.add(0.)
    possible = []
    for threshold in sorted(thresholds):
        counts = [int(any(cell is not None and cell[0] > threshold and cell[1] == expected
                          for cell in row["outputs"])) for _, row in outputs for expected in (True, False)]
        valid, invalid = sum(counts[::2]), sum(counts[1::2])
        if invalid <= int(n * MAX_CALIBRATION_WRONG):
            possible.append((valid, -invalid, threshold))
    return max(possible)[-1]


def summarize(rows, calibration):
    thresholds = {arm: calibrate([(group, case[arm]) for group, case in calibration]) for arm in ARMS}
    grades = {}
    for arm in ARMS:
        outputs = [case[arm] for _, case in rows]
        cut = thresholds[arm]
        accepted = [[cell is not None and cell[0] > cut for cell in out["outputs"]] for out in outputs]
        true = [is_on[0] and row["outputs"][0][1] for is_on, row in zip(accepted, outputs, strict=True)]
        wrong = [sum(int(flag and not cell[1]) for flag, cell in zip(is_on, row["outputs"], strict=True))
                 for is_on, row in zip(accepted, outputs, strict=True)]
        grades[arm] = dict(
            full=outputs, true=true, wrong=wrong, threshold=cut, supported=sum(true), invalid=sum(wrong),
            wrong_person=sum(flags[1] for flags in accepted),
            misses_without_threshold=sum(row["outputs"][0] is None or not row["outputs"][0][1] for row in outputs),
            false_attachment_query=sum(bool(value) for value in wrong),
            clean_success=sum(bool(valid and not invalid) for valid, invalid in zip(true, wrong, strict=True)),
            model_pairs=sum(row["pairs"] for row in outputs), truncated=sum(row["clipped"] for row in outputs),
            component_p95_ms=float(np.quantile([row["ms"] for row in outputs], .95)),
        )
    baseline = grades["all_person"]
    groups = [group for group, _ in rows]
    for values in grades.values():
        for field in ("true", "wrong"):
            values[f"{field}_difference_vs_all"] = clustered_paired_interval(
                [int(a) - int(b) for a, b in zip(values[field], baseline[field], strict=True)], groups)
    return {arm: {name: val for name, val in values.items() if name not in ("full", "true", "wrong")}
            for arm, values in grades.items()}


def run(old, transfer_path, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    panels = legacy.prepare(old)
    calibration, _ = panels["calibration"]
    transfer_rows, names, _ = transfer.cohort(old, transfer_path)
    reader = Reranker(checkpoint)
    reader.score_pairs([("Warm up", "A statement")] * 4, task=TOPICAL)
    fitted = collect(calibration, list(range(len(calibration))), reader, 0)
    first = collect(panels["test"][0], list(range(len(panels["test"][0]))), reader, 1)
    external = collect(transfer_rows, transfer.clusters(names), reader, 1)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), calibration=len(fitted),
        capture_sha256=digest(old / "capture.json"), transfer_sha256=digest(transfer_path / "capture.json"),
        script_sha256=digest(Path(__file__)), reranker_revision=reader.provenance["revision"],
        panels={name: dict(papers=len(samples), arms=summarize(samples, fitted))
                for name, samples in (("test", first), ("transfer", external))},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "transfer_path", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
