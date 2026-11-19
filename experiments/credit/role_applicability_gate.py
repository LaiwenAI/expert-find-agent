"""Compare closed-role confidence with an applicability gate on NISO examples."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import anchored_clause_rescue as clauses
from experiments.credit import anchored_role_margin as prior
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit.role_language_router import NISO_SHA256
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


# These selected guide examples describe the narrower legacy publisher headings directly.
POSITIVE = {
    "Methodology": ("design", (3,)),
    "Investigation": ("execution", (1, 2)),
    "Formal analysis": ("analysis", (1, 2)),
    "Resources": ("resources", (1,)),
    "Writing – original draft": ("writing", (1, 2)),
}
NEGATIVE = {
    "Funding acquisition": (1, 2, 3),
    "Project administration": (1, 3, 4),
    "Data curation": (1, 2, 3, 4),
    "Supervision": (1, 2, 3, 4),
    "Visualization": (1, 2),
}
EXCLUDED_PROTOTYPES = (
    "Conceptualization", "Data curation", "Funding acquisition", "Project administration",
    "Software", "Supervision", "Validation", "Visualization", "Writing – review & editing",
)
METHODS = ("absolute", "relative", "outside_contrast", "definition_contrast")


def query(phrase):
    return f"Instruct: {prior.TASK}\nQuery: {phrase}"


def score(phrase, encoder, bank, definitions, excluded):
    prediction, confidence, elapsed, _, _ = score_with_definitions(phrase, encoder, bank, definitions, excluded)
    return prediction, confidence, elapsed


def score_with_definitions(phrase, encoder, bank, definitions, excluded):
    start = perf_counter()
    vec = encoder.encode([query(phrase)])[0]
    scores = bank @ vec
    order = np.argsort(-scores, kind="stable")[:2]
    best, second = map(float, scores[order])
    outside = float((excluded @ vec).max())
    definition_scores = definitions @ vec
    defined = int(definition_scores.argmax())
    outputs = dict(absolute=best, relative=best - second, outside_contrast=best - outside,
                   definition_contrast=float(definition_scores[defined]) - outside)
    roles = dict.fromkeys(METHODS, prior.ROLES[order[0]])
    roles["definition_contrast"] = prior.ROLES[defined]
    return roles, outputs, (perf_counter() - start) * 1000, definition_scores, vec


def gate_threshold(positives, negatives, method):
    candidates = {
        float(np.nextafter(row[1][method], -np.inf)) for row in (*positives, *negatives)
    } | {row[1][method] for row in (*positives, *negatives)}
    feasible = []
    for cut in sorted(candidates):
        selected = sum(predicted[method] == gold and scores[method] > cut for predicted, scores, gold in positives)
        adverse = sum(scores[method] > cut for _, scores, _ in negatives)
        if adverse <= int(.10 * len(negatives)):
            feasible.append((selected, -adverse, cut))
    return max(feasible)[-1]


def attach(case, role):
    attached = [(owner, idx) for owner in case["owners"]
                for idx, (declared, scope) in enumerate(zip(case["roles"], case["scope"], strict=True))
                if declared == role and owner in scope]
    return bool(attached) and role == case["missing"], bool(attached) and role != case["missing"]


def evaluate(rows, thresholds):
    results = {}
    clean = {}
    for method, cut in thresholds.items():
        counts = dict(valid=0, invalid=0, abstained=0, forced_correct=0)
        clean[method] = []
        for _, experimental, predictions, scores, case, _ in rows:
            prediction = predictions[method]
            passed = scores[method] > cut
            counts["forced_correct"] += int(prediction == experimental)
            valid, invalid = (attach(case, prediction) if case is not None else
                              (prediction == experimental, prediction != experimental))
            counts["valid"] += int(passed and valid)
            counts["invalid"] += int(passed and invalid)
            counts["abstained"] += int(not (passed and (valid or invalid)))
            clean[method].append(int(passed and valid))
        results[method] = counts
    assert all(len(rows) == value["valid"] + value["invalid"] + value["abstained"]
               for value in results.values())
    groups = [group for group, *_ in rows]
    for method, values in results.items():
        values["valid_delta_vs_absolute"] = clustered_paired_interval(
            [a - b for a, b in zip(clean[method], clean["absolute"], strict=True)], groups)
    return dict(cases=len(rows), controller_p95_ms=float(np.quantile([row[-1] for row in rows], .95)),
                methods=results)


def run(old, external, niso_path, embedding, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a Slurm CUDA job and a fresh output")
    if digest(niso_path) != NISO_SHA256:
        raise ValueError("Changed guide input")
    guide = json.loads(niso_path.read_bytes())
    positives = [(gold, row) for key, (gold, ordinals) in POSITIVE.items() for row in guide["examples"]
                 if row["role"] == key and row["ordinal"] in ordinals]
    negatives = [row for row in guide["examples"]
                 if row["ordinal"] in NEGATIVE.get(row["role"], ())]
    assert len(positives) == 8 and len(negatives) == 16
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    bank = encoder.encode(list(prior.DESCRIPTIONS))
    crosswalk = {role: name for name, (role, _) in POSITIVE.items()}
    definitions = encoder.encode([f"{heading}. {guide['definitions'][crosswalk[role]]}"
                                  for heading, role in zip(prior.DESCRIPTIONS, prior.ROLES, strict=True)])
    excluded = encoder.encode([f"{name}: {guide['definitions'][name]}" for name in EXCLUDED_PROTOTYPES])

    def scored(group, gold, phrase, case=None):
        predicted, scores, elapsed = score(phrase, encoder, bank, definitions, excluded)
        return group, gold, predicted, scores, case, elapsed

    encoder.encode([query("Warm up")])
    cal_pos = []
    cal_neg = []
    for role in prior.ROLES:
        pred, scores, _ = score(legacy.PHRASES[role][0], encoder, bank, definitions, excluded)
        cal_pos.append((pred, scores, role))
    for phrase in prior.OUTSIDE:
        pred, scores, _ = score(phrase, encoder, bank, definitions, excluded)
        cal_neg.append((pred, scores, None))
    cuts = {method: gate_threshold(cal_pos, cal_neg, method) for method in METHODS}
    atomic_positive = [scored(row["role"], role, row["text"]) for role, row in positives]
    atomic_negative = [scored(row["role"], None, row["text"]) for row in negatives]

    held, authors, _ = transfer.cohort(old, external)
    binding_good, binding_bad = [], []
    for row, group in zip(held, transfer.clusters(authors), strict=True):
        case = clauses.make_case(row, 1)
        if case is None:
            continue
        gold = case["missing"]
        candidates = [item for role, item in positives if role == gold]
        example = min(candidates, key=lambda x: legacy.stable("niso-positive", case["doi"], x["ordinal"]))
        binding_good.append(scored(group, gold, example["text"], case))
        example = min(negatives, key=lambda x: legacy.stable("niso-negative", case["doi"], x["role"], x["ordinal"]))
        unsupported = dict(case, missing=None)
        binding_bad.append(scored(group, None, example["text"], unsupported))

    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), thresholds=cuts,
        calibration={"eligible_phrases": len(cal_pos), "unsupported_phrases": len(cal_neg)},
        examples_sha256=digest(niso_path), capture_sha256=digest(external / "capture.json"),
        code_sha256=digest(Path(__file__)), embedding_revision=provenance["revision"],
        panels={name: evaluate(rows, cuts) for name, rows in (
            ("human_supported", atomic_positive), ("human_unsupported", atomic_negative),
            ("publication_supported", binding_good), ("publication_unsupported", binding_bad))},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "external", "niso_path", "embedding", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
