"""Choose a role before binding its original publisher clause to a person."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import anchored_clause_rescue as base
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ROLES = tuple(legacy.LABELS.values())
DESCRIPTIONS = tuple(legacy.LABELS)
TASK = "Identify the research contribution activity described by the request."
ARMS = ("scope_reranker", "embedding_cosine", "embedding_margin")
PARAPHRASES = {
    "design": "Helped decide the study setup and plan the experimental protocol.",
    "execution": "Was involved in doing the hands-on research procedures.",
    "analysis": "Interpreted the measurements and examined the observations.",
    "writing": "Contributed to drafting the paper that describes the study.",
    "resources": "Supplied research inputs or the tools the study depended on.",
}


OUTSIDE = (
    "Reconciled travel reimbursement receipts and expense claims.",
    "Processed the team's journal subscription renewals.",
    "Organized the conference venue and booked accommodation.",
    "Managed permission requests to reproduce third-party figures.",
    "Maintained a mailing list for announcements about the study.",
)


def route(case, mode, encoder, bank):
    start = perf_counter()
    vector = encoder.encode([f"Instruct: {TASK}\nQuery: {case['phrase']}"])[0]
    scores = bank @ vector
    order = np.argsort(-scores, kind="stable")[:2]
    role = ROLES[order[0]]
    score = float(scores[order[0]] if mode == "embedding_cosine" else scores[order[0]] - scores[order[1]])
    citations = []
    for owner in case["owners"]:
        match = next((idx for idx, (observed, scope) in enumerate(zip(case["roles"], case["scope"], strict=True))
                      if observed == role and owner in scope), None)
        citations.append(None if match is None else (score, role == case["missing"]))
    return dict(outputs=citations, ms=(perf_counter() - start) * 1000, role_correct=role == case["missing"])


def collect(rows, groups, version, reader, encoder, bank):
    observations = []
    for i, (row, group) in enumerate(zip(rows, groups, strict=True)):
        case = base.make_case(row, min(version, 1))
        if case is None:
            continue
        if version == 2:
            case["phrase"] = PARAPHRASES[case["missing"]]
        elif version == 3:
            case["phrase"] = min(OUTSIDE, key=lambda phrase: legacy.stable("outside", case["doi"], phrase))
            case["missing"] = "outside_inventory"
        rotation = i % len(ARMS)
        result = {}
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            result[arm] = (base.propose(reader, case, "top2_scope") if arm == "scope_reranker"
                           else route(case, arm, encoder, bank))
        observations.append((group, result))
    return observations


def summarize(rows, calibration):
    outcomes = {}
    signals = {}
    for arm in ARMS:
        cutoff = base.calibrate([(group, result[arm]) for group, result in calibration])
        observed = [result[arm] for _, result in rows]
        true, wrong, clean = [], [], []
        for source in observed:
            accepted = [cell for cell in source["outputs"] if cell is not None and cell[0] > cutoff]
            valid = any(cell[1] for cell in accepted)
            invalid = any(not cell[1] for cell in accepted)
            true.append(int(valid))
            wrong.append(int(invalid))
            clean.append(int(valid and not invalid))
        signals[arm] = clean
        outcomes[arm] = dict(
            threshold=cutoff, valid=sum(true), wrong_papers=sum(wrong), clean=sum(clean),
            p95_ms=float(np.quantile([row["ms"] for row in observed], .95)),
        )
        if "role_correct" in observed[0]:
            outcomes[arm]["unfiltered_role_matches"] = sum(row["role_correct"] for row in observed)
    for arm, result in outcomes.items():
        result["clean_delta_vs_scoped"] = clustered_paired_interval(
            [current - control for current, control in zip(signals[arm], signals["scope_reranker"], strict=True)],
            [group for group, _ in rows],
        )
    return dict(papers=len(rows), arms=outcomes)


def run(old, external, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    embed_provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    bank = encoder.encode(list(DESCRIPTIONS))
    reader.score_pairs([("Warm up", "A contribution clause")] * 4, task=base.TOPICAL)
    encoder.encode([f"Instruct: {TASK}\nQuery: warm up"])
    pool = legacy.prepare(old)
    cal = pool["calibration"][0]
    dev = pool["test"][0]
    held, authors, _ = transfer.cohort(old, external)
    groups = transfer.clusters(authors)
    calibration = collect(cal, range(len(cal)), 0, reader, encoder, bank)
    panels = {}
    for name, rows, keys, version in (
        ("test", dev, range(len(dev)), 1),
        ("transfer", held, groups, 1),
        ("transfer_new_paraphrase", held, groups, 2),
        ("transfer_outside_inventory", held, groups, 3),
    ):
        panels[name] = summarize(collect(rows, keys, version, reader, encoder, bank), calibration)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), calibration=len(calibration),
        old_capture_sha256=digest(old / "capture.json"), external_capture_sha256=digest(external / "capture.json"),
        code_sha256=digest(Path(__file__)), embedding_revision=embed_provenance["revision"],
        reranker_revision=reader.provenance["revision"], panels=panels,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "external", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
