"""Factor author scope and role semantics on same candidate-window pairs."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import anchored_clause_rescue as anchored
from experiments.credit import anchored_role_margin as wording
from experiments.credit import full_article_role_windows as base
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit.compile_legacy_notes import compile_note
from experiments.followir.core17_reranking import digest, write_json

SELECTORS = ("owner_postings", "section_lexical")
ARMS = ("joint", "scoped_joint", "scoped_role", "scoped_head")
ROLE_TASK = ("Does the supplied contribution describe the requested activity? "
             "A different activity or shared topic is insufficient.")
NO_SCORE = -1e9


def local_scope(text, author):
    spans = compile_note(text)
    if len(spans) == 1 and spans[0].start == 0 and spans[0].end == len(text) and author in spans[0].owners:
        return spans[0].action
    return None


def observe(inputs, reader, outside=False):
    results = {selector: [] for selector in SELECTORS}
    for j, (article, phrase, role, authors, cluster) in enumerate(inputs):
        if outside:
            phrase = wording.OUTSIDE[j % len(wording.OUTSIDE)]
            role = None
        for selector in SELECTORS[j % 2:] + SELECTORS[:j % 2]:
            selections = base.rank(article, phrase, authors, selector)
            windows = [article.text[i] for i in selections]
            oracle = [article.original.get(i) for i in selections]
            valid = [role is not None and label is not None and label[0] == role and author in label[1]
                     for author, label in zip(authors, oracle, strict=True)]
            scoped = [local_scope(text, author) is not None for author, text in zip(authors, windows, strict=True)]
            rows = {}
            offset = j % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                started = perf_counter()
                heads = [local_scope(text, author) for author, text in zip(authors, windows, strict=True)] if arm != "joint" else []
                inputs_for_reader, positions = [], []
                for i, (author, text) in enumerate(zip(authors, windows, strict=True)):
                    if arm != "joint" and heads[i] is None:
                        continue
                    if arm in ("joint", "scoped_joint"):
                        query = f"Author initials: {author}. Requested activity: {phrase}"
                        payload = text
                    else:
                        query = phrase
                        payload = heads[i] if arm == "scoped_head" else text
                    positions.append(i)
                    inputs_for_reader.append((query, payload))
                instruction = anchored.PERSONAL if arm in ("joint", "scoped_joint") else ROLE_TASK
                scores, clipped = reader.score_pairs(inputs_for_reader, task=instruction) if positions else ([], 0)
                if clipped or not np.isfinite(scores).all():
                    raise ValueError("Invalid verifier outputs")
                margins = [NO_SCORE] * len(authors)
                for i, value in zip(positions, scores, strict=True):
                    margins[i] = value
                rows[arm] = dict(
                    ids=selections, scores=margins, good=valid, pairs=len(positions), scoped=scoped,
                    ownership=tuple(None if label is None else author in label[1]
                                    for author, label in zip(authors, oracle, strict=True)),
                    ms=(perf_counter() - started) * 1000,
                )
            results[selector].append((cluster, rows))
    return results


def summarize(rows, thresholds, frozen_joint_threshold):
    summaries = {}
    clean_by_arm = {}
    groups = [cluster for cluster, _ in rows]
    for arm in (*ARMS, "scoped_joint_frozen"):
        source_arm = "scoped_joint" if arm == "scoped_joint_frozen" else arm
        cut = frozen_joint_threshold if arm == "scoped_joint_frozen" else thresholds[arm]
        records = [record[source_arm] for _, record in rows]
        flags = [[score != NO_SCORE and score > cut for score in row["scores"]] for row in records]
        supported = [is_on[0] and row["good"][0] for is_on, row in zip(flags, records, strict=True)]
        adverse = [any(is_on and not good for is_on, good in zip(on, row["good"], strict=True))
                   for on, row in zip(flags, records, strict=True)]
        clean_by_arm[arm] = [int(good and not bad) for good, bad in zip(supported, adverse, strict=True)]
        summaries[arm] = dict(
            queries=len(records), nominated_correct=sum(row["good"][0] for row in records),
            scope_preserved_correct=sum(row["scoped"][0] and row["good"][0] for row in records),
            accepted_correct=sum(supported), adverse_papers=sum(adverse),
            accepted_partial_author=sum(on[1] for on in flags),
            accepted_wrong_owner=sum(allowed and owner is False
                                     for on, row in zip(flags, records, strict=True)
                                     for allowed, owner in zip(on, row["ownership"], strict=True)),
            accepted_wrong_role_same_owner=sum(allowed and owner is True and not good
                                               for on, row in zip(flags, records, strict=True)
                                               for allowed, owner, good in zip(on, row["ownership"], row["good"], strict=True)),
            accepted_outside_reference=sum(allowed and owner is None
                                           for on, row in zip(flags, records, strict=True)
                                           for allowed, owner in zip(on, row["ownership"], strict=True)),
            clean_papers=sum(clean_by_arm[arm]), reader_pairs=sum(row["pairs"] for row in records),
            verifier_p95_ms=float(np.quantile([row["ms"] for row in records], .95)),
        )
    for arm, clean in clean_by_arm.items():
        summaries[arm]["clean_delta_vs_joint"] = clustered_paired_interval(
            [a - b for a, b in zip(clean, clean_by_arm["joint"], strict=True)], groups,
        )
    return summaries


def run(old, external, checkpoint, output):
    import torch
    from experiments.followir.core17_verifier import Reranker

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    previous = json.loads((old / "capture.json").read_bytes())
    newer = json.loads((external / "capture.json").read_bytes())
    panels = legacy.prepare(old)
    later, names, _ = transfer.cohort(old, external)
    sources = {
        split: list(base.cases(rows, range(len(rows)), old / split, previous[split]["checksums"], variant))
        for (split, (rows, _)), variant in zip(panels.items(), (0, 1), strict=True)
    }
    sources["transfer"] = list(base.cases(later, transfer.clusters(names), external, newer["checksums"], 1))
    reader = Reranker(checkpoint)
    observe(sources["calibration"][:3], reader)
    calibration = observe(sources["calibration"], reader)
    thresholds = {selector: {arm: base.threshold(rows, arm) for arm in ARMS}
                  for selector, rows in calibration.items()}

    def panel(observations):
        return {selector: summarize(rows, thresholds[selector], thresholds[selector]["joint"])
                for selector, rows in observations.items()}

    results = {split: panel(observe(sources[split], reader)) for split in ("test", "transfer")}
    results["unsupported_transfer"] = panel(observe(sources["transfer"], reader, outside=True))
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=reader.provenance["revision"], code_sha256=digest(Path(__file__)),
        selector_sha256=digest(Path(base.__file__)), original_capture_sha256=digest(old / "capture.json"),
        transfer_capture_sha256=digest(external / "capture.json"),
        thresholds=thresholds, calibration=panel(calibration), panels=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("old", "external", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
