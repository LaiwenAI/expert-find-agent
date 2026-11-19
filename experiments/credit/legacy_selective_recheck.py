"""Reuse a paid attribution verdict and schedule a bounded number of rechecks."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit import legacy_body_confirmation as body
from experiments.credit import legacy_body_transfer as transfer
from experiments.credit import legacy_open_attribution as base
from experiments.credit import legacy_source_expansion as expansion


REFERENCE = "f055fd1809aaee4f677c5412ced8476020ada88b521a32f05cbb3110b41ff766"
ARMS = ("character", "threshold_only", "reuse_original", "reuse", "first_one", "margin_one", "first_two", "margin_two", "full")


def record(article, doi, tasks, gate, engine):
    blocks = []
    for start in range(0, len(article.authors), 2):
        owners = article.authors[start:start+2]
        row = (doi, article.note, tasks, owners[0], owners[-1])
        begin = perf_counter()
        feature = base.cascade.score_pair(row, article.entry, 1, None, gate.features)
        automatic, pending = base.cascade.first_stage(feature, gate)
        fast_ms = (perf_counter() - begin) * 1000
        begin = perf_counter()
        verdict = base.cascade.reader_pair(row, 1, engine)
        reader_ms = (perf_counter() - begin) * 1000
        n = len(owners)
        blocks.append((owners, automatic[:n], pending[:n], feature[:n, 0],
                       verdict[:n], fast_ms, reader_ms))
    return blocks


def choose_checks(blocks, budget, margin_first):
    eligible = [i for i, (_, auto, pending, _, _, _, _) in enumerate(blocks) if auto.any() and not pending.any()]
    if margin_first:
        eligible.sort(key=lambda i: float(blocks[i][3][blocks[i][1]].min()))
    return eligible[:budget]


def decide(blocks, arm, original_cut, revised_cut):
    if arm.startswith(("first_", "margin_")):
        budget = {"one": 1, "two": 2}[arm.split("_")[1]]
        additional = set(choose_checks(blocks, budget, arm.startswith("margin")))
    else:
        additional = set()
    answers, elapsed, calls = set(), 0., 0
    for i, (owners, auto, pending, _, verdict, fast, reader) in enumerate(blocks):
        paid = pending.any()
        if arm == "full":
            accepted = verdict > revised_cut
            elapsed += reader
            calls += 1
        else:
            elapsed += fast
            if paid or i in additional:
                elapsed += reader
                calls += 1
            if arm in ("character", "threshold_only"):
                threshold = original_cut if arm == "character" else revised_cut
                accepted = auto | (pending & (verdict > threshold))
            elif arm == "reuse_original" and paid:
                accepted = (auto | pending) & (verdict > original_cut)
            elif paid or i in additional:
                accepted = (auto | pending) & (verdict > revised_cut)
            else:
                accepted = auto
        answers.update(a for a, keep in zip(owners, accepted, strict=True) if keep)
    return answers, elapsed, calls


def evaluate(index, bodies, questions, groups, form, view, dense_cut, body_cut,
             gate, engine, cutoff, stricter, cache):
    withheld = view == "withheld"
    all_results = {arm: [] for arm in ARMS}
    oracle = Counter()
    for doi, tasks in questions:
        shortlist, picked, scores, rank_ms = expansion.inspect(index, bodies, doi, form, withheld, dense_cut, body_cut)
        source, _, source_ms = expansion.select("top_one", shortlist, picked, scores, {}, body_cut)
        blocks = []
        if source is not None:
            if (source, tasks) not in cache:
                cache[source, tasks] = record(index.notes[source], source, tasks, gate, engine)
            blocks = cache[source, tasks]
        correct = source is not None and not withheld and source == doi
        truth = set.intersection(*(index.notes[doi].memberships[t] for t in tasks)) if correct else set()
        for arm in ARMS:
            selected, elapsed, calls = decide(blocks, arm, cutoff, stricter)
            all_results[arm].append(dict(
                clean=int(bool(selected) and selected <= truth),
                supported=int(bool(selected & truth)),
                wrong_author_on_correct_source=int(correct and bool(selected - truth)),
                wrong_source_recommendation=int(bool(selected) and not correct),
                calls=calls, ms=rank_ms + source_ms + elapsed,
            ))
        if blocks:
            before = decide(blocks, "reuse", cutoff, stricter)[0]
            possible = [before]
            for i in choose_checks(blocks, len(blocks), False):
                owners, auto, _, _, verdict, _, _ = blocks[i]
                repaired = (before - set(owners)) | {a for a, keep in zip(
                    owners, auto & (verdict > stricter), strict=True) if keep}
                possible.append(repaired)
            was_clean = bool(before) and before <= truth
            oracle["one_block_clean_gain"] += int(not was_clean and any(s and s <= truth for s in possible))
            oracle["one_block_wrong_reduction"] += int(bool(before - truth) and any(s <= truth for s in possible))
            oracle["queries_with_extra_candidates"] += int(len(possible) > 1)
            oracle["different_one_pair_choices"] += int(
                choose_checks(blocks, 1, True) != choose_checks(blocks, 1, False))
    summary = {}
    metrics = ("clean", "wrong_author_on_correct_source", "wrong_source_recommendation")
    for arm, rows in all_results.items():
        totals = Counter()
        for row in rows:
            totals.update({key: val for key, val in row.items() if key != "ms"})
        control = "first_" + arm.split("_", 1)[1] if arm.startswith("margin_") else "character"
        intervals = {
            metric: clustered_paired_interval([a[metric]-b[metric] for a, b in
                         zip(rows, all_results[control], strict=True)], groups)["percentile_95"]
            for metric in metrics
        }
        summary[arm] = dict(totals, factored_p95_ms=float(np.quantile([r["ms"] for r in rows], .95)),
                            comparison=control, paired_95_interval=intervals)
    return dict(arms=summary, oracle=dict(oracle))


def run(old_capture, first, fresh, embedding, reader, open_policy, body_policy, author_policy, reference, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh CUDA Slurm run")
    for path, expected in ((open_policy, body.REFERENCE), (body_policy, transfer.BODY_RESULTS),
                           (author_policy, base.transfer.REFERENCE), (reference, REFERENCE)):
        if base.indexed.digest(path) != expected:
            raise ValueError("Unexpected frozen input")
    opened, lexical, frozen, prior = [json.loads(p.read_bytes()) for p in
                                      (open_policy, body_policy, author_policy, reference)]
    notes, bodies, questions, groups, _ = transfer.prepare(old_capture, first, fresh)
    encoder = base.indexed.QwenTorchEmbedding(embedding, batch_size=12)
    encoder.model.config.use_cache = False
    index = base.Search(notes, encoder)
    tokenizer, model, meta = base.indexed.load_reader(reader)
    if (base.indexed.verify_checkpoint(embedding)["revision"] != frozen["embedding_revision"]
            or meta["revision"] != frozen["reader_revision"]):
        raise ValueError("Changed embedding or reader checkpoint")
    digits = [tokenizer.encode(x, add_special_tokens=False) for x in ("1", "0")]
    if any(len(d) != 1 for d in digits) or digits[0] == digits[1]:
        raise ValueError("Unexpected verdict encoding")
    engine = model, tokenizer, [x[0] for x in digits]
    gate = base.transfer.load_gate(frozen["routes"]["character"]["selected"]["gate"], base.cascade.Gate)
    doi, tasks = questions[0]
    record(notes[doi], doi, tasks, gate, engine)
    stricter = prior["calibration"]["thresholds"]["full"]
    report, cache = {}, {}
    fields = ("clean", "supported", "wrong_author_on_correct_source", "wrong_source_recommendation", "calls")
    for form in base.FORMS:
        for view in ("present", "withheld"):
            key = f"{form}/{view}"
            report[key] = evaluate(index, bodies, questions, groups, form, view, opened["thresholds"][form],
                                   lexical["thresholds"][form]["lexical_body"]["threshold"], gate, engine,
                                   frozen["reader_cutoff"], stricter, cache)
            for arm in ("character", "full"):
                baseline = prior["report"][key]["top_one"][arm]
                if any(report[key]["arms"][arm][field] != baseline[field] for field in fields):
                    raise ValueError("Changed pinned control decisions")
            for n in ("one", "two"):
                if report[key]["arms"][f"first_{n}"]["calls"] != report[key]["arms"][f"margin_{n}"]["calls"]:
                    raise ValueError("Unmatched model-call budgets")
    base.indexed.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), reference_sha256=REFERENCE,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), queries=len(questions),
        clusters=len(set(groups)), cutoff=stricter, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old_capture", "first", "fresh", "embedding", "reader", "open_policy",
                 "body_policy", "author_policy", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
