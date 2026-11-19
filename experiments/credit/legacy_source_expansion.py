"""Measure whether a bounded follow-up can recover a different publication."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit import legacy_body_confirmation as body
from experiments.credit import legacy_body_transfer as transfer
from experiments.credit import legacy_open_attribution as base


REFERENCE = "65d399aed4fc821fcd070c7c939f687a63736a55bd142c01900a7ac00c826f9a"
DEPTH = 4
ARMS = ("top_one", "loose_four", "strict_two", "strict_four", "flat_four", "relative_four")


def rank(index, clue, withheld):
    vector = index.encoder.encode([f"Instruct: {base.TASK}\nQuery: {clue}"])[0]
    dense = index.vectors @ vector
    missing = index.pos.get(withheld)
    order = [i for i in np.argsort(-dense, kind="stable") if i != missing]
    terms = set(tokens(clue))
    overlap = [sum(index.idf[w] for w in sorted(row & terms)) for row in index.index]
    lexical = sorted((i for i in order if overlap[i] > 0), key=lambda i: (-overlap[i], i))
    scores = {i: 1 / (60 + rank) for rank, i in enumerate(order, 1)}
    for position, i in enumerate(lexical, 1):
        scores[i] += 1 / (60 + position)
    shortlist = sorted(scores, key=scores.get, reverse=True)[:DEPTH]
    med = float(np.median(dense[order[:DEPTH]]))
    return [(index.keys[i], float(dense[i]), float(dense[i]) - med) for i in shortlist]


def inspect(index, bodies, doi, form, withheld, dense_cut, body_cut):
    clue = index.clue(doi, form)
    start = perf_counter()
    candidates = rank(index, clue, doi if withheld else None)
    rank_ms = 1e3 * (perf_counter() - start)
    first, dense, _ = candidates[0]
    if dense > dense_cut:
        return candidates, first, [], rank_ms
    scores = []
    for candidate, _, _ in candidates:
        start = perf_counter()
        value, _ = body.select_windows(clue, bodies[candidate], index.idf)
        scores.append((value if value is not None else -1.0, 1e3 * (perf_counter() - start)))
        if len(scores) == 1 and scores[0][0] > body_cut:
            return candidates, first, scores, rank_ms
    return candidates, None, scores, rank_ms


def calibrate(index, bodies, questions, form, dense_cut, body_cut):
    wrong_body, wrong_relative = [], []
    for doi in questions:
        for withheld in (False, True):
            candidates, fixed, scores, _ = inspect(index, bodies, doi, form, withheld, dense_cut, body_cut)
            if fixed is not None:
                continue
            for (candidate, _, relative), (coverage, _) in zip(candidates[1:], scores[1:], strict=True):
                if withheld or candidate != doi:
                    wrong_body.append(coverage)
                    wrong_relative.append(relative)
    return dict(body=max(wrong_body), relative=max(wrong_relative), incorrect_proposals=len(wrong_body))


def select(arm, candidates, fixed, scores, cutoffs, body_cut):
    costs = [ms for _, ms in scores]
    first_cost = sum(costs[:1])
    if fixed is not None or arm == "top_one":
        return fixed, len(costs[:1]), first_cost
    if arm == "relative_four":
        chosen = next((cid for cid, _, relative in candidates[1:] if relative > cutoffs["relative"]), None)
        return chosen, 1, first_cost
    cutoff = body_cut if arm == "loose_four" else cutoffs["body"]
    limit = 2 if arm == "strict_two" else DEPTH
    if arm == "flat_four":
        strongest = max(range(1, DEPTH), key=lambda i: scores[i][0])
        return (candidates[strongest][0] if scores[strongest][0] > cutoff else None), DEPTH, sum(costs)
    for i in range(1, limit):
        if scores[i][0] > cutoff:
            return candidates[i][0], i + 1, sum(costs[:i+1])
    return None, limit, sum(costs[:limit])


def evaluate(index, bodies, rows, groups, form, withheld, dense_cut, body_cut, cutoffs, engine, gate, author_cut):
    outputs = {arm: [] for arm in ARMS}
    oracle = Counter()
    for doi, tasks in rows:
        shortlist, fixed, scores, rank_ms = inspect(index, bodies, doi, form, withheld, dense_cut, body_cut)
        keys = [cid for cid, _, _ in shortlist]
        if not withheld:
            oracle.update({
                "top_one": int(doi == keys[0]), "top_two": int(doi in keys[:2]), "top_four": int(doi in keys),
                "reachable_on_abstention": int(fixed is None and doi in keys[1:]),
                "wrong_locked_first": int(fixed is not None and fixed != doi and doi in keys[1:]),
            })
        decisions = {a: select(a, shortlist, fixed, scores, cutoffs, body_cut) for a in ARMS}
        verified = {}
        for cid in sorted({cid for cid, _, _ in decisions.values()} - {None}):
            start = perf_counter()
            authors, calls = base.verify(index.notes[cid], cid, tasks, True, engine, gate, author_cut)
            verified[cid] = set(authors), calls, 1e3 * (perf_counter() - start)
        expected = set.intersection(*(index.notes[doi].memberships[t] for t in tasks))
        for arm, (cid, reads, ms) in decisions.items():
            actual, calls, author_ms = verified.get(cid, (set(), 0, 0.0))
            correct = not withheld and cid == doi
            allowed = expected if correct else set()
            outputs[arm].append(dict(
                correct_source=int(correct), wrong_source=int(cid is not None and not correct),
                supported=int(bool(actual & allowed)), clean_support=int(bool(actual) and actual <= allowed),
                wrong_author_on_correct_source=int(correct and bool(actual - allowed)),
                wrong_source_recommendation=int(bool(actual) and not correct),
                author_calls=calls, body_reads=reads, ms=rank_ms + ms + author_ms,
            ))
    results = {}
    for arm, items in outputs.items():
        totals = Counter()
        for item in items:
            totals.update({k: v for k, v in item.items() if k != "ms"})
        intervals = {}
        for metric in ("clean_support", "wrong_source_recommendation"):
            delta = [a[metric] - b[metric] for a, b in zip(items, outputs["top_one"], strict=True)]
            intervals[metric] = clustered_paired_interval(delta, groups)["percentile_95"]
        results[arm] = dict(totals, factored_p95_ms=float(np.quantile([item["ms"] for item in items], .95)),
                            paired_95_interval=intervals)
    return dict(results=results, oracle=dict(oracle))


def run(old, first, fresh, embedding, reader, open_policy, body_policy, author_policy, reference, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh CUDA Slurm run")
    for path, expected in ((open_policy, body.REFERENCE), (body_policy, transfer.BODY_RESULTS),
                           (author_policy, base.transfer.REFERENCE), (reference, REFERENCE)):
        if base.indexed.digest(path) != expected:
            raise ValueError("Unexpected frozen input")
    opened, lexical, frozen, prior = [json.loads(p.read_bytes()) for p in
                                      (open_policy, body_policy, author_policy, reference)]
    if base.indexed.digest(fresh / "capture.json") != prior["capture_sha256"]:
        raise ValueError("Unexpected transfer capture")
    notes, bodies, questions, groups, _ = transfer.prepare(old, first, fresh)
    calibration = [doi for doi, *_ in base.legacy.prepare(old)["calibration"][0]]
    encoder = base.indexed.QwenTorchEmbedding(embedding, batch_size=12)
    encoder.model.config.use_cache = False
    index = base.Search(notes, encoder)
    tokenizer, model, reader_meta = base.indexed.load_reader(reader)
    if (base.indexed.verify_checkpoint(embedding)["revision"] != frozen["embedding_revision"]
            or reader_meta["revision"] != frozen["reader_revision"]):
        raise ValueError("Changed embedding or verification model")
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("1", "0")]
    if any(len(i) != 1 for i in ids) or ids[0] == ids[1]:
        raise ValueError("Unexpected verifier outputs")
    engine = model, tokenizer, [i[0] for i in ids]
    gate = base.transfer.load_gate(frozen["routes"]["character"]["selected"]["gate"], base.cascade.Gate)
    warm, duties = questions[0]
    base.cascade.reader_pair((warm, notes[warm].note, duties, *notes[warm].authors[:2]), 1, engine)
    report, thresholds = {}, {}
    for form in base.FORMS:
        dense_cut = opened["thresholds"][form]
        body_cut = lexical["thresholds"][form]["lexical_body"]["threshold"]
        thresholds[form] = calibrate(index, bodies, calibration, form, dense_cut, body_cut)
        for state in ("present", "withheld"):
            key = f"{form}/{state}"
            view = evaluate(index, bodies, questions, groups, form, state == "withheld",
                            dense_cut, body_cut, thresholds[form], engine, gate, frozen["reader_cutoff"])
            control = prior["report"][key]["results"]["lexical_body"]
            if any(view["results"]["top_one"][metric] != control[metric] for metric in
                   ("correct_source", "wrong_source", "supported", "clean_support",
                    "wrong_source_recommendation", "wrong_author_on_correct_source", "author_calls")):
                raise ValueError("Top-one control changed from the pinned test")
            report[key] = view
    base.indexed.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), reference_sha256=REFERENCE,
        catalog=len(notes), test_queries=len(questions), clusters=len(set(groups)),
        calibration_queries=len(calibration), thresholds=thresholds, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "first", "fresh", "embedding", "reader", "open_policy", "body_policy",
                 "author_policy", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
