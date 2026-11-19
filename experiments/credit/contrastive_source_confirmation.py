"""Confirm an exact study by testing its textual clues against retrieved alternatives."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit import legacy_body_transfer
from experiments.credit import open_person_postings as base
from experiments.credit.legacy_body_confirmation import select_windows
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


REFERENCE = "bb7b2ad74465c308b3612cc3f3eae1cf1a1832ac253474475b117a7a6468764f"
ARMS = ("fixed_body_top4", "body_peak", "body_gap", "distinctive_gap", "deferred_distinctive")
WINDOW_ARMS = ARMS[1:4]


def window_scores(index, bodies, clue, shortlist):
    query = set(tokens(clue))
    weights = {word: index.idf.get(word, 1.) for word in sorted(query)}
    total = sum(weights.values()) or 1.
    peaks = []
    evidence = []
    for doi in shortlist:
        peak, windows = select_windows(clue, bodies[doi], index.idf)
        peaks.append(peak or 0.)
        evidence.append(query.intersection(tokens(windows[0])) if windows else set())
    collisions = Counter(word for witness in evidence for word in witness)
    distinctive = [sum(weights[word] / collisions[word] for word in sorted(witness)) / total
                   for witness in evidence]
    return np.array(peaks), np.array(distinctive)


def best_score(shortlist, values, use_gap):
    order = np.argsort(-values, kind="stable")
    best = int(order[0])
    score = float(values[best])
    if use_gap:
        score *= float(values[best] - values[order[1]])
    return shortlist[best], score


def dense_gap(shortlist, confidence):
    return confidence[shortlist[0]] - max(confidence[doi] for doi in shortlist[1:])


def calibrate(index, bodies, development, form, dense_threshold):
    bad = defaultdict(list)
    correct = Counter()
    dense_errors = []
    for doi in development:
        clue = index.clue(doi, form)
        for missing in (False, True):
            goal = None if missing else doi
            ranking, confidence, _ = base.candidates(index, clue, doi if missing else None)
            shortlist = ranking[:4]
            peak, distinctive = window_scores(index, bodies, clue, shortlist)
            for arm, values in (("body_peak", peak), ("body_gap", peak), ("distinctive_gap", distinctive)):
                proposed, score = best_score(shortlist, values, arm != "body_peak")
                if proposed != goal:
                    bad[arm].append(score)
                else:
                    correct[arm] += 1
            if shortlist[0] != goal and confidence[shortlist[0]] > dense_threshold:
                dense_errors.append(dense_gap(shortlist, confidence))
    cuts = {arm: max(bad[arm]) for arm in WINDOW_ARMS}
    cuts["dense_gap"] = max([0., *dense_errors])
    return cuts, dict(correct_before_cut=dict(correct), wrong_before_cut={k: len(v) for k, v in bad.items()})


def reference(index, bodies, clue, shortlist, confidence, dense_threshold, body_threshold):
    checks = 0
    for doi in shortlist:
        if confidence[doi] > dense_threshold:
            return doi, checks
        checks += 1
        if base.body_score(index, bodies, clue, doi) > body_threshold:
            return doi, checks
    return None, checks


def route(arm, index, bodies, clue, shortlist, confidence, fixed, cuts):
    if arm == "fixed_body_top4":
        return reference(index, bodies, clue, shortlist, confidence, *fixed)
    if (arm == "deferred_distinctive" and confidence[shortlist[0]] > fixed[0]
            and dense_gap(shortlist, confidence) > cuts["dense_gap"]):
        return shortlist[0], 0
    peak, distinctive = window_scores(index, bodies, clue, shortlist)
    policy = "distinctive_gap" if arm == "deferred_distinctive" else arm
    proposed, score = best_score(shortlist, distinctive if policy == "distinctive_gap" else peak,
                                 policy != "body_peak")
    return (proposed if score > cuts[policy] else None), len(shortlist)


def evaluate(index, bodies, tests, groups, form, fixed, cuts, modern):
    rows = {state: [] for state in base.STATES}
    coverage = Counter()
    for i, (doi, group) in enumerate(zip(tests, groups, strict=True)):
        clue = index.clue(doi, form)
        duties = base.select_pairs(doi, index.notes[doi].roles, index.notes[doi].people) if modern else {}
        for state in base.STATES:
            goal = doi if state == "present" else None
            ranked, confidence, search_ms = base.candidates(index, clue, None if goal else doi)
            shortlist = ranked[:4]
            coverage[state] += int(goal in shortlist)
            results = {}
            shift = i % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                start = perf_counter()
                selected, reads = route(arm, index, bodies, clue, shortlist, confidence, fixed, cuts)
                latency = search_ms + (perf_counter() - start) * 1000
                correct = selected is not None and selected == goal
                wrong = selected is not None and selected != goal
                row = dict(correct_source=int(correct), wrong_source=int(wrong), body_checks=reads, ms=latency)
                for kind, pair in duties.items():
                    owners = index.notes[selected].people.get(pair, set()) if selected is not None else set()
                    row[f"{kind}_clean"] = int(correct and bool(owners))
                    row[f"{kind}_wrong"] = int(wrong and bool(owners))
                results[arm] = row
            rows[state].append((group, results))

    panels = {}
    for state, records in rows.items():
        keys = {key for _, values in records for key in values[ARMS[0]] if key != "ms"}
        methods = {}
        for arm in ARMS:
            observed = [values[arm] for _, values in records]
            totals = {key: sum(row.get(key, 0) for row in observed) for key in sorted(keys)}
            paired = {
                key: clustered_paired_interval(
                    [r[arm][key] - r["fixed_body_top4"][key] for _, r in records], [group for group, _ in records],
                ) for key in ("correct_source", "wrong_source")
            }
            methods[arm] = dict(totals, component_p95_ms=float(np.quantile([r["ms"] for r in observed], .95)),
                                paired_vs_fixed=paired)
        panels[state] = dict(requests=len(records), exact_in_top4=coverage[state], arms=methods)
    return panels


def run(prior, external, legacy_old, legacy_first, legacy_fresh, embedding, previous, output, folders):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    if digest(previous) != REFERENCE:
        raise ValueError("Changed fixed body reference")
    provenance = verify_checkpoint(embedding)
    baseline = json.loads(previous.read_bytes())
    if provenance["revision"] != baseline["embedding_revision"]:
        raise ValueError("Changed embedding checkpoint")
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    notes, bodies, development, tests = base.load(prior, external, folders)
    old_notes, old_bodies, old_rows, old_groups, _ = legacy_body_transfer.prepare(legacy_old, legacy_first, legacy_fresh)
    if set(notes) & {doi for doi, _ in old_rows}:
        raise ValueError("External source transfer overlaps modern catalog")
    primary = base.Search(notes, encoder)
    transfer = base.Search(old_notes, encoder)
    results = {}
    for form in base.FORMS:
        panel = baseline["panels"][form]
        fixed = (panel["dense_cutoff"], panel["body_cutoff"])
        cuts, calibration = calibrate(primary, bodies, development, form, fixed[0])
        main = evaluate(primary, bodies, tests, tests, form, fixed, cuts, True)
        for state, outcomes in main.items():
            for kind in ("supported", "split_between_authors"):
                expected = panel["cases"][f"{state}/{kind}"]["arms"]["body_top4"]
                actual = outcomes["arms"]["fixed_body_top4"]
                if (actual[f"{kind}_clean"], actual[f"{kind}_wrong"]) != (expected["clean"], expected["wrong_source"]):
                    raise AssertionError(f"Fixed body replay diverged for {form}/{state}/{kind}")
        results[form] = dict(
            thresholds=cuts, source_thresholds=fixed, calibration=calibration, modern=main,
            legacy=evaluate(transfer, old_bodies, [doi for doi, _ in old_rows], old_groups, form, fixed, cuts, False),
        )
    write_json(output, dict(
        code_sha256=digest(Path(__file__)), baseline_sha256=digest(previous),
        modern_catalog=len(notes), legacy_catalog=len(old_notes), calibration=len(development),
        embedding_revision=provenance["revision"], job=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prior", "external", "legacy_old", "legacy_first", "legacy_fresh", "embedding", "previous", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    run(**vars(parser.parse_args()))
