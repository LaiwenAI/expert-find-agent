"""Test a source-blind one-option expansion when recheck cannot start."""

import argparse
from collections import Counter, defaultdict
from copy import copy
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.distilled_role_recheck import student_values
from experiments.credit.elife_one_role_transfer import build as development
from experiments.credit.elife_prompt_holdout import build as holdout
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, Catalog, candidates, phrase_bank, select,
)
from experiments.credit.engineering_one_role_recheck import RoleReader, pick, replace, state
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_issuance_audit import outcome
from experiments.credit.role_recheck_opportunity import approvals
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


def expansion(catalog, scope, sets, scores, allowed, width, feasibility_first):
    opportunities = []
    for slot in (0, 1):
        values = student_values("dense_agreement", None, scores, {}, slot)
        order = sorted(range(len(ROLES)), key=lambda i: (-values[i], i))
        top = order[:width]
        alternatives = [idx for idx in top if ROLES[idx] not in allowed.get(slot, ())]
        if not feasibility_first:
            alternatives = alternatives[:1]
        for idx in alternatives:
            role = ROLES[idx]
            if candidates(catalog, scope["papers"], replace(sets, slot, role)):
                opportunities.append((float(values[idx]), -slot, role, {ROLES[i] for i in top}))
                break
    if not opportunities:
        return None
    _, neg_slot, role, options = max(opportunities)
    return -neg_slot, role, options


def timed_read(reader, phrase, options, cache, slot):
    key = slot, frozenset(options)
    if key not in cache:
        start = perf_counter()
        value = reader.read(phrase, options)
        cache[key] = value, (perf_counter() - start) * 1000
    return cache[key]


def evaluate(articles, rows, encoder, banks, wording, reader, bands, cutoff, full_controls, reverse):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts = defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        target_pair = tuple(row["papers"])
        excerpts = [articles[paper]["clue"] for paper in target_pair]
        tasks = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(x) for x in excerpts])):
            cache = {}
            start = perf_counter()
            vectors = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{text}" for text in clues] +
                                     [f"Instruct: {TASK}\nQuery:{task}" for task in tasks])
            scores = {name: vectors[2:] @ banks[name].T for name in bands}
            embed_ms = (perf_counter() - start) * 1000
            available = (None, *target_pair) if full_controls else (None,)
            for status, masked in zip(AVAILABILITY, available):
                start = perf_counter()
                topic = catalog.rank(vectors[:2], clues, masked)
                pair, _ = select(catalog, topic, {}, [set(), set()], "anchored", masked)
                scope = {"papers": pair, "roles": (None, None)}
                sets, initial, branches = [set(), set()], {}, {}
                if pair:
                    sets, initial, branches = state(articles, scope, scores["ridge"], bands)
                approved = approvals(catalog, scope, sets, branches, scores, bands) if branches else {}
                slot = pick(branches, scores["ridge"])["proof_aware_gated"]
                normal = approved.get(slot)
                shared_ms = embed_ms + (perf_counter() - start) * 1000
                baseline_sets, read_ms = sets, 0.
                if normal:
                    (role, margin, _), read_ms = timed_read(reader, tasks[slot], sets[slot], cache, slot)
                    if role in normal and margin > cutoff:
                        baseline_sets = replace(sets, slot, role)
                start = perf_counter()
                baseline, _ = outcome(catalog, row, pair, baseline_sets, "base", None)
                baseline_ms = shared_ms + read_ms + (perf_counter() - start) * 1000
                for policy, width, feasibility_first in (("language_top3", 3, False), ("feasible_top5", 5, True)):
                    novel, oracle, confidence = baseline, False, 0.
                    added_calls, added_tokens, extra_ms = 0, 0, 0.
                    if pair and not initial and not normal:
                        start = perf_counter()
                        proposal = expansion(catalog, scope, sets, scores, approved, width, feasibility_first)
                        extra_ms = (perf_counter() - start) * 1000
                        if proposal:
                            other, new_role, options = proposal
                            (chosen, gap, added_tokens), reader_ms = timed_read(reader, tasks[other], options, cache, other)
                            start = perf_counter()
                            repaired, _ = outcome(catalog, row, pair, replace(sets, other, new_role), "expanded", other)
                            oracle = repaired["proof"]
                            if chosen == new_role:
                                novel, confidence = repaired, gap
                            extra_ms += reader_ms + (perf_counter() - start) * 1000
                            added_calls = 1
                    if i:
                        assert int(bool(normal)) + added_calls <= 1
                        counts[view, status, policy].append(dict(
                            baseline=baseline, novel=novel, confidence=confidence,
                            base_calls=int(bool(normal)), extra_calls=added_calls, extra_tokens=added_tokens,
                            elapsed=baseline_ms, extra_ms=extra_ms, proposed_exact=oracle))
    return counts


def sum_results(rows, cutoff):
    counters = Counter()
    for row in rows:
        approved = cutoff is not None and row["confidence"] > cutoff
        verdict = row["novel"] if approved else row["baseline"]
        counters.update(verdict, proposed_exact=row["proposed_exact"], extra_calls=row["extra_calls"],
                        extra_tokens=row["extra_tokens"], base_calls=row["base_calls"], requests=1)
    return dict(counters)


def fit_margin(development_rows):
    best = None
    for cutoff in [0., *sorted({row["confidence"] for row in development_rows if row["confidence"] > 0}), None]:
        counts = sum_results(development_rows, cutoff)
        if counts["adverse"] > .05 * counts["issued"]:
            continue
        target = counts["proof"], -counts["adverse"], -cutoff if cutoff is not None else -float("inf")
        if best is None or target > best[0]:
            best = target, dict(cutoff=cutoff, counts=counts)
    return None if best is None else best[1]


def summarize(rows, fit):
    results = {}
    for (view, status, policy), decisions in rows.items():
        arms = ((policy, 0.),) if policy == "language_top3" else (
            ("original", None), (policy, 0.), ("calibrated_top5", fit["cutoff"] if fit else None))
        for arm, cutoff in arms:
            active = cutoff is not None
            totals = sum_results(decisions, cutoff)
            if not active:
                totals.update(extra_calls=0, extra_tokens=0)
            times = [row["elapsed"] + (row["extra_ms"] if active else 0.) for row in decisions]
            results["|".join((view, status, arm))] = dict(totals=totals, p95_ms=float(np.quantile(times, .95)))
    return dict(sorted(results.items()))


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    model = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    full = RoleReader(args.reader, definitions)
    names = copy(full)
    names.definitions = ROLES
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    dev, queries_dev, census = development(args.development)
    observations = evaluate(dev, queries_dev, encoder, banks, wording, names, bands, cutoff, False, args.reverse)
    fit = fit_margin([row for key, values in observations.items() if key[2] == "feasible_top5" for row in values])
    articles, cases, held = holdout(args.holdout, json.loads(args.manifest.read_bytes()))
    results = evaluate(articles, cases, encoder, banks, wording, names, bands, cutoff, True, args.reverse)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), dev_sha256=digest(args.development), holdout_sha256=digest(args.holdout),
        embedding_revision=model["revision"], reader_revision=full.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, fit=fit, development_panel=census, held_panel=held,
        development=summarize(observations, fit), results=summarize(results, fit),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "holdout", "manifest", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
