"""Attribute issuance errors before applying stricter source and role guards."""

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
from experiments.credit.distilled_role_recheck import route
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, Catalog, candidates, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_one_role_recheck import RoleReader, replace
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


def outcome(catalog, row, pair, bands, stage, reread):
    people = set(candidates(catalog, pair, bands)) if pair is not None else set()
    source = tuple(row["papers"]) == pair
    expected = set(row["gold"])
    missing = [i for i, role in enumerate(row["roles"]) if role not in bands[i]]
    if not people:
        error = None
    elif not source:
        error = "source"
    elif missing:
        error = "both_roles" if len(missing) > 1 else (
            "reviewed_role" if missing[0] == reread else "unreviewed_role")
    elif people - expected:
        error = "owner"
    else:
        error = None
    return dict(
        issued=bool(people), proof=bool(expected) and source and people == expected and not missing,
        adverse=error is not None, swap_issued=bool(people) and row["variant"] == "swap",
        substitute=bool(people) and not source,
    ), None if error is None else f"{stage}:{error}"


def guard(pair, topic, sets, prototype, stage, reread):
    both_leaders = pair is not None and all(
        paper == side["fused"][0] for paper, side in zip(pair, topic, strict=True))
    # A confirmed clause keeps the independently checked reader decision.
    # Every other clause must support both embedding bands' possibilities.
    union = [a if stage == "reader" and i == reread else a | b
             for i, (a, b) in enumerate(zip(sets, prototype, strict=True))]
    return {
        "name_only": (pair, sets),
        "two_leaders": (pair if both_leaders else None, sets),
        "unreviewed_union": (pair, union),
        "combined": (pair if both_leaders else None, union),
    }


def evaluate(articles, cohort, encoder, banks, wording, readers, bands, cutoff, reverse):
    catalog = Catalog(articles, encoder)
    phrase_table = phrase_bank(wording)
    rows = cohort[::-1] if reverse else cohort
    counts, errors, timings = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    for i, row in enumerate([rows[0], *rows]):
        intended = tuple(row["papers"])
        clues = [articles[paper]["clue"] for paper in intended]
        tasks = [phrase_table[r][int(row["family"][:16], 16) % len(phrase_table[r])] for r in row["roles"]]
        for view, text in (("excerpt", clues), ("keywords", [catalog.keywords(item) for item in clues])):
            cache = {}
            start = perf_counter()
            vectors = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{c}" for c in text] +
                                     [f"Instruct: {TASK}\nQuery:{t}" for t in tasks])
            scores = {name: vectors[2:] @ banks[name].T for name in bands}
            encode_ms = (perf_counter() - start) * 1000
            for availability, removed in zip(AVAILABILITY, (None, *intended), strict=True):
                start = perf_counter()
                topic = catalog.rank(vectors[:2], text, removed)
                pair, _ = select(catalog, topic, {}, [set(), set()], "anchored", removed)
                scope = {"papers": pair, "roles": (None, None)}
                proto = required_roles(scope, scores, "prototype_band", bands)
                search_ms = encode_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                sets, slot, approved = [set(), set()], None, set()
                if pair is not None:
                    sets, _, slot, approved = route(articles, scope, scores, bands)
                gates_ms = (perf_counter() - start) * 1000
                decisions = {"static": (pair, proto, "initial", None, search_ms, 0)}
                for model, reader in readers.items():
                    calls, read_ms, stage, kept = 0, 0., "initial", sets
                    if approved:
                        if (model, slot) not in cache:
                            start = perf_counter()
                            response = reader.read(tasks[slot], sets[slot])
                            cache[model, slot] = response, (perf_counter() - start) * 1000
                        (role, margin, _), read_ms = cache[model, slot]
                        calls = 1
                        if role in approved and margin > cutoff:
                            kept = replace(sets, slot, role)
                            stage = "reader"
                    total_ms = search_ms + gates_ms + read_ms
                    reviewed = slot if stage == "reader" else None
                    if model == "full":
                        decisions[model] = pair, kept, stage, reviewed, total_ms, calls
                    else:
                        start = perf_counter()
                        repaired = guard(pair, topic, kept, proto, stage, reviewed)
                        extra_ms = (perf_counter() - start) * 1000
                        for name, (papers, selected) in repaired.items():
                            decisions[name] = papers, selected, stage, reviewed, total_ms + extra_ms, calls
                for arm, (papers, roles, stage, reviewed, ms, calls) in decisions.items():
                    start = perf_counter()
                    report, category = outcome(catalog, row, papers, roles, stage, reviewed)
                    ms += (perf_counter() - start) * 1000
                    if i:
                        key = "|".join((view, availability, arm))
                        counts[key].update(report, requests=1, calls=calls)
                        if category:
                            errors[key][category] += 1
                        timings[key].append(ms)
    return {key: dict(totals=value, errors=dict(errors[key]), p95_ms=float(np.quantile(timings[key], .95)))
            for key, value in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    ckpt = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    full = RoleReader(args.reader, definitions)
    names = copy(full)
    names.definitions = ROLES
    readers = {"full": full, "name_only": names}
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    articles, cases, census = build(args.projection, json.loads(args.manifest.read_bytes()))
    stats = evaluate(articles, cases, encoder, banks, wording, readers, bands, cutoff, args.reverse)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), manifest_sha256=digest(args.manifest), projection_sha256=digest(args.projection),
        panel=census, embedding_revision=ckpt["revision"], reader_revision=full.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, results=stats,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "manifest", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
