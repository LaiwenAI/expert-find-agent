"""Transfer one-clause proof repair to a disjoint publisher search catalog."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.engineering_cross_source_panel import SENTENCE, requests
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, Catalog, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_joint_calibration import judge
from experiments.credit.engineering_one_role_recheck import RoleReader, combined, pick, resolve, state
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


PROJECTION = "4ce67b1ff52d869f133bd5318bad2573502d12aae065c75c9407c14754171c89"
SOURCE_MODES = ("anchored", "oracle_sources")
ROLE_MODES = ("ridge_band", "prototype_band", "left", "proof_aware", "proof_aware_gated")


def build(path):
    if digest(path) != PROJECTION:
        raise ValueError("Changed eLife projection")
    articles, owners = {}, defaultdict(set)
    dropped = Counter()
    for row in json.loads(path.read_bytes())["sources"]:
        doi = row["doi"]
        title, abstract = row["text"].split("\n", 1)
        sentences = SENTENCE.split(abstract.replace("\n", " "))
        if not row["declaration_complete"]:
            dropped["incomplete_declaration"] += 1
            continue
        if len(sentences) < 3:
            dropped["short_abstract"] += 1
            continue
        roles = {author["person"] or f"{doi}#{author['ordinal']}": author["roles"] for author in row["authors"]}
        if len(roles) != len(row["authors"]):
            dropped["duplicate_owner"] += 1
            continue
        half = len(sentences) // 2
        articles[doi] = dict(indexed=f"{title}\n" + " ".join(sentences[:half]),
                             clue=" ".join(sentences[half:]), roles=roles)
        for author in row["authors"]:
            if author["person"]:
                owners[author["person"]].add(doi)
    links = [(pair, key) for key, docs in owners.items() for pair in combinations(sorted(docs), 2)]
    cohort = {(row["family"], row["variant"]): row for row in requests(articles, links)}
    rows = [cohort[key] for key in sorted(cohort)]
    if not rows:
        raise ValueError("No eligible eLife scope swaps")
    return articles, rows, dict(
        admitted=len(articles), dropped=dict(dropped), links=len(links),
        paper_pairs=len({tuple(row["papers"]) for row in rows}),
        independent_people=len({key for _, key in links}), variants=dict(Counter(row["variant"] for row in rows)),
    )


def evaluate(articles, rows, encoder, banks, wording, reader, bands, cutoff, reverse):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts, times = defaultdict(Counter), defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        intended = tuple(row["papers"])
        base_clues = [articles[paper]["clue"] for paper in intended]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view in ("excerpt", "keywords"):
            clues = base_clues if view == "excerpt" else [catalog.keywords(clue) for clue in base_clues]
            start = perf_counter()
            prompts = [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
            prompts += [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in text]
            v = encoder.encode(prompts)
            scores = {mode: v[2:] @ banks[mode].T for mode in bands}
            encode_ms = (perf_counter() - start) * 1000
            cache = {}
            for availability, missing in zip(AVAILABILITY, (None, *intended), strict=True):
                start = perf_counter()
                topic = catalog.rank(v[:2], clues, missing)
                retrieval_ms = (perf_counter() - start) * 1000
                for mode in SOURCE_MODES:
                    start = perf_counter()
                    pair, _ = select(catalog, topic, row, [set(), set()], mode, missing)
                    target = pair == intended
                    case = row | {"papers": pair, "gold": row["gold"] if target else []}
                    shared_ms = (perf_counter() - start) * 1000 + encode_ms + (retrieval_ms if mode == "anchored" else 0)
                    start = perf_counter()
                    if pair is None:
                        sets, initial, choices = None, {}, dict.fromkeys(ROLE_MODES)
                    else:
                        sets, initial, branches = state(articles, case, scores["ridge"], bands)
                        choices = pick(branches, scores["ridge"])
                    stage_ms = (perf_counter() - start) * 1000
                    for role_mode in ROLE_MODES:
                        start = perf_counter()
                        actual, calls, tokens, extra_ms, measured_ms = initial, 0, 0, 0., 0.
                        static = role_mode.endswith("_band")
                        if static and pair is not None:
                            required = required_roles(case, scores, role_mode, bands)
                            actual = resolve(articles, case, required)
                        slot = choices.get(role_mode)
                        if slot is not None:
                            if slot not in cache:
                                read_start = perf_counter()
                                result = reader.read(text[slot], sets[slot])
                                measured_ms = (perf_counter() - read_start) * 1000
                                cache[slot] = result, measured_ms
                            (role, margin, tokens), extra_ms = cache[slot]
                            calls = 1
                            if role is not None and margin > cutoff:
                                actual = combined(articles, case, sets, slot, role)
                        outcome = judge(case, actual, 0.)
                        json.dumps(outcome)
                        role_ms = (perf_counter() - start) * 1000
                        if i:
                            key = "|".join((view, availability, mode, role_mode))
                            counts[key].update(outcome, requests=1, source_pair_correct=target,
                                               substitutions=outcome["issued"] and not target,
                                               swap_issued=row["variant"] == "swap" and outcome["issued"],
                                               calls=calls, tokens=tokens)
                            # Cache replay still charges the selected model call.
                            # A fresh call was already timed inside this loop.
                            times[key].append(shared_ms + (0. if static else stage_ms) + extra_ms + role_ms - measured_ms)
    return {key: dict(totals=val, mean_ms=float(np.mean(times[key])),
                      p95_ms=float(np.quantile(times[key], .95))) for key, val in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, rows, panel = build(args.projection)
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    reader = RoleReader(args.reader, definitions)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    results = evaluate(articles, rows, encoder, banks, wording, reader, bands, cutoff, args.reverse)
    write_json(args.output, dict(
        projection_sha256=PROJECTION, script_sha256=digest(Path(__file__)), panel=panel,
        embedding_revision=checkpoint["revision"], reader_revision=reader.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, bands=bands, cutoff=cutoff,
        prior_bands_sha256=digest(args.bands), cutoff_source_sha256=digest(args.frozen), results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
