"""Transfer frozen source acceptance to short clues and another publisher."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.independent_source_confirmation import VERIFY, weaker_side
from experiments.credit.keyword_author_rescue import UNGATED, PostingCatalog, at_gate
from experiments.credit.role_issuance_audit import outcome
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("anchor", "ungated", "weakest", "absolute")
VIEWS = ("all_keywords", "leading_four", "trailing_four")


def render(index, raw, view):
    tokens = index.keywords(raw).split()
    if view == "all_keywords":
        return " ".join(tokens)
    return " ".join(tokens[:4] if view == "leading_four" else tokens[-4:])


def verify(reranker, index, text, pair, side):
    inputs = [(text[i], index.articles[pair[i]]["indexed"]) for i in side]
    values, clipped = reranker.score_pairs(inputs, task=VERIFY)
    if clipped:
        raise ValueError("Source comparison unexpectedly clipped")
    return min(values)


def evaluate(articles, rows, encoder, banks, wording, bands, reranker, cutoffs, reverse):
    index = PostingCatalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts, times, unique = defaultdict(Counter), defaultdict(list), defaultdict(set)
    if reverse:
        rows = rows[::-1]
    for j, row in enumerate([rows[0], *rows]):
        target = tuple(row["papers"])
        text = [articles[key]["clue"] for key in target]
        roles = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view in VIEWS:
            clues = [render(index, phrase, view) for phrase in text]
            if any(not clue for clue in clues):
                raise ValueError("Expected at least one indexed keyword for each clue")
            start = perf_counter()
            vector = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                    [f"Instruct: {TASK}\nQuery:{role}" for role in roles])
            scores = {"prototype": vector[2:] @ banks["prototype"].T}
            role_sets = required_roles(row, scores, "prototype_band", bands)
            known_role_sets = [{role} for role in row["roles"]]
            embedding_ms = (perf_counter() - start) * 1000
            for availability, hidden in zip(AVAILABILITY, (None, *target), strict=True):
                start = perf_counter()
                ranked = index.full_rank(vector[:2], clues, hidden)
                initial, _ = select(index, [s | {"fused": s["fused"][:6]} for s in ranked],
                                    {}, [set(), set()], "anchored", hidden)
                base_ms = embedding_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                proposal = at_gate(index.linked_options(ranked)[0], *UNGATED) if initial is None else None
                rescue_ms = (perf_counter() - start) * 1000
                arms = {"anchor": (initial, base_ms, 0), "ungated": (initial or proposal, base_ms + rescue_ms, 0)}
                for arm in ("weakest", "absolute"):
                    decided, extra_ms, reads = initial, 0., 0
                    if proposal:
                        start = perf_counter()
                        sides = (weaker_side(ranked, proposal),) if arm == "weakest" else (0, 1)
                        score = verify(reranker, index, clues, proposal, sides)
                        decided = proposal if score > cutoffs[arm] else None
                        extra_ms = (perf_counter() - start) * 1000
                        reads = len(sides)
                    arms[arm] = (decided, base_ms + rescue_ms + extra_ms, reads)
                for arm, (selected, ms, reads) in arms.items():
                    actual, _ = outcome(index, row, selected, role_sets, "transfer", None)
                    oracle, _ = outcome(index, row, selected, known_role_sets, "transfer", None)
                    if j:
                        key = "|".join((view, availability, arm))
                        counts[key].update(
                            pair_correct=selected == target, pair_wrong=selected is not None and selected != target,
                            proof=actual["proof"], adverse=actual["adverse"], role_substitution=actual["substitute"],
                            oracle_proof=oracle["proof"], oracle_substitution=oracle["substitute"], requests=1,
                            source_reads=reads, verifier_calls=int(reads > 0))
                        times[key].append(ms)
                        if selected is not None and selected != target:
                            unique[key].add((target, selected, availability))
    return {key: dict(totals=value, unique_wrong_source_pairs=len(unique[key]),
                      p95_ms=float(np.quantile(times[key], .95))) for key, value in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    reranker = Reranker(args.reranker)
    baseline = json.loads(args.frozen.read_bytes())["runs"][0]
    if baseline["verifier"]["revision"] != reranker.provenance["revision"]:
        raise ValueError("Source verifier revision changed")
    cutoffs = {arm: baseline["thresholds"]["keywords"][arm]["cutoff"] for arm in ("weakest", "absolute")}
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    articles, rows, info = build(args.elife, json.loads(args.manifest.read_bytes()))
    engineering = json.loads(args.engineering.read_bytes())
    panels = {"elife": (articles, rows), "engineering": (engineering["articles"], engineering["queries"])}
    results = {key: evaluate(*panel, encoder, banks, wording, bands, reranker, cutoffs, args.reverse)
               for key, panel in panels.items()}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), elife_sha256=digest(args.elife), engineering_sha256=digest(args.engineering),
        frozen_sha256=digest(args.frozen), embedding_revision=checkpoint["revision"], verifier=reranker.provenance,
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, cutoffs=cutoffs, elife_panel=info, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("elife", "engineering", "manifest", "niso", "training", "bands", "frozen", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
