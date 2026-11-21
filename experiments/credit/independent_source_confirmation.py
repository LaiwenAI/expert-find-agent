"""Verify an author-linked fallback using only requested topic and paper text."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.elife_one_role_transfer import build as development
from experiments.credit.elife_prompt_holdout import build as holdout
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, TOPIC_TASK, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.keyword_author_rescue import UNGATED, PostingCatalog, at_gate
from experiments.credit.role_issuance_audit import outcome
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


VERIFY = ("Given a topic clue, determine whether the scientific paper reports the "
          "specific study described by that clue. Distinguish closely related studies.")
ARMS = ("weakest", "absolute", "relative")


def weaker_side(topic, pair):
    return max(range(2), key=lambda i: (topic[i]["best"] - topic[i]["cosine"][pair[i]], -i))


def confirmation(reranker, articles, clues, topic, pair):
    start = perf_counter()
    weakest = weaker_side(topic, pair)
    first, truncated = reranker.score_pairs([(clues[weakest], articles[pair[weakest]]["indexed"])], task=VERIFY)
    if truncated:
        raise ValueError("Source read exceeded the benchmark context cap")
    weakest_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    tasks = [(clue, articles[paper]["indexed"]) for clue, paper in zip(clues, pair, strict=True)]
    scores, truncated = reranker.score_pairs(tasks, task=VERIFY)
    if truncated:
        raise ValueError("Source read exceeded the benchmark context cap")
    absolute = min(scores)
    absolute_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    rivals = []
    for side, target in zip(topic, pair, strict=True):
        same_topic_rival = next((paper for paper in side["fused"] if paper != target), None)
        rivals.append(same_topic_rival)
    tasks.extend((clue, articles[paper]["indexed"]) for clue, paper in zip(clues, rivals, strict=True))
    scores, truncated = reranker.score_pairs(tasks, task=VERIFY)
    if truncated:
        raise ValueError("Source read exceeded the benchmark context cap")
    return dict(weakest=first[0], absolute=absolute, relative=min(a - b for a, b in zip(scores[:2], scores[2:], strict=True)),
                ms={"weakest": weakest_ms, "absolute": absolute_ms, "relative": (perf_counter() - start) * 1000})


def measure(articles, rows, encoder, banks, wording, bands, reranker, reverse, calibration=False):
    index = PostingCatalog(articles, encoder)
    text_bank = phrase_bank(wording)
    records = defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        target = tuple(row["papers"])
        excerpts = [articles[doc]["clue"] for doc in target]
        roles = [text_bank[role][int(row["family"][:16], 16) % len(text_bank[role])] for role in row["roles"]]
        for kind, clues in (("excerpt", excerpts), ("keywords", [index.keywords(text) for text in excerpts])):
            start = perf_counter()
            vec = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                 [f"Instruct: {TASK}\nQuery:{role}" for role in roles])
            scores = {"prototype": vec[2:] @ banks["prototype"].T}
            required = required_roles(row, scores, "prototype_band", bands)
            encode_ms = (perf_counter() - start) * 1000
            verified = {}
            for availability, masked in zip(AVAILABILITY, (None, *target), strict=True):
                start = perf_counter()
                topic = index.full_rank(vec[:2], clues, masked)
                base, _ = select(index, [side | {"fused": side["fused"][:6]} for side in topic],
                                 {}, [set(), set()], "anchored", masked)
                baseline_ms = encode_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                proposal = base or at_gate(index.linked_options(topic)[0], *UNGATED)
                proposal_ms = (perf_counter() - start) * 1000
                if proposal and (base is None or calibration):
                    key = proposal, tuple(tuple(side["fused"][:2]) for side in topic), weaker_side(topic, proposal)
                    if key not in verified:
                        verified[key] = confirmation(reranker, articles, clues, topic, proposal)
                    signals = verified[key]
                else:
                    signals = None
                if i:
                    def verdict(pair, role_sets):
                        result, _ = outcome(index, row, pair, role_sets, "source", None)
                        return result
                    oracle = [{role} for role in row["roles"]]
                    records[kind, availability].append(dict(
                        target=target, proposal=proposal, base=base, signals=signals,
                        base_ms=baseline_ms, proposal_ms=proposal_ms,
                        oracle_base=verdict(base, oracle), oracle_proposal=verdict(proposal, oracle),
                        static_base=verdict(base, required), static_proposal=verdict(proposal, required)))
    return records


def fit(records):
    result = {}
    for kind in ("keywords", "excerpt"):
        unique = {}
        for (view, status), rows in records.items():
            if view != kind:
                continue
            for row in rows:
                if row["signals"] is None:
                    continue
                key = row["target"], status
                current = row["proposal"], {arm: row["signals"][arm] for arm in ARMS}
                if key in unique and unique[key] != current:
                    raise ValueError("Role wording changed a source verification score")
                unique[key] = current
        result[kind] = {}
        for arm in ARMS:
            positives, adverse = [], []
            for (target, _), (proposal, scores) in unique.items():
                (positives if target == proposal else adverse).append(scores[arm])
            if not positives or not adverse:
                raise ValueError("Need matched and incorrect development proposals")
            cutoff = max(adverse)
            result[kind][arm] = {"cutoff": cutoff, "positive": len(positives), "adverse": len(adverse),
                                 "positive_pass": sum(score > cutoff for score in positives)}
    return result


def summary(records, fits):
    result = {}
    for (kind, status), rows in sorted(records.items()):
        measures = {}
        for arm in ("baseline", "ungated", *ARMS):
            totals, times = Counter(), []
            for row in rows:
                read = not row["base"] and row["signals"] and arm in ARMS
                accepted = arm == "ungated" or (read and row["signals"][arm] > fits[kind][arm]["cutoff"])
                route = "proposal" if accepted else "base"
                pair = row["proposal"] if accepted else row["base"]
                o, s = row[f"oracle_{route}"], row[f"static_{route}"]
                totals.update(
                    correct_pair=pair == row["target"], pair_emitted=pair is not None,
                    oracle_proof=o["proof"], oracle_substitute=o["substitute"],
                    static_proof=s["proof"], static_adverse=s["adverse"], static_substitute=s["substitute"],
                    verified=int(bool(read)), requests=1)
                ms = row["base_ms"]
                if arm != "baseline":
                    ms += row["proposal_ms"]
                if read:
                    ms += row["signals"]["ms"][arm]
                times.append(ms)
            measures[arm] = dict(counts=dict(totals), p95_ms=float(np.quantile(times, .95)))
        result[f"{kind}|{status}"] = measures
    return result


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    model = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    verifier = Reranker(args.reranker)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    dev, dev_rows, census = development(args.development)
    collected = measure(dev, dev_rows, encoder, banks, wording, bands, verifier, args.reverse, calibration=True)
    thresholds = fit(collected)
    articles, cases, held = holdout(args.holdout, json.loads(args.manifest.read_bytes()))
    records = measure(articles, cases, encoder, banks, wording, bands, verifier, args.reverse)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), dev_sha256=digest(args.development), holdout_sha256=digest(args.holdout),
        embedding_revision=model["revision"], verifier=verifier.provenance, gpu=torch.cuda.get_device_name(),
        reverse=args.reverse, development_panel=census, held_panel=held, thresholds=thresholds,
        development=summary(collected, thresholds), results=summary(records, thresholds),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "holdout", "manifest", "niso", "training", "bands", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
