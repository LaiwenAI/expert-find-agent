"""Measure selective rechecking of one role in a two-source proof."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_cross_source_search import phrase_bank, required_roles
from experiments.credit.engineering_joint_calibration import SUPPORT, judge, open_calibration
from experiments.credit.engineering_role_banks import encode, projections, queries
from experiments.credit.engineering_role_comparison import offers
from experiments.credit.scoped_credit_language import SYMBOLS
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.paired_reader import load_reader


ROUTES = ("left", "right", "smallest_margin", "proof_aware", "proof_aware_gated")


class RoleReader:
    def __init__(self, path, definitions):
        self.tokenizer, self.model, self.provenance = load_reader(path)
        self.definitions = definitions
        digits = [self.tokenizer.encode(tag, add_special_tokens=False) for tag in SYMBOLS[:15]]
        if any(len(digit) != 1 for digit in digits) or len({digit[0] for digit in digits}) != len(digits):
            raise ValueError("Expected distinct option tokens")
        self.digits = [digit[0] for digit in digits]

    @torch.inference_mode()
    def read(self, phrase, roles):
        ordered = sorted(roles, key=lambda role: sha256(f"{phrase}\n{role}".encode()).digest())
        choices = [self.definitions[SUPPORT[role]] for role in ordered]
        choices.append("None of these roles describes the activity.")
        options = "\n".join(f"{tag}. {choice}" for tag, choice in zip(SYMBOLS, choices))
        messages = [
            dict(role="system", content="Classify a research activity under CRediT. Return exactly one option letter."),
            dict(role="user", content=f"Which role best matches this activity?\n{phrase}\n\n{options}"),
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        scores = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1, self.digits[:len(choices)]]
        values = scores.float().cpu().numpy()
        first, second = np.argsort(-values, kind="stable")[:2]
        role = ordered[first] if first < len(ordered) else None
        return role, float(values[first] - values[second]), int(inputs["input_ids"].shape[1])


def resolve(articles, row, sets):
    intended = tuple(row["roles"])
    cited = intended if all(role in options for role, options in zip(intended, sets, strict=True)) else ("", "")
    return offers(articles, row["papers"], sets, cited)


def replace(sets, slot, role):
    result = list(sets)
    result[slot] = {role}
    return result


def state(articles, row, score, bands):
    sets = required_roles(row, {"ridge": score}, "ridge_band", bands)
    initial = resolve(articles, row, sets)
    if initial:
        return sets, initial, {}
    owners = [articles[paper]["roles"] for paper in row["papers"]]
    eligible = owners[0].keys() & owners[1].keys()
    branches = {}
    for slot in (0, 1):
        if len(sets[slot]) <= 1:
            continue
        other = 1 - slot
        branches[slot] = sum(any(role in owners[slot][key] and sets[other] <= set(owners[other][key])
                                  for key in eligible) for role in sets[slot]) / len(sets[slot])
    return sets, initial, branches


def pick(branches, score):
    if not branches:
        return dict.fromkeys(ROUTES)
    sorted_scores = np.sort(score, axis=1)
    gaps = sorted_scores[:, -1] - sorted_scores[:, -2]
    useful = max(branches, key=lambda slot: (branches[slot], -gaps[slot], -slot))
    return dict(left=min(branches), right=max(branches),
                smallest_margin=min(branches, key=lambda slot: (gaps[slot], slot)),
                proof_aware=useful, proof_aware_gated=useful if branches[useful] else None)


def combined(articles, row, sets, slot, role):
    return resolve(articles, row, replace(sets, slot, role))


def cutoff_for(articles, calibration, scores, reader, bands):
    observations = []
    for row in calibration:
        indices = [SUPPORT[role] for role in row["roles"]]
        matrix = scores[indices]
        sets, initial, branches = state(articles, row, matrix, bands)
        selected = pick(branches, matrix)["proof_aware"]
        confidence = 0.
        rescued = initial
        if selected is not None:
            role, confidence, _ = reader.read(VIEWS["original"][indices[selected]], sets[selected])
            if role is not None:
                rescued = combined(articles, row, sets, selected, role)
        observations.append((confidence, judge(row, initial, 0.), judge(row, rescued, 0.)))
    best = None
    for cutoff in sorted({0., *(margin for margin, _, _ in observations)}):
        totals = Counter()
        for margin, before, after in observations:
            totals.update(after if margin > cutoff else before)
        if totals["adverse"] <= len(calibration) // 20:
            key = totals["exact_proof"], -totals["adverse"], cutoff
            if best is None or key > best[0]:
                best = key, dict(cutoff=cutoff, totals=totals)
    return None if best is None else best[1]


def evaluate(panel, wording, encoder, ridge, reader, bands, cutoff, reverse):
    articles = panel["articles"]
    rows = sorted(panel["queries"], key=lambda row: (row["family"], row["variant"]), reverse=reverse)
    phrases = phrase_bank(wording)
    totals, timing = {}, {}
    for i, row in enumerate([rows[0], *rows]):
        texts = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        start = perf_counter()
        scores = encode(encoder, texts) @ ridge.T
        sets, initial, branches = state(articles, row, scores, bands)
        chosen = pick(branches, scores)
        shared_ms = (perf_counter() - start) * 1000
        reads = {}
        for slot in sorted(branches):
            start = perf_counter()
            reading = reader.read(texts[slot], sets[slot])
            reads[slot] = reading, (perf_counter() - start) * 1000
        evidence = {"ridge_band": (judge(row, initial, 0.), 0, 0, shared_ms)}
        oracle_scores = {}
        for arm, slot in chosen.items():
            started = perf_counter()
            current = initial
            if slot is None:
                tokens = extra_ms = 0
            else:
                (role, margin, tokens), extra_ms = reads[slot]
                if role is not None and cutoff is not None and margin > cutoff:
                    current = combined(articles, row, sets, slot, role)
            measured = judge(row, current, 0.)
            json.dumps(measured)
            evidence[arm] = measured, int(slot is not None), tokens, shared_ms + extra_ms + (perf_counter() - started) * 1000
            if slot not in oracle_scores:
                intended = None if slot is None else row["roles"][slot]
                resolved = initial
                if slot is not None and intended in sets[slot]:
                    resolved = combined(articles, row, sets, slot, intended)
                oracle_scores[slot] = judge(row, resolved, 0.)
            evidence[f"{arm}_oracle"] = oracle_scores[slot], int(slot is not None), 0, None
        evidence["best_single_oracle"] = max(oracle_scores.values(), key=lambda d: (d["exact_proof"], not d["adverse"])), int(bool(branches)), 0, None
        if i:
            for arm, (metrics, calls, tokens, ms) in evidence.items():
                totals.setdefault(arm, Counter()).update(
                    metrics, calls=calls, tokens=tokens,
                    swap_issued=row["variant"] == "swap" and metrics["issued"],
                )
                if ms is not None:
                    timing.setdefault(arm, []).append(ms)
    return {
        arm: dict(totals=value, p95_ms=float(np.quantile(timing[arm], .95)) if arm in timing else None,
                  mean_ms=float(np.mean(timing[arm])) if arm in timing else None)
        for arm, value in totals.items()
    }


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, _, calibration = open_calibration(args.plos)
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    ridge = projections(encoder, definitions, training)["ridge"]
    reader = RoleReader(args.reader, definitions)
    bands = json.loads(args.published.read_bytes())["runs"][0]["bands"]
    fit = cutoff_for(articles, calibration, encode(encoder, VIEWS["original"]) @ ridge.T, reader, bands)
    panel = json.loads(args.panel.read_bytes())
    output = evaluate(panel, wording, encoder, ridge, reader, bands, None if fit is None else fit["cutoff"], args.reverse)
    write_json(args.output, dict(
        panel_sha256=digest(args.panel), calibration_sha256=digest(args.plos), script_sha256=digest(Path(__file__)),
        embedding_revision=checkpoint["revision"], reader_revision=reader.provenance["revision"],
        gpu=torch.cuda.get_device_name(), calibration_cases=len(calibration), target_cases=len(panel["queries"]),
        reverse=args.reverse, fit=fit, results=output,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "plos", "niso", "training", "published", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
