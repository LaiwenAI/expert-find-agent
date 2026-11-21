"""Resolve two ambiguous role clauses within one verifier forward pass."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.budgeted_source_role_search import role_check
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_cross_source_search import TOPIC_TASK, candidates, phrase_bank, required_roles
from experiments.credit.engineering_joint_calibration import SUPPORT, open_calibration
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import TASK, encode, projections, queries
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.scoped_credit_language import SYMBOLS
from experiments.credit.source_role_factorial_diagnosis import payload, plan
from experiments.credit.transfer_source_role_resolution import checked_original, load_engineering
from experiments.credit.unquoted_budgeted_source_role import sample_families
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


METHODS = ("top_one", "two_serial", "two_batched", "packed_pair")


def keyed_order(phrase, roles):
    return sorted(roles, key=lambda role: sha256(f"{phrase}\n{role}".encode()).digest())


class PairedReader(RoleReader):
    @torch.inference_mode()
    def option_logits(self, prompts, *, per_row=False):
        tokenizer = self.tokenizer
        rendered = [tokenizer.apply_chat_template([
            dict(role="system", content=system), dict(role="user", content=user),
        ], tokenize=False, add_generation_prompt=True) for system, user in prompts]
        side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(rendered, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        finally:
            tokenizer.padding_side = side
        positions = (inputs["attention_mask"].cumsum(1) - 1).clamp(min=0)
        logits = self.model(**inputs, position_ids=positions, use_cache=False, logits_to_keep=1).logits
        scores = logits[:, -1, self.digits].float().cpu().numpy()
        mask = inputs["attention_mask"]
        counts = mask.sum(dim=1).tolist() if per_row else int(mask.sum().item())
        return scores, counts

    def read_options(self, prompts, choices):
        scores, tokens = self.option_logits(prompts)
        return self.decode_options(scores, choices, tokens)

    @staticmethod
    def decode_options(scores, choices, tokens):
        decisions, gaps = [], []
        for row, options in zip(scores, choices, strict=True):
            best, second = np.argsort(-row[:len(options)], kind="stable")[:2]
            decisions.append(options[best])
            gaps.append(float(row[best] - row[second]))
        return decisions, min(gaps), tokens

    def format_batched(self, phrases, pools):
        prompts, choices = [], []
        for phrase, pool in zip(phrases, pools, strict=True):
            options = keyed_order(phrase, pool)
            definitions = [self.definitions[SUPPORT[role]] for role in options]
            definitions.append("None of these roles describes the activity.")
            listing = "\n".join(f"{label}. {text}" for label, text in zip(SYMBOLS, definitions))
            prompts.append(("Classify a research activity under CRediT. Return exactly one option letter.",
                            f"Which role best matches this activity?\n{phrase}\n\n{listing}"))
            choices.append(options + [None])
        return prompts, choices

    def batched(self, phrases, pools):
        prompts, choices = self.format_batched(phrases, pools)
        return self.read_options(prompts, choices)

    def batched_requests(self, requests):
        prompts, choices = [], []
        for phrases, pools in requests:
            rows, options = self.format_batched(phrases, pools)
            prompts.extend(rows)
            choices.extend(options)
        scores, counts = self.option_logits(prompts, per_row=True)
        return [self.decode_options(scores[i:i + 2], choices[i:i + 2], sum(counts[i:i + 2]))
                for i in range(0, len(prompts), 2)]

    def packed(self, phrases, pools):
        glossary = sorted(set(pools[0]) | set(pools[1]))
        labels = {role: f"R{i + 1}" for i, role in enumerate(glossary)}
        definitions = "\n".join(f"{labels[r]}: {self.definitions[SUPPORT[r]]}" for r in glossary)
        pairs = sorted(product(*pools), key=lambda pair: sha256(repr((phrases, pair)).encode()).digest())
        options = [f"{labels[a]} then {labels[b]}" for a, b in pairs]
        options.append("At least one activity has no matching role.")
        listing = "\n".join(f"{tag}. {text}" for tag, text in zip(SYMBOLS, options))
        out, gap, tokens = self.read_options([
            ("Classify two research activities independently under CRediT. Select one ordered pair or no match. "
             "Return exactly one option letter.",
             f"Activity 1: {phrases[0]}\nActivity 2: {phrases[1]}\n\nRole definitions:\n{definitions}"
             f"\n\nOptions, in activity order:\n{listing}"),
        ], [pairs + [None]])
        return out[0], gap, tokens


def alternative_roles(scores):
    normalized = []
    for bank in ("ridge", "prototype"):
        values = scores[bank]
        normalized.append((values - values.mean(axis=1, keepdims=True)) /
                          values.std(axis=1, keepdims=True).clip(.001))
    return [[ROLES[i] for i in np.argsort(-row, kind="stable")[:3]]
            for row in (normalized[0] + normalized[1])]


def available(catalog, pair, pools):
    left, right = [catalog.articles[doc]["roles"] for doc in pair]
    allowed = [set(roles) for roles in pools]
    return any(allowed[0].intersection(left[owner]) and allowed[1].intersection(right[owner])
               for owner in left.keys() & right.keys())


def proposals(catalog, pair, bases, phrases, scores, reader, methods):
    start = perf_counter()
    pools = alternative_roles(scores)
    ready = bool(pair and not candidates(catalog, pair, bases) and available(catalog, pair, pools))
    planning_ms = (perf_counter() - start) * 1000
    outcomes = {}
    for method in methods:
        start = perf_counter()
        choice, margin, forwards, tokens = None, 0., 0, 0
        if ready:
            if method == "top_one":
                choice, margin = [pool[0] for pool in pools], 1.
            elif method == "two_serial":
                reads = [reader.read(phrase, pool) for phrase, pool in zip(phrases, pools, strict=True)]
                choice, gaps, budgets = zip(*reads, strict=True)
                margin, tokens, forwards = min(gaps), sum(budgets), 2
            elif method == "two_batched":
                choice, margin, tokens = reader.batched(phrases, pools)
                forwards = 1
            elif method == "packed_pair":
                choice, margin, tokens = reader.packed(phrases, pools)
                forwards = 1
        if choice is None or None in choice:
            choice = None
        outcomes[method] = dict(choice=choice, margin=margin, forwards=forwards, tokens=tokens,
                                ms=planning_ms + (perf_counter() - start) * 1000, pools=pools,
                                attempted=ready)
    return outcomes


def apply(catalog, pair, base, observation, cutoff, source_calls=0):
    choice = observation["choice"]
    roles = [{value} for value in choice] if choice and cutoff is not None and observation["margin"] > cutoff else base
    return payload(catalog, pair, roles, source_calls, observation["forwards"], observation["tokens"])


def calibrate(path, encoder, banks, bands, reader):
    articles, _, rows = open_calibration(path)
    catalog = SimpleNamespace(articles=articles)
    vectors = encode(encoder, VIEWS["original"])
    observations = defaultdict(list)
    for row in rows:
        pair = tuple(row["papers"])
        labels = [SUPPORT[r] for r in row["roles"]]
        phrases = [VIEWS["original"][i] for i in labels]
        scores = {bank: vectors[labels] @ banks[bank].T for bank in bands}
        base = required_roles(row, scores, "ridge_band", bands)
        tests = proposals(catalog, pair, base, phrases, scores, reader, METHODS)
        for name, output in tests.items():
            observations[name].append((row, pair, base, output))
    fitted = {}
    for name, examples in observations.items():
        points = []
        for cutoff in sorted({0., *(item[-1]["margin"] for item in examples)}):
            tally = Counter()
            for row, pair, base, observation in examples:
                outcome = apply(catalog, pair, base, observation, cutoff)
                tally.update(evidence(outcome, row))
            if tally["adverse"] <= len(rows) // 20:
                points.append((tally["strict"], -tally["adverse"], cutoff, dict(tally)))
        best = max(points) if points else None
        fitted[name] = dict(cutoff=best[2], counts=best[3]) if best else dict(cutoff=None)
    return fitted


def benchmark(service, rows, wording, fit, reverse):
    rows = rows[::-1] if reverse else rows
    table = phrase_bank(wording)
    counts, times = defaultdict(Counter), defaultdict(list)
    for step, row in enumerate([rows[0], *rows]):
        docs = tuple(row["papers"])
        excerpts = [service.index.articles[p]["clue"] for p in docs]
        keywords = [service.index.keywords(x) for x in excerpts]
        roles = [table[r][int(row["family"][:16], 16) % len(table[r])].strip().rstrip(".?!") for r in row["roles"]]
        for view, clues in (("keywords", keywords), ("excerpt", excerpts)):
            start = perf_counter()
            vectors = service.encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                             [f"Instruct: {TASK}\nQuery:{role}" for role in roles])
            scores = {bank: vectors[2:] @ service.banks[bank].T for bank in service.bands}
            embed_ms = (perf_counter() - start) * 1000
            for availability, hidden in zip(("present", "left_hidden", "right_hidden"), (None, *docs), strict=True):
                start = perf_counter()
                topic = service.index.full_rank(vectors[:2], clues, hidden)
                state = plan(service, topic, scores, clues, roles, hidden, "screened", run_role=False)
                pair, base = state["pair"], state["roles"]
                source_calls = state["response"]["source_calls"]
                timing = embed_ms + (perf_counter() - start) * 1000
                start = perf_counter()
                if pair and not source_calls:
                    checked, forwards, tokens, _ = role_check(service.index, pair, scores, service.bands,
                                                               service.role, roles, service.cutoffs["role"])
                    baseline = payload(service.index, pair, checked, role_calls=forwards, tokens=tokens)
                else:
                    baseline = state["response"]
                outputs = {"baseline": (baseline, timing + (perf_counter() - start) * 1000, False, False)}
                attempts = proposals(service.index, pair if not source_calls else None, base, roles,
                                     scores, service.role, METHODS)
                for name, prediction in attempts.items():
                    start = perf_counter()
                    response = apply(service.index, pair, base, prediction, fit[name]["cutoff"], source_calls)
                    duration = timing + prediction["ms"] + (perf_counter() - start) * 1000
                    correct_choice = prediction["choice"] is not None and tuple(prediction["choice"]) == tuple(row["roles"])
                    outputs[name] = response, duration, prediction["attempted"], correct_choice
                # A one-slot call is reserved solely from the pre-read proof branches.
                # Replay the batched rescue only if that branch would make no call.
                batched = attempts["two_batched"]
                if source_calls or baseline["role_calls"] or baseline["owners"] or not batched["attempted"]:
                    outputs["reserve_batch"] = outputs["reserve_agreement"] = outputs["baseline"]
                else:
                    response, duration, attempted, correct = outputs["two_batched"]
                    guard_ms = max(0., outputs["baseline"][1] - timing)
                    outputs["reserve_batch"] = response, duration + guard_ms, attempted, correct
                    dense = attempts["top_one"]["choice"]
                    if batched["choice"] is not None and tuple(batched["choice"]) == tuple(dense):
                        outputs["reserve_agreement"] = outputs["reserve_batch"]
                    else:
                        abstain = payload(service.index, pair, base, role_calls=batched["forwards"],
                                          tokens=batched["tokens"])
                        outputs["reserve_agreement"] = abstain, duration + guard_ms, attempted, correct
                if hidden is None:
                    request = SimpleNamespace(payload=json.dumps({"requirements": [
                        dict(topic=topic, role=role) for topic, role in zip(clues, roles, strict=True)]}))
                    if baseline != json.loads(service.search(request, "screened_one_call")):
                        raise ValueError("Baseline deviates from service")
                if step:
                    for name, (response, duration, attempted, correct_choice) in outputs.items():
                        key = f"{view}|{availability}|{name}"
                        if name != "two_serial" and response["calls"] > 1:
                            raise ValueError("Verifier forward cap exceeded")
                        counts[key].update(evidence(response, row), expression_gate=attempted,
                                           exact_interpretation=correct_choice, tokens=response["tokens"],
                                           two_forwards=response["calls"] > 1)
                        times[key].append(duration)
    return {k: dict(totals=dict(c), p95_ms=float(np.quantile(times[k], .95))) for k,c in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    current, current_rows, _ = build(args.current, json.loads(args.current_manifest.read_bytes()))
    earlier, earlier_rows, _ = checked_original(args.earlier, args.earlier_manifest, current)
    engineering, engineering_rows = load_engineering(args.engineering, args.engineering_manifest)
    definitions, training, wording = queries(args.niso, args.training)
    revision = verify_checkpoint(args.embedding)["revision"]
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    source = Reranker(args.source_reader)
    reader = PairedReader(args.role_reader, definitions)
    reader.definitions = ROLES
    frozen = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != frozen["verifier"]["revision"]:
        raise ValueError("Source verifier revision changed")
    cutoffs = dict(source=frozen["thresholds"]["keywords"]["weakest"]["cutoff"],
                   role=json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"])
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    fit = calibrate(args.plos, encoder, banks, bands, reader)
    print(f"Calibrated role routes: {fit}", flush=True)
    results = {}
    for name, articles, rows in (
        ("elife_current", current, current_rows), ("elife_earlier", earlier, earlier_rows),
        ("engineering", engineering, engineering_rows),
    ):
        index = PostingCatalog(articles, encoder)
        service = Service(index, encoder, banks, bands, source, reader, cutoffs)
        selected = sample_families(rows)
        results[name] = dict(pairs=len(selected)//2, articles=len(articles),
                             results=benchmark(service, selected, wording, fit, args.reverse))
        print(f"{name} complete", flush=True)
    names = ("current", "current_manifest", "earlier", "earlier_manifest", "engineering",
             "engineering_manifest", "plos", "niso", "training", "bands", "source_frozen", "role_frozen")
    modules = ("source_role_factorial_diagnosis", "budgeted_source_role_service")
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), inputs={k: digest(getattr(args, k)) for k in names},
        modules={name: digest(Path(__file__).with_name(f"{name}.py")) for name in modules},
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, embedding_revision=revision,
        source_reader_revision=source.provenance["revision"], role_reader_revision=reader.provenance["revision"],
        cutoffs=cutoffs, fit=fit, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("current", "current_manifest", "earlier", "earlier_manifest", "engineering",
                     "engineering_manifest", "plos", "niso", "training", "bands", "source_frozen", "role_frozen",
                     "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
