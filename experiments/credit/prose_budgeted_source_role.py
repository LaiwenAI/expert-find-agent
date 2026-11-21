"""Measure source-role binding errors before a fixed one-call search path."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import phrase_bank
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import projections, queries
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.scoped_credit_language import SYMBOLS
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


OPTIONS = tuple(((a, c), (b, d)) if not swap else ((a, d), (b, c))
                for a, b in combinations(range(4), 2)
                for c, d in [sorted(set(range(4)) - {a, b})]
                for swap in (False, True))
FORMATS = ("interleaved", "reversed_references", "role_first")
QUOTES = re.compile(r"“([^”]+)”")


def prose(clues, roles, style):
    a, b = clues
    x, y = roles
    if any("“" in s or "”" in s for s in (*clues, *roles)):
        raise ValueError("Unexpected quote delimiters inside prepared spans")
    if style == "interleaved":
        return (f"Find one researcher for two kinds of work. The paper about “{a}” "
                f"must credit them for “{x}”. In a separate paper about “{b}”, "
                f"they must also be credited for “{y}”.")
    if style == "role_first":
        return (f"I need one researcher credited for “{x}” in the paper about “{a}” "
                f"and credited for “{y}” in the one about “{b}”.")
    return (f"Two papers are relevant, one about “{a}” and another about “{b}”. The same "
            f"person must be credited for “{y}” in the latter and for “{x}” in the former.")


def bind_structural(request):
    fragments = QUOTES.split(request)
    contexts = fragments[::2]
    topics = [i for i, before in enumerate(contexts[:-1]) if re.search(r"\babout\s*$", before)]
    if len(topics) != 2:
        return None
    associations = []
    for activity in sorted(set(range(4)) - set(topics)):
        after = contexts[activity + 1]
        ref = re.match(r"\s*in the (former|latter)\b", after)
        if ref:
            source = topics[int(ref[1] == "latter")]
        elif re.match(r"\s*in the (paper|one) about\s*$", after) and activity + 1 in topics:
            source = activity + 1
        elif activity - 1 in topics:
            source = activity - 1
        else:
            return None
        associations.append((source, activity))
    edges = tuple(sorted(associations))
    return edges if edges in OPTIONS else None


class SpanParser:
    def __init__(self, reader):
        self.reader = reader
        self.options = "\n".join(
            f"{code}: topic [{a + 1}] with activity [{c + 1}]; topic [{b + 1}] with activity [{d + 1}]"
            for code, ((a, c), (b, d)) in zip(SYMBOLS, OPTIONS))
        self.none = SYMBOLS[len(OPTIONS)]

    @torch.inference_mode()
    def parse(self, request):
        numbered = QUOTES.sub(lambda _m, step=iter(range(1, 5)): f"[{next(step)}] {_m[0]}", request)
        messages = [
            dict(role="system", content="Bind each study topic to the exact activity required in that study. Return one option letter."),
            dict(role="user", content=f"Request: {numbered}\n\n{self.options}\n{self.none}: unclear"),
        ]
        tokenizer = self.reader.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        logits = self.reader.model(**tokens, use_cache=False, logits_to_keep=1).logits[0, -1, self.reader.digits[:len(OPTIONS) + 1]]
        chosen = int(logits.argmax().item())
        return (OPTIONS[chosen] if chosen < len(OPTIONS) else None), int(tokens.input_ids.shape[1])


    @torch.inference_mode()
    def factorized(self, request):
        numbered = QUOTES.sub(lambda match, step=iter(range(1, 5)): f"[{next(step)}] {match[0]}", request)
        topic_pairs = tuple(combinations(range(4), 2))
        topics = "\n".join(f"{tag}: [{i+1}] and [{j+1}]" for tag, (i, j) in zip(SYMBOLS, topic_pairs))
        questions = (
            f"Which two spans are paper topics?\n{topics}\nG: unclear",
            "Which span binds to [1] as topic and activity of the same paper? "
            "A: [2]\nB: [3]\nC: [4]\nD: unclear",
        )
        tokenizer = self.reader.tokenizer
        tokenizer.padding_side = "left"
        prompts = [tokenizer.apply_chat_template([
            dict(role="system", content="Read the request and return only the selected option letter."),
            dict(role="user", content=f"Request: {numbered}\n\n{question}"),
        ], tokenize=False, add_generation_prompt=True) for question in questions]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        positions = (inputs["attention_mask"].cumsum(1) - 1).clamp(min=0)
        scores = self.reader.model(**inputs, position_ids=positions, use_cache=False,
                                   logits_to_keep=1).logits[:, -1, self.reader.digits[:7]]
        topic_option = int(scores[0, :7].argmax().item())
        mate = int(scores[1, :4].argmax().item()) + 1
        tokens = int(inputs["attention_mask"].sum().item())
        if topic_option >= len(topic_pairs) or mate == 4:
            return None, tokens
        topic = set(topic_pairs[topic_option])
        groups = ((0, mate), tuple(sorted(set(range(4)) - {0, mate})))
        if any(len(topic & set(group)) != 1 for group in groups):
            return None, tokens
        edges = tuple(sorted((a, b) if a in topic else (b, a) for a, b in groups))
        return edges, tokens


def evidence(response, row):
    expected = set(row["gold"])
    people = set(response["owners"])
    exact_source = response["papers"] == list(row["papers"])
    presented = response["roles"]
    exact_roles = len(presented) == 2 and all(role in options for role, options in zip(row["roles"], presented))
    return dict(
        issued=bool(people), strict=bool(expected) and people == expected and exact_source and exact_roles,
        adverse=bool(people) and (not exact_source or not exact_roles or bool(people - expected)),
        wrong_source=bool(people) and not exact_source, requests=1, positives=bool(expected),
        search_calls=response["calls"], source_calls=response["source_calls"], role_calls=response["role_calls"])


def test(service, rows, wording, parser, reverse):
    bank = phrase_bank(wording)
    rows = rows[::-1] if reverse else rows
    counts, times = defaultdict(Counter), defaultdict(list)
    for i, row in enumerate([rows[0], *rows]):
        topics = [service.index.keywords(service.index.articles[p]["clue"]) for p in row["papers"]]
        activities = [bank[role][int(row["family"][:16], 16) % len(bank[role])] for role in row["roles"]]
        expected = list(zip(topics, activities, strict=True))
        for style in FORMATS:
            request = prose(topics, activities, style)
            for arm in ("supplied", "adjacent", "structural", "span_model", "factorized"):
                start = perf_counter()
                spans = QUOTES.findall(request)
                if len(spans) != 4:
                    raise ValueError("Four delimited fragments required")
                parse_tokens = 0
                if arm == "supplied":
                    assigned = expected
                else:
                    if arm == "span_model":
                        edges, parse_tokens = parser.parse(request)
                    elif arm == "factorized":
                        edges, parse_tokens = parser.factorized(request)
                    elif arm == "structural":
                        edges = bind_structural(request)
                    else:
                        edges = ((0, 1), (2, 3))
                    assigned = [(spans[a], spans[b]) for a, b in edges] if edges else []
                matched = assigned == expected
                parsed_topics = tuple(x for x, _ in assigned)
                correct_topic = parsed_topics == tuple(topics)
                parse_ms = (perf_counter() - start) * 1000
                if assigned:
                    payload = SimpleNamespace(payload=json.dumps({"requirements": [
                        dict(topic=topic, role=activity) for topic, activity in assigned]}))
                    response = json.loads(service.search(payload, "screened_one_call"))
                else:
                    response = dict(papers=None, owners=[], roles=[], calls=0, source_calls=0, role_calls=0)
                result = evidence(response, row)
                elapsed = (perf_counter() - start) * 1000
                if i:
                    key = "|".join((style, arm))
                    counts[key].update(result, correct_bindings=matched, correct_topic_spans=correct_topic,
                                       parse_calls=int(arm in ("span_model", "factorized")), parse_tokens=parse_tokens)
                    times[key].append((parse_ms, elapsed))
    return {key: dict(totals=dict(value),
                      parse_p95_ms=float(np.quantile([a for a, _ in times[key]], .95)),
                      p95_ms=float(np.quantile([b for _, b in times[key]], .95)))
            for key, value in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, rows, census = build(args.projection, json.loads(args.manifest.read_bytes()))
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    source = Reranker(args.source_reader)
    full = RoleReader(args.role_reader, definitions)
    full.definitions = ROLES
    source_config = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != source_config["verifier"]["revision"]:
        raise ValueError("Source verifier revision changed")
    cutoffs = {"source": source_config["thresholds"]["keywords"]["weakest"]["cutoff"],
               "role": json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"]}
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    index = PostingCatalog(articles, encoder)
    service = Service(index, encoder, banks, bands, source, full, cutoffs)
    results = test(service, rows, wording, SpanParser(full), args.reverse)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection), manifest_sha256=digest(args.manifest),
        embedding_revision=checkpoint["revision"], source_reader_revision=source.provenance["revision"],
        role_reader_revision=full.provenance["revision"], gpu=torch.cuda.get_device_name(), reverse=args.reverse,
        panel=census, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
