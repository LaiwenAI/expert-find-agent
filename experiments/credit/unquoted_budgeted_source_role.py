"""Measure literal requirement extraction and expert proof search on unquoted requests."""

import argparse
from collections import Counter, defaultdict
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
from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.engineering_cross_source_search import phrase_bank
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import projections, queries
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


TEMPLATES = {
    "local_markers": ("I am looking for one researcher. For the paper about {a}, show their contribution to {x}. "
                      "In a separate paper about {b}, show their contribution to {y}."),
    "reversed_references": ("There are studies on {a} and on {b}. Find one researcher whose work in the "
                            "latter included {y}, and in the former included {x}."),
    "shifted_markers": ("Can a single researcher be credited with {x} in work involving {a}, while also "
                        "helping with {y} in a different study concerning {b}?"),
}
MARKERS = (
    re.compile(r".*?For the paper about (?P<a>.+?), show their contribution to (?P<x>.+?)\. "
               r"In a separate paper about (?P<b>.+?), show their contribution to (?P<y>.+?)\."),
    re.compile(r".*?studies on (?P<a>.+?) and on (?P<b>.+?)\. Find one researcher whose work in the "
               r"latter included (?P<y>.+?), and in the former included (?P<x>.+?)\."),
)


def rule_parse(request):
    for pattern in MARKERS:
        if match := pattern.fullmatch(request):
            return ((match["a"], match["x"]), (match["b"], match["y"]))
    return ()


def validate_copy(raw, request):
    """Check both requirements use exact, distinct, nonoverlapping query spans."""
    try:
        pairs = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(pairs, list) or len(pairs) != 2:
        return ()
    fields = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            return ()
        for value in pair:
            if not isinstance(value, str) or not value or value != value.strip():
                return ()
            start = request.find(value)
            if start < 0 or request.find(value, start + 1) >= 0:
                return ()
            fields.append((start, start + len(value)))
    spans = sorted(fields)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        return ()
    return tuple(tuple(pair) for _, pair in sorted(zip(fields[::2], pairs, strict=True)))


@torch.inference_mode()
def model_parse(reader, request, *, allow_empty):
    tokenizer = reader.tokenizer
    suffix = ("If the request does not give both pairs, return []." if allow_empty else
              "Each request in this task contains both pairs. Extract both.")
    messages = [
        dict(role="system", content=(
            "Extract the exact words for each paper topic and the activity the researcher must have done "
            "in that same paper. Copy each complete topic and activity verbatim from the request. "
            "Exclude linking language and punctuation around the copied phrases. "
            'Return only compact JSON: [["topic1","activity1"],["topic2","activity2"]]. '
            f"{suffix}")),
        dict(role="user", content=request),
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    input_tokens = inputs.input_ids.shape[1]
    output = reader.model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id)[0, input_tokens:].tolist()
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    return validate_copy(decoded, request), input_tokens, len(output)


def run_panel(service, rows, wording, reader, reverse):
    bank = phrase_bank(wording)
    rows = rows[::-1] if reverse else rows
    counts, times = defaultdict(Counter), defaultdict(list)
    for index, row in enumerate([rows[0], *rows]):
        topics = tuple(service.index.keywords(service.index.articles[p]["clue"]) for p in row["papers"])
        activities = tuple(bank[r][int(row["family"][:16], 16) % len(bank[r])].strip().rstrip(".?!")
                           for r in row["roles"])
        expected = tuple(zip(topics, activities, strict=True))
        for style, template in TEMPLATES.items():
            request = template.format(a=topics[0], b=topics[1], x=activities[0], y=activities[1])
            start = perf_counter()
            rules = rule_parse(request)
            rule_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            extracted, input_tokens, output_tokens = model_parse(reader, request, allow_empty=True)
            model_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            forced, forced_input, forced_output = model_parse(reader, request, allow_empty=False)
            forced_ms = (perf_counter() - start) * 1000
            arms = (
                ("supplied", expected, 0., 0, 0),
                ("marker_rules", rules, rule_ms, 0, 0),
                ("model", extracted, model_ms, input_tokens, output_tokens),
                ("model_required", forced, forced_ms, forced_input, forced_output),
                ("hybrid", rules, rule_ms, 0, 0) if rules else
                ("hybrid", extracted, rule_ms + model_ms, input_tokens, output_tokens),
                ("hybrid_required", rules, rule_ms, 0, 0) if rules else
                ("hybrid_required", forced, rule_ms + forced_ms, forced_input, forced_output),
            )
            for arm, parsed, parse_ms, read_tokens, produced_tokens in arms:
                start = perf_counter()
                if parsed:
                    payload = SimpleNamespace(payload=json.dumps({"requirements": [
                        dict(topic=topic, role=activity) for topic, activity in parsed]}))
                    response = json.loads(service.search(payload, "screened_one_call"))
                else:
                    response = dict(papers=None, owners=[], roles=[], calls=0, source_calls=0, role_calls=0)
                elapsed = parse_ms + (perf_counter() - start) * 1000
                if index:
                    key = f"{style}|{arm}"
                    counts[key].update(evidence(response, row), binding=parsed == expected,
                                       topics=tuple(topic for topic, _ in parsed) == topics,
                                       roles=tuple(role for _, role in parsed) == activities,
                                       parsed=bool(parsed), parse_calls=int(read_tokens > 0),
                                       parse_input_tokens=read_tokens, parse_output_tokens=produced_tokens)
                    times[key].append((parse_ms, elapsed))
        if index and index % 5 == 0:
            print(f"Complete {index}/{len(rows)} cases", flush=True)
    return {key: dict(totals=dict(value),
                      parser_p95_ms=float(np.quantile([a for a, _ in times[key]], .95)),
                      total_p95_ms=float(np.quantile([b for _, b in times[key]], .95)))
            for key, value in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    articles, cases, census = build(args.projection, json.loads(args.manifest.read_bytes()))
    rows = sample_families(cases)
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    source = Reranker(args.source_reader)
    reader = RoleReader(args.role_reader, definitions)
    reader.definitions = ROLES
    source_config = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != source_config["verifier"]["revision"]:
        raise ValueError("Source verifier revision changed")
    cutoffs = dict(source=source_config["thresholds"]["keywords"]["weakest"]["cutoff"],
                   role=json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"])
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    service = Service(PostingCatalog(articles, encoder), encoder, banks, bands, source, reader, cutoffs)
    results = run_panel(service, rows, wording, reader, args.reverse)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), projection_sha256=digest(args.projection),
        manifest_sha256=digest(args.manifest), score_bands_sha256=digest(args.bands),
        source_frozen_sha256=digest(args.source_frozen), role_frozen_sha256=digest(args.role_frozen),
        embedding_revision=checkpoint["revision"], source_reader_revision=source.provenance["revision"],
        role_reader_revision=reader.provenance["revision"], gpu=torch.cuda.get_device_name(),
        reverse=args.reverse, catalog=census, sampled_families=sorted({row["family"] for row in rows}),
        results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("projection", "manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                     "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
