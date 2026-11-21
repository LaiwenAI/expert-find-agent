"""Separate first-query source nomination and role recognition under supplied spans."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.budgeted_source_role_search import check_source, role_check
from experiments.credit.budgeted_source_role_service import Service
from experiments.credit.elife_prompt_holdout import build
from experiments.credit.engineering_cross_source_search import TOPIC_TASK, candidates, phrase_bank, required_roles, select
from experiments.credit.engineering_one_role_recheck import RoleReader
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.keyword_author_rescue import UNGATED, PostingCatalog, at_gate
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.transfer_source_role_resolution import checked_original, load_engineering
from experiments.credit.unquoted_budgeted_source_role import sample_families
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ROUTES = ("screened", "screen_four", "anchor_twenty_four")


def payload(index, pair, roles, source_calls=0, role_calls=0, tokens=0):
    owners = candidates(index, pair, roles) if pair else []
    return dict(papers=list(pair) if owners else None, owners=owners,
                roles=[sorted(x) for x in roles] if owners else [], calls=source_calls + role_calls,
                source_calls=source_calls, role_calls=role_calls, tokens=tokens)


def plan(service, topic, scores, clues, phrases, hidden, route):
    index = service.index
    shallow = [side | {"fused": side["fused"][:6]} for side in topic]
    pair, _ = select(index, shallow, {}, [set(), set()], "anchored", hidden)
    options = index.linked_options(topic)[0] if pair is None else []
    role_sets = required_roles({}, scores, "prototype_band", service.bands)
    if route == "anchor_twenty_four" and pair is None:
        pair = at_gate(options, True, .04, 0.)
    original = at_gate(options, *UNGATED)
    proposal = original
    if route == "screen_four" and pair is None:
        proposals = dict.fromkeys(option[1] for option in options if at_gate([option], *UNGATED))
        proposal = next((p for p in list(proposals)[:4] if candidates(index, p, role_sets)), None)
    source_calls = role_calls = tokens = 0
    if pair is None and proposal and candidates(index, proposal, role_sets):
        pair, _ = check_source(index, service.source, topic, proposal, clues, service.cutoffs["source"])
        source_calls = 1
    if pair and not source_calls:
        role_sets, role_calls, tokens, _ = role_check(index, pair, scores, service.bands, service.role,
                                                      phrases, service.cutoffs["role"])
    assert source_calls + role_calls <= 1
    return dict(pair=pair, roles=role_sets, proposal=proposal, options={option[1] for option in options},
                response=payload(index, pair, role_sets, source_calls, role_calls, tokens))


def diagnose(row, topic, state):
    target = tuple(row["papers"])
    if evidence(state["response"], row)["strict"]:
        return "strict"
    if state["pair"] == target:
        if any(role not in values for role, values in zip(row["roles"], state["roles"], strict=True)):
            return "role_missing_required_label"
        if not state["response"]["owners"]:
            return "role_extra_labels_blocked_owner"
        return "role_extra_labels_changed_owner_set"
    if any(paper not in side["rank"] for side, paper in zip(topic, target, strict=True)):
        return "source_outside_top24"
    if any(paper not in side["fused"][:6] for side, paper in zip(topic, target, strict=True)):
        if target not in state["options"]:
            return "source_outside_shortlist_and_postings"
    if state["proposal"] == target:
        return "source_verifier_rejection" if state["response"]["source_calls"] else "source_role_prescreen"
    if state["pair"] is not None:
        return "source_selected_alternate"
    if target not in state["options"]:
        return "source_no_anchored_path"
    return "source_lower_ranked_link"


def oracles(service, row, state, scores, phrases):
    index = service.index
    target = tuple(row["papers"])
    gold_roles = [{role} for role in row["roles"]]
    predicted, calls, tokens, _ = role_check(index, target, scores, service.bands, service.role, phrases,
                                              service.cutoffs["role"])
    if state["response"]["source_calls"]:
        matched = required_roles({}, scores, "prototype_band", service.bands)
        matched_calls = 0
    else:
        matched, matched_calls = predicted, calls
    return dict(
        roles_supplied=payload(index, state["pair"], gold_roles),
        sources_supplied_matched_role_budget=payload(index, target, matched, role_calls=matched_calls),
        sources_supplied_with_role_call=payload(index, target, predicted, role_calls=calls, tokens=tokens),
        sources_and_roles_supplied=payload(index, target, gold_roles),
    )


def evaluate(service, rows, wording, reverse):
    index = service.index
    bank = phrase_bank(wording)
    rows = rows[::-1] if reverse else rows
    totals = defaultdict(Counter)
    timing = defaultdict(list)
    stages = defaultdict(Counter)
    failures = defaultdict(Counter)
    factorial = defaultdict(Counter)
    for i, row in enumerate([rows[0], *rows]):
        target = tuple(row["papers"])
        excerpts = [index.articles[p]["clue"] for p in target]
        keywords = [index.keywords(text) for text in excerpts]
        phrases = [bank[role][int(row["family"][:16], 16) % len(bank[role])].strip().rstrip(".?!")
                   for role in row["roles"]]
        for view, clues in (("keywords", keywords), ("excerpt", excerpts)):
            start = perf_counter()
            vectors = service.encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                             [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in phrases])
            scores = {name: vectors[2:] @ service.banks[name].T for name in service.bands}
            embedding_ms = (perf_counter() - start) * 1000
            for status, hidden in zip(("present", "left_hidden", "right_hidden"), (None, *target), strict=True):
                start = perf_counter()
                topic = index.full_rank(vectors[:2], clues, hidden)
                common_ms = embedding_ms + (perf_counter() - start) * 1000
                states = {}
                for route in ROUTES:
                    start = perf_counter()
                    state = plan(service, topic, scores, clues, phrases, hidden, route)
                    took = common_ms + (perf_counter() - start) * 1000
                    states[route] = state
                    if i:
                        key = f"{view}|{status}|{route}"
                        totals[key].update(evidence(state["response"], row),
                                           selected_target=state["pair"] == target,
                                           selected_alternate=state["pair"] is not None and state["pair"] != target)
                        timing[key].append(took)
                baseline = states["screened"]
                if hidden is not None:
                    continue
                requirements = [dict(topic=a, role=b) for a, b in zip(clues, phrases, strict=True)]
                request = SimpleNamespace(payload=json.dumps(dict(requirements=requirements)))
                if json.loads(service.search(request, "screened_one_call")) != baseline["response"]:
                    raise ValueError("Benchmark baseline diverged from the HTTP implementation")
                if not i:
                    continue
                reference = evidence(baseline["response"], row)
                supplied = oracles(service, row, baseline, scores, phrases)
                for arm, response in supplied.items():
                    totals[f"{view}|present|{arm}"].update(evidence(response, row))
                if row["gold"]:
                    stages[view].update(
                        positives=1, both_top6=all(p in s["fused"][:6] for p, s in zip(target, topic, strict=True)),
                        both_top24=all(p in s["rank"] for p, s in zip(target, topic, strict=True)),
                        selected_target=baseline["pair"] == target)
                    failures[view][diagnose(row, topic, baseline)] += 1
                    if not reference["strict"]:
                        source_only = evidence(supplied["sources_supplied_matched_role_budget"], row)["strict"]
                        role_only = evidence(supplied["roles_supplied"], row)["strict"]
                        category = "either" if source_only and role_only else (
                            "source_only" if source_only else "role_only" if role_only else "both_required")
                        factorial[view][category] += 1
    return dict(
        routes={key: dict(totals=dict(value), p95_ms=float(np.quantile(timing[key], .95)) if key in timing else None)
                for key, value in sorted(totals.items())},
        stages={k: dict(v) for k, v in sorted(stages.items())},
        failures={k: dict(v) for k, v in sorted(failures.items())},
        factorial={k: dict(v) for k, v in sorted(factorial.items())},
    )


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    current, current_rows, _ = build(args.current, json.loads(args.current_manifest.read_bytes()))
    earlier, earlier_rows, _ = checked_original(args.earlier, args.earlier_manifest, current)
    engineering, engineering_rows = load_engineering(args.engineering, args.engineering_manifest)
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    source = Reranker(args.source_reader)
    reader = RoleReader(args.role_reader, definitions)
    reader.definitions = ROLES
    frozen = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != frozen["verifier"]["revision"]:
        raise ValueError("Source verifier revision changed")
    cutoffs = dict(source=frozen["thresholds"]["keywords"]["weakest"]["cutoff"],
                   role=json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"])
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    results = {}
    for name, (articles, rows) in (
        ("elife_current", (current, current_rows)), ("elife_earlier", (earlier, earlier_rows)),
        ("engineering", (engineering, engineering_rows)),
    ):
        selected = sample_families(rows)
        service = Service(PostingCatalog(articles, encoder), encoder, banks, bands, source, reader, cutoffs)
        results[name] = evaluate(service, selected, wording, args.reverse)
        results[name]["articles"] = len(articles)
        results[name]["pairs"] = len(selected) // 2
        print(f"{name}: {len(selected) // 2} pairs, {len(articles)} articles", flush=True)
    input_names = ("current", "current_manifest", "earlier", "earlier_manifest", "engineering",
                   "engineering_manifest", "niso", "training", "bands", "source_frozen", "role_frozen")
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), inputs={k: digest(getattr(args, k)) for k in input_names},
        service_sha256=digest(Path(__file__).with_name("budgeted_source_role_service.py")),
        embedding_revision=checkpoint["revision"], source_reader_revision=source.provenance["revision"],
        role_reader_revision=reader.provenance["revision"], gpu=torch.cuda.get_device_name(),
        reverse=args.reverse, cutoffs=cutoffs, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("current", "current_manifest", "earlier", "earlier_manifest", "engineering",
                     "engineering_manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                     "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
