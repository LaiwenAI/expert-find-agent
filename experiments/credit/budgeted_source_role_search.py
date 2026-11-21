"""Allocate a shared first-query verification budget to source and role checks."""

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
    AVAILABILITY, TOPIC_TASK, candidates, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_one_role_recheck import RoleReader, replace
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.independent_source_confirmation import weaker_side
from experiments.credit.keyword_author_rescue import UNGATED, PostingCatalog, at_gate
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_issuance_audit import outcome
from experiments.credit.source_confirmation_transfer import VIEWS, render, verify
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("static", "source_only", "role_only", "both", "one_call", "screened_one_call")


def role_check(index, pair, scores, bands, reader, phrases, cutoff):
    start = perf_counter()
    scope = {"papers": pair, "roles": (None, None)}
    sets, _, slot, approved = route(index.articles, scope, scores, bands)
    if not approved:
        return sets, 0, 0, (perf_counter() - start) * 1000
    role, gap, tokens = reader.read(phrases[slot], sets[slot])
    if role in approved and gap > cutoff:
        sets = replace(sets, slot, role)
    return sets, 1, tokens, (perf_counter() - start) * 1000


def check_source(index, source, ranked, proposal, clues, source_cutoff):
    start = perf_counter()
    slot = weaker_side(ranked, proposal)
    score = verify(source, index, clues, proposal, (slot,))
    return (proposal if score > source_cutoff else None), (perf_counter() - start) * 1000


def evaluate(articles, rows, encoder, banks, wording, bands, source_reader, role_reader, cutoffs, reverse):
    index = PostingCatalog(articles, encoder)
    phrase_table = phrase_bank(wording)
    counts, timing = defaultdict(Counter), defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        target = tuple(row["papers"])
        excerpts = [articles[doc]["clue"] for doc in target]
        phrases = [phrase_table[role][int(row["family"][:16], 16) % len(phrase_table[role])] for role in row["roles"]]
        for view in VIEWS:
            clues = [render(index, raw, view) for raw in excerpts]
            start = perf_counter()
            vec = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues] +
                                 [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in phrases])
            scores = {name: vec[2:] @ banks[name].T for name in bands}
            prototype = required_roles(row, scores, "prototype_band", bands)
            embedding_ms = (perf_counter() - start) * 1000
            for availability, masked in zip(AVAILABILITY, (None, *target), strict=True):
                start = perf_counter()
                ranked = index.full_rank(vec[:2], clues, masked)
                baseline, _ = select(index, [s | {"fused": s["fused"][:6]} for s in ranked],
                                     {}, [set(), set()], "anchored", masked)
                common_ms = embedding_ms + (perf_counter() - start) * 1000
                results = {}
                if baseline:
                    roles, role_calls, tokens, role_ms = role_check(index, baseline, scores, bands, role_reader, phrases, cutoffs["role"])
                    for arm in ARMS:
                        use_role = arm not in ("static", "source_only")
                        results[arm] = (baseline, roles if use_role else prototype,
                                        0, role_calls if use_role else 0, tokens if use_role else 0,
                                        common_ms + (role_ms if use_role else 0.))
                else:
                    start = perf_counter()
                    proposal = at_gate(index.linked_options(ranked)[0], *UNGATED)
                    proposal_ms = (perf_counter() - start) * 1000
                    results = {arm: (None, prototype, 0, 0, 0, common_ms) for arm in ARMS}
                    if proposal:
                        start = perf_counter()
                        feasible = bool(candidates(index, proposal, prototype))
                        screening_ms = (perf_counter() - start) * 1000
                        accepted, source_ms = check_source(index, source_reader, ranked, proposal, clues, cutoffs["source"])
                        extra = common_ms + proposal_ms + source_ms
                        results["source_only"] = results["one_call"] = (accepted, prototype, 1, 0, 0, extra)
                        results["screened_one_call"] = (
                            accepted if feasible else None, prototype, int(feasible), 0, 0,
                            common_ms + proposal_ms + screening_ms + (source_ms if feasible else 0.))
                        if accepted:
                            roles, calls, tokens, role_ms = role_check(index, accepted, scores, bands, role_reader, phrases, cutoffs["role"])
                            results["both"] = accepted, roles, 1, calls, tokens, extra + role_ms
                        else:
                            results["both"] = None, prototype, 1, 0, 0, extra
                    else:
                        for arm in ("source_only", "both", "one_call", "screened_one_call"):
                            results[arm] = (None, prototype, 0, 0, 0, common_ms + proposal_ms)

                outcomes = {}
                for arm, (pair, roles, source_calls, role_calls, tokens, ms) in results.items():
                    start = perf_counter()
                    result, _ = outcome(index, row, pair, roles, "final", None)
                    local_ms = ms + (perf_counter() - start) * 1000
                    outcomes[arm] = result
                    if i:
                        key = "|".join((view, availability, arm))
                        counts[key].update(
                            result, source_calls=source_calls, role_calls=role_calls, role_input_tokens=tokens,
                            requests=1, over_one_call=source_calls + role_calls > 1,
                            correct_pair=pair == target, source_only_paper=bool(pair and not baseline))
                        timing[key].append(local_ms)
                if i:
                    emissions = []
                    for arm in ("one_call", "screened_one_call"):
                        pair, roles = results[arm][:2]
                        owners = candidates(index, pair, roles) if pair else []
                        emissions.append((pair, roles, owners) if owners else None)
                    if emissions[0] != emissions[1] or outcomes["one_call"] != outcomes["screened_one_call"]:
                        raise ValueError("Proof screening changed an evidence outcome")
                    if any(sum(results[arm][2:4]) > 1 for arm in ("one_call", "screened_one_call")):
                        raise ValueError("Exceeded one-call verifier limit")
    return {key: dict(totals=dict(value), mean_ms=float(np.mean(timing[key])),
                      p95_ms=float(np.quantile(timing[key], .95))) for key, value in sorted(counts.items())}


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    model = verify_checkpoint(args.embedding)
    source = Reranker(args.source_reader)
    role = RoleReader(args.role_reader, definitions)
    role.definitions = ROLES
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    frozen = json.loads(args.source_frozen.read_bytes())["runs"][0]
    if source.provenance["revision"] != frozen["verifier"]["revision"]:
        raise ValueError("Frozen source verifier revision changed")
    margin = json.loads(args.role_frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    cutoffs = {"source": frozen["thresholds"]["keywords"]["weakest"]["cutoff"], "role": margin}
    articles, cases, census = build(args.elife, json.loads(args.manifest.read_bytes()))
    eng = json.loads(args.engineering.read_bytes())
    cohorts = {"elife": (articles, cases), "engineering": (eng["articles"], eng["queries"])}
    results = {name: evaluate(*panel, encoder, banks, wording, bands, source, role, cutoffs, args.reverse)
               for name, panel in cohorts.items()}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), elife_sha256=digest(args.elife), engineering_sha256=digest(args.engineering),
        source_frozen_sha256=digest(args.source_frozen), role_frozen_sha256=digest(args.role_frozen),
        embedding_revision=model["revision"], source_reader_revision=source.provenance["revision"],
        role_reader_revision=role.provenance["revision"], cutoffs=cutoffs, elife_panel=census,
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("elife", "engineering", "manifest", "niso", "training", "bands", "source_frozen", "role_frozen",
                "embedding", "source_reader", "role_reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
