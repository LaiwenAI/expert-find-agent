"""Audit one-role headroom and choose viable slots before running the reader."""

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
from experiments.credit.contrastive_role_recheck import shortlist
from experiments.credit.distilled_role_recheck import student_values
from experiments.credit.elife_one_role_transfer import build as development
from experiments.credit.elife_prompt_holdout import build as holdout
from experiments.credit.engineering_cross_source_search import (
    TOPIC_TASK, Catalog, candidates, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_joint_calibration import SUPPORT
from experiments.credit.engineering_one_role_recheck import pick, replace, state, RoleReader
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.role_issuance_audit import outcome
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("original", "agreement_first", "weighted_agreement")


def approvals(catalog, scope, role_sets, branches, scores, bands):
    prototype = required_roles(scope, scores, "prototype_band", bands)
    return {
        slot: {role for role in role_sets[slot] & prototype[slot]
               if candidates(catalog, scope["papers"], replace(role_sets, slot, role))}
        for slot in branches if branches[slot]
    }


def pick_slot(role_sets, allowed, scores, weighted):
    priority = {}
    for slot, roles in allowed.items():
        if not roles:
            continue
        if weighted:
            values = student_values("dense_agreement", None, scores, {}, slot)
            indices = [SUPPORT[role] for role in sorted(role_sets[slot])]
            weights = np.exp(values[indices] - max(values[indices]))
            total = sum(weights[i] for i, idx in enumerate(indices) if ROLES[idx] in roles) / weights.sum()
        else:
            total = len(roles) / len(role_sets[slot])
        gaps = sorted(scores["ridge"][slot], reverse=True)
        priority[slot] = total, -(gaps[0] - gaps[1]), -slot
    return max(priority, key=priority.get, default=None)


def headroom(catalog, row, pair, sets, allowed, slot, scores):
    if not row["gold"]:
        return {}
    strict_baseline, _ = outcome(catalog, row, pair, sets, "initial", None)
    counts = dict(positive=1, source_pair=tuple(row["papers"]) == pair, initial=strict_baseline["proof"])
    for key in ("pair_choice_oracle", "chosen_slot_oracle", "any_slot_oracle", "expanded_one_oracle", "both_roles_oracle"):
        counts[key] = False
    if not counts["source_pair"]:
        return counts
    intended = row["roles"]

    def exact(role_sets):
        result, _ = outcome(catalog, row, pair, role_sets, "diagnostic", None)
        return result["proof"]

    initial = counts["initial"]
    chosen = slot is not None and intended[slot] in allowed.get(slot, set()) and exact(replace(sets, slot, intended[slot]))
    counts["pair_choice_oracle"] = initial or (chosen and intended[slot] in shortlist(scores, slot, sets[slot]))
    counts["chosen_slot_oracle"] = initial or chosen
    counts["any_slot_oracle"] = initial or any(
        intended[s] in roles and exact(replace(sets, s, intended[s])) for s, roles in allowed.items())
    counts["expanded_one_oracle"] = initial or any(exact(replace(sets, s, intended[s])) for s in range(2))
    counts["both_roles_oracle"] = exact([{role} for role in intended])
    chain = [counts[key] for key in ("initial", "pair_choice_oracle", "chosen_slot_oracle", "any_slot_oracle",
                                    "expanded_one_oracle", "both_roles_oracle", "source_pair")]
    if chain != sorted(chain) or not counts["both_roles_oracle"]:
        raise ValueError("Inconsistent nested counterfactual")
    return counts


def evaluate(articles, rows, encoder, banks, wording, reader, bands, cutoff, reverse):
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    counts, opportunities, times = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    if reverse:
        rows = rows[::-1]
    for i, row in enumerate([rows[0], *rows]):
        excerpts = [articles[p]["clue"] for p in row["papers"]]
        text = [phrases[role][int(row["family"][:16], 16) % len(phrases[role])] for role in row["roles"]]
        for view, clues in (("excerpt", excerpts), ("keywords", [catalog.keywords(clue) for clue in excerpts])):
            start = perf_counter()
            vectors = encoder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{c}" for c in clues] +
                                     [f"Instruct: {TASK}\nQuery:{t}" for t in text])
            scores = {name: vectors[2:] @ banks[name].T for name in bands}
            topic = catalog.rank(vectors[:2], clues, None)
            pair, _ = select(catalog, topic, {}, [set(), set()], "anchored", None)
            scope = {"papers": pair, "roles": (None, None)}
            sets, branches = [set(), set()], {}
            if pair is not None:
                sets, _, branches = state(articles, scope, scores["ridge"], bands)
            common_ms = (perf_counter() - start) * 1000

            start = perf_counter()
            slot = pick(branches, scores["ridge"])["proof_aware_gated"]
            prior_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            approved = approvals(catalog, scope, sets, branches, scores, bands) if branches else {}
            approval_ms = (perf_counter() - start) * 1000
            chosen = {"original": (slot if approved.get(slot) else None, prior_ms)}
            for arm, weighted in (("agreement_first", False), ("weighted_agreement", True)):
                start = perf_counter()
                selected = pick_slot(sets, approved, scores, weighted)
                chosen[arm] = selected, (perf_counter() - start) * 1000

            if i:
                opportunities[view].update(headroom(catalog, row, pair, sets, approved, slot, scores))
            reads = {}
            for arm in ARMS:
                selected, pick_ms = chosen[arm]
                if selected is not None and selected not in reads:
                    start = perf_counter()
                    answer = reader.read(text[selected], sets[selected])
                    reads[selected] = answer, (perf_counter() - start) * 1000
                start = perf_counter()
                current, read_ms, tokens = sets, 0., 0
                if selected is not None:
                    (role, gap, tokens), read_ms = reads[selected]
                    if role in approved[selected] and gap > cutoff:
                        current = replace(sets, selected, role)
                result, _ = outcome(catalog, row, pair, current, "report", None)
                elapsed = common_ms + approval_ms + pick_ms + read_ms + (perf_counter() - start) * 1000
                if i:
                    key = "|".join((view, arm))
                    counts[key].update(result, calls=int(selected is not None), tokens=tokens,
                                       changed_slots=selected != chosen["original"][0], requests=1)
                    times[key].append(elapsed)
    return dict(opportunities={key: dict(value) for key, value in opportunities.items()},
                outcomes={key: dict(totals=dict(value), p95_ms=float(np.quantile(times[key], .95)))
                          for key, value in sorted(counts.items())})


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    definitions, training, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, definitions, training)
    full = RoleReader(args.reader, definitions)
    reader = copy(full)
    reader.definitions = ROLES
    bands = json.loads(args.bands.read_bytes())["runs"][0]["bands"]
    cutoff = json.loads(args.frozen.read_bytes())["runs"][0]["fit"]["cutoff"]
    dev = development(args.development)
    test = holdout(args.holdout, json.loads(args.manifest.read_bytes()))
    results = {key: evaluate(articles, cases, encoder, banks, wording, reader, bands, cutoff, args.reverse)
               for key, (articles, cases, _) in (("development", dev), ("held", test))}
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), dev_sha256=digest(args.development), holdout_sha256=digest(args.holdout),
        embedding_revision=checkpoint["revision"], reader_revision=full.provenance["revision"],
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "holdout", "manifest", "niso", "training", "bands", "frozen", "embedding", "reader", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
