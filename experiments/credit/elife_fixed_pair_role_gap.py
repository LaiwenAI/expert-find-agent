"""Measure role recognition with frozen two-read source decisions."""

import argparse
from collections import Counter, defaultdict
from copy import copy
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.compact_role_prompts import COMPACT
from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.elife_role_feedback_reads import GUARDS, menus
from experiments.credit.engineering_cross_source_search import TOPIC_TASK, candidates, phrase_bank
from experiments.credit.engineering_role_banks import encode, projections, queries
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_prospective_service import FROZEN_SHA, clusters
from experiments.credit.paired_role_verification import PairedReader, alternative_roles
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint

ARMS = ("exact_role_reference", "ridge_top_one", "prototype_top_one", "teacher_top_three", "teacher_all_roles")


MENU_ARMS = ("teacher_weak_slot", "teacher_compact_all")


def recognize_menus(scores, pools, reader, compact_reader):
    # Compare alternatives independently of the two source declarations.
    # The weak slot is where the preferred shortlist role has the least
    # separation from the runner-up across the two embedding projections.
    normalized = []
    for values in scores.values():
        centered = values - values.mean(axis=1, keepdims=True)
        normalized.append(centered / values.std(axis=1, keepdims=True).clip(.001))
    ranked = np.sort(sum(normalized), axis=1)
    slot = int(np.argmin(ranked[:, -1] - ranked[:, -2]))
    weak_options = pools.copy()
    weak_options[slot] = ROLES
    return {
        "teacher_weak_slot": (reader, weak_options),
        "teacher_compact_all": (compact_reader, [ROLES, ROLES]),
    }


def judge(catalog, row, pair, labels):
    if labels is None or not pair:
        return False, False
    owners = candidates(catalog, pair, [{label} for label in labels])
    strict = (tuple(pair) == tuple(row["papers"]) and tuple(labels) == tuple(row["roles"])
              and bool(set(owners) & set(row["gold"])))
    return bool(owners), strict


def recognize(row, bank, embedder, role_model, glossary, cutoff, compact_reader):
    phrases = [glossary[role][int(row["family"][:16],16) % len(glossary[role])] for role in row["roles"]]
    features = encode(embedder, phrases)
    scores = {name: features @ bank[name].T for name in ("ridge","prototype")}
    pools = alternative_roles(scores)
    outputs = {"exact_role_reference": (row["roles"],0,0)}
    for kind in ("ridge","prototype"):
        outputs[f"{kind}_top_one"] = ([ROLES[int(a)] for a in scores[kind].argmax(axis=1)],0,0)
    teacher_routes = {
        "teacher_top_three": (role_model, pools),
        "teacher_all_roles": (role_model, [ROLES, ROLES]),
    }
    if compact_reader is not None:
        teacher_routes.update(recognize_menus(scores, pools, role_model, compact_reader))
    unfiltered = {}
    for name, (reader, options) in teacher_routes.items():
        labels, margin, tokens = reader.batched(phrases, options)
        unfiltered[name] = labels
        outputs[name] = (labels if margin > cutoff and None not in labels else None, 1, tokens)
    reachable = {name: all(role in allowed for role, allowed in zip(row["roles"], options, strict=True))
                 for name, (_, options) in teacher_routes.items()}
    return outputs, reachable, unfiltered


def run(projection, manifest, embedding, role_checkpoint, definition_file, training, paired_reference, output, reverse, menu_ablation=False):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require Slurm CUDA and a fresh output")
    if digest(paired_reference) != FROZEN_SHA:
        raise ValueError("Role configuration is not from the frozen paired benchmark")
    reference = json.loads(paired_reference.read_bytes())["runs"][0]
    if any(digest(path) != reference["inputs"][key] for key,path in (("niso",definition_file),("training",training))):
        raise ValueError("Changed role glossary or training phrases")
    frozen = json.loads(manifest.read_bytes())
    articles, rows, census = build(projection, frozen)
    if census != frozen["panel"]:
        raise ValueError("Changed publisher roles or projection")
    selected = sample_families(rows)
    groups = clusters(selected, articles)
    if sorted(set(row["family"] for row in selected)) != frozen["chosen_families"]:
        raise ValueError("Changed result families")
    definitions, phrases, wording = queries(definition_file, training)
    embed_rev = verify_checkpoint(embedding)["revision"]
    if embed_rev != reference["embedding_revision"]:
        raise ValueError("Changed frozen embedding revision")
    torch.backends.cuda.matmul.allow_tf32 = False
    embedder = QwenTorchEmbedding(embedding)
    banks = projections(embedder, definitions, phrases)
    catalog = PostingCatalog(articles, embedder)
    model = PairedReader(role_checkpoint, ROLES)
    if model.provenance["revision"] != reference["role_reader_revision"]:
        raise ValueError("Changed role reader revision")
    compact_reader = None
    arms = ARMS
    if menu_ablation:
        compact_reader = copy(model)
        compact_reader.definitions = [f"{role}: {COMPACT[role]}" for role in ROLES]
        arms += MENU_ARMS
    cutoff = reference["fit"]["two_batched"]["cutoff"]
    glossary = phrase_bank(wording)
    byline = {doc:set(r["roles"]) for doc,r in articles.items()}
    totals = defaultdict(Counter)
    records = {}
    for i,row in enumerate(reversed(selected) if reverse else selected,1):
        target = tuple(row["papers"])
        texts = [articles[doc]["clue"] for doc in target]
        decoded = None
        for view,clues in (("keyword",[catalog.keywords(t) for t in texts]),("excerpt",texts)):
            points = embedder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues])
            for mask,withheld in enumerate((None,*target)):
                state = "one_withheld" if withheld else "both_present"
                ranking = catalog.full_rank(points, clues, withheld)
                pair_order = menus(ranking,byline,GUARDS[1])[3]
                pair = pair_order[0] if pair_order else None
                if pair and withheld in pair:
                    raise ValueError("Withheld document entered read pair")
                if pair and decoded is None:
                    decoded, reachable, unfiltered = recognize(row, banks, embedder, model, glossary, cutoff, compact_reader)
                exact_view = bool(pair) and tuple(pair)==target
                key = row["family"], row["variant"], view, mask
                records[key] = {}
                for arm in arms:
                    predictions, forwards, tokens = decoded[arm] if pair else (None,0,0)
                    issued, strict = judge(catalog,row,pair,predictions)
                    stats = dict(issued=issued,strict=strict,adverse=issued and not strict,
                                 source_reads=2*bool(pair),forwards=forwards,tokens=tokens,
                                 exact_source_menu_pair=exact_view,role_correct=bool(pair) and tuple(predictions or ())==tuple(row["roles"]),
                                 top_three_slots_reachable=bool(pair) and reachable["teacher_top_three"])
                    if menu_ablation:
                        stats["exact_proof_reachable_in_menu"] = (
                            exact_view and row["variant"] == "present" and reachable.get(arm, True)
                        )
                        positive_exact_pair = exact_view and row["variant"] == "present"
                        stats["positive_role_parsed"] = positive_exact_pair and stats["role_correct"]
                        stats["positive_raw_role_parsed"] = positive_exact_pair and (
                            tuple(unfiltered.get(arm, predictions) or ()) == tuple(row["roles"])
                        )
                    records[key][arm] = stats
                    totals[state,view,arm].update({k:int(v) for k,v in stats.items()})
        if i % 40 == 0:
            print(i,"structured positive or swap queries",flush=True)
    report = defaultdict(dict)
    for (state,view,arm), values in totals.items():
        report[state].setdefault(view,{})[arm] = dict(values)
        if state=="both_present":
            keys = [key for key in sorted(records) if key[1:] == ("present",view,0)]
            report[state][view][arm]["paired_delta_vs_ridge"] = clustered_paired_interval([
                int(records[key][arm]["strict"]) - int(records[key]["ridge_top_one"]["strict"])
                for key in keys], [groups[key[0]] for key in keys])
    write_json(output, dict(report=report, gpu=torch.cuda.get_device_name(), reverse=reverse,
        authorship_groups=len(set(groups.values())),
        cases=len(selected), encoder=embed_rev, role_checkpoint=model.provenance["revision"],
        manifest_sha256=digest(manifest), projection_sha256=digest(projection),
        runner_sha256=digest(Path(__file__)), frozen_reference_sha256=digest(paired_reference),
        slurm_job=os.environ["SLURM_JOB_ID"],
        signature=sha256(json.dumps(sorted(records.items()),sort_keys=True).encode()).hexdigest()))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection","manifest","embedding","role_checkpoint","definition_file","training","paired_reference","output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    cli.add_argument("--menu-ablation", action="store_true")
    run(**vars(cli.parse_args()))
