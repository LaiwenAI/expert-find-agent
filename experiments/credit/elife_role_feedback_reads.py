"""Allocate reads to complete a two-paper same-author role proof."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path

import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit.elife_fresh_source_evaluation import top_three_directional
from experiments.credit.elife_prompt_holdout import build, sample_families
from experiments.credit.engineering_cross_source_search import TOPIC_TASK
from experiments.credit.joint_byline_depth import pair_key
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_prospective_service import clusters
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint

DEPTH = 12
BUDGETS = (2, 3, 4)
ARMS = ("round_robin", "static_byline", "role_feedback", "expected_proof")
GUARDS = ("unfiltered", "directional_top_three")


def menus(ranking, byline, guard):
    ordered = [side["fused"][:DEPTH] for side in ranking]
    ranks = [{doc: side["rank"][doc] for doc in row} for side, row in zip(ranking, ordered, strict=True)]
    reads = list(dict.fromkeys(doc for pair in zip(*ordered, strict=True) for doc in pair))
    pairs = {}
    for a, b in product(*ordered):
        if a == b:
            continue
        shared = byline[a] & byline[b]
        if shared and (guard == GUARDS[0] or top_three_directional((a, b), ranking)):
            pairs[a, b] = shared
    pair_order = sorted(pairs, key=lambda p: pair_key(p, ranks))
    return reads, ranks, pairs, pair_order


def next_read(arm, reads, ranks, pairs, pair_order, opened, requirements, remaining):
    if not pair_order or arm == "round_robin":
        return next(doc for doc in reads if doc not in opened)
    first = sorted(pair_order[0], key=lambda doc: (min(side.get(doc, DEPTH) for side in ranks), doc))
    if arm == "static_byline" or (arm == "role_feedback" and not opened):
        return next(doc for doc in first + reads if doc not in opened)

    choices = []
    for pair in pair_order:
        unseen = [i for i, doc in enumerate(pair) if doc not in opened]
        if not 0 < len(unseen) <= remaining:
            continue
        owners = pairs[pair].copy()
        for i, doc in enumerate(pair):
            if doc in opened:
                owners.intersection_update(owner for owner, roles in opened[doc].items() if requirements[i] in roles)
        if not owners:
            continue
        if arm == "expected_proof":
            p = math.prod(1 / (1+ranks[i][pair[i]]) for i in unseen)
            gain = (1 - (1-p)**len(owners)) / len(unseen)
            key = (-gain, *pair_key(pair, ranks))
        else:
            key = (len(unseen), *pair_key(pair, ranks))
        choices.append((key, pair, unseen))
    if choices:
        _, pair, unread = min(choices)
        side = min(unread, key=lambda i: (ranks[i][pair[i]], pair[i]))
        return pair[side]
    return next(doc for doc in reads if doc not in opened)


def schedule(arm, budget, prepared, roles, reveal):
    order, ranks, pairs, pair_order = prepared
    opened = {}
    for n in range(budget):
        doc = next_read(arm, order, ranks, pairs, pair_order, opened, roles, budget-n)
        if doc in opened:
            raise ValueError("Repeated a source instead of spending a new read")
        opened[doc] = reveal(doc)
    for left, right in pair_order:
        if left not in opened or right not in opened:
            continue
        owners = sorted(owner for owner in pairs[left, right]
                        if roles[0] in opened[left][owner] and roles[1] in opened[right][owner])
        if owners:
            return tuple(opened), (left, right), tuple(owners)
    return tuple(opened), None, ()


def evaluate(articles, selected, embedder, reverse):
    catalog = PostingCatalog(articles, embedder)
    byline = {doc: set(row["roles"]) for doc,row in articles.items()}
    groups = clusters(selected, articles)
    cases = defaultdict(dict)
    identities = {}
    families = sorted({case["family"] for case in selected})
    by_family = defaultdict(list)
    for row in selected:
        by_family[row["family"]].append(row)
    for number, family in enumerate(reversed(families) if reverse else families, 1):
        panel = by_family[family]
        target = tuple(panel[0]["papers"])
        if not all(tuple(row["papers"]) == target for row in panel):
            raise ValueError("Inconsistent paired control")
        excerpts = [articles[doc]["clue"] for doc in target]
        for view, clues in (("keyword", [catalog.keywords(text) for text in excerpts]), ("excerpt", excerpts)):
            vectors = embedder.encode([f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues])
            for withheld in (None, *target):
                ranking = catalog.full_rank(vectors, clues, withheld)
                availability = "both_present" if withheld is None else "one_withheld"
                for guard in GUARDS:
                    prepared = menus(ranking, byline, guard)
                    _, _, menu, _ = prepared
                    if withheld is not None and any(withheld in doc for doc in menu):
                        raise ValueError("Withheld source found in candidate menus")
                    for row in panel:
                        key = family, view, row["variant"], withheld or "", guard
                        identities[key] = dict(eligible_exact_pair=target in menu, positive=bool(row["gold"]),
                                               availability=availability, group=groups[family])
                        for budget in BUDGETS:
                            for arm in ARMS:
                                # Only this callback reveals a read article's declared roles.
                                observed, pair, persons = schedule(
                                    arm, budget, prepared, row["roles"], lambda d: articles[d]["roles"])
                                if withheld in observed:
                                    raise ValueError("Withheld article was read")
                                strict = pair == target and bool(set(persons) & set(row["gold"]))
                                cases[key][budget, arm] = dict(
                                    reads=observed, issued=pair is not None, exact_pair=set(target) <= set(observed),
                                    strict=strict, adverse=bool(pair) and not strict,
                                )
        if number % 20 == 0:
            print(number, "matched families scored", flush=True)

    totals = defaultdict(Counter)
    opportunity = defaultdict(Counter)
    for key, measures in sorted(cases.items()):
        _, view, _, _, guard = key
        metadata = identities[key]
        scope = metadata["availability"], view, guard
        opportunity[scope].update(cases=1, positives=metadata["positive"],
                                  eligible_positive=metadata["positive"] and metadata["eligible_exact_pair"])
        for (budget, arm), observed in measures.items():
            totals[*scope, str(budget), arm].update(
                strict=int(observed["strict"]), issued=int(observed["issued"]),
                adverse=int(observed["adverse"]), reads=len(observed["reads"]),
                correct_papers_read=int(observed["exact_pair"]),
            )
    report = {}
    for scope, value in opportunity.items():
        presence, view, guard = scope
        output = report.setdefault(presence, {}).setdefault(view, {}).setdefault(guard, {})
        output["opportunity"] = dict(value)
        for budget in BUDGETS:
            output[str(budget)] = {arm: dict(totals[*scope, str(budget), arm]) for arm in ARMS}
            if presence == "both_present":
                positive = [k for k,v in sorted(identities.items()) if k[1] == view and k[-1] == guard
                            and v["availability"] == presence and v["positive"]]
                for arm in ARMS[1:]:
                    diff = [int(cases[k][budget,arm]["strict"])-int(cases[k][budget,ARMS[1]]["strict"])
                            for k in positive]
                    output[str(budget)][arm]["delta_vs_static_byline"] = clustered_paired_interval(
                        diff, [identities[k]["group"] for k in positive])
    ordered = [(key, [(b, a, case[b,a]) for b in BUDGETS for a in ARMS]) for key,case in sorted(cases.items())]
    return dict(families=len(families), groups=len(set(groups.values())), report=report,
                signature=sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest())


def run(projection, manifest, embedding, output, reverse):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a Slurm CUDA allocation")
    frozen = json.loads(manifest.read_bytes())
    articles, raw, census = build(projection, frozen)
    if census != frozen["panel"]:
        raise ValueError("Publisher projection differs from its frozen panel")
    selected = sample_families(raw)
    verify = verify_checkpoint(embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = evaluate(articles, selected, QwenTorchEmbedding(embedding), reverse)
    write_json(output, dict(result, reverse=reverse, manifest_sha256=digest(manifest),
        projection_sha256=digest(projection), runner_sha256=digest(Path(__file__)),
        slurm_job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        encoder=verify["revision"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "embedding", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(**vars(cli.parse_args()))
