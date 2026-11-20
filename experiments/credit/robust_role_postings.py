"""Stress-test fixed-budget joint nomination under missing and misassigned roles."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import joint_nomination_blind_review as audit
from experiments.credit import live_joint_role_postings as typed
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


SCENARIOS = ((25, 0), (50, 0), (50, 20), (100, 0), (100, 20))
MASKS = ("", "second-mask", "third-mask")
METHODS = ("typed", "known_pair", "competitive_pair")


def bucket(label, percent):
    number = int(sha256(label.encode()).hexdigest()[:8], 16)
    return number < percent * (1 << 32) // 100


def project(postings, catalogue, coverage, corruption, mask):
    owners = {}
    exposed = sampled = corrupted = 0
    for doi in catalogue.ids:
        if not bucket(mask + doi, coverage):
            continue
        original = postings.personal[doi]
        exposed += 1
        if not original:
            continue
        people = sorted(p for p, duties in original.items() if duties)
        if len(people) > 1 and bucket("corruption:" + mask + doi, corruption):
            sampled += 1
            changed = dict(original)
            for person, previous in zip(people, people[-1:] + people[:-1], strict=True):
                changed[person] = original[previous]
            corrupted += changed != original
            owners[doi] = changed
        else:
            owners[doi] = original
    return owners, dict(visible_sources=exposed, sampled_for_rotation=sampled, changed_sources=corrupted)


def facets_for(rankings, atoms, byline, visible):
    facets = []
    for ranking, (_, role) in zip(rankings, atoms, strict=True):
        supports = {}
        for doi in ranking:
            people = visible.get(doi)
            if people is None:
                supports[doi] = set(), byline[doi]
                continue
            supports[doi] = ({p for p, roles in people.items() if role in roles},
                             {p for p, roles in people.items() if not roles})
        facets.append(supports)
    return facets


def anchors_confirmed(plan, facets):
    a, b = plan[:2]
    return a in facets[0] and b in facets[1] and not facets[0][a][0].isdisjoint(facets[1][b][0])


def competitive(plan, baseline, ranks):
    def weakest(pair):
        return max(ranks[i].get(doi, live.DEPTH + 1) for i, doi in enumerate(pair[:2]))

    return weakest(plan) <= max(8, 2 * weakest(baseline))


def verify(index, plan, atoms, ranks, subjects):
    loaded = index.open_sources(plan)
    if len(plan) != 3 or set(plan) != loaded.keys():
        raise ValueError("Expected three actual distinct declaration reads")
    evidence = SimpleNamespace(ownership=loaded)
    candidates = live.BASE.choose_proofs(live.BASE.read_articles(evidence, plan, atoms, ranks), ranks)
    match = candidates[0] if candidates else None
    anchor_has_proof = any(candidate[1:] == tuple(plan[:2]) for candidate in candidates)
    if match is None:
        return dict(match=None, known=False, priority=None), anchor_has_proof
    positions = [ranks[i][doi] for i, doi in enumerate(match[1:])]
    known = all(topic in subjects[doi] for (topic, _), doi in zip(atoms, match[1:], strict=True))
    return dict(match=match, known=known, priority=(max(positions), sum(positions))), anchor_has_proof


def compare(rows):
    outcomes = {}
    for method, estimates in rows.items():
        policies = audit.select(estimates)
        for policy, masks in policies.items():
            stats = audit.summarize(estimates, masks)
            if policy == "natural":
                outcomes[method] = {}
            outcomes[method][policy] = dict(
                control=stats["arms"]["owner_gap"], proposed=stats["arms"]["joint"],
                common_requests=stats["both_issue"], overlap_changes=stats["overlap_changes"],
            )
    return outcomes


def evaluate(index, subjects, cases, offset):
    postings = typed.RolePostings(index.catalogue)
    bank = []
    planning = []
    for atoms in cases:
        ranks = index.rank([topic for topic, _ in atoms])
        initial, extras = live.simple_order(ranks)
        fallback = [*initial, extras[0]]
        start = perf_counter_ns()
        baseline = live.joint.posting_plan(postings.byline, set().union(*ranks), ranks, fallback, live.DEPTH)
        planning.append((perf_counter_ns() - start) / 1e6)
        record, _ = verify(index, baseline, atoms, ranks, subjects)
        bank.append((atoms, ranks, fallback, baseline, record))
    outputs = {}
    for coverage, corruption in SCENARIOS:
        name = f"{coverage}-{corruption}"
        outputs[name] = []
        for salt in MASKS:
            rows = {method: [] for method in METHODS}
            choices = {method: Counter() for method in METHODS}
            typed_ms = []
            visible, counts = project(postings, index.catalogue, coverage, corruption, salt)
            for j, (atoms, ranks, fallback, baseline, control) in enumerate(bank):
                began = perf_counter_ns()
                facets = facets_for(ranks, atoms, postings.byline, visible)
                typed_plan = typed.source_plan(ranks, facets, fallback, live.DEPTH)
                exact = anchors_confirmed(typed_plan, facets)
                good_rank = exact and competitive(typed_plan, baseline, ranks)
                typed_ms.append((perf_counter_ns() - began) / 1e6)
                selected = dict(typed=True, known_pair=exact, competitive_pair=good_rank)
                rotation = (j + offset) % len(METHODS)
                for method in METHODS[rotation:] + METHODS[:rotation]:
                    adopt = selected[method]
                    plan = typed_plan if adopt else baseline
                    record, anchor_true = verify(index, plan, atoms, ranks, subjects)
                    choices[method].update(
                        typed_adoptions=int(adopt), changed_plans=int(plan != baseline),
                        indexed_pairs_without_original_proof=int(adopt and exact and not anchor_true),
                        lost_known=int(control["known"] and not record["known"]),
                        gained_known=int(record["known"] and not control["known"]),
                    )
                    rows[method].append(dict(
                        atoms=atoms, topic_pair=json.dumps([topic for topic, _ in atoms]),
                        arms=dict(owner_gap=control, joint=record),
                    ))
            outputs[name].append(dict(
                salt=salt, projection=counts, comparisons=compare(rows), choices=choices,
                typed_planning_p95_ms=float(np.quantile(typed_ms, .95)),
            ))
    return dict(
        queries=len(cases), byline_known=sum(row[-1]["known"] for row in bank),
        byline_issued=sum(row[-1]["match"] is not None for row in bank),
        byline_planning_p95_ms=float(np.quantile(planning, .95)), scenarios=outputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on a Slurm GPU")
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.manifest) != expected["elife_manifest_sha256"] or live.DEPTH != expected["query_depth"]
    ):
        raise ValueError("Frozen inputs differ from reference")
    plos_subjects = base.publisher_topics([args.plos_xml])
    plos_cases = base.query_panel(base.annotation_postings(plos, plos_subjects))
    life_subjects, _, life_cases = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_subjects, plos_cases), elife=(life, life_subjects, life_cases))
    encoder = QwenTorchEmbedding(args.embedding)
    results = {}
    with TemporaryDirectory() as scratch:
        for name, (catalogue, subjects, cases) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / name)
            panel = evaluate(index, subjects, cases, args.order_offset)
            control = expected["panels"][name]["arms"]["structured"]["joint"]
            if (panel["byline_known"], panel["byline_issued"]) != (control["known"], control["issued"]):
                raise ValueError(f"Byline replay changed for {name}")
            results[name] = panel
    write_json(args.output, dict(
        order_offset=args.order_offset, gpu=torch.cuda.get_device_name(), embedding_checkpoint=checkpoint,
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)), panels=results,
    ))


if __name__ == "__main__":
    main()
