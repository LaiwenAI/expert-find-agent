"""Test author-bound role postings under the same live three-source budget."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import product
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
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


ARM_CONFIGURATION = {
    "owner_gap": (0, False),
    "byline": (0, False),
    "article_roles": (100, False),
    "person_roles_25": (25, True),
    "person_roles_50": (50, True),
    "person_roles_100": (100, True),
}


class RolePostings:
    def __init__(self, catalogue):
        self.byline = catalogue.byline
        self.personal = {}
        self.article = {}
        for doi, source in catalogue.ownership.items():
            people = defaultdict(set)
            roles = set()
            for author in source["authors"]:
                roles.update(author["roles"])
                if author["person"]:
                    people[author["person"]].update(author["roles"])
            self.personal[doi] = people
            self.article[doi] = roles
        self.order = {doi: int(sha256(doi.encode()).hexdigest()[:8], 16) for doi in self.byline}

    def facet(self, ranking, role, coverage, personal):
        cutoff = coverage * (1 << 32) // 100
        supports = {}
        for doi in ranking:
            if self.order[doi] >= cutoff:
                supports[doi] = (set(), self.byline[doi])
            elif personal:
                authors = self.personal[doi]
                supports[doi] = ({p for p, duties in authors.items() if role in duties},
                                 {p for p, duties in authors.items() if not duties})
            else:
                duties = self.article[doi]
                supports[doi] = (self.byline[doi] if role in duties else set(),
                                 set() if duties else self.byline[doi])
        return supports

    def inventory(self):
        return {
            str(coverage): dict(
                indexed_sources=sum(number < coverage * (1 << 32) // 100 for number in self.order.values()),
                person_role_postings=sum(
                    len(roles) for doi, people in self.personal.items()
                    if self.order[doi] < coverage * (1 << 32) // 100 for roles in people.values()
                ),
            )
            for coverage in (25, 50, 100)
        }


def first_pair(ranks, facets):
    owners = defaultdict(lambda: ([], []))
    for side, sources in enumerate(facets):
        for doi, (matched, unknown) in sources.items():
            for known, people in ((True, matched), (False, unknown)):
                for person in people:
                    owners[person][side].append((not known, ranks[side][doi], doi))

    def pair_key(pair):
        (u, a, d), (v, b, e) = pair
        return u + v, max(a, b), a + b, d, e

    return min(
        (pair for left, right in owners.values() for pair in product(sorted(left)[:2], sorted(right)[:2])
         if pair[0][2] != pair[1][2]),
        key=pair_key, default=None,
    )


def source_plan(ranks, facets, fallback, depth):
    first = first_pair(ranks, facets)
    if first is None:
        return fallback
    anchors = [entry[2] for entry in first]
    extras = sorted(set().union(*ranks) - set(anchors))
    if not extras:
        return anchors

    def gain(doi):
        exact = possible = 0
        for anchor in anchors:
            for i in range(2):
                if anchor not in facets[i] or doi not in facets[1-i]:
                    continue
                anchor_known, anchor_unknown = facets[i][anchor]
                extra_known, extra_unknown = facets[1-i][doi]
                exact += not anchor_known.isdisjoint(extra_known)
                possible += not (anchor_known | anchor_unknown).isdisjoint(extra_known | extra_unknown)
        return exact, possible, -min(side.get(doi, depth+1) for side in ranks)

    return [*anchors, max(extras, key=gain)]


def choose(index, postings, atoms, ranks, arm):
    if arm in ("owner_gap", "byline"):
        return live.inspect(index, atoms, ranks, "owner_gap" if arm == "owner_gap" else "joint")
    coverage, personal = ARM_CONFIGURATION[arm]
    facets = tuple(postings.facet(ranking, role, coverage, personal)
                   for ranking, (_, role) in zip(ranks, atoms, strict=True))
    initial, extras = live.simple_order(ranks)
    plan = source_plan(ranks, facets, [*initial, extras[0]], live.DEPTH)
    return plan, index.open_sources(plan)


def evaluate(index, subjects, queries, offset):
    started = perf_counter_ns()
    postings = RolePostings(index.catalogue)
    build_ms = (perf_counter_ns() - started) / 1e6
    rows = []
    times = defaultdict(list)
    names = list(ARM_CONFIGURATION)
    for j, atoms in enumerate(queries):
        ranks = index.rank([topic for topic, _ in atoms])
        row = dict(atoms=atoms, topic_pair=json.dumps([topic for topic, _ in atoms]), arms={})
        rotation = (j + offset) % len(names)
        for arm in names[rotation:] + names[:rotation]:
            start = perf_counter_ns()
            plan, loaded = choose(index, postings, atoms, ranks, arm)
            candidates = live.BASE.choose_proofs(
                live.BASE.read_articles(SimpleNamespace(ownership=loaded), plan, atoms, ranks), ranks,
            )
            times[arm].append((perf_counter_ns() - start) / 1e6)
            if len(plan) != 3 or set(plan) != loaded.keys():
                raise ValueError("The comparison requires exactly three distinct declaration reads")

            def known(match):
                return all(topic in subjects[doi] for (topic, _), doi in zip(atoms, match[1:], strict=True))

            match = candidates[0] if candidates else None
            positions = [ranks[i][doi] for i, doi in enumerate(match[1:])] if match else []
            row["arms"][arm] = dict(match=match, priority=(max(positions), sum(positions)) if match else None,
                                    known=bool(match) and known(match), any_known=any(map(known, candidates)))
        rows.append(row)
    comparison = {}
    for arm in names[1:]:
        comparison[arm] = {}
        for control in ("owner_gap", "byline"):
            if control == arm:
                continue
            normalized = [dict(atoms=row["atoms"], topic_pair=row["topic_pair"], arms={
                "owner_gap": row["arms"][control], "joint": row["arms"][arm],
            }) for row in rows]
            comparison[arm][control] = {
                policy: audit.summarize(normalized, mask) for policy, mask in audit.select(normalized).items()
            }
    primary = {arm: dict(
        reads=3 * len(rows), known=sum(row["arms"][arm]["known"] for row in rows),
        issued=sum(row["arms"][arm]["match"] is not None for row in rows),
        known_proof_loaded=sum(row["arms"][arm]["any_known"] for row in rows),
        selection_and_reads_p95_ms=float(np.quantile(times[arm], .95)),
    ) for arm in names}
    return dict(queries=len(rows), inventory=postings.inventory(), index_build_ms=build_ms,
                arms=primary, matched_comparisons=comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a new output on a Slurm GPU")
    reference = json.loads(args.reference.read_bytes())["runs"][0]
    base = live.BASE
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != reference["embedding_checkpoint"] or projection != reference["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != reference["plos_manifest_sha256"]
        or digest(args.manifest) != reference["elife_manifest_sha256"] or live.DEPTH != reference["query_depth"]
    ):
        raise ValueError("Frozen inputs differ from the reference")
    plos_subjects = base.publisher_topics([args.plos_xml])
    life_subjects, _, life_queries = live.elife.build_requests(life)["topics"]
    plos_queries = base.query_panel(base.annotation_postings(plos, plos_subjects))
    panels = dict(plos=(plos, plos_subjects, plos_queries), elife=(life, life_subjects, life_queries))
    encoder = QwenTorchEmbedding(args.embedding)
    results = {}
    with TemporaryDirectory() as scratch:
        for name, (catalogue, subjects, queries) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / name)
            results[name] = evaluate(index, subjects, queries, args.order_offset)
            frozen = reference["panels"][name]["arms"]["structured"]
            for current, prior in (("owner_gap", "owner_gap"), ("byline", "joint")):
                for metric in ("known", "issued", "reads"):
                    if results[name]["arms"][current][metric] != frozen[prior][metric]:
                        raise ValueError(f"Reference {name}/{prior}/{metric} did not reproduce")
    write_json(args.output, dict(
        gpu=torch.cuda.get_device_name(), order_offset=args.order_offset, embedding_checkpoint=checkpoint,
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)), panels=results,
    ))


if __name__ == "__main__":
    main()
