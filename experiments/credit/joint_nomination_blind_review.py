"""Compare first-round evidence plans at matched issuance and freeze blind cards."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import blind_joint_topic_review as review
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("owner_gap", "joint")


def collect(index, queries, subjects, offset):
    rows = []
    for number, atoms in enumerate(queries):
        ranks = index.rank([topic for topic, _ in atoms])
        row = dict(atoms=atoms, topic_pair=json.dumps([topic for topic, _ in atoms]), arms={})
        for arm in (ARMS if (number + offset) % 2 == 0 else ARMS[::-1]):
            plan, loaded = live.inspect(index, atoms, ranks, arm)
            if len(plan) != 3 or set(plan) != loaded.keys():
                raise ValueError("The experiment requires exactly three distinct declaration reads")
            matches = json.loads(live.render(loaded, plan, atoms, ranks))["matches"]
            match = matches[0] if matches else None
            article_ranks = [ranks[i][doi] for i, doi in enumerate(match[1:])] if match else []
            row["arms"][arm] = dict(
                match=match,
                priority=(max(article_ranks), sum(article_ranks)) if match else None,
                known=bool(match) and all(topic in subjects[doi]
                                          for (topic, _), doi in zip(atoms, match[1:], strict=True)),
            )
        rows.append(row)
    return rows


def select(rows):
    natural = {arm: {i for i, row in enumerate(rows) if row["arms"][arm]["match"]} for arm in ARMS}
    common = set.intersection(*natural.values())
    count = min(map(len, natural.values()))

    def order(arm, i):
        row = rows[i]
        request_hash = sha256(json.dumps(row["atoms"], ensure_ascii=True).encode()).hexdigest()
        return row["arms"][arm]["priority"], request_hash

    quota = {arm: set(sorted(natural[arm], key=lambda i: order(arm, i))[:count]) for arm in ARMS}
    return dict(natural=natural, common={arm: common for arm in ARMS}, quota=quota)


def summarize(rows, masks):
    groups = [row["topic_pair"] for row in rows]
    flags = {}
    arms = {}
    for arm, selected in masks.items():
        flags[arm] = np.array([int(i in selected and row["arms"][arm]["known"])
                               for i, row in enumerate(rows)])
        known = int(flags[arm].sum())
        arms[arm] = dict(issued=len(selected), known=known, tag_unresolved=len(selected)-known)
    left, right = (masks[arm] for arm in ARMS)
    overlap = left & right
    changed = Counter(same_response=0, same_person_changed_sources=0, different_person=0,
                      known_gain=0, known_loss=0)
    for i in overlap:
        before, after = [rows[i]["arms"][arm]["match"] for arm in ARMS]
        if before == after:
            changed["same_response"] += 1
        elif before[0] == after[0]:
            changed["same_person_changed_sources"] += 1
        else:
            changed["different_person"] += 1
        change = int(flags["joint"][i]) - int(flags["owner_gap"][i])
        if change:
            changed["known_gain" if change == 1 else "known_loss"] += 1
    difference = flags["joint"] - flags["owner_gap"]

    def interval(indices):
        if not indices:
            return None
        return dict(
            clustered_paired_interval(difference[list(indices)], [groups[i] for i in indices]),
            sampling_unit="topic pair; query-weighted mean",
        )

    return dict(
        arms=arms, both_issue=len(overlap), overlap_changes=dict(changed),
        issued_only={ARMS[0]: len(left - right), ARMS[1]: len(right - left)},
        paired_known_delta_per_query=interval(range(len(rows))) if left or right else None,
        paired_known_delta_on_overlap=interval(sorted(overlap)),
    )


def bind_cards(panel, rows, policies, catalogue, paths, cards):
    views = []
    for i, row in enumerate(rows):
        atoms = row["atoms"]
        views.append({})
        for arm in ARMS:
            if not any(i in masks[arm] for masks in policies.values()):
                continue
            match = row["arms"][arm]["match"]
            if not match:
                continue
            person, a, b = match
            uid = review.stable_id(panel, atoms, person, a, b)
            views[-1][arm] = uid
            if uid not in cards:
                cards[uid] = dict(
                    case_id=uid,
                    requirements=[dict(topic=topic, role=role) for topic, role in atoms],
                    sources=[review.citation(catalogue, doi, person, paths) for doi in (a, b)],
                )
    return {
        f"{panel}_{policy}": dict(
            baseline="owner_gap", queries=len(rows),
            rows=[
                dict(topic_pair=row["topic_pair"], outputs={
                    arm: view.get(arm) if i in masks[arm] else None for arm in ARMS
                })
                for i, (row, view) in enumerate(zip(rows, views, strict=True))
            ],
        )
        for policy, masks in policies.items()
    }


def build(args):
    reference = json.loads(args.reference.read_bytes())["runs"]
    reference_run = reference[0]
    checkpoint = live.verify_embedding(args.embedding)
    base = live.BASE
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos_xml])
    life, projection = live.elife.load_catalogue(args.elife_xml, args.manifest)
    if (
        checkpoint != reference_run["embedding_checkpoint"]
        or projection != reference_run["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != reference_run["plos_manifest_sha256"]
        or digest(args.manifest) != reference_run["elife_manifest_sha256"]
        or live.DEPTH != reference_run["query_depth"]
    ):
        raise ValueError("Frozen input mismatch")
    plos_subjects = base.publisher_topics([args.plos_xml])
    plos_queries = base.query_panel(base.annotation_postings(plos, plos_subjects))
    life_subjects, _, life_queries = live.elife.build_requests(life)["topics"]
    panels = dict(plos=(plos, plos_subjects, plos_queries, args.plos_xml),
                  elife=(life, life_subjects, life_queries, args.elife_xml))
    cards, key, summaries = {}, {}, {}
    encoder = QwenTorchEmbedding(args.embedding)
    with TemporaryDirectory() as scratch:
        for panel, (catalogue, subjects, queries, xml) in panels.items():
            index = live.LocalIndex(catalogue, encoder, Path(scratch) / panel)
            rows = collect(index, queries, subjects, args.order_offset)
            policies = select(rows)
            summaries[panel] = {name: summarize(rows, masks) for name, masks in policies.items()}
            for run in reference:
                expected = run["panels"][panel]
                actual = summaries[panel]["natural"]["arms"]
                if len(rows) != expected["queries"] or any(
                    value != expected["arms"]["structured"][arm][metric]
                    for arm, stats in actual.items() for metric, value in stats.items()
                ):
                    raise ValueError(f"Reference issuance did not reproduce for {panel}")
            key.update(bind_cards(panel, rows, policies, catalogue, review.source_paths(xml), cards))
    blinded = sorted(cards.values(), key=lambda card: card["case_id"])
    packet_id = review.stable_id(blinded)
    write_json(args.packet, dict(packet_id=packet_id, dimensions=review.DIMENSIONS, cards=blinded))
    write_json(args.key, dict(packet_id=packet_id, panels=key, reference_sha256=digest(args.reference)))
    write_json(args.summary, dict(
        distinct_blind_cards=len(blinded), panels=summaries,
        embedding_checkpoint=checkpoint, order_offset=args.order_offset, gpu=torch.cuda.get_device_name(),
        reference_sha256=digest(args.reference), script_sha256=digest(Path(__file__)),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "fixture", "elife_xml", "manifest", "embedding", "reference", "summary", "packet", "key"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    outputs = (args.summary, args.packet, args.key)
    if (
        not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
        or len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs)
    ):
        raise ValueError("Require three fresh outputs on a Slurm GPU")
    build(args)


if __name__ == "__main__":
    main()
