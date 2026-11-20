"""Replay first-query role search when source identities are hidden from the index."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import cross_source_identity as source_ids
from experiments.credit import live_joint_role_postings as typed
from experiments.credit import owner_evidence_bound as bound
from experiments.followir.core17_reranking import digest, write_json


POLICIES = ("visible", "name", "coauthor", "exact")
SEEDS = (20261121, 20261122, 20261123)
DEGREE = (25, 50)
NAME_MODES = ("full", "abbreviated")


def read_names(directory, catalogue):
    names = {}
    for path in source_ids.xml_sources(directory):
        meta = ET.fromstring(path.read_bytes()).find("./front/article-meta")
        doi = meta.findtext("article-id[@pub-id-type='doi']")
        if doi not in catalogue.ownership:
            continue
        authors = meta.findall("contrib-group/contrib[@contrib-type='author']")
        if len(authors) != len(catalogue.ownership[doi]["authors"]):
            raise ValueError(f"Unaligned author records in {doi}")
        rows = []
        for author in authors:
            full = source_ids.normalized_name(author)
            given = author.findtext("name/given-names", "")
            surname = author.findtext("name/surname", "")
            reduced = "".join(char for char in (given[:1]+surname).casefold() if char.isalnum()) or full
            rows.append((full, reduced))
        names[doi] = rows
    if names.keys() != catalogue.ownership.keys():
        raise ValueError("Missing named source inventory")
    return names


def mask_rows(catalogue, names, percent, seed, mode):
    view = {}
    for doi, source in catalogue.ownership.items():
        publisher = doi.split("/", 1)[0]
        masked = int.from_bytes(sha256(f"{seed}:{doi}".encode()).digest()[:4], "big") < percent * (1 << 32) // 100
        view[doi] = [
            dict(row, match_name=name[mode], original_name=name[0],
                 person=None if masked else row["person"], namespace=publisher)
            for row, name in zip(source["authors"], names[doi], strict=True)
        ]
    return view


def cohort_names(rows, ordinal):
    return {row["original_name"] for row in rows
            if row["ordinal"] != ordinal and row["original_name"]}


def link(view, policy):
    anchors = defaultdict(list)
    for doi, authors in view.items():
        for row in authors:
            if row["person"] and row["match_name"]:
                collaborators = cohort_names(authors, row["ordinal"]) - {row["original_name"]}
                anchors[row["namespace"], row["match_name"]].append((row["person"], doi, collaborators))

    resolved = {}
    for doi, authors in view.items():
        present = {row["person"] for row in authors if row["person"]}
        proposals = {}
        for row in authors:
            if row["person"] or not row["match_name"]:
                continue
            others = cohort_names(authors, row["ordinal"]) - {row["original_name"]}
            viable = {
                person for person, prior_doi, group in anchors[row["namespace"], row["match_name"]]
                if prior_doi != doi and person not in present and (policy == "name" or others & group)
            }
            if len(viable) == 1:
                proposals[row["ordinal"]] = next(iter(viable))
        repeats = Counter(proposals.values())
        resolved[doi] = {
            "authors": [
                dict(row, person=proposals[row["ordinal"]])
                if row["ordinal"] in proposals and repeats[proposals[row["ordinal"]]] == 1 else row
                for row in authors
            ]
        }
    return resolved


def link_counts(indexed, visible, original):
    counts = Counter()
    for doi, source in indexed.items():
        for actual, masked, candidate in zip(original[doi]["authors"], visible[doi], source["authors"], strict=True):
            if not masked["person"] and actual["person"]:
                counts["hidden_known"] += 1
            if masked["person"] or not candidate["person"]:
                continue
            counts["proposed"] += 1
            if not actual["person"]:
                counts["unresolved"] += 1
            elif candidate["person"] == actual["person"]:
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
    return dict(counts)


def postings(indexed):
    byline = {doi: {row["person"] for row in src["authors"] if row["person"]}
              for doi, src in indexed.items()}
    return typed.RolePostings(SimpleNamespace(byline=byline, ownership=indexed))


def best_plan(indexed, atoms, ranks, depth):
    facets = tuple(indexed.facet(rank, role, 100, True)
                   for (_, role), rank in zip(atoms, ranks, strict=True))
    pair = typed.first_pair(ranks, facets)
    return [item[2] for item in pair] if pair else bound.live.simple_order(ranks, depth=depth)[0]


def inspect_proof(proof, atoms, original, proposed):
    if proof is None:
        return "abstained"
    person, *docs = proof
    results = []
    for (_, duty), doi in zip(atoms, docs, strict=True):
        eligible = [a for a in proposed[doi]["authors"] if a["person"] == person and duty in a["roles"]]
        actual = {a["ordinal"]: a["person"] for a in original[doi]["authors"]}
        if any(actual[a["ordinal"]] == person for a in eligible):
            results.append("supported")
        elif any(not actual[a["ordinal"]] for a in eligible):
            results.append("unresolved")
        else:
            results.append("wrong")
    if "wrong" in results:
        return "wrong"
    return "unresolved" if "unresolved" in results else "supported"


def query_result(index, indexed, view, original, query, depth):
    atoms, ranks, subjects, base_ms = query
    start = perf_counter()
    docs = best_plan(indexed, atoms, ranks, depth)
    loaded = index.open_sources(docs)
    tentative = {}
    for doi, source in loaded.items():
        hypotheses = {row["ordinal"]: row["person"] for row in view[doi]["authors"]}
        tentative[doi] = {"authors": [dict(row, person=hypotheses[row["ordinal"]]) for row in source["authors"]]}
    nominal = bound.verified(docs, tentative, atoms, ranks)[1]
    verified = bound.verified(docs, loaded, atoms, ranks)[1]
    elapsed = (perf_counter() - start) * 1000 + base_ms
    if len(set(docs)) != 2 or set(docs) != loaded.keys():
        raise AssertionError("Each policy must read two distinct stored declarations")

    def topic_match(proof):
        return bool(proof) and all(topic in subjects.get(doi, ())
                                   for (topic, _), doi in zip(atoms, proof[1:], strict=True))

    status = inspect_proof(nominal, atoms, original, tentative)
    return dict(
        ms=elapsed, reads=len(docs), direct_issued=bool(nominal),
        direct_supported=status == "supported", direct_unresolved=status == "unresolved",
        direct_wrong=status == "wrong", direct_known=status == "supported" and topic_match(nominal),
        verified_issued=bool(verified), verified_known=topic_match(verified),
        changed_by_verification=nominal != verified,
    )


def evaluate(index, queries, names, depth):
    original = index.catalogue.ownership
    grouped = {}
    for mode in NAME_MODES:
        for percent in DEGREE:
            folds = []
            for seed in SEEDS:
                visible = mask_rows(index.catalogue, names, percent, seed, NAME_MODES.index(mode))
                views = {name: link(visible, name) for name in ("name", "coauthor")}
                views["exact"] = original
                views["visible"] = {doi: {"authors": rows} for doi, rows in visible.items()}
                linked = {policy: postings(view) for policy, view in views.items()}
                counts = {policy: link_counts(view, visible, original) for policy, view in views.items()}
                topics = [json.dumps([topic for topic, _ in item[0]]) for item in queries]
                outcomes = {policy: [] for policy in POLICIES}
                for number, row in enumerate(queries):
                    offset = (number + seed) % len(POLICIES)
                    for policy in POLICIES[offset:] + POLICIES[:offset]:
                        outcomes[policy].append(query_result(index, linked[policy], views[policy], original, row, depth))
                metrics = {}
                reference = np.asarray([row["verified_known"] for row in outcomes["visible"]], dtype=int)
                for policy, records in outcomes.items():
                    counter = Counter()
                    for record in records:
                        counter.update({k: v for k, v in record.items() if k != "ms"})
                    differences = np.asarray([x["verified_known"] for x in records], dtype=int) - reference
                    interval = clustered_paired_interval(differences, topics)
                    interval["sampling_unit"] = "ordered topic pair; query-weighted mean"
                    metrics[policy] = dict(counter, p95_ms=float(np.percentile([x["ms"] for x in records], 95)),
                                           verified_delta_vs_visible=interval)
                folds.append(dict(seed=seed, linked=counts, arms=metrics))
            grouped[f"{mode}:{percent}"] = folds
    return grouped


def main(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a Slurm compute node and a fresh output file")
    base, live = bound.BASE, bound.live
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = base.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.elife_manifest) != expected["elife_manifest_sha256"]
        or bound.raw_identifiers(args.plos) & bound.raw_identifiers(args.elife)
    ):
        raise ValueError("Changed publisher snapshots or embedding checkpoint")
    names = read_names(args.plos, plos) | read_names(args.elife, elife)
    plos_subjects = base.publisher_topics([args.plos])
    elife_subjects, _, elife_queries = live.elife.build_requests(elife)["topics"]
    panels = (
        ("plos", plos_subjects, base.query_panel(base.annotation_postings(plos, plos_subjects))),
        ("elife", elife_subjects, elife_queries),
    )
    output = {}
    with TemporaryDirectory(dir=args.output.resolve().parent) as temp:
        index = live.LocalIndex(bound.merge(plos, elife), QwenTorchEmbedding(args.embedding), Path(temp) / "mixed")
        index.rank([topic for topic, _ in panels[0][2][0]], depth=args.depth)
        for name, subjects, queries in panels:
            prepared = bound.prepare(index, queries, args.depth)
            workload = [(atoms, ranks, subjects, base_ms)
                        for atoms, (ranks, base_ms) in zip(queries, prepared, strict=True)]
            output[name] = dict(requests=len(queries), measurements=evaluate(index, workload, names, args.depth))
    write_json(args.output, dict(depth=args.depth, embedding_checkpoint=checkpoint,
                                 fixture_digest=expected["plos_manifest_sha256"], projection=projection, panels=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--depth", type=int, default=128)
    main(parser.parse_args())
