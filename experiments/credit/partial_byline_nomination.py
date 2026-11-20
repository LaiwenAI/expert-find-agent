"""Stress-test cross-source nomination with missing publisher identity metadata."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from expert_search.statistics import clustered_paired_interval
from experiments.credit import cross_source_identity as identities
from experiments.credit import elife_cross_publication_transfer as elife
from experiments.credit import joint_byline_depth as joint
from experiments.followir.core17_reranking import digest, write_json


BASE = joint.base
DEPTH = 32
COVERAGE = (1.0, .75, .50, .25, 0.)
SEEDS = (11, 29, 53)
ARMS = ("topic", "visible_orcid", "name", "name_with_coauthor", "orcid_first_then_name")


def byline_names(paths):
    names = {}
    for path in paths:
        meta = ET.fromstring(path.read_bytes()).find("./front/article-meta")
        doi = meta.findtext("article-id[@pub-id-type='doi']")
        authors = meta.findall("contrib-group/contrib[@contrib-type='author']")
        names[doi] = tuple(identities.normalized_name(author) for author in authors)
    return names


def visible_snapshot(catalogue, names, fraction, seed):
    ownership, tokens, records = {}, {}, {}
    for doi, source in catalogue.ownership.items():
        authors = []
        term_count = Counter(names[doi])
        known = {}
        for author, name in zip(source["authors"], names[doi], strict=True):
            person = author["person"]
            row_hash = sha256(f"{seed}|{doi}|{author['ordinal']}".encode()).digest()
            revealed = int.from_bytes(row_hash[:8], "big") / 2**64 < fraction
            if not revealed:
                person = None
            authors.append(dict(author, person=person))
            if name and term_count[name] == 1:
                known[name] = person
        tokens[doi] = {row["person"] for row in authors if row["person"]}
        records[doi] = known
        ownership[doi] = dict(source, authors=authors)
    return SimpleNamespace(ownership=ownership, byline=tokens, names=records)


def candidate_link(view, a, b, corroborate):
    if not view.byline[a].isdisjoint(view.byline[b]):
        return True
    shared = view.names[a].keys() & view.names[b].keys()
    if len(shared) < (2 if corroborate else 1):
        return False
    for name in shared:
        # Two conflicting visible IDs defeat an otherwise identical name.
        if view.names[a][name] is None or view.names[b][name] is None:
            if not corroborate or any(
                other != name and (view.names[a][other] is None or view.names[b][other] is None
                                   or view.names[a][other] == view.names[b][other])
                for other in shared
            ):
                return True
    return False


def topical_plan(ranks):
    first = list(dict.fromkeys(next(iter(facet)) for facet in ranks if facet))
    rest = sorted(set().union(*ranks) - set(first), key=lambda doi: (
        min(rank.get(doi, DEPTH + 1) for rank in ranks),
        sum(rank.get(doi, DEPTH + 1) for rank in ranks), doi,
    ))
    return [*first, *rest[:3-len(first)]]


def selected_links(view, original, docs):
    if len(docs) < 2:
        return 0, 0
    left, right = docs[:2]
    wrong = unresolved = 0
    for term in view.names[left].keys() & view.names[right].keys():
        if view.names[left][term] and view.names[right][term]:
            continue
        a, b = original.names[left][term], original.names[right][term]
        wrong += int(bool(a and b and a != b))
        unresolved += int(a is None or b is None)
    return wrong, unresolved


def answer(catalogue, docs, atoms, ranks, subjects):
    pairs = BASE.choose_proofs(BASE.read_articles(catalogue, docs, atoms, ranks), ranks)
    known = bool(pairs) and all(atoms[i][0] in subjects[pairs[0][i + 1]] for i in range(2))
    return dict(known=int(known), issued=int(bool(pairs)), unresolved=int(bool(pairs) and not known))


def run_query(catalogue, view, original_names, atoms, ranking, subjects):
    ranks = tuple(ranking[topic] for topic, _ in atoms)
    pool = set().union(*ranks)
    topical = topical_plan(ranks)
    plans = dict(
        topic=topical,
        visible_orcid=joint.posting_plan(view.byline, pool, ranks, topical, DEPTH),
        name=BASE.joint_sources(pool, ranks, topical,
                                lambda a, b: candidate_link(view, a, b, False), DEPTH),
        name_with_coauthor=BASE.joint_sources(pool, ranks, topical,
                                             lambda a, b: candidate_link(view, a, b, True), DEPTH),
    )
    exact = plans["visible_orcid"]
    linked = len(exact) >= 2 and not view.byline[exact[0]].isdisjoint(view.byline[exact[1]])
    plans["orcid_first_then_name"] = exact if linked else plans["name"]
    output = {}
    for arm, docs in plans.items():
        confirmed = answer(catalogue, docs, atoms, ranks, subjects)
        available = answer(view, docs, atoms, ranks, subjects)
        # Audit every compromised name match in the selected first paper
        # pair, not claims that the plan actually selected a particular name.
        wrong, unjudged = selected_links(view, original_names, docs)
        output[arm] = dict(
            restored_known=confirmed["known"], restored_issued=confirmed["issued"],
            visible_known=available["known"], visible_issued=available["issued"],
            visible_tag_unresolved=available["unresolved"], reads=len(docs),
            selected_pair_conflicting_name_links=wrong,
            selected_pair_unjudged_name_links=unjudged,
        )
    return output


def evaluate(catalogue, subjects, names, ranks, queries):
    outcomes = {}
    original_names = visible_snapshot(catalogue, names, 1., 0)
    clusters = ["|".join((left[0], right[0])) for left, right in queries]
    for fraction in COVERAGE:
        observations = []
        for seed in SEEDS if 0 < fraction < 1 else SEEDS[:1]:
            view = visible_snapshot(catalogue, names, fraction, seed)
            observations.append([run_query(catalogue, view, original_names, case, ranks, subjects)
                                 for case in queries])
        by_arm = {}
        for arm in ARMS:
            counts = []
            for repeat in observations:
                totals = Counter()
                for row in repeat:
                    totals.update(row[arm])
                counts.append(dict(totals))
            delta = [sum(repeat[i][arm]["restored_known"] - repeat[i]["visible_orcid"]["restored_known"]
                         for repeat in observations) / len(observations) for i in range(len(queries))]
            by_arm[arm] = dict(runs=counts, paired_delta_vs_visible_orcid=clustered_paired_interval(delta, clusters))
        outcomes[str(fraction)] = by_arm
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plos_xml", "plos_fixture", "elife_xml", "elife_manifest", "checkpoint", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh result on a Slurm compute node")
    plos, _ = BASE.pinned.load_fixture(args.plos_fixture, [args.plos_xml])
    life, checksum = elife.load_catalogue(args.elife_xml, args.elife_manifest)
    documents = dict(plos=(plos, byline_names(identities.xml_sources(args.plos_xml))),
                     elife=(life, byline_names(args.elife_xml.glob("elife-*.xml"))))
    plos_subjects = BASE.publisher_topics([args.plos_xml])
    plos_gold = BASE.annotation_postings(plos, plos_subjects)
    panels = {"plos": {"leaf_topics": (plos_subjects, plos_gold, BASE.query_panel(plos_gold))},
              "elife": elife.build_requests(life)}
    results = {}
    model = None
    for publisher, data in panels.items():
        catalogue, names = documents[publisher]
        topics = sorted({atom[0] for _, _, queries in data.values() for pair in queries for atom in pair})
        views, model = BASE.topic_rankings(catalogue, topics, args.checkpoint, width=DEPTH)
        results[publisher] = {
            field: dict(queries=len(queries), outcomes=evaluate(catalogue, subjects, names, views["hybrid"], queries))
            for field, (subjects, _, queries) in data.items() if queries
        }
    write_json(args.output, dict(
        depth=DEPTH, seeds=SEEDS, embedding_checkpoint=model, elife_projection=checksum,
        script_sha256=digest(Path(__file__)), plos_manifest_sha256=digest(args.plos_fixture / "manifest.json"),
        elife_manifest_sha256=digest(args.elife_manifest), panels=results,
    ))


if __name__ == "__main__":
    main()
