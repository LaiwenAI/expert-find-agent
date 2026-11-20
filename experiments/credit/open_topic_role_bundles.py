"""Measure cross-publication evidence recovery from topic-and-role requests."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import cross_source_gap_headroom as pinned
from experiments.credit.cross_source_identity import xml_sources


TOP_K = 8
QUERY_CAP = 12
ARMS = ("initial", "parallel_topic", "parallel_byline", "pooled_gap", "owner_gap",
        "joint_byline", "joint_role_union")


def publisher_topics(folders):
    result = {}
    for folder in folders:
        for path in xml_sources(folder):
            root = ET.fromstring(path.read_bytes()).find("./front/article-meta")
            doi = root.findtext("article-id[@pub-id-type='doi']")
            tags = set()
            for group in root.findall("article-categories/subj-group[@subj-group-type='Discipline-v3']"):
                for node in group.iter("subj-group"):
                    if node.find("subj-group") is None:
                        text = node.findtext("subject")
                        if text:
                            tags.add(" ".join(text.casefold().split()))
            result[doi] = tags
    return result


def annotation_postings(catalogue, subjects):
    coverage = Counter(topic for terms in subjects.values() for topic in terms)
    witness = defaultdict(lambda: defaultdict(set))
    for doi, paper in catalogue.ownership.items():
        for author in paper["authors"]:
            if author["person"]:
                for topic in subjects[doi]:
                    if coverage[topic] >= 4:
                        for role in author["roles"]:
                            witness[topic, role][author["person"]].add(doi)
    return witness


def query_panel(postings):
    shared = defaultdict(list)
    for atom, contributors in postings.items():
        for person in contributors:
            shared[person].append(atom)
    selected = {}
    for person, atoms in shared.items():
        if len({doi for atom in atoms for doi in postings[atom][person]}) < 2:
            continue
        cases = []
        for a, b in combinations(sorted(atoms), 2):
            if a[0] == b[0] or a[1] == b[1]:
                continue
            overlapping = postings[a].keys() & postings[b].keys()
            # Require every qualified person to cite two different papers.
            if any(postings[a][owner] & postings[b][owner] for owner in overlapping):
                continue
            digest = sha256(json.dumps([a, b]).encode()).hexdigest()
            cases.append((digest, a, b))
        for digest, a, b in sorted(cases)[:QUERY_CAP]:
            selected[digest] = (a, b)
    return [selected[key] for key in sorted(selected)]


def topic_rankings(catalogue, topics, checkpoint):
    eligible = {i for i, doi in enumerate(catalogue.ids) if catalogue.kinds[doi] is None}
    views = {"bm25": {}}
    dense_scores = None
    provenance = None
    if checkpoint is not None:
        receipt = json.loads((checkpoint / "checkpoint-provenance.json").read_bytes())
        for name, expected in receipt["files"].items():
            if pinned.search.digest(checkpoint / name) != expected:
                raise ValueError(f"Changed embedding checkpoint: {name}")
        provenance = receipt["revision"]
        encoder = QwenTorchEmbedding(checkpoint)
        bank = encoder.encode(["\n".join(catalogue.documents[doi]) for doi in catalogue.ids])
        queries = encoder.encode(topics, query=True)
        dense_scores = queries @ bank.T
        views["hybrid"] = {}
    for row, topic in enumerate(topics):
        scores = catalogue.lexical.document_scores(topic)
        lexical = sorted((int(i) for i in scores if int(i) in eligible),
                         key=lambda i: (-scores[str(i)], i))
        views["bm25"][topic] = {catalogue.ids[i]: rank for rank, i in enumerate(lexical[:TOP_K], 1)}
        if dense_scores is not None:
            dense = sorted(eligible, key=lambda i: (-dense_scores[row, i], i))
            fused = Counter()
            for ranking in (lexical, dense):
                for rank, i in enumerate(ranking, 1):
                    fused[i] += 1 / (60 + rank)
            top = sorted(fused, key=lambda i: (-fused[i], i))[:TOP_K]
            views["hybrid"][topic] = {catalogue.ids[i]: rank for rank, i in enumerate(top, 1)}
    return views, provenance


def choose_proofs(people, ranks):
    found = set()
    for person, (left, right) in people.items():
        for a, b in product(left, right):
            if a != b:
                found.add((person, a, b))
    return sorted(found, key=lambda triple: (
        max(ranks[0][triple[1]], ranks[1][triple[2]]),
        ranks[0][triple[1]] + ranks[1][triple[2]], triple,
    ))


def read_articles(catalogue, titles, atoms, ranks):
    shown = defaultdict(lambda: (set(), set()))
    for doi in titles:
        for author in catalogue.ownership[doi]["authors"]:
            person = author["person"]
            if person:
                for i, (_, role) in enumerate(atoms):
                    if role in author["roles"] and doi in ranks[i]:
                        shown[person][i].add(doi)
    return shown


def joint_sources(pool, ranks, fallback, can_cover):
    candidates = [
        (a, b) for a, b in product(*ranks) if a != b and can_cover(a, b)
    ]
    if not candidates:
        return fallback
    first = min(candidates, key=lambda pair: (
        max(ranks[0][pair[0]], ranks[1][pair[1]]),
        ranks[0][pair[0]] + ranks[1][pair[1]], pair,
    ))
    second = sorted(set(pool) - set(first))
    if not second:
        return list(first)
    def gain(doi):
        return (sum(doi in pair and any(part in first for part in pair) for pair in candidates),
                -min(ranks[0].get(doi, TOP_K+1), ranks[1].get(doi, TOP_K+1)))
    return [*first, max(second, key=gain)]


def evaluate(catalogue, subjects, atoms, gold, ranks_by_topic, role_union):
    ranks = tuple(ranks_by_topic[topic] for topic, _ in atoms)
    present = [facet for facet in ranks if facet]
    first = list(dict.fromkeys(next(iter(facet)) for facet in present))
    more = dict.fromkeys(doi for facet in ranks for doi in facet if doi not in first)
    first.extend(list(more)[:2-len(first)])
    extras = sorted((ranks[0].keys() | ranks[1].keys()) - set(first))
    observed = read_articles(catalogue, first, atoms, ranks)

    def rank(doi, side):
        return ranks[side].get(doi, TOP_K + 1)

    def topical(doi):
        return (min(rank(doi, 0), rank(doi, 1)), rank(doi, 0) + rank(doi, 1), doi)

    def byline(doi):
        gains = sum(
            len(catalogue.byline[doi] & catalogue.byline[witness])
            for witness in first for i in range(2)
            if witness in ranks[i] and doi in ranks[1 - i]
        )
        return (-gains, *topical(doi))

    def owner_gap(doi):
        contributions = [
            1 / min(rank(witness, i) for witness in witnesses)
            for person, sides in observed.items() for i, witnesses in enumerate(sides)
            if person in catalogue.byline[doi] and witnesses and doi in ranks[1 - i]
            and not any(a != b for a, b in product(*sides))
        ]
        return (-sum(contributions), *topical(doi))

    side = min(range(2), key=lambda i: (sum(bool(sides[i]) for sides in observed.values()), i))
    keys = dict(parallel_topic=topical, parallel_byline=byline,
                pooled_gap=lambda doi: (rank(doi, side), *topical(doi)), owner_gap=owner_gap)
    proposals = {arm: [*first, min(extras, key=key)] if extras else first for arm, key in keys.items()}
    proposals["initial"] = first
    pool = set().union(*ranks)
    proposals["joint_byline"] = joint_sources(
        pool, ranks, proposals["parallel_topic"],
        lambda a, b: bool(catalogue.byline[a] & catalogue.byline[b]),
    )
    proposals["joint_role_union"] = joint_sources(
        pool, ranks, proposals["joint_byline"],
        lambda a, b: (bool(catalogue.byline[a] & catalogue.byline[b])
                      and atoms[0][1] in role_union[a] and atoms[1][1] in role_union[b]),
    )
    ref = {person: (gold[atoms[0]][person] & ranks[0].keys(), gold[atoms[1]][person] & ranks[1].keys())
           for person in gold[atoms[0]].keys() & gold[atoms[1]].keys()}

    def reaches(docs):
        return any(left & docs and right & docs for left, right in ref.values())

    results = {}
    for arm, docs in proposals.items():
        pairs = choose_proofs(read_articles(catalogue, docs, atoms, ranks), ranks)
        supported = bool(pairs) and all(atoms[i][0] in subjects[pairs[0][i + 1]] for i in range(2))
        # A topic outside the publisher tag list is unresolved.
        results[arm] = dict(reads=len(docs), issued=int(bool(pairs)), known=int(supported),
                            tag_unresolved=int(bool(pairs) and not supported),
                            proof_reached=int(reaches(set(docs))))
    first_reachable = reaches(set(first))
    return dict(
        answers=len(ref), any_in_pool=int(reaches(set().union(*ranks))),
        empty_facet=int(not all(ranks)),
        first_reached=int(first_reachable), one_more=int(any(reaches({*first, doi}) for doi in extras)),
        arms=results,
    )


def summarize(results, groups):
    topic_total, statistics = Counter(), {}
    for item in results:
        topic_total.update({key: value for key, value in item.items() if key != "arms"})
    for arm in ARMS:
        counts = Counter()
        for result in results:
            counts.update(result["arms"][arm])
        difference = [r["arms"][arm]["known"] - r["arms"]["parallel_topic"]["known"] for r in results]
        statistics[arm] = dict(counts, paired_known_delta_vs_parallel_topic=clustered_paired_interval(
            difference, groups,
        ))
    return dict(headroom=dict(topic_total), arms=statistics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--embedding", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    catalogue, _ = pinned.load_fixture(args.fixture, args.sources)
    subjects = publisher_topics(args.sources)
    if subjects.keys() != catalogue.ownership.keys():
        raise ValueError("Changed topic inventory")
    postings = annotation_postings(catalogue, subjects)
    queries = query_panel(postings)
    if not queries:
        raise ValueError("No cross-source requests with two qualified topics")
    topics = sorted({a[0] for case in queries for a in case})
    rankings, checkpoint = topic_rankings(catalogue, topics, args.embedding)
    groups = ["|".join(sorted((a[0], b[0]))) for a, b in queries]
    role_union = {doi: {role for a in item["authors"] for role in a["roles"]}
                  for doi, item in catalogue.ownership.items()}
    results = {}
    for view, ranks in rankings.items():
        estimates = [evaluate(catalogue, subjects, case, postings, ranks, role_union) for case in queries]
        results[view] = summarize(estimates, groups)
    with args.output.open("x") as handle:
        json.dump(dict(
            sources=len(catalogue.ids), queries=len(queries), topic_pairs=len(set(groups)),
            topic_labels=len(topics), embedding_checkpoint=checkpoint,
            script_sha256=pinned.search.digest(Path(__file__)),
            manifest_sha256=pinned.search.digest(args.fixture / "manifest.json"),
            methods=results,
        ), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
