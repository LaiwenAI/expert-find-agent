"""Compare exact role-state read completion at matched initial source budgets."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import live_joint_role_postings as typed
from experiments.credit import live_joint_source_request as live
from experiments.credit.mixed_publisher_joint import merge, raw_identifiers, reference_in_pool
from experiments.followir.core17_reranking import digest, write_json


ARMS = (
    "joint_byline", "static_two", "adaptive_two", "static_partial",
    "adaptive_forced", "adaptive_certified", "full_roles",
)
BASE = live.BASE


def verified(plan, files, atoms, ranks):
    people = BASE.read_articles(SimpleNamespace(ownership=files), plan, atoms, ranks)
    proofs = BASE.choose_proofs(people, ranks)
    return people, proofs[0] if proofs else None


def proof_key(proof, ranks):
    person, left, right = proof
    a, b = ranks[0][left], ranks[1][right]
    return max(a, b), a + b, person, left, right


def candidates(allowed, observed, ranks, extras, existing):
    """Optimistically assign every permitted missing role to a byline member."""
    current = proof_key(existing, ranks) if existing is not None else None
    best_per_doc = {}
    for side, facet in enumerate(allowed):
        opposite = 1 - side
        for doc in extras:
            if doc not in facet:
                continue
            known_people, possible_people = facet[doc]
            best = best_per_doc.get(doc)
            for person in known_people | possible_people:
                if person not in observed:
                    continue
                for witness in observed[person][opposite]:
                    proof = (person, doc, witness) if side == 0 else (person, witness, doc)
                    quality = proof_key(proof, ranks)
                    if current is None or quality < current:
                        best = quality if best is None else min(best, quality)
            if best is not None:
                best_per_doc[doc] = best
    return best_per_doc



def next_anchor(facets, observed, ranks, fallback, initial):
    """Prefer indexed matches to the opened declaration, then optimistic byline matches."""
    confirmed = tuple({doi: (matched, set()) for doi, (matched, _) in facet.items()} for facet in facets)
    extras = set().union(*ranks) - {initial}
    choices = candidates(confirmed, observed, ranks, extras, None)
    if not choices:
        choices = candidates(facets, observed, ranks, extras, None)
    return min(choices, key=choices.__getitem__) if choices else fallback


def plan(index, postings, atoms, ranks, coverage, arm, depth):
    if arm == "joint_byline":
        docs, files = live.inspect(index, atoms, ranks, "joint", depth=depth)
        return docs, files, None
    available = 100 if arm == "full_roles" else coverage
    facets = tuple(postings.facet(ranking, duty, available, True)
                   for ranking, (_, duty) in zip(ranks, atoms, strict=True))
    if arm in {"static_partial", "full_roles"}:
        first, extra = live.simple_order(ranks, depth=depth)
        static = typed.source_plan(ranks, facets, [*first, extra[0]], depth)
        return static, index.open_sources(static), None

    pair = typed.first_pair(ranks, facets)
    anchors = [entry[2] for entry in pair] if pair is not None else live.simple_order(ranks, depth=depth)[0]
    if arm == "static_two":
        return anchors, index.open_sources(anchors), None

    if arm == "adaptive_two":
        first, fallback = anchors
        opened = index.open_sources([first])
        observed, _ = verified([first], opened, atoms, ranks)
        second = next_anchor(facets, observed, ranks, fallback, first)
        return [first, second], opened | index.open_sources([second]), None

    opened = index.open_sources(anchors)
    observed, existing = verified(anchors, opened, atoms, ranks)
    remaining = sorted(set().union(*ranks) - set(anchors))
    prospective = candidates(facets, observed, ranks, remaining, existing)
    if arm == "adaptive_certified" and not prospective:
        return anchors, opened, remaining
    if prospective:
        third = min(prospective, key=prospective.__getitem__)
    else:
        first, extra = live.simple_order(ranks, depth=depth)
        static = typed.source_plan(ranks, facets, [*first, extra[0]], depth)
        third = static[-1]
    docs = [*anchors, third]
    return docs, opened | index.open_sources([third]), None


def scope_proof(docs, files, rivals, index, first, atoms, ranks):
    """Independently replay every skipped article's true declaration offline."""
    for rival in rivals:
        truth = files | {rival: index.catalogue.ownership[rival]}
        if verified([*docs, rival], truth, atoms, ranks)[1] != first:
            raise AssertionError("A bypassed declaration could change the reference result")


def summarize(outcomes, queries):
    clusters = [json.dumps([topic for topic, _ in atoms]) for atoms in queries]
    known = {name: np.asarray([row["known"] for row in rows], dtype=int) for name, rows in outcomes.items()}
    report = {}
    for name, rows in outcomes.items():
        counts = Counter()
        for row in rows:
            counts.update({k: v for k, v in row.items() if k not in {"ms", "selected", "anchors"}})
        counts["issued_topic_unresolved"] = counts["issued"] - counts["known"]
        report[name] = dict(
            **counts,
            p95_ms=float(np.percentile([row["ms"] for row in rows], 95)),
            known_delta_vs_static=clustered_paired_interval(known[name] - known["static_partial"], clusters),
        )
        report[name]["known_delta_vs_static"]["sampling_unit"] = "ordered topic pair; query-weighted mean"
    two_read_delta = clustered_paired_interval(known["adaptive_two"] - known["static_two"], clusters)
    two_read_delta["sampling_unit"] = "ordered topic pair; query-weighted mean"
    report["adaptive_two"]["known_delta_vs_two_source"] = two_read_delta
    report["adaptive_two"]["changed_vs_two_source"] = 0
    for static, adapted in zip(outcomes["static_two"], outcomes["adaptive_two"], strict=True):
        if static["anchors"][0] != adapted["anchors"][0]:
            raise AssertionError("The two-read arms must share their first article")
        report["adaptive_two"]["changed_vs_two_source"] += static["selected"] != adapted["selected"]
    matched = ("static_two", "static_partial", "adaptive_forced", "adaptive_certified")
    report["adaptive_certified"]["changed_vs_two_source"] = 0
    report["adaptive_certified"]["changed_vs_static"] = 0
    for two, static, forced, stopped in zip(*(outcomes[arm] for arm in matched), strict=True):
        if not (two["anchors"] == static["anchors"] == forced["anchors"] == stopped["anchors"]):
            raise AssertionError("The matched arms must share both starting articles")
        if forced["selected"] != stopped["selected"]:
            raise AssertionError("Early stop changed the forced output or witness")
        report["adaptive_certified"]["changed_vs_two_source"] += stopped["selected"] != two["selected"]
        report["adaptive_certified"]["changed_vs_static"] += stopped["selected"] != static["selected"]
    return report


def evaluate(index, subjects, queries, rank_cache, depth, rotation):
    postings = typed.RolePostings(index.catalogue)
    report = {}
    for coverage in (0, 25, 50, 100):
        outcomes = {name: [] for name in ARMS}
        upper = 0
        for ordinal, (atoms, (ranks, base_ms)) in enumerate(zip(queries, rank_cache, strict=True)):
            upper += reference_in_pool(postings, subjects, atoms, ranks)
            phase = (ordinal + rotation) % len(ARMS)
            for arm in ARMS[phase:] + ARMS[:phase]:
                begin = perf_counter()
                docs, files, rivals = plan(index, postings, atoms, ranks, coverage, arm, depth)
                selected = verified(docs, files, atoms, ranks)[1]
                response = json.dumps({"matches": [selected] if selected else []}, separators=(",", ":"))
                elapsed = (perf_counter() - begin) * 1000 + base_ms
                if set(docs) != files.keys() or len(docs) != len(files) or not 2 <= len(docs) <= 3:
                    raise ValueError("Each planned source is read once within the budget")
                if rivals is not None:
                    scope_proof(docs, files, rivals, index, selected, atoms, ranks)
                exact = bool(selected) and all(topic in subjects.get(doi, ()) for (topic, _), doi in zip(
                    atoms, selected[1:], strict=True,
                ))
                outcomes[arm].append(dict(
                    reads=len(docs), issued=bool(selected), known=exact,
                    bypasses=int(rivals is not None), alternate_reads_checked=len(rivals or ()),
                    verified_two_source=bool(selected) and len(set(selected[1:])) == 2,
                    reply_bytes=len(response), selected=selected, anchors=docs[:2], ms=elapsed,
                ))
        report[str(coverage)] = dict(known_proof_possible=upper, arms=summarize(outcomes, queries))
    return report


def prepare(index, queries, depth):
    result = []
    for atoms in queries:
        begin = perf_counter()
        request = json.loads(live.request(atoms))
        ranks = index.rank([row["topic"] for row in request["requirements"]], depth=depth)
        result.append((ranks, (perf_counter() - begin) * 1000))
    return result


def main(args):
    expected = json.loads(args.reference.read_bytes())["runs"][0]
    checkpoint = live.verify_embedding(args.embedding)
    plos, _ = BASE.pinned.load_fixture(args.fixture, [args.plos])
    elife, projection = live.elife.load_catalogue(args.elife, args.elife_manifest)
    if (
        checkpoint != expected["embedding_checkpoint"] or projection != expected["elife_projection_sha256"]
        or digest(args.fixture / "manifest.json") != expected["plos_manifest_sha256"]
        or digest(args.elife_manifest) != expected["elife_manifest_sha256"]
    ):
        raise ValueError("The source snapshots no longer match")
    if raw_identifiers(args.plos) & raw_identifiers(args.elife):
        raise ValueError("Unexpected cross-publisher ORCID overlap")
    plos_subjects = BASE.publisher_topics([args.plos])
    elife_subjects, _, elife_queries = live.elife.build_requests(elife)["topics"]
    panels = (
        ("plos", plos_subjects, BASE.query_panel(BASE.annotation_postings(plos, plos_subjects))),
        ("elife", elife_subjects, elife_queries),
    )
    encoder = QwenTorchEmbedding(args.embedding)
    output = {}
    with TemporaryDirectory(dir=args.output.resolve().parent) as directory:
        index = live.LocalIndex(merge(plos, elife), encoder, Path(directory) / "mixed")
        for name, subjects, queries in panels:
            index.rank([topic for topic, _ in queries[0]], depth=args.depth)
            rank_cache = prepare(index, queries, args.depth)
            output[name] = dict(queries=len(queries), measurements=evaluate(
                index, subjects, queries, rank_cache, args.depth, args.order_offset,
            ))
    write_json(args.output, dict(
        topics_depth=args.depth, order_offset=args.order_offset, embedding_checkpoint=checkpoint,
        plos_fixture_sha256=expected["plos_manifest_sha256"], elife_projection_sha256=projection, panels=output,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--order-offset", type=int, default=0)
    main(parser.parse_args())
