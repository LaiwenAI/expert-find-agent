"""Compare local and catalog-wide rescue while holding recommendation counts fixed."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.credit import role_first_query_equal_reads as live
from experiments.credit import role_first_query_factorial as factorial
from experiments.credit import role_first_query_router as router
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json

ARMS = ("topic_fixed", "topic_global_fixed")
MODES = ("first_stage_count", "minimum_final_count")


def compare_lists(local, extended, tagged, k):
    if min(len(local), len(extended)) < k:
        raise ValueError("Unable to preserve the pre-specified output count")
    control, candidate = local[:k], extended[:k]
    left, right = set(control), set(candidate)
    removed, added = left - right, right - left
    delta = sum(source in tagged for _, source in added) - sum(source in tagged for _, source in removed)
    return dict(
        issued_each=k, changed_ranked_list=int(control != candidate),
        changed_people=len({person for person, _ in added} - {person for person, _ in removed}),
        replaced_witnesses=len(added), removed_tagged=sum(source in tagged for _, source in removed),
        added_tagged=sum(source in tagged for _, source in added), tagged_delta=delta,
    ), removed, added


def case_id(request, doi, ordinal):
    return sha256(json.dumps([request, doi, ordinal], ensure_ascii=True).encode()).hexdigest()[:24]


def review_cards(packet, request, catalog, pair, witnesses):
    ids = []
    for person, source in sorted(witnesses):
        article = catalog.articles[source]
        author = next(a for a in article["authors"]
                      if (a["person"] or f'{article["doi"]}#{a["ordinal"]}') == person)
        uid = case_id(request, article["doi"], author["ordinal"])
        packet[uid] = dict(case_id=uid, topic=request[2], requested_roles=pair, doi=article["doi"],
                           author_ordinal=author["ordinal"], roles=author["roles"],
                           title_and_abstract=article["text"])
        ids.append(uid)
    return ids


def evaluate(catalog, publisher, cutoff, packet, scores_by_contrast):
    selected = router.panels(catalog, publisher)
    topics = {topic for _, queries in selected.values() for topic, _, _ in queries}
    orders = {topic: catalog.rank(topic) for topic in sorted(topics)}
    known = {i for i, row in enumerate(catalog.articles) if row.get("declaration_complete", True)}
    score_cache = {}

    def score(topic, sources):
        key = topic, tuple(sources)
        if key not in score_cache:
            score_cache[key] = catalog.verify(topic, sources)
        return score_cache[key]

    panels = {}
    for field, (lookup, queries) in selected.items():
        if publisher == "elife":
            queries = [row for row in queries if int.from_bytes(
                pilot.stable("factorial-calibration-v1", row[0])[:8], "big",
            ) % 3 != 0]
        for mask, inventory in catalog.visible.items():
            totals = {mode: Counter() for mode in MODES}
            gains = {mode: [] for mode in MODES}
            for topic, rarity, pair in queries:
                proxy = SimpleNamespace(rank=lambda query: orders[query], verify=score, person=catalog.person,
                                        bindings=catalog.bindings, topics=lookup)
                local, extended = (
                    live.replay(proxy, topic, pair, known & inventory, cutoff, arm,
                                include_witnesses=True) for arm in ARMS
                )
                if local["first_witnesses"] != extended["first_witnesses"]:
                    raise ValueError("Rescue arms must share the same first-stage evidence")
                caps = (len(local["first_witnesses"]), min(local["issued"], extended["issued"]))
                for mode, cap in zip(MODES, caps, strict=True):
                    counts, removed, added = compare_lists(local["witnesses"], extended["witnesses"],
                                                           lookup[topic], cap)
                    totals[mode].update(counts)
                    gains[mode].append(counts["tagged_delta"] / pilot.TOP)
                    if removed:
                        request = publisher, field, topic, rarity, pair
                        direction = dict(remove=review_cards(packet, request, catalog, pair, removed),
                                         add=review_cards(packet, request, catalog, pair, added))
                        scores_by_contrast.append(dict(publisher=publisher, field=field, mask=mask,
                                                       mode=mode, topic=topic, **direction))
            panels[f"{field}:{mask}"] = {
                mode: dict(totals[mode], queries=len(queries), paired_p5_delta=pilot.clustered_paired_interval(
                    gains[mode], [topic for topic, _, _ in queries],
                )["percentile_95"])
                for mode in MODES
            }
    return panels


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("elife", "plos", "plos_manifest", "reference", "embedding", "reranker", "output"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--review-packet", type=Path)
    cli.add_argument("--private-key", type=Path)
    args = cli.parse_args()
    if bool(args.review_packet) != bool(args.private_key):
        raise ValueError("Supply both review-packet and private-key paths, or neither")
    spec = json.loads(args.reference.read_bytes())
    plos = json.loads(args.plos_manifest.read_bytes())
    if digest(args.elife) != spec["projection_sha256"] or digest(args.plos) != plos["projection_sha256"]:
        raise ValueError("Unexpected publisher projection")
    for p, field in ((args.embedding, "embedding_sha256"),
                     (args.reranker / factorial.Verifier.name, "reranker_sha256")):
        if {name: digest(p / name) for name in spec[field]} != spec[field]:
            raise ValueError("Unexpected checkpoint")

    encoder, verifier = factorial.Encoder(args.embedding), factorial.Verifier(args.reranker)
    packet, key, result = {}, [], {}
    for pub in ("elife", "plos"):
        projection = getattr(args, pub)
        catalog = pilot.Catalog(json.loads(projection.read_bytes())["sources"], encoder, verifier)
        result[pub] = evaluate(catalog, pub, spec["cutoff"], packet, key)
    if args.review_packet:
        write_json(args.review_packet, [packet[uid] for uid in sorted(packet)])
        write_json(args.private_key, key)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), runner_sha256=digest(Path(live.__file__)),
        reference_sha256=digest(args.reference), elife_sha256=digest(args.elife),
        plos_sha256=digest(args.plos), review_cards=len(packet), review_contrasts=len(key), results=result,
    ))


if __name__ == "__main__":
    main()
