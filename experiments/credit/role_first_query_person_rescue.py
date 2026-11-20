"""Spend a fixed rescue budget on useful candidate identities, not duplicate reads."""

import argparse
import json
from pathlib import Path

from experiments.credit import role_first_query_equal_reads as live
from experiments.credit import role_first_query_factorial as factorial
from experiments.credit import role_first_query_swaps as swaps
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json


TOPICAL_LIMIT = 16


def selector(catalog, pair, visible):
    def choose(ranked, available, incumbent_witnesses):
        incumbent = {person for person, _ in incumbent_witnesses}
        considered = set(ranked[:TOPICAL_LIMIT])
        shortlist = [source for source in available if source in visible and source in considered]
        picked = []
        for source in shortlist:
            owners = set(catalog.bindings.get((source, pair), ()))
            if owners - incumbent:
                picked.append(source)
                incumbent.update(owners)
                if len(picked) == live.EXTRA:
                    break
        picked_set = set(picked)
        picked.extend(source for source in available if source not in picked_set)
        return picked
    return choose


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
    reports = {}
    packet, contrast_key = {}, []
    for pub in ("elife", "plos"):
        projection = getattr(args, pub)
        catalog = pilot.Catalog(json.loads(projection.read_bytes())["sources"], encoder, verifier)
        cache = {}
        ordinary = swaps.evaluate(catalog, pub, spec["cutoff"], {}, [], cached_scores=cache)
        directed = swaps.evaluate(catalog, pub, spec["cutoff"], packet, contrast_key,
                                  selector_factory=selector, cached_scores=cache)
        reports[pub] = dict(ordinary=ordinary, new_person_priority=directed)
    if args.review_packet:
        write_json(args.review_packet, [packet[key] for key in sorted(packet)])
        write_json(args.private_key, contrast_key)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), swap_runner_sha256=digest(Path(swaps.__file__)),
        read_runner_sha256=digest(Path(live.__file__)), model_reference_sha256=digest(args.reference),
        topical_limit=TOPICAL_LIMIT, reports=reports,
        review_cards=len(packet), review_contrasts=len(contrast_key),
    ))


if __name__ == "__main__":
    main()
