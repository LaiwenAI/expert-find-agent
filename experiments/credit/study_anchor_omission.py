"""Test recovery from one missing interior title token under catalog outages."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.credit.study_anchor_lookup import Index, load_catalog, queries
from experiments.followir.core17_reranking import digest, write_json


def omission_signatures(span):
    for i in range(1, len(span) - 1):
        yield span[:i] + span[i + 1:]


def omission_matches(literal, clue, exact):
    # Every interior omission in a six-token span leaves one end trigram intact.
    shortlist = set(literal.spans.get(clue[:3], ())).union(literal.spans.get(clue[-3:], ()))
    matches = set(exact)
    for doi in shortlist - matches:
        title = literal.titles[doi]
        if any(clue in omission_signatures(title[start:start + 6])
               for start in range(len(title) - 5)):
            matches.add(doi)
    return matches, len(shortlist)


def materialize_omissions(literal):
    expanded = defaultdict(set)
    for doi, title in literal.titles.items():
        for start in range(len(title) - 5):
            for signature in omission_signatures(title[start:start + 6]):
                expanded[signature].add(doi)
    return expanded


def unique_available(matches, unavailable, visible_only=False):
    eligible = matches - {unavailable} if visible_only else matches
    if len(eligible) == 1:
        doi = next(iter(eligible))
        return doi if doi != unavailable else None
    return None


def damaged_views(catalog, literal):
    frequent = sorted(literal.terms, key=lambda term: (-len(literal.terms[term]), term))[:32]
    for doi, group, style, clue in queries(catalog, literal):
        if style != "damaged":
            continue
        seed = int.from_bytes(sha256(("substitution:" + doi).encode()).digest()[:8], "big")
        choices = [term for term in frequent if term not in clue]
        position = 1 + seed % 3
        replacement = choices[(seed // 3) % len(choices)]
        yield doi, group, "one_omission", clue
        yield doi, group, "omission_plus_substitution", (
            clue[:position] + (replacement,) + clue[position + 1:]
        )


def compare(catalog):
    literal = Index(catalog)
    materialized = materialize_omissions(literal)
    panel = list(damaged_views(catalog, literal))
    calibration = defaultdict(list)
    for doi, group, style, clue in panel:
        if group == 0:
            calibration[style].append(literal.rank(clue, doi)[1])
    cutoffs = {style: float(np.quantile(scores, .95, method="higher"))
               for style, scores in calibration.items()}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    collisions = defaultdict(Counter)
    for doi, group, style, clue in panel:
        if group == 0:
            continue
        exact = literal.matches(clue)
        expanded, shortlist_size = omission_matches(literal, clue, exact)
        if expanded != exact.union(materialized.get(clue, ())):
            raise AssertionError("Trigram shortcut and materialized signatures disagree")
        collisions[style].update(
            queries=1, checked_title_upper_bound=shortlist_size,
            target_in_matches=int(doi in expanded),
            ambiguous=int(len(expanded) > 1), unique_other=int(len(expanded) == 1 and doi not in expanded),
        )
        for mode in ("available", "registered_outage", "absent_from_inventory"):
            unavailable = None if mode == "available" else doi
            visible_only = mode == "absent_from_inventory"
            ranked, score = literal.rank(clue, unavailable)
            selected = {
                "bm25_calibrated": ranked if score > cutoffs[style] else None,
                "frozen_literal": unique_available(exact, unavailable, visible_only),
                "visible_omission": unique_available(expanded, unavailable, True),
                "frozen_omission": unique_available(expanded, unavailable, visible_only),
            }
            for arm, target in selected.items():
                correct = target == doi and mode == "available"
                results[style][mode][arm].update(
                    cases=1, correct=int(correct), wrong=int(target is not None and not correct),
                    unresolved=int(target is None),
                )

    return dict(
        title_articles=len(catalog), calibration_cutoffs=cutoffs,
        trigram_keys=len(literal.spans),
        materialized_omission_keys=len(materialized),
        materialized_omission_postings=sum(len(postings) for postings in materialized.values()),
        collisions=collisions, results=results,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("captures", type=Path, nargs="+")
    args = cli.parse_args()
    catalog, receipts = load_catalog(args.captures)
    write_json(args.output, dict(capture_sha256=receipts, script_sha256=digest(Path(__file__)), **compare(catalog)))
