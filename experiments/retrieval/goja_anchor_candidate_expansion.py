"""Isolate boundary expansion and cue-free spans in raw GOJA extraction."""

import argparse
from itertools import chain
import os
from pathlib import Path

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.goja_experience_binding import ARCHIVE, Entity, load_ads
from experiments.retrieval.goja_raw_experience_scope import CUES, anchor_candidates, anchor_features, evaluate
from experiments.retrieval.goja_scope_nomination import TOKEN


def compound_anchor_candidates(text, vocab):
    for word in TOKEN.finditer(text):
        surface = word.group()
        for trailing in (False, True):
            for width in range(len(surface) - 1, 2, -1):
                raw = surface[-width:] if trailing else surface[:width]
                term = raw.casefold()
                if term not in vocab:
                    continue
                start, end = (word.end() - width, word.end()) if trailing else (word.start(), word.start() + width)
                span = Entity(start, end, "candidate")
                marked = any(cue in term for cue in CUES)
                yield span, (term,), anchor_features(text, span, (term,), marked, True)
                break


def with_compounds(text, vocab):
    return chain(anchor_candidates(text, vocab), compound_anchor_candidates(text, vocab))


def with_cue_free(text, vocab):
    return chain(anchor_candidates(text, vocab, include_cue_free=True), compound_anchor_candidates(text, vocab))


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a Slurm allocation and a fresh result path")
    ads = list(load_ads(args.archive, minimum_links=0))
    if args.reverse:
        ads.reverse()
    ads.sort(key=lambda row: row[0])

    routes, traces = {}, {}
    for arm, generator in (
        ("standard", anchor_candidates), ("compounds", with_compounds), ("cue_free", with_cue_free),
    ):
        traces[arm] = {}
        routes[arm] = evaluate(ads, generator, trace=traces[arm])
    matched = "compounds_matched_budget"
    traces[matched] = {}
    routes[matched] = evaluate(ads, with_compounds, trace=traces[matched],
                               anchor_limits=traces["standard"]["learned_anchor_counts"])
    groups = traces["standard"]["groups"]
    comparisons = {}
    for arm in ("compounds", "cue_free", matched):
        policies = ("learned",) if arm == matched else ("learned", "learned_best_fixed", "learned_calibrated")
        for policy in policies:
            if traces[arm]["groups"] != groups:
                raise ValueError("Changed the held-out grouping")
            left, right = (traces[name]["edge_outcomes"][policy] for name in ("standard", arm))
            comparisons[f"{arm}_vs_standard/{policy}"] = clustered_paired_interval(
                [int(b) - int(a) for a, b in zip(left, right, strict=True)], groups
            )
    write_json(args.output, dict(routes=routes, pair_recall_deltas=comparisons, source=ARCHIVE,
                                runner=digest(Path(__file__)), slurm=os.environ["SLURM_JOB_ID"], reverse=args.reverse))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
