"""Diagnose whether within-document entity grounding helps recover conjunctive evidence."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.docred import conjunctive_proof_windows as proofs
from experiments.docred import person_bound_evidence as entities
from experiments.followir.core17_reranking import digest, write_json


MODES = ("canonical", "distinct_aliases", "annotated_mentions")
POLICIES = ("global_names", "conjunction")


def signals(case, document, mode):
    global_names, scoped = proofs.features(case)
    if mode == "canonical":
        return global_names, scoped
    terms = list(dict.fromkeys(name for pair in case["names"] for name in pair))
    origins = defaultdict(list)
    owners = defaultdict(set)
    for i, cluster in enumerate(document["vertexSet"]):
        origins[entities.canonical(cluster)].append(cluster)
        for mention in cluster:
            owners[tuple(entities.tokens(mention["name"]))].add(i)
    if any(len(origins[name]) != 1 for name in terms):
        return None
    words, _, _ = entities.profile(case["sentences"])
    for pos, name in enumerate(terms):
        cluster = origins[name][0]
        if mode == "annotated_mentions":
            indexes = {m["sent_id"] for m in cluster}
        else:
            aliases = {
                tuple(entities.tokens(m["name"])) for m in cluster
                if m["name"] and m["name"][0].isupper() and len(m["name"]) >= 4
            }
            aliases = {form for form in aliases if form and len(owners[form]) == 1}
            indexes = {i for i, words_i in enumerate(words)
                       if any(entities.contains(words_i, form) for form in aliases)}
        for idx in indexes:
            global_names[idx, pos] = 1.
    features = dict(zip(terms, global_names.T[:-1], strict=True))
    cue = global_names[:, -1]
    scoped = np.column_stack([
        .65*np.minimum(features[work], features[owner])
        + .2*np.maximum(features[work], features[owner]) + .15*cue
        for work, owner in case["names"]
    ])
    return global_names, scoped


def run(release, output):
    if output.exists():
        raise FileExistsError(output)
    pools, hashes = proofs.load(release)
    documents = {name: {d["title"]: d for d in json.loads((release / f"{split}_revised.json").read_bytes())}
                 for name, split in (("calibration", "dev"), ("test", "test"), ("external", "train"))}
    results = {}
    for name, panel in pools.items():
        table = defaultdict(Counter)
        bounds = Counter()
        skipped = 0
        for case in panel:
            views = {mode: signals(case, documents[name][case["doc"]], mode) for mode in MODES}
            if any(view is None for view in views.values()):
                skipped += 1
                continue
            for lhs, rhs in proofs.PAIR:
                a, b = (set(case["annotated"][i]) for i in (lhs, rhs))
                bounds.update(both_annotated=int(bool(a and b)),
                              fits_windows=int(bool(a and b and len(a) <= 2 and len(b) <= 2)))
            for mode, (global_names, scoped) in views.items():
                for policy in POLICIES:
                    picked = proofs.choose_windows(global_names, scoped, policy)
                    assigned = proofs.assign_windows(scoped, picked)
                    attainable = [bool(gold) and set(gold).issubset(window)
                                  for gold, window in zip(case["annotated"], assigned, strict=True)]
                    table[f"{mode}/{policy}"].update(
                        atom_available=sum(attainable),
                        whole_proof=sum(all(attainable[i] for i in pair) for pair in proofs.PAIR),
                    )
        results[name] = dict(cases=len(panel), skipped_ambiguous=skipped, bounds=bounds, arms=table)
    write_json(output, dict(script_sha256=digest(Path(__file__)), input_hashes=hashes, groups=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.release, args.output)
