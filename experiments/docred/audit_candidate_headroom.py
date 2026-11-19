"""Locate retrieval and ranking losses before last-slot relationship verification."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import compare_candidate_evidence as comparison
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposal
from experiments.docred import noisy_role_source_routing as routing


def audit(snapshot, projection):
    documents = auto.load_documents(snapshot)
    extracted = json.loads(projection.read_bytes())
    if extracted["inputs"] != index.HASHES or len(extracted["annotations"]) != len(documents):
        raise ValueError("Incorrect NER projection")
    spans = extracted["annotations"]
    _, annotated = index.mentions_index(documents)
    aliases, profiles = auto.auto_index(documents, spans, annotated)
    lookup = routing.AliasLookup(aliases)
    postings = index.bm25_index(documents)
    identities = auto.person_alias_judgments(documents)

    def resolve(pair):
        doc, name = pair
        return doc, identities[doc].get(name, name)

    panels = defaultdict(Counter)
    for query in auto.prepare_views(documents):
        counts = panels[query["style"]]
        counts["queries"] += 1
        docs = proposal.noisy_sources(query, postings, lookup)
        baseline, _ = index.rank_people(query, docs, profiles)
        if not baseline:
            counts["empty_baseline"] += 1
            continue
        positives = query["positive"]
        if resolve(baseline[-1]) in positives:
            counts["last_slot_already_positive"] += 1
            continue
        unseen = positives - set(map(resolve, baseline))
        if not unseen:
            counts["no_additional_annotated_positive"] += 1
        elif not any(doc in docs for doc, _ in unseen):
            counts["additional_positive_source_missed"] += 1
        elif not any(
            resolve((doc, name)) in unseen
            for doc in docs
            for name, _, _ in proposal.pool(documents[doc], query, profiles[doc], spans[doc])
        ):
            counts["additional_positive_person_missing_from_proposals"] += 1
        else:
            challenger = comparison.alternative(query, docs, documents, profiles, spans, baseline)
            if challenger is not None and resolve(challenger) in unseen:
                counts["additional_positive_is_top_challenger"] += 1
            else:
                counts["additional_positive_ranked_below_top_challenger"] += 1

    return {
        "code_sha256": index.digest(Path(__file__)),
        "comparison_code_sha256": index.digest(Path(comparison.__file__)),
        "projection_sha256": index.digest(projection),
        "inputs": index.HASHES,
        "panels": {panel: dict(sorted(counts.items())) for panel, counts in sorted(panels.items())},
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "projection", "output"):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Use a new output file")
    index.write_json(args.output, audit(args.snapshot, args.projection))
