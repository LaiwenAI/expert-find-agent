"""Audit known-positive headroom before the staged role-verification model."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposal
from experiments.docred import noisy_role_source_routing as routing

FIRST = 3


def run(snapshot, extraction, output):
    if output.exists():
        raise FileExistsError(output)
    docs = auto.load_documents(snapshot)
    raw = json.loads(extraction.read_bytes())
    if raw["inputs"] != index.HASHES or len(raw["annotations"]) != len(docs):
        raise ValueError("Changed NER projection")
    spans = raw["annotations"]
    _, original = index.mentions_index(docs)
    aliases, profiles = auto.auto_index(docs, spans, original)
    lookup = routing.AliasLookup(aliases)
    postings = index.bm25_index(docs)
    judgments = auto.person_alias_judgments(docs)
    examples = index.study_queries(docs, 500)
    views = [dict(query, style="exact") for query in examples] + routing.query_styles(examples, docs)

    def known(pairs, positives):
        return {(doc, judgments[doc].get(name, name)) for doc, name in pairs} & positives

    panels = defaultdict(Counter)
    for query in views:
        counts = panels[query["style"]]
        counts["queries"] += 1
        sources, _, uncertain = routing.route(query, postings, lookup, "swap_then_guard")
        initial = index.rank_people(query, sources, profiles)[0]
        slots = max(0, len(initial) - FIRST)
        if not slots:
            continue
        counts["queries_with_second_stage"] += 1
        counts["uncertain_second_stage_queries"] += uncertain
        positives = query["positive"]
        already = known(initial[:FIRST], positives)
        depth = known(initial[FIRST:], positives) - already
        expanded = {}
        for doc in sources:
            _, work, cues = profiles[doc]
            expanded[doc] = (
                [(name, tuple(sorted(positions))) for name, positions, _ in
                 proposal.pool(docs[doc], query, profiles[doc], spans[doc]) if (doc, name) not in initial],
                work, cues,
            )
        people = known(index.rank_people(query, sources, expanded)[0], positives) - already
        extra = [doc for doc in routing.bm25_scores(query, postings) if doc not in sources]
        from_sources = known(index.rank_people(query, extra, profiles)[0], positives) - already
        counts["depth_positives"] += len(depth)
        counts["identity_pool_positives_not_in_depth"] += len(people - depth)
        counts["source_pool_positives_not_in_depth"] += len(from_sources - depth)
        max_supported = min(slots, len(depth | people | from_sources))
        counts["oracle_possible_extra_over_depth"] += max_supported - len(depth)
        counts["queries_with_oracle_gain"] += max_supported > len(depth)
    index.write_json(output, dict(
        audit_sha256=index.digest(Path(__file__)), projection_sha256=index.digest(extraction),
        inputs=index.HASHES, phases={a: dict(sorted(b.items())) for a, b in sorted(panels.items())},
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("snapshot", "extraction", "output"):
        cli.add_argument(k, type=Path)
    run(**vars(cli.parse_args()))
