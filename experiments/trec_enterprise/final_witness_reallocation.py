"""Test one-check reallocation with matched documents and verification work."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise import joint_owner_source_plan as reference

ARMS = ("legacy", "fixed_last_check", "open_last_check")


def finished(plan, order, source_labels, expert_labels):
    confirmed = {owner for owner, doc in plan if source_labels.get(owner, {}).get(doc) == 2}
    issued = [owner for owner in order if owner in confirmed][:5]
    supported = sum(expert_labels.get(owner) == 2 for owner in issued)
    return dict(supported=supported, issued=len(issued), wrong=len(issued)-supported,
                reads=len({doc for _, doc in plan}), checks=len(plan),
                unknown=sum(source_labels.get(owner, {}).get(doc) is None for owner, doc in plan))


def alternatives(trace, fixed, wide, scores, source_labels):
    if not trace:
        return dict.fromkeys(ARMS, None), 0, False
    last = trace[-1]
    prior = set(trace[:-1])
    confirmed = {owner for owner, doc in prior if source_labels.get(owner, {}).get(doc) == 2}
    fixed_owners = set(fixed)
    ranks = {owner: i for i, owner in enumerate(wide)}
    if len(confirmed) >= 5:
        return dict.fromkeys(ARMS, last), 0, False
    # Pin the last source itself to match reads as well as checks.
    choices = [pair for pair in scores if pair[1] == last[1] and pair not in prior and pair[0] not in confirmed]
    key = lambda pair: (-scores[pair], ranks[pair[0]])
    fixed_choice = min((pair for pair in choices if pair[0] in fixed_owners), key=key)
    wide_choice = min(choices, key=key)
    return dict(legacy=last, fixed_last_check=fixed_choice, open_last_check=wide_choice), len(choices), any(
        owner not in fixed_owners for owner, _ in choices
    )


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use an allocated Slurm compute node and fresh output")
    experts, docs = reference.load(args.qrels)
    topics = sorted(experts, key=int)
    vectors = reference.lexical_vectors(topics, reference.read_topics(args.qrels / "ent06.expert.topics"))
    archive = json.loads(args.archive.read_bytes())
    archived_legacy = archive["runs"][0]["results"]["fitted_legacy"]
    results = {}
    diagnostic = Counter()
    for topic in (topics[::-1] if args.reverse else topics):
        graph = reference.HistoricalIndex({held: labels for held, labels in docs.items() if held != topic},
                                          topics, vectors)
        legacy, wide_model, _ = reference.fit_models(graph, topics, topic, experts, docs)
        sections = reference.nominations(graph, topic)
        people, sources = sections["expert_first"]
        wide, fixed = people[:reference.UNIVERSE], people[:reference.PERSON_CAP]
        doc_order, doc_sources = sections["document_first"]
        src, exp = docs[topic], experts[topic]
        starter = reference.allocate(doc_order[:reference.PERSON_CAP], doc_sources, src, exp, 3, "rank_first")
        cap = reference.allocate(fixed, sources, src, exp, starter["reads"], "reuse_first")
        keys, x = reference.features(graph, topic, fixed, sources)
        legacy_scores = dict(zip(keys, map(float, reference.estimate(legacy, x)), strict=True))
        keys, x = reference.features(graph, topic, wide, sources)
        wide_scores = dict(zip(keys, map(float, reference.estimate(wide_model, x)), strict=True))
        trace = []
        expected = reference.schedule(fixed, legacy_scores, src, exp, cap["reads"], False, cap["checks"], trace=trace)
        edit, width, expandable = alternatives(trace, fixed, wide, wide_scores, src)
        diagnostic.update(last_source_pairs=width, open_owner_alternative=int(expandable))
        per_topic = {}
        for name, action in edit.items():
            new_plan = [*trace[:-1], action] if action is not None else []
            # Existing verified results keep their fixed priority. An added
            # owner can fill an empty position or replace the changed action.
            metrics = finished(new_plan, list(dict.fromkeys([*fixed, *wide])), src, exp)
            if (metrics["reads"], metrics["checks"]) != (expected["reads"], expected["checks"]):
                raise ValueError("Changed exact comparison budget")
            metrics["changed_last_check"] = int(bool(trace) and action != trace[-1])
            metrics["new_nominee_checked"] = int(action is not None and action[0] not in fixed)
            per_topic[name] = metrics
        if any(per_topic["legacy"][key] != expected[key] for key in expected):
            raise ValueError("Reference trace did not reproduce")
        results[topic] = per_topic

    totals = {name: dict(sum((Counter(results[t][name]) for t in topics), Counter())) for name in ARMS}
    if any(totals["legacy"][key] != archived_legacy[key] for key in ("reads", "checks", "unknown", "issued", "supported")):
        raise ValueError("Archived reference did not reproduce")
    comparisons = {name: {
        control: paired_interval([20 * (results[t][name]["supported"] - results[t][control]["supported"])
                                  for t in topics])
        for control in ("legacy", "fixed_last_check")
    } for name in ARMS[1:]}
    write_json(args.output, dict(
        topics=len(topics), totals=totals, diagnostics=diagnostic, supported_p5_delta_pp=comparisons,
        outcomes_sha256=sha256(json.dumps(results, sort_keys=True).encode()).hexdigest(),
        previous=digest(args.archive), scripts={str(path.name): digest(path) for path in
                                                 (Path(__file__), Path(reference.__file__))},
        inputs=reference.FILES | {"ent06.expert.topics": reference.TOPIC_SHA256},
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("qrels", "archive", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reverse", action="store_true")
    run(cli.parse_args())
