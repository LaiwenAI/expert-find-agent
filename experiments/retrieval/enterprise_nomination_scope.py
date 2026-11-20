"""Cross frozen first-round expert nomination with qualification interpretation."""

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.retrieval.enterprise_condition_nomination import ARMS
from experiments.retrieval.enterprise_joint_headroom import scope_axes, verify_all
from experiments.retrieval.enterprise_qualification_replay import load_scope, score_profile, support
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write


def score_run(record, criteria, additions, axes, sources, verifier):
    selections = {owner: tuple(map(tuple, references))
                  for owner, references in zip(record["shortlist"], record["plans"], strict=True)}
    if len(selections) != len(record["shortlist"]):
        raise ValueError("An owner occupies more than one verification slot")
    first = verify_all(selections, criteria, sources, verifier)
    if len(first) != record["pairs"]:
        raise ValueError("Original verifier budget changed")
    extra = verify_all(selections, additions, sources, verifier)
    bank = first | {(owner, ref, i + len(criteria)): score for (owner, ref, i), score in extra.items()}
    profiles = {owner: score_profile(owner, refs, bank, len(criteria) + len(additions))
                for owner, refs in selections.items()}
    passed = {name: {owner for owner, profile in profiles.items() if support(profile, conditions)}
              for name, conditions in axes.items()}
    if passed["original"] != {item["expert"] for item in record["issued"]}:
        raise ValueError("Original model-issued experts did not reproduce")
    return passed, len(bank)


def run(args):
    import torch
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.enterprise_first_query import CRITERION_CHARS

    require_cuda(args.output)
    anchor = json.loads(args.base_report.read_bytes())
    nominations = json.loads(args.nomination_report.read_bytes())
    supplied = (
        (args.queries, anchor["queries_sha256"]),
        (args.sources, anchor["source_sha256"]),
        (args.base_trace, anchor["trace_sha256"]),
        (args.nomination_trace, nominations["trace_sha256"]),
    )
    if any(digest(path) != expected for path, expected in supplied):
        raise ValueError("Changed pinned development inputs")
    if (nominations["query_sha256"], nominations["source_sha256"]) != (
            anchor["queries_sha256"], anchor["source_sha256"]):
        raise ValueError("Mixed development projections")
    queries = {r["id"]: r["text"] for r in rows(args.queries)}
    sources = {r["id"]: r for r in rows(args.sources)}
    old = {r["id"]: r["arms"]["hybrid_prefixes"]
           for r in json.loads(args.base_trace.read_bytes())["observations"]}
    new = {r["id"]: r["arms"] for r in json.loads(args.nomination_trace.read_bytes())["observations"]}
    scope = load_scope(anchor, json.loads(args.scope_report.read_bytes()), args.scope_trace, queries)
    if queries.keys() != old.keys() or queries.keys() != new.keys() or scope.keys() != {
            key for key, record in old.items() if record["qualifications"]}:
        raise ValueError("Changed multi-condition development panel")
    torch.backends.cuda.matmul.allow_tf32 = False
    verifier = Reranker(args.verifier)
    if verifier.provenance["revision"] != anchor["models"]["verifier"]:
        raise ValueError("Changed verifier checkpoint")

    totals = {arm: {name: Counter(issued_requests=0, people=0, people_missing_original_checks=0,
                                  gained_requests_vs_hybrid=0, lost_requests_vs_hybrid=0)
                    for name in ("original", "in_both", "direct_gate", "union", "topic_only")}
              for arm in ARMS}
    checks = Counter()
    for key in sorted(scope, reverse=bool(args.order_offset)):
        text = queries[key]
        qualifiers = old[key]["qualifications"]
        extra, axes = scope_axes(qualifiers, scope[key])
        criteria = (text.strip().splitlines()[0][:CRITERION_CHARS], *(text[a:b] for a, b in qualifiers))
        additions = tuple(text[a:b] for a, b in extra)
        if any(not 0 < len(clause) <= CRITERION_CHARS for clause in criteria + additions):
            raise ValueError("Unexpected clause length")
        verified = {arm: score_run(new[key][arm], criteria, additions, axes, sources, verifier)
                    for arm in ARMS}
        control = verified["hybrid_prefix"][0]
        for arm, (outcomes, budget) in verified.items():
            checks[arm] += budget
            for name, passed in outcomes.items():
                totals[arm][name].update(
                    issued_requests=bool(passed), people=len(passed),
                    people_missing_original_checks=len(passed - outcomes["original"]),
                    gained_requests_vs_hybrid=bool(passed) and not control[name],
                    lost_requests_vs_hybrid=not passed and bool(control[name]),
                )
    if len(set(checks.values())) != 1:
        raise ValueError("Joint verification exposure differs across nomination arms")
    write(args.output, dict(
        arms=totals, verification_checks=dict(checks), queries=len(scope),
        scope="Model outputs for pinned development nominations and sampled qualification interpretations.",
        nomination_trace_sha256=digest(args.nomination_trace), qualification_report_sha256=digest(args.scope_report),
        script_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "sources", "base_report", "base_trace", "nomination_report", "nomination_trace",
                 "scope_report", "scope_trace", "verifier", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    run(parser.parse_args())
