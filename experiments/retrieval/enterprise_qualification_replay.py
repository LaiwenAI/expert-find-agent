"""Replay competing eligibility interpretations on the same people and evidence."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.retrieval import enterprise_role_first_query as roles
from experiments.retrieval.enterprise_joint_headroom import verify_all
from experiments.retrieval.enterprise_routed_review import digest, rows, write

ARMS = ("original", "in_both", "direct_gate", "union", "topic_only")


def interpretations(original, compared):
    original, compared = set(original), set(compared)
    return dict(original=sorted(original), in_both=sorted(original & compared),
                direct_gate=sorted(compared), union=sorted(original | compared), topic_only=[])


def score_profile(owner, references, bank, criteria_count):
    return [max((bank[owner, ref, i] for ref in references), default=float("-inf")) > 0
            for i in range(criteria_count)]


def support(profile, indices):
    return profile[0] and all(profile[i] for i in indices)


def run(args):
    import torch

    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.enterprise_first_query import CRITERION_CHARS

    roles.require_cuda(args.output)
    baseline = json.loads(args.baseline.read_bytes())
    direct = json.loads(args.scope_aggregate.read_bytes())
    for expected, path in ((baseline["trace_sha256"], args.trace),
                           (baseline["queries_sha256"], args.queries),
                           (baseline["source_sha256"], args.corpus),
                           (direct["private_sha256"], args.scope_trace)):
        if digest(path) != expected:
            raise ValueError("Changed frozen diagnostics")
    if direct["trace_sha256"] != baseline["trace_sha256"] or direct["source_sha256"] != baseline["queries_sha256"]:
        raise ValueError("Scope comparison used another request population")
    origin = {r["id"]: r["text"] for r in rows(args.queries)}
    corpus = {r["id"]: r for r in rows(args.corpus)}
    traced = json.loads(args.trace.read_bytes())["observations"]
    audit = json.loads(args.scope_trace.read_bytes())
    by_query = defaultdict(dict)
    for row, outcome in zip(audit["cases"], audit["new_gate"], strict=True):
        start, end = row["start"], row["end"]
        text = origin[row["id"]]
        if row["target"] != text[start:end] or row["full_request"] != text or type(outcome) is not bool:
            raise ValueError("Scope decisions do not bind the original request")
        key = start, end
        if key in by_query[row["id"]]:
            raise ValueError("Duplicate scope comparison excerpt")
        by_query[row["id"]][key] = row["chosen"], outcome

    reader = Reranker(args.reranker)
    if reader.provenance["revision"] != baseline["models"]["verifier"]:
        raise ValueError("Changed enterprise verifier")
    torch.backends.cuda.matmul.allow_tf32 = False
    totals = {key: Counter(issued_requests=0, provisionals=0, clauses=0,
                           requests_with_no_qualification=0, retained_original_checks=0,
                           provisionals_without_all_original=0,
                           requests_with_pass_on_any_retained_qualification=0) for key in ARMS}
    one_drop = Counter(recoverable_requests=0, recovered_people=0, requests_with_multiple_relaxations=0)
    pair_count = 0
    if args.clarification:
        from experiments.retrieval.enterprise_qualification_questions import examine

        questions, cost = Counter(), defaultdict(float)

    for row in traced:
        key = row["id"]
        if key not in by_query:
            continue
        original = row["arms"]["hybrid_prefixes"]
        previous = [tuple(p) for p in original["qualifications"]]
        votes = by_query[key]
        if sorted(previous) != sorted(span for span,(selected, _) in votes.items() if selected) or original["issued"]:
            raise ValueError("Changed baseline requirement set or original abstention")
        index = {span: i + 1 for i, span in enumerate(previous)}
        newly_qualified = [span for span, (_, positive) in votes.items() if positive]
        additional = sorted(set(newly_qualified) - index.keys())
        index.update({span: len(previous) + i + 1 for i, span in enumerate(additional)})
        original_indices = [index[span] for span in previous]
        compare_indices = [index[span] for span in newly_qualified]
        text = origin[key]
        topic = text.strip().splitlines()[0][:CRITERION_CHARS]
        first = (topic, *(text[a:b] for a,b in previous))
        extra = tuple(text[a:b] for a,b in additional)
        if any(len(criterion) > CRITERION_CHARS for criterion in first + extra):
            raise ValueError("Unexpected overlength clause in frozen replay")
        options = {
            expert: tuple(map(tuple, refs))
            for expert, refs in zip(original["shortlist"], original["plan"], strict=True)
        }
        # Preserve the original forward batches for all reused scores.
        bank = verify_all(options, first, corpus, reader)
        if extra:
            bank.update({(expert, ref, len(first) + i): margin for (expert, ref, i), margin
                         in verify_all(options, extra, corpus, reader).items()})
        pair_count += len(bank)
        profiles = [score_profile(expert, refs, bank, len(first) + len(extra))
                    for expert, refs in options.items()]
        if args.clarification:
            observed, weights = examine(profiles, original_indices, compare_indices)
            questions.update(observed)
            for k, value in weights.items():
                cost[k] += value
        for arm, indices in interpretations(original_indices, compare_indices).items():
            approved = [profile for profile in profiles if support(profile, indices)]
            if arm == "original" and approved:
                raise ValueError("Original abstention did not reproduce")
            totals[arm].update(issued_requests=int(bool(approved)), provisionals=len(approved), clauses=len(indices),
                               requests_with_no_qualification=int(not indices),
                               retained_original_checks=len(set(indices) & set(original_indices)),
                               provisionals_without_all_original=sum(not support(p, original_indices) for p in approved),
                               requests_with_pass_on_any_retained_qualification=int(any(
                                   any(p[i] for i in indices) for p in profiles
                               )))
        disputed = set(original_indices) - set(compare_indices)
        recovered = set()
        switches = 0
        for omitted in disputed:
            others = [i for i in original_indices if i != omitted]
            diagnosed = {k for k, profile in enumerate(profiles) if support(profile, others)}
            switches += bool(diagnosed)
            recovered.update(diagnosed)
        one_drop.update(recoverable_requests=int(bool(recovered)), recovered_people=len(recovered),
                        requests_with_multiple_relaxations=int(switches > 1))

    if len(by_query) != baseline["multi_condition"] or len(traced) != len(origin):
        raise ValueError("Changed replay panel")
    clarification = {"clarification": dict(questions, **cost)} if args.clarification else {}
    write(args.output, dict(
        arms=totals, **clarification, single_disputed_clause_omission=one_drop, distinct_scored_pairs=pair_count,
        fixed_requests=len(by_query), original_trace_sha256=digest(args.trace),
        scope_aggregate_sha256=digest(args.scope_aggregate), script_sha256=digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), checkpoint=reader.provenance["revision"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "trace", "baseline", "scope_trace", "scope_aggregate", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--clarification", action="store_true")
    run(parser.parse_args())
