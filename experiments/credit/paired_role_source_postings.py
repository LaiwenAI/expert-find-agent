"""Join role-feasible sources before spending the remaining verifier forward."""

from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path

from experiments.credit.budgeted_source_role_search import role_check
from experiments.credit.engineering_cross_source_search import candidates, required_roles
from experiments.credit.independent_source_confirmation import weaker_side
from experiments.credit.paired_role_stage_audit import query_state, signatures
from experiments.credit.paired_role_verification import alternative_roles, apply
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_confirmation_transfer import verify
from experiments.credit.source_role_factorial_diagnosis import payload
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("stock", "postings_weak_side", "postings_both_sides")


def pairs_for(index, ranking, required):
    postings = []
    for side, roles in zip(ranking, required, strict=True):
        by_owner = defaultdict(list)
        for paper in side["fused"]:
            for owner, declared in index.articles[paper]["roles"].items():
                if roles.issubset(declared):
                    by_owner[owner].append(paper)
        postings.append(by_owner)
    first, second = postings
    pairs = {
        (a, b)
        for owner in first.keys() & second.keys()
        for a, b in product(first[owner], second[owner]) if a != b
    }

    def priority(pair):
        gaps = [side["best"] - side["cosine"][paper] for side, paper in zip(ranking, pair, strict=True)]
        fused_rank = sum(side["rank"][paper] for side, paper in zip(ranking, pair, strict=True))
        return max(gaps), fused_rank, pair

    return sorted(pairs, key=priority)


def exact_role(service, state, scores, phrases, cutoff):
    index = service.index
    pair, base, response = state["pair"], state["roles"], state["response"]
    if not pair or response["calls"] or response["owners"]:
        return response
    checked, calls, tokens, _ = role_check(index, pair, scores, service.bands, service.role,
                                          phrases, service.cutoffs["role"])
    if calls or candidates(index, pair, checked):
        return payload(index, pair, checked, role_calls=calls, tokens=tokens)
    pools = alternative_roles(scores)
    eligible = [options[0] for options in pools]
    if cutoff is None or not candidates(index, pair, [{role} for role in eligible]):
        return response
    choice, margin, tokens = service.role.batched(phrases, pools)
    observation = dict(choice=choice if choice == eligible else None, margin=margin, forwards=1, tokens=tokens)
    return apply(index, pair, base, observation, cutoff)


def responses(service, request, withheld, cutoff):
    state, ranking, scores, clues, phrases = query_state(service, request, withheld)
    stock = exact_role(service, state, scores, phrases, cutoff)
    results = dict.fromkeys(ARMS, stock)
    if state["pair"] is not None or stock["calls"]:
        return results, None, None
    index = service.index
    required = required_roles({}, scores, "prototype_band", service.bands)
    options = pairs_for(index, ranking, required)
    proposal = options[0] if options else None
    if proposal is not None:
        weak = weaker_side(ranking, proposal)
        # Score the one-row and two-row policies separately to retain their
        # actual batches. Each policy spends one forward.
        weak_score = verify(service.source, index, clues, proposal, (weak,))
        both_score = verify(service.source, index, clues, proposal, (0, 1))
        for arm, score in (("postings_weak_side", weak_score), ("postings_both_sides", both_score)):
            accepted = proposal if score > service.cutoffs["source"] else None
            results[arm] = payload(index, accepted, required, source_calls=1)
    return results, options, (ranking, required)


def run_postings(service, labels, cutoff, reference, output, reverse):
    requests = list(labels)
    counts = defaultdict(lambda: defaultdict(Counter))
    selection = defaultdict(Counter)
    recorded = defaultdict(dict)
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        target = tuple(label["papers"])
        for availability, withheld in (("present", None), ("left_missing", target[0]), ("right_missing", target[1])):
            outcomes, options, context = responses(service, request, withheld, cutoff)
            scope = f"{label['view']}|{availability}"
            if options is not None:
                top = options[0] if options else None
                selection[scope].update(
                    eligible=1, positives=bool(label["gold"]), candidates=len(options),
                    proposed=bool(top),
                    target_feasible=bool(label["gold"]) and target in options,
                    target_first=bool(label["gold"]) and target == top,
                )
                if withheld is not None and any(withheld in pair for pair in options):
                    raise ValueError("Withheld source entered a proposal")
                if withheld is None and label["gold"]:
                    ranking, required = context
                    found = [paper in side["rank"] for paper, side in zip(target, ranking, strict=True)]
                    role_ok = bool(candidates(service.index, target, required))
                    selection[scope].update(
                        both_target_sources_in_top24=all(found),
                        at_least_one_target_source_in_top24=any(found),
                        target_pair_passes_static_roles=role_ok,
                        top24_target_blocked_by_static_roles=all(found) and not role_ok,
                    )
            for arm, response in outcomes.items():
                if response["calls"] > 1 or response["calls"] != response["source_calls"] + response["role_calls"]:
                    raise ValueError("Exceeded the one-forward budget")
                extra_row = arm == "postings_both_sides" and response["source_calls"] > outcomes["stock"]["source_calls"]
                counts[scope][arm].update(evidence(response, label),
                                          source_rows=response["source_calls"] + extra_row)
                if availability == "present":
                    recorded[arm][request] = response
    proof_hashes = signatures(recorded, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]["reserve_prescreen"]
    if proof_hashes["stock"] != expected:
        raise ValueError("Stock response differs from the archived evidence")
    write_json(output, dict(counts=counts, source_selection=selection, signatures=proof_hashes,
                            distinct_requests=len(requests), reverse=reverse,
                            reference_sha256=digest(reference), script_sha256=digest(Path(__file__))))
