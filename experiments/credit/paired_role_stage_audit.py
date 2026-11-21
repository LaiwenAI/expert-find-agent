"""Audit when agreement blocks one-forward role recovery on the held panel."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.credit.budgeted_source_role_search import role_check
from experiments.credit.engineering_cross_source_search import TOPIC_TASK, candidates
from experiments.credit.engineering_role_banks import TASK
from experiments.credit.paired_role_verification import alternative_roles, apply, available
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_role_factorial_diagnosis import payload, plan
from experiments.followir.core17_reranking import digest, write_json


POLICIES = {"exact_agreement": 0, "one_disagreement": 1, "any_feasible_pair": 2}


def evaluate(service, request, cutoff):
    requirements = json.loads(request.payload)["requirements"]
    clues = [item["topic"] for item in requirements]
    phrases = [item["role"] for item in requirements]
    vectors = service.encoder.encode(
        [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in clues]
        + [f"Instruct: {TASK}\nQuery:{phrase}" for phrase in phrases]
    )
    scores = {name: vectors[2:] @ service.banks[name].T for name in service.bands}
    index = service.index
    ranking = index.full_rank(vectors[:2], clues, None)
    state = plan(service, ranking, scores, clues, phrases, None, "screened", run_role=False)
    pair, base, response = state["pair"], state["roles"], state["response"]
    if not pair or response["calls"] or response["owners"]:
        return dict.fromkeys(POLICIES, response), pair, "initial"
    checked, calls, tokens, _ = role_check(index, pair, scores, service.bands, service.role,
                                          phrases, service.cutoffs["role"])
    if calls or candidates(index, pair, checked):
        response = payload(index, pair, checked, role_calls=calls, tokens=tokens)
        return dict.fromkeys(POLICIES, response), pair, "single_role"

    pools = alternative_roles(scores)
    leaders = [pool[0] for pool in pools]
    if cutoff is None or not available(index, pair, pools):
        return dict.fromkeys(POLICIES, response), pair, "no_feasible_role_pair"
    feasible = {
        "exact_agreement": bool(candidates(index, pair, [{role} for role in leaders])),
        "one_disagreement": any(candidates(index, pair, [{a}, {b}])
                                for a, b in [(role, leaders[1]) for role in pools[0]]
                                + [(leaders[0], role) for role in pools[1]]),
        "any_feasible_pair": True,
    }
    choice, margin, tokens = service.role.batched(phrases, pools)
    if choice and None not in choice:
        disagreement = sum(role != leader for role, leader in zip(choice, leaders, strict=True))
    else:
        disagreement = 3
    outcomes = {}
    for name, allowed in POLICIES.items():
        if feasible[name]:
            accepted = choice if disagreement <= allowed else None
            observation = dict(choice=accepted, margin=margin, forwards=1, tokens=tokens)
            outcomes[name] = apply(index, pair, base, observation, cutoff)
        else:
            outcomes[name] = response
    return outcomes, pair, f"batched_disagreement_{disagreement}"


def signatures(responses, requests):
    return {name: sha256(json.dumps([
        [items[request]["papers"], items[request]["owners"], items[request]["roles"]]
        for request in requests
    ], ensure_ascii=False).encode()).hexdigest() for name, items in responses.items()}


def audit(service, labels, cutoff, reference, output, reverse):
    requests = list(labels)
    responses = defaultdict(dict)
    counts = defaultdict(lambda: defaultdict(Counter))
    failures = defaultdict(Counter)
    rescues = defaultdict(lambda: defaultdict(Counter))
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        outcomes, pair, stage = evaluate(service, request, cutoff)
        base = evidence(outcomes["exact_agreement"], label)
        view = label["view"]
        if label["gold"] and not base["strict"]:
            if pair is None:
                failure = "no_source_pair"
            elif list(pair) != list(label["papers"]):
                failure = "different_source_pair"
            elif outcomes["exact_agreement"]["source_calls"]:
                failure = "source_used_the_forward"
            else:
                failure = stage
            failures[view][failure] += 1
        for arm, response in outcomes.items():
            if response["calls"] > 1 or response["calls"] != response["role_calls"] + response["source_calls"]:
                raise ValueError("Policy exceeded its one-forward limit")
            record = evidence(response, label)
            responses[arm][request] = response
            counts[view][arm].update(record | {"role_tokens": response["tokens"]})
            rescues[view][arm].update(
                strict_added=bool(record["strict"] and not base["strict"]),
                strict_lost=bool(base["strict"] and not record["strict"]),
                adverse_added=bool(record["adverse"] and not base["adverse"]),
                adverse_removed=bool(base["adverse"] and not record["adverse"]),
            )
    proof_hashes = signatures(responses, requests)
    stock = json.loads(reference.read_bytes())
    expected = stock["response_signatures"]["reserve_prescreen"]
    if proof_hashes["exact_agreement"] != expected:
        raise ValueError("Exact-agreement control does not match prior service evidence")
    write_json(output, dict(counts=counts, missed_positive_stages=failures, changes=rescues,
                            signatures=proof_hashes, requests=len(requests), reverse=reverse,
                            reference_sha256=digest(reference), script_sha256=digest(Path(__file__))))
