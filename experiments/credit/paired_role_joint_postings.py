"""Use resident role predictions to unlock a batched source confirmation."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.credit.distilled_role_recheck import options
from experiments.credit.engineering_cross_source_search import required_roles
from experiments.credit.paired_role_source_postings import exact_role, pairs_for
from experiments.credit.paired_role_stage_audit import query_state, signatures
from experiments.credit.paired_role_verification import alternative_roles
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_confirmation_transfer import verify
from experiments.credit.source_role_factorial_diagnosis import payload
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("stock", "static_postings", "dense_singletons", "student_agreement")


def required_by_policy(service, scores, vectors, heads, cutoff):
    pools = alternative_roles(scores)
    leaders = [pool[0] for pool in pools]
    singleton = [{role} for role in leaders]
    static = required_roles({}, scores, "prototype_band", service.bands)
    weights, bias = heads["distilled_head"]
    decisions = [options(weights @ vector + bias, pool)
                 for vector, pool in zip(vectors, pools, strict=True)]
    roles, margins = zip(*decisions, strict=True)
    supported = cutoff is not None and list(roles) == leaders and min(margins) > cutoff
    return {"static_postings": static, "dense_singletons": singleton,
            "student_agreement": singleton if supported else None}


def responses(service, request, withheld, role_cutoff, heads, student_cutoff):
    state, ranking, scores, clues, phrases = query_state(service, request, withheld)
    baseline = exact_role(service, state, scores, phrases, role_cutoff)
    result = dict.fromkeys(ARMS, baseline)
    if state["pair"] is not None or baseline["calls"]:
        return result, None
    requirements = required_by_policy(service, scores, state["role_vectors"], heads, student_cutoff)
    proposals, verified = {}, {}
    for arm, required in requirements.items():
        ordered = pairs_for(service.index, ranking, required) if required is not None else []
        proposals[arm] = ordered
        if not ordered:
            continue
        pair = ordered[0]
        if pair not in verified:
            verified[pair] = verify(service.source, service.index, clues, pair, (0, 1))
        accepted = pair if verified[pair] > service.cutoffs["source"] else None
        result[arm] = payload(service.index, accepted, required, source_calls=1)
    return result, proposals


def run_joint(service, labels, role_cutoff, heads, student_cutoff, reference, output, reverse):
    requests = list(labels)
    counts = defaultdict(lambda: defaultdict(Counter))
    nomination = defaultdict(lambda: defaultdict(Counter))
    recorded = defaultdict(dict)
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        target = tuple(label["papers"])
        positive = bool(label["gold"])
        for condition, withheld in (("present", None), ("left_missing", target[0]), ("right_missing", target[1])):
            outcomes, proposals = responses(service, request, withheld, role_cutoff, heads, student_cutoff)
            scope = f"{label['view']}|{condition}"
            if proposals is not None:
                for arm, candidates in proposals.items():
                    if withheld is not None and any(withheld in pair for pair in candidates):
                        raise ValueError("A withheld paper was proposed")
                    chosen = candidates[0] if candidates else None
                    nomination[scope][arm].update(
                        eligible=1, positives=positive, proposed=bool(chosen),
                        target_reachable=positive and target in candidates,
                        target_first=positive and chosen == target,
                    )
            for arm, response in outcomes.items():
                if response["calls"] > 1 or response["calls"] != response["role_calls"] + response["source_calls"]:
                    raise ValueError("Policy exceeded the forward budget")
                extra_row = arm != "stock" and response["source_calls"] > outcomes["stock"]["source_calls"]
                counts[scope][arm].update(evidence(response, label),
                                          source_rows=response["source_calls"] + extra_row)
                if condition == "present":
                    recorded[arm][request] = response
    proofs = signatures(recorded, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]["reserve_prescreen"]
    if proofs["stock"] != expected:
        raise ValueError("Stock response differs from the archived proof")
    write_json(output, dict(counts=counts, nomination=nomination, signatures=proofs, student_cutoff=student_cutoff,
                            distinct_requests=len(requests), reverse=reverse,
                            reference_sha256=digest(reference), script_sha256=digest(Path(__file__))))
