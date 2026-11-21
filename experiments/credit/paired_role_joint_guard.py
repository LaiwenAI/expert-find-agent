"""Classify source-recovery errors and test resident-only agreement gates."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.credit.paired_role_joint_postings import dense_source_fallback, required_by_policy
from experiments.credit.paired_role_prospective_service import execute
from experiments.credit.paired_role_stage_audit import query_state, signatures
from experiments.credit.paired_role_verification import alternative_roles
from experiments.credit.plos_role_inventory import ROLES
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("cascade", "dense_joint", "student_guard", "two_bank_guard")


def error_type(response, gold):
    if not response["owners"]:
        return None
    if response["papers"] != list(gold["papers"]):
        return "wrong_source"
    bad_owner = bool(set(response["owners"]) - set(gold["gold"]))
    bad_role = any(expected not in actual for expected, actual in zip(gold["roles"], response["roles"], strict=True))
    if bad_role and bad_owner:
        return "wrong_role_and_contributor"
    if bad_role:
        return "wrong_role"
    if bad_owner:
        return "wrong_contributor"
    if set(response["owners"]) != set(gold["gold"]):
        return "incomplete_contributor_set"
    return None


def evaluate(service, request, cutoff, lightweight):
    base = json.loads(execute(service, request, "student_then_teacher", cutoff, lightweight))
    outcomes = dict.fromkeys(ARMS, base)
    if base["calls"] or base["owners"]:
        return outcomes, None

    state, ranking, scores, clues, _ = query_state(service, request)
    if state["pair"] is not None or state["response"]["calls"]:
        return outcomes, None

    heads, _, fit = lightweight
    student_cutoff = fit["reserve_student"]["cutoff"]
    student = required_by_policy(service, scores, state["role_vectors"], heads, student_cutoff)
    leaders = [pool[0] for pool in alternative_roles(scores)]
    independent = all([ROLES[int(row.argmax())] for row in scores[bank]] == leaders
                      for bank in ("ridge", "prototype"))
    guards = dict(student_guard=student["student_agreement"] is not None, two_bank_guard=independent)
    dense = dense_source_fallback(service, ranking, scores, clues)
    outcomes["dense_joint"] = dense
    for name in guards:
        if guards[name]:
            outcomes[name] = dense
    return outcomes, guards


def run_guard(service, labels, cutoff, lightweight, reference, output, reverse):
    requests = list(labels)
    counts = defaultdict(lambda: defaultdict(Counter))
    blocking = defaultdict(lambda: defaultdict(Counter))
    errors = defaultdict(Counter)
    recovered = []
    by_family = defaultdict(dict)
    responses = defaultdict(dict)
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        view = label["view"]
        arms, guards = evaluate(service, request, cutoff, lightweight)
        added = arms["dense_joint"]
        by_family[view, label["family"]][bool(label["gold"])] = (
            added["papers"], added["owners"], added["roles"]
        )
        cause = error_type(added, label) if guards is not None and added["owners"] else None
        if guards is not None and added["source_calls"]:
            dense_result = evidence(added, label)
            recovered.append(dict(
                case=sha256(request.payload.encode()).hexdigest(), view=view, strict=dense_result["strict"],
                issued=dense_result["issued"], adverse=dense_result["adverse"], error=cause, guards=guards,
                negative_control=not label["gold"],
            ))
            if cause is not None:
                errors[view][cause] += 1
            for arm, allowed in guards.items():
                if not allowed:
                    blocking[view][arm].update(
                        source_forwards=1, strict=dense_result["strict"], adverse=dense_result["adverse"],
                    )
        for arm, result in arms.items():
            if result["calls"] > 1 or result["calls"] != result["role_calls"] + result["source_calls"]:
                raise ValueError("Policy exceeded the verifier cap")
            counts[view][arm].update(evidence(result, label),
                                     source_rows=result.get("source_rows", result["source_calls"]))
            responses[arm][request] = result
    observed = signatures(responses, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]
    if any(observed[new] != expected[old] for new, old in (
        ("dense_joint", "dense_joint"), ("cascade", "student_then_teacher")
    )):
        raise ValueError("Diagnosed paths do not reproduce archived service responses")
    collisions = Counter(view for (view, _), variants in by_family.items()
                         if set(variants) == {False, True} and variants[True][1]
                         and variants[True] == variants[False])
    write_json(output, dict(requests=len(requests), counts=counts, guard_blocks=blocking, errors=errors,
                            identical_issued_proof_for_positive_and_swap=collisions,
                            recovered=sorted(recovered, key=lambda row: row["case"]), signatures=observed,
                            reverse=reverse, reference_sha256=digest(reference),
                            script_sha256=digest(Path(__file__))))
