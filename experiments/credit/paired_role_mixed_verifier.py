"""Confirm both sources and both role clauses in one mixed-task batch."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.credit.engineering_joint_calibration import SUPPORT
from experiments.credit.independent_source_confirmation import VERIFY
from experiments.credit.lightweight_paired_role import TASK as ROLE_TASK
from experiments.credit.paired_role_prospective_service import execute
from experiments.credit.paired_role_source_postings import pairs_for
from experiments.credit.paired_role_stage_audit import query_state, signatures
from experiments.credit.paired_role_verification import alternative_roles
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_role_factorial_diagnosis import payload
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("cascade", "dense_joint", "mixed_agreement", "mixed_relabel")


def availability(service, request, cutoff, lightweight):
    cascade = json.loads(execute(service, request, "student_then_teacher", cutoff, lightweight))
    if cascade["calls"] or cascade["owners"]:
        return cascade, None
    state, ranking, scores, clues, phrases = query_state(service, request)
    if state["pair"] is not None or state["response"]["calls"]:
        return cascade, None
    pools = alternative_roles(scores)
    required = [{pool[0]} for pool in pools]
    selected = pairs_for(service.index, ranking, required)
    if not selected:
        return cascade, None
    return cascade, (selected[0], pools, required, clues, phrases, state["role_vectors"], ranking)


def classify(values, pools):
    selected, margins = [], []
    for row, pool in zip(np.asarray(values).reshape(2, 3), pools, strict=True):
        top, runner_up = np.argsort(-row, kind="stable")[:2]
        selected.append(pool[top])
        margins.append(min(row[top], row[top] - row[runner_up]))
    return selected, float(min(margins))


def compare(service, request, cutoff, lightweight):
    cascade, plan = availability(service, request, cutoff, lightweight)
    outcomes = dict.fromkeys(ARMS, cascade)
    if plan is None:
        return outcomes, None, None
    _, definitions, calibration = lightweight
    pair, pools, required, clues, phrases, _, _ = plan
    source_rows = [(VERIFY, clue, service.index.articles[paper]["indexed"])
                   for clue, paper in zip(clues, pair, strict=True)]
    role_rows = [(ROLE_TASK, phrase, definitions[SUPPORT[role]])
                 for phrase, pool in zip(phrases, pools, strict=True) for role in pool]
    source_scores, clipped_source, source_tokens = service.source.score_tasks(source_rows, count_tokens=True)
    joint_scores, clipped, joint_tokens = service.source.score_tasks(source_rows + role_rows, count_tokens=True)
    separate_roles, clipped_roles, role_tokens = service.source.score_tasks(role_rows, count_tokens=True)
    if clipped_source or clipped or clipped_roles:
        raise ValueError("A source or role row was clipped")
    source_cutoff = service.cutoffs["source"]
    source_ok = min(source_scores) > source_cutoff
    mixed_ok = min(joint_scores[:2]) > source_cutoff
    labels, confidence = classify(joint_scores[2:], pools)
    separate_labels, separate_confidence = classify(separate_roles, pools)
    expected = [pool[0] for pool in pools]
    role_cutoff = calibration["reserve_reranker"]["cutoff"]
    if role_cutoff is None:
        raise ValueError("A calibrated role threshold is required")
    roles_ok = confidence > role_cutoff
    exact_role = roles_ok and labels == expected

    def response(accepted, selected, rows, tokens):
        return payload(service.index, pair if accepted else None, selected, source_calls=1) | {
            "check_rows": rows, "check_tokens": tokens
        }

    outcomes["dense_joint"] = response(source_ok, required, 2, source_tokens)
    outcomes["mixed_agreement"] = response(mixed_ok and exact_role, required, 8, joint_tokens)
    outcomes["mixed_relabel"] = response(mixed_ok and roles_ok, [{role} for role in labels], 8, joint_tokens)
    teacher_labels, gap, teacher_tokens = service.role.batched(phrases, pools)
    teacher_ok = cutoff is not None and gap > cutoff and None not in teacher_labels
    teacher_roles = [{role} for role in teacher_labels] if teacher_ok else required
    teacher = payload(service.index, pair if source_ok and teacher_ok else None,
                      teacher_roles, source_calls=1, role_calls=1, tokens=teacher_tokens)
    diagnostic = dict(
        mixed_source_decision_changes=source_ok != mixed_ok,
        mixed_role_label_changes=labels != separate_labels,
        mixed_role_threshold_changes=roles_ok != (separate_confidence > role_cutoff),
        separate_tokens=source_tokens + role_tokens,
        mixed_tokens=joint_tokens,
        role_agrees=exact_role,
        proposed=1,
    )
    return outcomes, diagnostic, teacher


def run_mixed(service, labels, cutoff, lightweight, reference, output, reverse):
    requests = list(labels)
    counts = defaultdict(lambda: defaultdict(Counter))
    diagnostics = defaultdict(Counter)
    responses = defaultdict(dict)
    changes = defaultdict(lambda: defaultdict(Counter))
    teacher_diagnostic = defaultdict(Counter)
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        view = label["view"]
        outcomes, report, teacher = compare(service, request, cutoff, lightweight)
        dense = evidence(outcomes["dense_joint"], label)
        if report is not None:
            diagnostics[view].update(report)
            teacher_diagnostic[view].update(evidence(teacher, label), role_tokens=teacher["tokens"])
        for arm, response in outcomes.items():
            if response["calls"] > 1 or response["calls"] != response["role_calls"] + response["source_calls"]:
                raise ValueError("Policy exceeded the online forward cap")
            result = evidence(response, label)
            counts[view][arm].update(result, fallback_rows=response.get("check_rows", 0),
                                     fallback_tokens=response.get("check_tokens", 0))
            changes[view][arm].update(strict_lost=dense["strict"] and not result["strict"],
                                      strict_added=result["strict"] and not dense["strict"],
                                      adverse_removed=dense["adverse"] and not result["adverse"],
                                      adverse_added=result["adverse"] and not dense["adverse"])
            responses[arm][request] = response
    proof_hashes = signatures(responses, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]
    if any(proof_hashes[name] != expected[previous] for name, previous in (
            ("cascade", "student_then_teacher"), ("dense_joint", "dense_joint"))):
        raise ValueError("Control proof differs from the archived service")
    write_json(output, dict(counts=counts, changes_from_dense=changes, diagnostics=diagnostics,
                            over_budget_teacher_on_fallbacks=teacher_diagnostic,
                            signatures=proof_hashes, requests=len(requests), reverse=reverse,
                            reference_sha256=digest(reference), script_sha256=digest(Path(__file__))))
