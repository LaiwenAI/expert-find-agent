"""Confirm a proposed role without reopening source search or author attribution."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.credit import anchored_role_margin as roles
from experiments.credit import legacy_body_confirmation as body
from experiments.credit import legacy_body_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit import legacy_open_attribution as source
from experiments.credit import open_role_applicability as base
from experiments.credit import role_applicability_gate as gate
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


HEADS = dict(zip(roles.ROLES, roles.DESCRIPTIONS, strict=True))
TASK = ("Determine whether the contribution heading directly covers the requested activity. "
        "A related activity with a different responsibility does not qualify.")
METHOD = "definition_contrast"
BASELINE_SHA256 = "a5325a3b1bb0e7c21019726f740c77b5cbf0cf7580c2fbce3afd3cc79dca2f5e"
ARMS = ("baseline", "single_head", "rival_heads")
SHORTLIST_ARMS = (*ARMS, "two_head_veto", "two_head_repair")
VERIFIER_SHA256 = "b84cfa9ca6f9cd3088ec2747a26ebb52d4c2c49d2729aa6afa3ae09200b74543"
METRICS = ("clean", "returned", "wrong_source", "wrong_role", "wrong_author")


def score(reader, phrase, heads):
    values, truncated = reader.score_pairs([(phrase, HEADS[head]) for head in heads], task=TASK)
    if truncated or not np.isfinite(values).all():
        raise ValueError("Role confirmation requires finite scores on complete inputs")
    return dict(zip(heads, values, strict=True))


def rival_gap(scores, chosen):
    return scores[chosen] - max(value for head, value in scores.items() if head != chosen)


def calibrate(reader):
    # These paraphrases predate the NISO evaluation and do not use its examples.
    examples = [(phrase, role) for role, phrase in roles.PARAPHRASES.items()]
    examples.extend((phrase, None) for phrase in roles.OUTSIDE)
    rows = {arm: [] for arm in ARMS[1:]}
    for phrase, gold in examples:
        rival = score(reader, phrase, tuple(HEADS))
        for head in HEADS:
            single = score(reader, phrase, (head,))[head]
            rows["single_head"].append((single, head == gold))
            rows["rival_heads"].append((min(rival[head], rival_gap(rival, head)), head == gold))
    calibration = {}
    for arm, cases in rows.items():
        cutoff = max(value for value, positive in cases if not positive)
        calibration[arm] = dict(
            cutoff=cutoff, correct_retained=sum(value > cutoff and positive for value, positive in cases),
            correct_pairs=sum(positive for _, positive in cases), negative_pairs=sum(not positive for _, positive in cases),
        )
    return calibration


def shortlist(definition_scores):
    order = np.argsort(-definition_scores, kind="stable")[:2]
    return tuple(roles.ROLES[i] for i in order)


def confirm_shortlist(reader, phrase, candidates):
    margins = score(reader, phrase, candidates)
    return {head: min(margin, rival_gap(margins, head)) for head, margin in margins.items()}


def calibrate_shortlist(reader, encoder, definitions):
    examples = [(phrase, role) for role, phrase in roles.PARAPHRASES.items()]
    examples.extend((phrase, None) for phrase in roles.OUTSIDE)
    annotated = []
    for phrase, gold in examples:
        vector = encoder.encode([gate.query(phrase)])[0]
        candidates = shortlist(definitions @ vector)
        verdicts = confirm_shortlist(reader, phrase, candidates)
        annotated.extend((value, head == gold) for head, value in verdicts.items())
    cutoff = max(value for value, correct in annotated if not correct)
    return dict(cutoff=cutoff, correct_retained=sum(val > max(0, cutoff) and good for val, good in annotated),
                correct_pairs_proposed=sum(good for _, good in annotated), correct_phrases=len(roles.PARAPHRASES),
                negative_pairs=sum(not good for _, good in annotated))


def shortlist_actions(reader, index, phrase, candidates, nominated, initial, expected, goal, gates_pass,
                      control, shared_ms, cutoff):
    start = perf_counter()
    selected = index.notes[nominated].memberships
    known_members = selected.get(initial, set())
    possible = {head for head in candidates if head != initial and known_members & selected.get(head, set())}
    repair_checked = gates_pass and bool(possible)
    repair_prep_ms = (perf_counter() - start) * 1000
    veto_checked = bool(control["returned"])

    start = perf_counter()
    verdicts = confirm_shortlist(reader, phrase, candidates) if repair_checked or veto_checked else {}
    verify_ms = (perf_counter() - start) * 1000
    proposed = candidates[0]
    if veto_checked and verdicts[proposed] <= cutoff:
        control = dict.fromkeys(METRICS, 0)
    veto_ms = verify_ms if veto_checked else 0.

    start = perf_counter()
    winner = max(verdicts, key=verdicts.get) if verdicts else proposed
    # The second duty must be distinct and a repair needs a unique semantic winner.
    repair = repair_checked and winner in possible and verdicts[winner] > max(0, cutoff)
    repaired = base.outcome(index, nominated, goal, initial, expected, winner, repair)
    repair_ms = repair_prep_ms + (verify_ms if repair_checked else 0.) + (perf_counter() - start) * 1000
    return {
        "two_head_veto": dict(control, verify_ms=verify_ms if veto_checked else None,
                              ms=shared_ms + veto_ms),
        "two_head_repair": dict(repaired, verify_ms=verify_ms if repair_checked else None,
                                ms=shared_ms + repair_ms, changed_role=int(repair and winner != proposed),
                                rescued=int(repair and winner != proposed and nominated == goal and winner == expected),
                                reachable=int(repair_checked and nominated == goal and expected in possible)),
    }

def summarize(rows, arms=ARMS):
    report = {}
    for arm in arms:
        observed = [records[arm] for _, records in rows]
        ref = [records["baseline"] for _, records in rows]
        counts = {key: sum(row[key] for row in observed) for key in METRICS}
        costs = [row["verify_ms"] for row in observed if row["verify_ms"] is not None]
        counts.update(
            verification_calls=len(costs),
            verification_p95_ms=float(np.quantile(costs, .95)) if costs else None,
            component_p95_ms=float(np.quantile([row["ms"] for row in observed], .95)),
        )
        if arm != "baseline":
            for key in ("clean", "wrong_role"):
                counts[f"{key}_delta"] = clustered_paired_interval(
                    [a[key] - b[key] for a, b in zip(observed, ref, strict=True)], [group for group, _ in rows],
                )
        if arm == "two_head_repair":
            counts.update(changed_role=sum(row["changed_role"] for row in observed),
                          rescued=sum(row["rescued"] for row in observed),
                          reachable=sum(row["reachable"] for row in observed))
        report[arm] = counts
    return report


def matched_coverage(rows, report):
    # Rank only original responses. Match the verifier's observed coverage for diagnosis,
    # without treating a test-selected rank cut as a deployable threshold.
    ranked = sorted((records["baseline"] for _, records in rows if records["baseline"]["returned"]),
                    key=lambda row: -row["proposal_score"])
    controls = {}
    for arm in ARMS[1:]:
        count = report[arm]["returned"]
        selected = ranked[:count]
        controls[arm] = {key: sum(row[key] for row in selected) for key in METRICS}
        if 0 < count < len(ranked):
            cutoff = ranked[count - 1]["proposal_score"]
            tied = [row for row in ranked if row["proposal_score"] == cutoff]
            fixed = [row for row in selected if row["proposal_score"] > cutoff]
            remaining = count - len(fixed)
            controls[arm]["ties_at_cut"] = len(tied)
            for key in ("clean", "wrong_role"):
                hits = sum(row[key] for row in tied)
                certain = sum(row[key] for row in fixed)
                controls[arm][f"{key}_range"] = [
                    certain + max(0, remaining - (len(tied) - hits)),
                    certain + min(remaining, hits),
                ]
    return controls

def run(old, first, fresh, niso, role_results, source_results, body_results, baseline_results,
        embedding, reranker, output, shortlist_reference=None):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    for path, expected in (
        (niso, gate.NISO_SHA256), (role_results, base.ROLE_RESULTS), (source_results, body.REFERENCE),
        (body_results, transfer.BODY_RESULTS), (baseline_results, BASELINE_SHA256),
    ):
        if digest(path) != expected:
            raise ValueError(f"Changed input: {path.name}")

    if shortlist_reference is not None and digest(shortlist_reference) != VERIFIER_SHA256:
        raise ValueError("Changed verifier comparison result")
    prior = json.loads(shortlist_reference.read_bytes()) if shortlist_reference is not None else None
    arms = SHORTLIST_ARMS if prior is not None else ARMS

    guide, controller, source_policy, body_policy, baseline = (
        json.loads(path.read_bytes()) for path in (niso, role_results, source_results, body_results, baseline_results)
    )
    provenance = verify_checkpoint(embedding)
    if controller["embedding_revision"] != provenance["revision"]:
        raise ValueError("Changed embedding checkpoint")
    reader = Reranker(reranker)
    if reader.provenance["revision"] != "e61197ed45024b0ed8a2d74b80b4d909f1255473":
        raise ValueError("Changed reranker checkpoint")
    calibration = calibrate(reader)

    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    notes, bodies, queries, groups, _ = transfer.prepare(old, first, fresh)
    index = source.Search(notes, encoder)
    positives = {}
    for key, (role, ordinals) in gate.POSITIVE.items():
        positives[role] = [row for row in guide["examples"] if row["role"] == key and row["ordinal"] in ordinals]
    negatives = [row for row in guide["examples"] if row["ordinal"] in gate.NEGATIVE.get(row["role"], ())]
    crosswalk = {role: name for name, (role, _) in gate.POSITIVE.items()}
    bank = encoder.encode(list(roles.DESCRIPTIONS))
    definitions = encoder.encode([f"{heading}. {guide['definitions'][crosswalk[role]]}"
                                  for role, heading in HEADS.items()])
    excluded = encoder.encode([f"{name}: {guide['definitions'][name]}" for name in gate.EXCLUDED_PROTOTYPES])

    if prior is not None:
        calibration["shortlist"] = calibrate_shortlist(reader, encoder, definitions)

    panels = {}
    for form in source.FORMS:
        cuts = source_policy["thresholds"][form], body_policy["thresholds"][form]["lexical_body"]["threshold"]
        observations = {(state, kind): [] for state in ("present", "withheld")
                        for kind in ("supported", "unsupported")}
        for position, ((doi, duties), group) in enumerate(zip(queries, groups, strict=True)):
            _, _, distractor = legacy.choose_case(doi, notes[doi].memberships)
            initial, = (role for role in duties if distractor in notes[doi].memberships[role])
            target, = (role for role in duties if role != initial)
            good = min(positives[target], key=lambda row: legacy.stable("open-good", doi, row["ordinal"]))
            bad = min(negatives, key=lambda row: legacy.stable("open-bad", doi, row["role"], row["ordinal"]))
            clue = index.clue(doi, form)
            for state in ("present", "withheld"):
                goal = doi if state == "present" else None
                start = perf_counter()
                nominated, dense = index.find(clue, None if goal else doi)
                retrieval_ms = (perf_counter() - start) * 1000
                source_pass, gate_ms = base.alternate(clue, nominated, index, bodies, dense, *cuts)
                for kind, example, expected in (("supported", good, target), ("unsupported", bad, None)):
                    phrase = example["text"]
                    proposed, signals, role_ms, definition_scores = gate.score_with_definitions(
                        phrase, encoder, bank, definitions, excluded,
                    )
                    chosen = proposed[METHOD]
                    start = perf_counter()
                    accepted = source_pass and signals[METHOD] > controller["thresholds"][METHOD]
                    control = base.outcome(index, nominated, goal, initial, expected, chosen, accepted)
                    shared_ms = retrieval_ms + gate_ms + role_ms + (perf_counter() - start) * 1000
                    records = {"baseline": dict(control, ms=shared_ms, verify_ms=None,
                                                proposal_score=signals[METHOD])}
                    verification_order = ARMS[1:] if position % 2 == 0 else tuple(reversed(ARMS[1:]))
                    if prior is not None:
                        rotation = position % 3
                        verification_order = (*ARMS[1:], "shortlist")
                        verification_order = verification_order[rotation:] + verification_order[:rotation]
                    for arm in verification_order:
                        if arm == "shortlist":
                            start = perf_counter()
                            candidates = shortlist(definition_scores)
                            shortlist_ms = (perf_counter() - start) * 1000
                            records.update(shortlist_actions(
                                reader, index, phrase, candidates, nominated, initial, expected, goal, accepted,
                                control, shared_ms + shortlist_ms, calibration["shortlist"]["cutoff"],
                            ))
                            continue
                        record = records["baseline"].copy()
                        if control["returned"]:
                            start = perf_counter()
                            if arm == "single_head":
                                verdict = score(reader, phrase, (chosen,))[chosen]
                            else:
                                scores = score(reader, phrase, tuple(HEADS))
                                verdict = min(scores[chosen], rival_gap(scores, chosen))
                            verify_ms = (perf_counter() - start) * 1000
                            if verdict <= calibration[arm]["cutoff"]:
                                record.update(dict.fromkeys(METRICS, 0))
                            record.update(ms=shared_ms + verify_ms, verify_ms=verify_ms)
                        records[arm] = record
                    observations[state, kind].append((group, records))

        panels[form] = {}
        for (state, kind), rows in observations.items():
            panel = f"{state}/{kind}"
            values = summarize(rows, arms)
            expected = baseline["results"][form]["panels"][panel]["definition_contrast/body"]
            if any(values["baseline"][k] != expected[k] for k in METRICS):
                raise AssertionError(f"Baseline replay diverged in {form}/{panel}")
            if prior is not None:
                for arm in ARMS:
                    previous = prior["panels"][form][panel][arm]
                    if any(values[arm][k] != previous[k] for k in METRICS):
                        raise AssertionError(f"Verifier replay diverged in {form}/{panel}/{arm}")
            values["proposal_at_matched_coverage"] = matched_coverage(rows, values)
            panels[form][panel] = values

    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(queries),
        author_groups=len(set(groups)), sources=len(notes), calibration=calibration,
        baseline_sha256=digest(baseline_results), code_sha256=digest(Path(__file__)),
        **({"verifier_reference_sha256": digest(shortlist_reference)} if prior is not None else {}),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"], panels=panels,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist-reference", type=Path)
    for key in ("old", "first", "fresh", "niso", "role_results", "source_results", "body_results",
                "baseline_results", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
