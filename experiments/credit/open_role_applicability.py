"""Cross source checks and role applicability in retrieval over a public catalog."""

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
from experiments.credit import role_applicability_gate as gate
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


ROLE_RESULTS = "f2c7dc8402082d253fcc493f857672d18ac5f141c37fc5bee682e110546416b6"
ARMS = (
    ("forced", "none"), ("relative", "body"), ("outside_contrast", "body"),
    ("definition_contrast", "body"), ("definition_contrast", "none"),
)


def alternate(clue, nominated, index, bodies, dense, dense_cut, body_cut):
    start = perf_counter()
    passed = dense > dense_cut
    if not passed:
        coverage, _ = body.select_windows(clue, bodies[nominated], index.idf)
        passed = coverage is not None and coverage > body_cut
    return passed, (perf_counter() - start) * 1000


def outcome(index, nominated, goal, initial, target, predicted, accepted):
    returned = set()
    if accepted:
        declared = index.notes[nominated].memberships
        returned = declared.get(initial, set()) & declared.get(predicted, set())
    truth = set()
    if goal is not None and target is not None:
        declared = index.notes[goal].memberships
        truth = declared[initial] & declared[target]
    actual = returned if nominated == goal and predicted == target else set()
    any_true = bool(actual & truth)
    return dict(
        clean=int(any_true and actual <= truth),
        returned=int(bool(returned)),
        wrong_source=int(bool(returned) and nominated != goal),
        wrong_role=int(bool(returned) and goal == nominated and predicted != target),
        wrong_author=int(bool(actual - truth)),
    )


def summarize(rows):
    output = {}
    reference = [row["forced/none"] for _, row in rows]
    groups = [group for group, _ in rows]
    for role_policy, source_policy in ARMS:
        name = f"{role_policy}/{source_policy}"
        records = [row[name] for _, row in rows]
        output[name] = {
            k: sum(row[k] for row in records)
            for k in ("clean", "returned", "wrong_source", "wrong_role", "wrong_author")
        }
        output[name]["component_p95_ms"] = float(np.quantile([row["ms"] for row in records], .95))
        output[name]["clean_delta_vs_forced"] = clustered_paired_interval(
            [a["clean"] - b["clean"] for a, b in zip(records, reference, strict=True)], groups)
    return output


def run(old, first, fresh, niso, role_results, source_results, body_results, embedding, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a Slurm CUDA job and a fresh output")
    for path, expected in (
        (niso, gate.NISO_SHA256), (role_results, ROLE_RESULTS), (source_results, body.REFERENCE),
        (body_results, transfer.BODY_RESULTS),
    ):
        if digest(path) != expected:
            raise ValueError(f"Changed input: {path.name}")
    guide, controller, source_policy, body_policy = (json.loads(path.read_bytes()) for path in
                                                      (niso, role_results, source_results, body_results))
    provenance = verify_checkpoint(embedding)
    if controller["embedding_revision"] != provenance["revision"]:
        raise ValueError("Changed applicability checkpoint")
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    notes, bodies, queries, groups, omissions = transfer.prepare(old, first, fresh)
    index = source.Search(notes, encoder)
    positives = {
        role: [row for row in guide["examples"] if row["role"] == key and row["ordinal"] in ordinals]
        for key, (role, ordinals) in gate.POSITIVE.items()
    }
    negatives = [row for row in guide["examples"] if row["ordinal"] in gate.NEGATIVE.get(row["role"], ())]
    crosswalk = {role: name for name, (role, _) in gate.POSITIVE.items()}
    bank = encoder.encode(list(roles.DESCRIPTIONS))
    definitions = encoder.encode([f"{heading}. {guide['definitions'][crosswalk[role]]}"
                                  for heading, role in zip(roles.DESCRIPTIONS, roles.ROLES, strict=True)])
    excluded = encoder.encode([f"{name}: {guide['definitions'][name]}" for name in gate.EXCLUDED_PROTOTYPES])

    all_results = {}
    for form in source.FORMS:
        cuts = source_policy["thresholds"][form], body_policy["thresholds"][form]["lexical_body"]["threshold"]
        observations = {(state, kind): [] for state in ("present", "withheld")
                        for kind in ("supported", "unsupported")}
        nominations = {state: 0 for state in ("present", "withheld")}
        for (doi, duties), group in zip(queries, groups, strict=True):
            _, _, positive_author, other = (doi, *legacy.choose_case(doi, notes[doi].memberships))
            initial = next(role for role in duties if other in notes[doi].memberships[role])
            target = next(role for role in duties if role != initial)
            if not all(positive_author in notes[doi].memberships[role] for role in duties):
                raise AssertionError("Inconsistent seeded author")
            good = min(positives[target], key=lambda row: legacy.stable("open-good", doi, row["ordinal"]))
            bad = min(negatives, key=lambda row: legacy.stable("open-bad", doi, row["role"], row["ordinal"]))
            signals = {kind: gate.score(example["text"], encoder, bank, definitions, excluded)
                       for kind, example in (("supported", good), ("unsupported", bad))}
            clue = index.clue(doi, form)
            for state in ("present", "withheld"):
                goal = doi if state == "present" else None
                started = perf_counter()
                nominated, dense = index.find(clue, None if goal else doi)
                retrieval_ms = (perf_counter() - started) * 1000
                nominations[state] += int(nominated == goal)
                source_pass, gate_ms = alternate(clue, nominated, index, bodies, dense, *cuts)
                for kind, (predicted, scores, role_ms) in signals.items():
                    expected = target if kind == "supported" else None
                    records = {}
                    for role_policy, source_check in ARMS:
                        started = perf_counter()
                        chosen = predicted["definition_contrast" if role_policy == "forced" else role_policy]
                        role_pass = role_policy == "forced" or scores[role_policy] > controller["thresholds"][role_policy]
                        ok = role_pass and (source_check == "none" or source_pass)
                        observation = outcome(index, nominated, goal, initial, expected, chosen, ok)
                        observation["ms"] = role_ms + retrieval_ms + (gate_ms if source_check == "body" else 0.)
                        observation["ms"] += (perf_counter() - started) * 1000
                        records[f"{role_policy}/{source_check}"] = observation
                    observations[state, kind].append((group, records))
        all_results[form] = {
            "correct_nominations": nominations,
            "panels": {f"{state}/{kind}": summarize(rows) for (state, kind), rows in observations.items()},
        }
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        queries=len(queries), author_groups=len(set(groups)), sources=len(notes), omissions=omissions,
        embedding_revision=provenance["revision"], capture_sha256=digest(fresh / "capture.json"),
        code_sha256=digest(Path(__file__)), results=all_results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old", "first", "fresh", "niso", "role_results", "source_results", "body_results", "embedding", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
