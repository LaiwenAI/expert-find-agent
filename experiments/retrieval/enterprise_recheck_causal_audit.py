"""Trace parser layout disagreements to individual qualification interventions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from experiments.retrieval import enterprise_packed_role_search as packed
from experiments.retrieval import enterprise_recheck_layouts as layout
from experiments.retrieval.enterprise_first_query import CRITERION_CHARS, ROLES, TASK
from experiments.retrieval.enterprise_isolated_parsing import spans
from experiments.retrieval.enterprise_packed_role_load import init_fixture
from experiments.retrieval.enterprise_role_first_query import recovery_options, require_cuda
from experiments.retrieval.enterprise_routed_review import digest, rows, write


class FixedParser:
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def evaluate(self, text, clauses, method, cutoff):
        if len(clauses) != len(self.labels):
            raise ValueError("Frozen labels do not cover the original request")
        return dict(labels=self.labels, rechecked=0)


def payload(result):
    return {field: result[field] for field in layout.FIELDS}


def recs(result):
    return result["initial"] + result["extra"]


def score_interventions(text, reference, variant, fixture):
    source = reference["labels"]
    other = variant["labels"]
    changed = [i for i, (a, b) in enumerate(zip(source, other, strict=True))
               if (ROLES[a] == "prior_qualification") != (ROLES[b] == "prior_qualification")]
    outcomes = {}
    for name, labels in (("whole", source), ("variant", other)):
        outcome = packed.one(text, "guarded_rescue", FixedParser(labels), *fixture[1:])
        expected = reference if name == "whole" else variant
        if payload(outcome) != payload(expected):
            raise ValueError("Frozen-role replay changed non-parser execution")
        outcomes[name] = payload(outcome)
    for i in changed:
        for name, original, alternate in (("adopt", source, other), ("restore", other, source)):
            labels = list(original)
            labels[i] = alternate[i]
            outcomes[f"{name}:{i}"] = payload(
                packed.one(text, "guarded_rescue", FixedParser(labels), *fixture[1:]))
    return changed, outcomes


def recovery_witness_gap(text, baseline, compared, fixture):
    _, _, _, sources, roles, reader = fixture
    losing = set(item["expert"] for item in baseline["extra"]) - {item["expert"] for item in recs(compared)}
    if not losing:
        return []
    candidates = {}
    for arm, result in (("whole", baseline), ("variant", compared)):
        criteria = [text.strip().splitlines()[0][:CRITERION_CHARS],
                    *(text[left:right] for left, right in result["qualifications"])]
        ranked = [row[0] for row in result["decisions"]]
        original = recovery_options(criteria, ranked, {row["expert"] for row in result["initial"]},
                                    roles, "global_on_abstain")
        candidates[arm] = [row["expert"] for row in original]
    baseline_offsets = set(baseline["qualifications"])
    changed = [pair for pair in compared["qualifications"] if pair not in baseline_offsets]
    output = []
    for item in baseline["extra"]:
        if item["expert"] not in losing:
            continue
        citations = item["citations"]
        phrases = [(left,right) for left,right in changed]
        signs = []
        if phrases:
            reader.change("guarded_rescue")
            inputs = [(text[a:b], sources[key]["text"][left:right]) for key,left,right in citations
                      for a,b in phrases]
            margins, clipped = reader.score_pairs(inputs, task=TASK)
            if not np.isfinite(margins).all():
                raise ValueError("Nonfinite witness audit")
            signs = [bool(margin > 0) for margin in margins] if not clipped else None
        output.append(dict(item=item, stayed_in_two_window_recovery=item["expert"] in candidates["variant"],
                           baseline_recovery_owners=candidates["whole"],
                           variant_recovery_owners=candidates["variant"],
                           newly_required_offsets=phrases, new_condition_support_signs=signs))
    return output


def run(queries, corpus, roles, vectors, meta, models, serial, prior, output, private):
    require_cuda(output, private)
    prior_results = json.loads(prior.read_bytes())["live"]
    if len({row["output_signature"] for row in prior_results}) != 1 or any(
            row["runner_sha256"] != digest(Path(layout.__file__)) for row in prior_results):
        raise ValueError("Layout methods or original signatures disagree")
    texts, base, provenance = init_fixture(queries, corpus, roles, vectors, meta, models, serial)
    fixture = (layout.LayoutParser(base[0]), *base[1:])
    warmup = "Find an expert in optical inspection. Prior testing with thin films is required."
    for arm in layout.ARMS:
        layout.search(warmup, arm, fixture)
    records = []
    proposals = []
    audit = Counter()
    for i, (query, text) in enumerate(zip(rows(queries), texts, strict=True)):
        shift = i % len(layout.ARMS)
        ordered = layout.ARMS[shift:] + layout.ARMS[:shift]
        current = {arm: layout.search(text, arm, fixture) for arm in ordered}
        records.append({arm: {**payload(current[arm]), "labels": current[arm]["labels"],
                      "recheck_counts": {key: count for key,count in current[arm]["parser_work"].items()
                                         if key != "fallback_ms"}} for arm in layout.ARMS})
        reference = current["whole_block"]
        changed_arms = {arm: value for arm,value in current.items() if arm != "whole_block"
                        and value["labels"] != reference["labels"]}
        if not changed_arms:
            continue
        audit["requests_with_any_layout_change"] += 1
        clause_coordinates = spans(text)
        variants = {}
        for arm, variant in changed_arms.items():
            differing, results = score_interventions(text, reference, variant, fixture)
            audit[f"{arm}_qualification_removals"] += len(set(reference["qualifications"]) - set(variant["qualifications"]))
            audit[f"{arm}_qualification_additions"] += len(set(variant["qualifications"]) - set(reference["qualifications"]))
            reference_recs = recs(reference)
            variant_recs = recs(variant)
            audit[f"{arm}_changed_recommendations"] += reference_recs != variant_recs
            sufficient, necessary = [], []
            if reference_recs != variant_recs:
                for idx in differing:
                    if recs(results[f"adopt:{idx}"]) == variant_recs:
                        sufficient.append(idx)
                    if recs(results[f"restore:{idx}"]) == reference_recs:
                        necessary.append(idx)
            audit[f"{arm}_single_clause_restores"] += bool(necessary)
            audit[f"{arm}_single_clause_reproduces_change"] += bool(sufficient)
            variants[arm] = dict(changed_qualification_indices=differing, sufficient=sufficient, necessary=necessary,
                                 interventions=results, lost_witnesses=recovery_witness_gap(text, reference, variant, fixture))
        proposals.append(dict(id=query["id"], clauses=clause_coordinates,
                              labels={arm: row["labels"] for arm,row in current.items()}, variants=variants))
    signature = sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if signature != prior_results[0]["output_signature"]:
        raise ValueError("Audit did not reproduce the measured non-timing outputs")
    write(private, dict(cases=proposals, output_signature=signature, queries_sha256=digest(queries),
                        source_sha256=digest(corpus)))
    write(output, dict(audit=dict(audit), total_requests=len(texts), output_signature=signature,
                       prior_sha256=digest(prior), private_sha256=digest(private),
                       runner_sha256=digest(Path(__file__)), slurm_job=os.environ["SLURM_JOB_ID"], **provenance))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "prior", "output", "private"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
