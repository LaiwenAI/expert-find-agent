"""Test whether an automatic role citation preserves the unresolved request."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
from string import ascii_uppercase
from time import perf_counter

import numpy as np

from expert_search import requirement_packets
from experiments.credit.study_credit_residual_packets import KINDS, exact_partition, probe
from experiments.credit.study_credit_scope_gate import CASES, SURFACE, available, scope_pick
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge


ANCHOR_TASK = (
    "Choose the exact clause that solely states a research contribution performed by the person. "
    "Choose NONE if no single clause does this. A clause combining the role with any measurement, "
    "operating condition, exclusivity, specified project, qualification, or future obligation "
    "does not count. Other clauses may contain additional requirements, which will be checked "
    "separately. Return only the choice letter."
)
JOIN = re.compile(r"\b(?:and|where|provided that|subject to|rather than|must|shall)\b", re.I)
ARMS = ("one", "both", "both_lexical", "both_local_proof")
MANDATORY = {"mandate_trailing", "mandate_leading", "mandate_attached",
             "quiet_attached", "mixed", "mixed_same_clause"}


def pick(judge, text, parts, labels, reverse):
    choices = tuple(range(len(parts))) + (None,)
    if len(choices) > len(labels):
        return None, 0.
    if reverse:
        choices = choices[::-1]
    values = {
        labels[i]: ("NONE" if option is None else parts[option].quote(text))
        for i, option in enumerate(choices)
    }
    answer, elapsed = judge(ANCHOR_TASK, dict(clauses=values), choices)
    return (parts[answer] if answer is not None else None), elapsed


def compile_routes(judge, text, labels, offset):
    start = perf_counter()
    parts = requirement_packets.clauses(text)
    split_ms = (perf_counter() - start) * 1000
    choices = {}
    for reverse in ((False, True) if offset % 2 == 0 else (True, False)):
        choices[reverse] = pick(judge, text, parts, labels, reverse)
    forward, a_ms = choices[False]
    reverse, b_ms = choices[True]
    both = forward if forward == reverse else None
    combined = split_ms + a_ms + b_ms
    candidates = {"one": (forward, split_ms + a_ms), "both": (both, combined)}
    start = perf_counter()
    lexical, scope = both, both
    if both is not None:
        quote = both.quote(text)
        if SURFACE.search(quote) or JOIN.search(quote):
            lexical = None
    candidates["both_lexical"] = (lexical, combined + (perf_counter() - start) * 1000)
    elapsed = 0.
    if both is not None:
        quote = both.quote(text)
        menu = ("role_only", "other")
        first, a = scope_pick(judge, quote, menu, labels, False)
        second, b = scope_pick(judge, quote, menu[::-1], labels, False)
        if first != "role_only" or second != "role_only":
            scope = None
        elapsed = a + b
    candidates["both_local_proof"] = (scope, combined + elapsed)
    return {arm: (requirement_packets.form_packet(text, anchor), spent)
            for arm, (anchor, spent) in candidates.items()}


def control_views():
    for _, base, extra in CASES:
        for kind in KINDS:
            text, bare = probe(kind, base, extra)
            yield kind, text, bare
        yield "quiet_attached", f"The candidate {base} in work where they {extra}.", None


def evaluate(judge, controls, offset):
    labels = available(judge)
    counts = defaultdict(lambda: {arm: Counter() for arm in ARMS})
    timing = defaultdict(list)
    for i, (kind, text, gold) in enumerate(control_views()):
        for arm, (packet, elapsed) in compile_routes(judge, text, labels, i + offset).items():
            exact_partition(text, packet)
            quote = packet.role.quote(text) if packet.role else None
            counts[kind][arm].update(
                cases=1, role_only=int(packet.role_only),
                correct_anchor=int(quote == gold), false_anchor=int(quote is not None and quote != gold),
                mandatory_bypass=int(kind in MANDATORY and packet.role_only),
            )
            timing[kind, arm].append(elapsed)

    public = {arm: Counter() for arm in ARMS}
    for i, row in enumerate(controls):
        text = row["text"]
        for arm, (packet, _) in compile_routes(judge, text, labels, i + offset).items():
            exact_partition(text, packet)
            public[arm].update(documents=1, role_only=int(packet.role_only))
            for quote, role in row["gold"]:
                if role == "other":
                    continue
                at = text.index(quote)
                kept = any(part.start <= at and part.end >= at + len(quote)
                           for part in packet.unresolved)
                public[arm].update(annotated_requirements=1, requirements_retained=int(kept))
    return dict(constructed={
        kind: {arm: dict(values, p95_ms=float(np.quantile(timing[kind, arm], .95)))
               for arm, values in group.items()}
        for kind, group in counts.items()
    }, bilingual_controls=public)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("controls", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()

    import torch

    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    controls = json.loads(args.controls.read_bytes())
    judge = Judge()
    report = evaluate(judge, controls, args.order_offset)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), controls_sha256=digest(args.controls),
        packet_builder_sha256=digest(Path(requirement_packets.__file__)),
        decision_model_sha256=judge.sha, order_offset=args.order_offset, **report,
    ))


if __name__ == "__main__":
    main()
