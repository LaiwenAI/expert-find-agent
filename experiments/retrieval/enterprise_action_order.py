"""Replay the order of evidence expansion and one qualification question."""

from collections import Counter
from functools import reduce
from operator import and_

from experiments.retrieval.enterprise_qualification_questions import candidate_masks, decision_table
from experiments.retrieval.enterprise_qualification_replay import score_profile


POLICIES = (
    "shallow_only", "shallow_ask", "wide_no_ask", "deep_no_ask",
    "wide_then_ask", "ask_then_wide", "deep_then_ask", "ask_then_deep",
    "state_route", "both_no_ask", "both_then_ask",
)


class Stage:
    def __init__(self, people, excerpts, bank, axes):
        union = axes["union"]
        profiles = [score_profile(owner, excerpts[owner], bank, 1 + len(union)) for owner in people]
        common = axes["in_both"]
        disputed = sorted(set(union) - set(common))
        blockers = candidate_masks(profiles, common, disputed)
        self.table = decision_table(blockers, len(disputed), full_set=True)
        self.excerpts = sum(len(excerpts[owner]) for owner in people)
        self.topic_available = any(profile[0] for profile in profiles)

    def safe(self, worlds):
        """Keep only people supported in every interpretation still possible."""
        return reduce(and_, (self.table[world] for world in worlds))


def select_question(stage, worlds):
    alternatives = []
    for bit in range((len(stage.table) - 1).bit_length()):
        off = tuple(world for world in worlds if not world & (1 << bit))
        on = tuple(world for world in worlds if world & (1 << bit))
        if not off or not on:
            continue
        gain = len(off) * bool(stage.safe(off)) + len(on) * bool(stage.safe(on))
        if gain:
            alternatives.append((-gain, bit, off, on))
    return min(alternatives) if alternatives else None


def replay(stages, policy, reality):
    stage = stages["shallow"]
    worlds = tuple(range(len(stage.table)))
    queries = 0

    def ask():
        nonlocal worlds, queries
        if queries or stage.safe(worlds):
            return
        question = select_question(stage, worlds)
        if question is None:
            return
        _, bit, off, on = question
        worlds = on if reality & (1 << bit) else off
        queries += 1

    if policy.startswith("ask_then") or policy in {"shallow_ask", "state_route"}:
        ask()
    expansion = None
    if not stage.safe(worlds):
        if policy == "state_route":
            potential = any(stage.table[world] for world in worlds)
            expansion = "wide" if potential or not stage.topic_available else "deep"
        else:
            expansion = next((name for name in ("wide", "deep", "both") if name in policy), None)
        if expansion:
            stage = stages[expansion]
    if policy.endswith("_ask") and not policy.endswith("no_ask"):
        ask()
    elif policy.startswith("ask_then") or policy == "state_route":
        ask()
    safe = stage.safe(worlds)
    if safe & ~stage.table[reality] or queries > 1:
        raise AssertionError("Recommendation exceeds the revealed interpretation")
    return dict(
        model_supported=bool(safe), excerpts=stage.excerpts, questions=queries,
        expanded=bool(expansion), wide=expansion == "wide", deep=expansion == "deep",
        all_conditions_known_model_possible=bool(stage.table[reality]),
    )


def examine(totals, order, initial, expanded, bank, axes):
    for owner in order:
        if not set(initial[owner]) <= set(expanded[owner]):
            raise AssertionError("Expanded read plan lost an already read excerpt")
    stages = {
        "shallow": Stage(order[:5], initial, bank, axes),
        "wide": Stage(order, initial, bank, axes),
        "deep": Stage(order[:5], expanded, bank, axes),
        "both": Stage(order, expanded, bank, axes),
    }
    worlds = len(stages["shallow"].table)
    totals["design"].update(
        requests=1, hypothetical_interpretations=worlds,
        requests_shallow_topic_support=stages["shallow"].topic_available,
        requests_shallow_one_question_informative=select_question(stages["shallow"], tuple(range(worlds))) is not None,
    )
    for policy in POLICIES:
        results = [replay(stages, policy, world) for world in range(worlds)]
        outcome = totals[policy]
        outcome.update(
            requests=1,
            requests_supported_in_some_worlds=any(row["model_supported"] for row in results),
            requests_supported_in_all_worlds=all(row["model_supported"] for row in results),
            sum_worst_case_excerpts=max(row["excerpts"] for row in results),
            sum_worst_case_questions=max(row["questions"] for row in results),
        )
        # World counts are powers of two. Dyadic means are exact and
        # independent of the order in which requests are processed.
        for field in results[0]:
            outcome[f"sum_uniform_mean_{field}"] += sum(row[field] for row in results) / worlds
