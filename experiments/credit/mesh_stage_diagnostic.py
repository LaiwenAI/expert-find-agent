"""Find where reachable role-compatible major-topic evidence is lost."""

from collections import Counter
import json
from pathlib import Path

from experiments.credit import elife_role_review as review
from experiments.credit import topic_role_postings as pilot


STAGES = ("selected_source", "passing_source", "issued_witness")


def stages(row, target):
    present = (
        bool(target.intersection(row["selected_sources"])),
        bool(target.intersection(row["passing_sources"])),
        any(source in target for _, source in row["witnesses"]),
    )
    if any(a < b for a, b in zip(present, present[1:])):
        raise ValueError("A downstream stage cannot recover a source absent from the previous one")
    return dict(zip(STAGES, present, strict=True))


def stage_counts(rows, targets, bindings):
    reached = [(row, target, stages(row, target))
               for row, target in zip(rows, targets, strict=True) if target]
    counts = {stage: sum(item[stage] for _, _, item in reached) for stage in STAGES}
    retained_other_source = omitted_owner = 0
    for row, target, item in reached:
        if not item["passing_source"] or item["issued_witness"]:
            continue
        nominated = {owner for owner, _ in row["witnesses"]}
        possible = {owner for source in target.intersection(row["passing_sources"])
                    for owner in bindings[source, row["role_pair"]]}
        if possible <= nominated:
            retained_other_source += 1
        elif len(nominated) == pilot.TOP:
            omitted_owner += 1
        else:
            raise ValueError("A passing new owner was omitted before the output budget was filled")
    return dict(
        **counts,
        no_target_in_reads=len(reached) - counts["selected_source"],
        lost_at_score_cutoff=counts["selected_source"] - counts["passing_source"],
        lost_at_output=counts["passing_source"] - counts["issued_witness"],
        all_major_owners_have_other_citation=retained_other_source,
        some_major_owner_omitted_at_full_budget=omitted_owner,
    )


def paired_changes(rows, base, targets):
    changes = {stage: Counter(gained=0, lost=0) for stage in STAGES}
    for left, right, target in zip(rows, base, targets, strict=True):
        if (left["topic"], left["role_pair"]) != (right["topic"], right["role_pair"]):
            raise ValueError("Diagnostic queries are not aligned")
        a, b = stages(left, target), stages(right, target)
        for stage in STAGES:
            changes[stage].update(gained=int(a[stage] and not b[stage]), lost=int(b[stage] and not a[stage]))
    return changes


def summarize(measurements, catalog, major):
    full = measurements[100]["topic_four"]
    targets = [major.get(row["topic"], set()) & catalog.person[row["role_pair"]] for row in full]
    ranks = [
        next(i for i, article in enumerate(row["topic_order"], 1) if article in target)
        for row, target in zip(full, targets, strict=True) if target
    ]
    summaries = {}
    for pct, arms in measurements.items():
        base = arms["topic_four"]
        for rows in arms.values():
            if [(r["topic"], r["role_pair"]) for r in rows] != [(r["topic"], r["role_pair"]) for r in full]:
                raise ValueError("Diagnostic queries differ from the common replay")
        summaries[pct] = {
            arm: dict(stages=stage_counts(rows, targets, catalog.bindings), vs_topic_four=paired_changes(rows, base, targets))
            for arm, rows in arms.items()
        }
    return dict(
        queries=len(full), reachable=len(ranks),
        first_major_target_rank=dict(
            first_four=sum(i <= 4 for i in ranks), ranks_five_to_eight=sum(4 < i <= 8 for i in ranks),
            beyond_eight=sum(i > 8 for i in ranks),
        ),
        masks=summaries,
    )


def write(output, diagnostic, current, reference):
    pinned = json.loads(reference.read_bytes())
    replayed = review.quality_record(json.loads(json.dumps(current)))
    if any(replayed != review.quality_record(pinned[name])
           for name in ("run", "repeat")):
        raise ValueError("Stage replay differs from the frozen scope metrics")
    if diagnostic["queries"] != current["queries"] or diagnostic["reachable"] != current["major_reachable_queries"]:
        raise ValueError("Diagnostic does not cover the frozen query set")
    pilot.write_json(output, dict(
        job=current["job"], scope_reference_sha256=pilot.digest(reference),
        script_sha256=pilot.digest(Path(__file__)),
        **diagnostic,
    ))
