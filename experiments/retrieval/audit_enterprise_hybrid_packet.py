"""Measure exact-view review burden and finite-panel P@5 bounds before annotation."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval.check_joint_review import FIELDS, check, load, sha
from experiments.retrieval.enterprise_hybrid_rescue import ARMS
from experiments.retrieval.enterprise_routed_review import rows, write
from experiments.retrieval.score_enterprise_evidence_gate import hit_bounds, paired_bounds, validated_views


CONTRASTS = (("hybrid", "dense"), ("global_on_abstain", "hybrid"),
             ("global_on_abstain", "deeper_on_abstain"))


def run(packet, queries, aggregate, output):
    """Use the existing strict scorer with every review field marked unreviewed."""
    if sha(queries) != json.loads(aggregate.read_bytes())["queries_sha256"]:
        raise ValueError("Require the pinned development requests")
    forms = packet / "blind-review.jsonl"
    validated = check(packet, forms)
    if any(set(counts) != {"missing"} for counts in validated["counts"].values()):
        raise ValueError("Expected the untouched frozen form")
    cases = load(forms)
    mapping = load(packet / "private-mapping.jsonl")
    frame = [dict(id=row["id"], group=row["id"], text_sha256=sha256(row["text"].encode()).hexdigest())
             for row in rows(queries)]
    texts = {item["id"]: item["text_sha256"] for item in frame}
    if any(sha256(form["need"].encode()).hexdigest() != texts[mapping[key]["query_id"]]
           for key, form in cases.items()):
        raise ValueError("An evidence view has the wrong request")
    outcomes = {key: {field: dict(label=None, status="missing_review") for field in FIELDS} for key in cases}
    views = {}
    uses = Counter()
    for arm in ARMS:
        decisions = rows(packet / f"{arm}-predictions.jsonl")
        arm_views, _ = validated_views(frame, decisions, mapping, outcomes, FIELDS, top_k=5)
        views[arm] = arm_views
        uses.update(case for view in arm_views.values() for case in view)
    if uses.keys() != mapping.keys():
        raise ValueError("Unexpected extra view in packet")
    comparisons = {}
    for left,right in CONTRASTS:
        stats = paired_bounds(frame, views[left], views[right], top_k=5)
        left_count = sum(map(len, views[left].values()))
        right_count = sum(map(len, views[right].values()))
        comparisons[f"{left}_vs_{right}"] = dict(
            shared_views=stats["shared_views"], hit_delta=stats["hit_delta"],
            candidate_only_views=left_count - stats["shared_views"],
            comparison_only_views=right_count - stats["shared_views"],
        )
    write(output, dict(requests=len(frame), expert_citation_views=len(uses), human_labels=0,
                       method_multiplicity=dict(sorted(Counter(uses.values()).items())),
                       arms={arm:hit_bounds(view, 5) for arm,view in views.items()},
                       contrast=comparisons))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("packet","queries","aggregate","output"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
