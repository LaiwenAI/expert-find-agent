"""Score group-level acceptance and evidence risk from frozen recommendations."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.enterprise_review_gate import assess
from experiments.retrieval.check_joint_review import FIELDS, LABELS, check, load, sha


STATES = {"issued", "abstained", "failed"}
LEVEL = {"supported": 0, "unresolved": 1, "adverse": 2}
RECONCILIATION_STATES = {
    None: {"missing_review", "disagreement"},
    "uncertain": {"agreed_uncertain", "adjudicated_uncertain"},
    "supported": {"agreed", "adjudicated"},
    "not_supported_by_view": {"agreed", "adjudicated"},
}


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def view_key(sources):
    result = set()
    for source in sources:
        key = (source["id"], source["start"], source["end"], source["sha256"])
        if not isinstance(key[0], str) or not key[0] or not 0 <= key[1] < key[2]:
            raise ValueError("Invalid citation span")
        result.add(key)
    if not result or len(result) != len(sources):
        raise ValueError("Expected each exact citation span once")
    return result


def evidence_status(outcome, required):
    if set(outcome) != set(FIELDS):
        raise ValueError("Unexpected review fields")
    labels = [outcome[field]["label"] for field in required]
    if any(label is not None and label not in LABELS for label in labels):
        raise ValueError("Unknown review label")
    if any(outcome[field]["status"] not in RECONCILIATION_STATES[label]
           for field, label in zip(required, labels, strict=True)):
        raise ValueError("Review labels and reconciliation status differ")
    if "not_supported_by_view" in labels:
        return "adverse"
    return "supported" if all(label == "supported" for label in labels) else "unresolved"


def validated_views(frame, outputs, mapping, outcomes, required, top_k):
    queries = {row["id"]: row for row in frame}
    predictions = {row["query_id"]: row for row in outputs}
    if (not queries or len(queries) != len(frame) or len(predictions) != len(outputs)
            or set(queries) != set(predictions)):
        raise ValueError("Each frozen query must have exactly one explicit output")
    if not required or len(set(required)) != len(required) or not set(required) <= set(FIELDS):
        raise ValueError("Unexpected required evidence fields")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid recommendation cutoff")
    if set(mapping) != set(outcomes):
        raise ValueError("Incomplete reconciliation")
    groups = {row["group"] for row in frame}
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("Missing frozen group linkage")
    run_status = Counter()
    by_query = {}
    for query_id, row in predictions.items():
        status, recommendations = row["status"], row["recommendations"]
        if (status not in STATES or not isinstance(recommendations, list)
                or (status == "issued") != bool(recommendations) or len(recommendations) > top_k):
            raise ValueError("Output status and recommendations differ")
        run_status[status] += 1
        experts = set()
        by_query[query_id] = {}
        for recommendation in recommendations:
            case, expert = recommendation["case_id"], recommendation["expert_id"]
            view = mapping.get(case)
            if (expert in experts or view is None or (view["query_id"], view["expert_id"]) != (query_id, expert)
                    or view_key(view["sources"]) != view_key(recommendation["sources"])):
                raise ValueError("Recommendation does not match the frozen reviewed evidence")
            experts.add(expert)
            level = LEVEL[evidence_status(outcomes[case], required)]
            by_query[query_id][case] = level
    return by_query, run_status


def tally_views(frame, views, run_status):
    issued = {}
    for row in frame:
        levels = views[row["id"]].values()
        if levels:
            group = row["group"]
            issued[group] = max(max(levels), issued.get(group, 0))
    return dict(
        queries=len(frame), groups=len({row["group"] for row in frame}), issued=len(issued), adverse=sum(level == 2 for level in issued.values()),
        unresolved=sum(level == 1 for level in issued.values()), query_status=dict(run_status),
    )


def counts(frame, outputs, mapping, outcomes, required, top_k):
    return tally_views(frame, *validated_views(frame, outputs, mapping, outcomes, required, top_k))


def hit_bounds(views, top_k):
    levels = [level for query in views.values() for level in query.values()]
    confirmed = levels.count(LEVEL["supported"])
    unresolved = levels.count(LEVEL["unresolved"])
    slots = len(views) * top_k
    return dict(confirmed=confirmed, unresolved=unresolved, slots=slots,
                lower=confirmed / slots, upper=(confirmed + unresolved) / slots)


def paired_bounds(frame, primary, reference, top_k):
    """Compute finite-panel differences while sharing labels for identical views."""
    supported, unresolved, adverse = (LEVEL[key] for key in ("supported", "unresolved", "adverse"))
    low = high = shared = 0
    group_views = [defaultdict(dict), defaultdict(dict)]
    for row in frame:
        a, b = (views[row["id"]] for views in (primary, reference))
        shared += len(a.keys() & b.keys())
        for this, other, sign in ((a, b, 1), (b, a, -1)):
            for key in this.keys() - other.keys():
                if this[key] == supported:
                    low += sign
                    high += sign
                elif this[key] == unresolved:
                    low += min(0, sign)
                    high += max(0, sign)
        for group, query in zip(group_views, (a, b), strict=True):
            group[row["group"]].update(query)

    def possible_excess(first, second):
        # An adverse group in the second arm rules out first-minus-second = +1.
        if adverse in second.values():
            return False
        return adverse in first.values() or any(
            level == unresolved and key not in second for key, level in first.items()
        )

    groups = {row["group"] for row in frame}
    risk_low = risk_high = 0
    for group in groups:
        a, b = (arm[group] for arm in group_views)
        # A nonzero difference is forced only if both group outcomes are fixed.
        forced_increase = adverse in a.values() and all(level == supported for level in b.values())
        forced_decrease = adverse in b.values() and all(level == supported for level in a.values())
        risk_low += int(forced_increase) - int(possible_excess(b, a))
        risk_high += int(possible_excess(a, b)) - int(forced_decrease)
    slots = len(frame) * top_k
    return dict(
        shared_views=shared,
        hit_delta=dict(lower=low / slots, upper=high / slots),
        adverse_group_rate_delta=dict(lower=risk_low / len(groups), upper=risk_high / len(groups)),
        scope="Primary minus reference. Hit denominator is all frozen Top-K slots. Risk denominator "
              "is all frozen groups, including abstentions and failures. Bounds describe unresolved "
              "labels on this panel and are distinct from sampling confidence intervals.",
    )


def load_execution(frame, protocol, outputs, execution, packet_sha):
    spec = json.loads(protocol.read_bytes())
    execution_manifest = json.loads(execution.read_bytes())
    if spec["frame_sha256"] != sha(frame):
        raise ValueError("Changed group frame")
    if execution_manifest != dict(protocol_sha256=sha(protocol), outputs_sha256=sha(outputs), packet_sha256=packet_sha):
        raise ValueError("Changed evaluation execution inputs")
    return spec, records(outputs)


def run(frame, protocol, outputs, execution, packet, reconciliation, compare=None):
    packet_manifest = packet / "manifest.json"
    outcome_path = reconciliation / "private-outcomes.json"
    review_manifest = json.loads((reconciliation / "summary.json").read_bytes())
    packet_sha = sha(packet_manifest)
    spec, primary_rows = load_execution(frame, protocol, outputs, execution, packet_sha)
    if review_manifest["packet_sha256"] != packet_sha or review_manifest["outcomes_sha256"] != sha(outcome_path):
        raise ValueError("Changed reconciliation or review packet")
    form = packet / "blind-review.jsonl"
    check(packet, form)
    mapping = load(packet / "private-mapping.jsonl")
    outcomes = json.loads(outcome_path.read_bytes())
    seen = set()
    for row in mapping.values():
        signature = row["query_id"], row["expert_id"], frozenset(view_key(row["sources"]))
        if signature in seen:
            raise ValueError("An exact view cannot have multiple review cases")
        seen.add(signature)
    frozen = records(frame)
    queries = {row["id"]: row for row in frozen}
    for case, row in load(form).items():
        query = queries.get(mapping[case]["query_id"])
        if query is None or sha256(row["need"].encode()).hexdigest() != query["text_sha256"]:
            raise ValueError("Reviewed request differs from the frozen input")
    required, top_k = spec["required_fields"], spec["top_k"]
    primary, run_status = validated_views(frozen, primary_rows, mapping, outcomes, required, top_k)
    tally = tally_views(frozen, primary, run_status)
    gate = assess(
        **{field: tally[field] for field in ("groups", "issued", "adverse", "unresolved")},
        **{field: spec[field] for field in ("coverage_floor", "risk_ceiling", "confidence")},
    )
    report = dict(
        evaluation_role=spec["evaluation_role"], policy=spec["policy"], tally=tally, gate=gate, hit_at_k=hit_bounds(primary, top_k),
        protocol_sha256=sha(protocol), frame_sha256=sha(frame), execution_sha256=sha(execution),
        packet_sha256=packet_sha, reconciliation_sha256=sha(reconciliation / "summary.json"),
        evaluator_sha256=sha(Path(__file__)),
    )

    if compare is not None:
        reference_protocol, reference_outputs, reference_execution = compare
        other, other_rows = load_execution(frame, reference_protocol, reference_outputs, reference_execution, packet_sha)
        if (type(other["top_k"]) is not int or other["top_k"] != top_k
                or sorted(other["required_fields"]) != sorted(required)):
            raise ValueError("Comparison requires the same cutoff and review criteria")
        reference, _ = validated_views(frozen, other_rows, mapping, outcomes, required, top_k)
        report["comparison"] = paired_bounds(frozen, primary, reference, top_k) | dict(
            policy=other["policy"], hit_at_k=hit_bounds(reference, top_k),
            execution_sha256=sha(reference_execution),
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frame", "protocol", "outputs", "execution", "packet", "reconciliation"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--compare", nargs=3, type=Path, metavar=("PROTOCOL", "OUTPUTS", "EXECUTION"))
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
