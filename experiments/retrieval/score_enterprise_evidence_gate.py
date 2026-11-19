"""Score group-level acceptance and evidence risk from frozen recommendations."""

import argparse
from collections import Counter
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


def counts(frame, outputs, mapping, outcomes, required, top_k):
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
    issued = {}
    for query_id, row in predictions.items():
        status, recommendations = row["status"], row["recommendations"]
        if (status not in STATES or not isinstance(recommendations, list)
                or (status == "issued") != bool(recommendations) or len(recommendations) > top_k):
            raise ValueError("Output status and recommendations differ")
        run_status[status] += 1
        experts = set()
        group = queries[query_id]["group"]
        for recommendation in recommendations:
            case, expert = recommendation["case_id"], recommendation["expert_id"]
            view = mapping.get(case)
            if (expert in experts or view is None or (view["query_id"], view["expert_id"]) != (query_id, expert)
                    or view_key(view["sources"]) != view_key(recommendation["sources"])):
                raise ValueError("Recommendation does not match the frozen reviewed evidence")
            experts.add(expert)
            level = LEVEL[evidence_status(outcomes[case], required)]
            issued[group] = max(level, issued.get(group, 0))
    return dict(
        queries=len(frame), groups=len(groups), issued=len(issued), adverse=sum(level == 2 for level in issued.values()),
        unresolved=sum(level == 1 for level in issued.values()), query_status=dict(run_status),
    )


def run(frame, protocol, outputs, execution, packet, reconciliation):
    spec = json.loads(protocol.read_bytes())
    execution_manifest = json.loads(execution.read_bytes())
    packet_manifest = packet / "manifest.json"
    outcome_path = reconciliation / "private-outcomes.json"
    review_manifest = json.loads((reconciliation / "summary.json").read_bytes())
    packet_sha = sha(packet_manifest)
    if spec["frame_sha256"] != sha(frame):
        raise ValueError("Changed group frame")
    if execution_manifest != dict(protocol_sha256=sha(protocol), outputs_sha256=sha(outputs), packet_sha256=packet_sha):
        raise ValueError("Changed evaluation execution inputs")
    if review_manifest["packet_sha256"] != packet_sha or review_manifest["outcomes_sha256"] != sha(outcome_path):
        raise ValueError("Changed reconciliation or review packet")
    form = packet / "blind-review.jsonl"
    check(packet, form)
    mapping = load(packet / "private-mapping.jsonl")
    outcomes = json.loads(outcome_path.read_bytes())
    frozen = records(frame)
    queries = {row["id"]: row for row in frozen}
    for case, row in load(form).items():
        query = queries.get(mapping[case]["query_id"])
        if query is None or sha256(row["need"].encode()).hexdigest() != query["text_sha256"]:
            raise ValueError("Reviewed request differs from the frozen input")
    tally = counts(frozen, records(outputs), mapping, outcomes, spec["required_fields"], spec["top_k"])
    gate = assess(
        **{field: tally[field] for field in ("groups", "issued", "adverse", "unresolved")},
        **{field: spec[field] for field in ("coverage_floor", "risk_ceiling", "confidence")},
    )
    return dict(
        evaluation_role=spec["evaluation_role"], policy=spec["policy"], tally=tally, gate=gate,
        protocol_sha256=sha(protocol), frame_sha256=sha(frame), execution_sha256=sha(execution),
        packet_sha256=packet_sha, reconciliation_sha256=sha(reconciliation / "summary.json"),
        evaluator_sha256=sha(Path(__file__)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frame", "protocol", "outputs", "execution", "packet", "reconciliation"):
        parser.add_argument(name, type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
