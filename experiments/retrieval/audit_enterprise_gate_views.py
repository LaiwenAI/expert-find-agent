"""Dry-run the gate on development requests with frozen dense and union citation views."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.retrieval.score_enterprise_evidence_gate import FIELDS, records, run, sha, view_key


def audit(source, packet, reconciliation):
    published = json.loads((source / "manifest.json").read_bytes())["files_sha256"]
    def development(name):
        path = source / name
        if sha(path) != published[name]:
            raise ValueError("Changed development source inventory")
        return (row for row in records(path) if row["split"] == "development")

    queries = {row["id"]: row for row in development("queries.jsonl")}
    pools = {row["query_id"]: row for row in development("pools.jsonl")}
    snapshots = {(row["query_id"], row["expert_id"]): row for row in development("snapshots.jsonl")}
    mapped = {(row["query_id"], row["expert_id"]): row for row in records(packet / "private-mapping.jsonl")}
    frame, predictions = [], {"union": [], "original_dense": []}
    counts = Counter(matching_views=0, different_views=0, queries_with_different_view=0)
    for query in sorted(queries):
        frame.append(dict(id=query, group=query, text_sha256=sha256(queries[query]["text"].encode()).hexdigest()))
        union, dense = [], []
        mismatched = False
        for expert in pools[query]["dense_order"][:5]:
            reviewed = mapped[query, expert]
            original = [dict(id=row["source_id"], start=row["start"], end=row["end"], sha256=row["sha256"])
                        for row in snapshots[query, expert]["sources"]]
            same = view_key(original) == view_key(reviewed["sources"])
            counts["matching_views" if same else "different_views"] += 1
            mismatched |= not same
            common = dict(expert_id=expert, case_id=reviewed["case_id"])
            union.append(common | {"sources": reviewed["sources"]})
            dense.append(common | {"sources": original})
        counts["queries_with_different_view"] += mismatched
        for name, recommendations in (("union", union), ("original_dense", dense)):
            predictions[name].append(dict(query_id=query, status="issued" if recommendations else "abstained",
                                          recommendations=recommendations))

    with TemporaryDirectory(prefix=".enterprise-gate-dry-run-", dir=packet.parent) as directory:
        root = Path(directory)

        def dump(name, value, *, lines=False):
            path = root / name
            path.write_text("".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value) + "\n")
            return path

        frame_path = dump("frame.jsonl", frame, lines=True)
        spec = dump("protocol.json", dict(
            frame_sha256=sha(frame_path), policy="development-union-replay",
            evaluation_role="Wiring check with placeholder groups",
            required_fields=FIELDS, top_k=5, coverage_floor=.60, risk_ceiling=.05, confidence=.95,
        ))

        def evaluate(name):
            outputs = dump(f"{name}.jsonl", predictions[name], lines=True)
            execution = dump(f"{name}-execution.json", dict(
                protocol_sha256=sha(spec), outputs_sha256=sha(outputs), packet_sha256=sha(packet / "manifest.json"),
            ))
            return run(frame_path, spec, outputs, execution, packet, reconciliation)

        report = evaluate("union")
        rejected = False
        try:
            evaluate("original_dense")
        except ValueError as error:
            if "frozen reviewed evidence" not in str(error):
                raise
            rejected = True
        if rejected != bool(counts["different_views"]):
            raise AssertionError("View audit and execution validation disagree")
    return dict(
        fixed_view_audit=dict(counts) | {"dense_replay_rejected": rejected},
        development_tally=report["tally"], gate_passed=report["gate"]["passes"],
        packet_sha256=report["packet_sha256"], reconciliation_sha256=report["reconciliation_sha256"],
        evaluator_sha256=report["evaluator_sha256"], dense_snapshot_sha256=published["snapshots.jsonl"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "packet", "reconciliation"):
        parser.add_argument(name, type=Path)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
