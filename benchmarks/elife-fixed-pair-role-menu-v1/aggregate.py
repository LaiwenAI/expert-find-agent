"""Verify task replications and compare unchanged arms with archived results."""

import json
from pathlib import Path
import sys

from collections.abc import Iterable


def aggregate(inputs: Iterable[Path], archived: Path, output: Path):
    old = json.loads(archived.read_bytes())["role_gap_replicas"]
    replicas = [json.loads(path.read_bytes()) for path in inputs]
    if len(replicas) != 4 or len(old) != 4:
        raise ValueError("Expected two fixed publisher panels and both traversal orders")
    if any(key in {"records", "queries"} for run in replicas for key in run):
        raise ValueError("Expected aggregate-only publisher output")
    for i, row in enumerate(replicas):
        baseline = old[i]
        for key in ("cases", "encoder", "role_checkpoint", "frozen_reference_sha256", "manifest_sha256",
                    "projection_sha256", "reverse", "authorship_groups"):
            if row[key] != baseline[key]:
                raise ValueError(f"Archived panel metadata changed: {key}")
        for state, views in baseline["report"].items():
            for view, arms in views.items():
                for arm, metrics in arms.items():
                    run = row["report"][state][view][arm]
                    for name, value in metrics.items():
                        if run[name] != value:
                            raise ValueError(f"Baseline result changed: {state}/{view}/{arm}/{name}")
    for forward, reverse in ((0, 1), (2, 3)):
        for key in ("signature", "report", "runner_sha256"):
            if replicas[forward][key] != replicas[reverse][key]:
                raise ValueError(f"Traversal order changed {key}")
    if output.exists():
        raise ValueError("Require fresh aggregate output")
    output.write_text(json.dumps({"replicas": replicas}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    arguments = list(map(Path, sys.argv[1:]))
    aggregate(arguments[2:], arguments[0], arguments[1])
