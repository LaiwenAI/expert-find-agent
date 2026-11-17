"""Paired group bootstrap for bilingual fusion overlap diagnostics."""
import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path


def analyze(root: Path, replicates: int = 2000, seed: int = 20261117) -> dict:
    if replicates < 40:
        raise ValueError("At least 40 bootstrap replicates are required")
    summary = json.loads((root / "summary.json").read_text())
    payload = (root / "evaluation.jsonl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != summary["evaluation_sha256"]:
        raise ValueError("Evaluation hash does not match frozen summary")
    groups = collections.defaultdict(list)
    rows = [json.loads(line) for line in payload.splitlines()]
    if not rows or len(rows) != summary["pairs"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate evaluation records")
    for row in rows:
        if not row.get("group_id"):
            raise ValueError("Every pair must have a group identifier")
        def overlap(method):
            zh = row["languages"]["zh"][method]["ranking"]
            en = row["languages"]["en"][method]["ranking"]
            if len(zh) != 10 or len(en) != 10 or len(set(zh)) != 10 or len(set(en)) != 10:
                raise ValueError("Expected unique top-ten rankings")
            return len(set(zh) & set(en)) / 10
        groups[row["group_id"]].append(overlap("exact_channels_rrf") - overlap("raw"))
    values = list(groups.values())
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(replicates):
        sample = [rng.choice(values) for _ in values]
        bootstrap.append(sum(map(sum, sample)) / sum(map(len, sample)))
    bootstrap.sort()
    return {"pairs": len(rows), "groups": len(values),
            "mean_paired_difference": statistics.mean(x for group in values for x in group),
            "group_bootstrap_replicates": replicates, "seed": seed,
            "percentile_interval_95": [bootstrap[int(0.025 * replicates) - 1], bootstrap[int(0.975 * replicates) - 1]],
            "evaluation_sha256": summary["evaluation_sha256"],
            "scope": "Construction pairs; overlap is not relevance; same retrieval pool and cached embeddings."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))
