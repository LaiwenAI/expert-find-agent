"""Localize changes between cached panel ranking and live topic encoding."""

import argparse
from collections import Counter
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import live_joint_source_request as live
from experiments.followir.core17_reranking import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("elife_xml", "manifest", "embedding", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output on a Slurm compute node")
    checkpoint = live.verify_embedding(args.embedding)
    model = QwenTorchEmbedding(args.embedding)
    catalogue, checksum = live.elife.load_catalogue(args.elife_xml, args.manifest)
    panels = live.elife.build_requests(catalogue)
    subjects, _, queries = panels["topics"]
    topics = sorted({atom[0] for _, _, rows in panels.values() for pair in rows for atom in pair})
    query_bank = dict(zip(topics, model.encode(topics, query=True), strict=True))
    singles = {topic: model.encode([topic], query=True)[0] for topic in topics}
    active = {atom[0] for pair in queries for atom in pair}
    pairs = sorted({tuple(atom[0] for atom in pair) for pair in queries})
    two = {pair: model.encode(list(pair), query=True) for pair in pairs}
    variants = ("cached_panel", "single", "live_pair")
    metrics = {variant: Counter() for variant in variants}
    comparisons = {variant: Counter() for variant in ("single", "live_pair")}
    with TemporaryDirectory() as root:
        index = live.LocalIndex(catalogue, model, Path(root) / "elife")
        for atoms in queries:
            topic_pair = tuple(topic for topic, _ in atoms)
            vectors = {
                "cached_panel": np.stack([query_bank[topic] for topic in topic_pair]),
                "single": np.stack([singles[topic] for topic in topic_pair]),
                "live_pair": two[topic_pair],
            }
            plans = {}
            for variant in variants:
                index.encoder = SimpleNamespace(encode=lambda *args, **kwargs: vectors[variant])
                ranks = index.rank(list(topic_pair))
                plan, read = live.inspect(index, atoms, ranks, "joint")
                result = live.render(read, plan, atoms, ranks)
                matches = live.json.loads(result)["matches"]
                good = bool(matches) and all(topic_pair[i] in subjects[matches[0][i+1]] for i in range(2))
                metrics[variant].update(issued=bool(matches), known=good, unresolved=bool(matches) and not good)
                plans[variant] = (ranks, plan, result)
            base_ranks, base_plan, base_answer = plans["cached_panel"]
            for variant, comparison in comparisons.items():
                ranks, plan, answer = plans[variant]
                comparison.update(
                    changed_rankings=sum(tuple(a) != tuple(b) for a, b in zip(base_ranks, ranks, strict=True)),
                    changed_sets=sum(a.keys() != b.keys() for a, b in zip(base_ranks, ranks, strict=True)),
                    changed_plans=int(base_plan != plan), changed_answers=int(base_answer != answer),
                )
    write_json(args.output, dict(
        queries=len(queries), unique_heading_terms=len(active), unique_heading_pairs=len(pairs),
        projection_sha256=checksum, embedding_checkpoint=checkpoint,
        script_sha256=digest(Path(__file__)),
        max_abs_cached_vs_live_query_vector=float(max(
            np.max(np.abs(two[pair][i] - query_bank[topic]))
            for pair in pairs for i, topic in enumerate(pair)
        )),
        max_abs_cached_vs_single_query_vector=float(max(
            np.max(np.abs(singles[topic] - query_bank[topic])) for topic in active
        )),
        results=metrics, relative_to_cached=comparisons,
    ))


if __name__ == "__main__":
    main()
