"""Transfer joint contributor nomination to an independent eLife catalog."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.credit import elife_role_nomination as publisher
from experiments.credit import joint_byline_depth as sweep
from experiments.followir.core17_reranking import digest, write_json


BASE = sweep.base
DEPTHS = 8, 16, 32, 64


def load_catalogue(xml, manifest):
    with TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / "projection.json"
        publisher.prepare(xml, manifest, snapshot)
        rows = json.loads(snapshot.read_bytes())["sources"]
        checksum = digest(snapshot)
    ownership = {row["doi"]: row for row in rows}
    documents = {row["doi"]: tuple(row["text"].split("\n", 1)) for row in rows}
    return BASE.pinned.Catalogue(documents, ownership), checksum


def build_requests(catalogue):
    panels = {}
    for field in ("topics", "keywords"):
        subjects = {
            doi: {" ".join(label.casefold().split()) for label in row[field]}
            for doi, row in catalogue.ownership.items()
        }
        gold = BASE.annotation_postings(catalogue, subjects)
        queries = BASE.query_panel(gold)
        panels[field] = subjects, gold, queries
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("xml", "manifest", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh result on a Slurm compute node")
    catalogue, checksum = load_catalogue(args.xml, args.manifest)
    panels = build_requests(catalogue)
    topics = sorted({atom[0] for _, _, queries in panels.values() for pair in queries for atom in pair})
    if not topics:
        write_json(args.output, dict(
            sources=len(catalogue.ids), projection_sha256=checksum, topic_terms=0,
            manifest_sha256=digest(args.manifest), script_sha256=digest(Path(__file__)),
            panels={name: {"queries":0} for name in panels},
        ))
        return
    rankings, checkpoint = BASE.topic_rankings(catalogue, topics, args.checkpoint, width=max(DEPTHS))
    union = {doi: {role for a in row["authors"] for role in a["roles"]}
             for doi, row in catalogue.ownership.items()}
    nominator = partial(sweep.posting_plan, catalogue.byline)
    results = {}
    for field, (subjects, gold, queries) in panels.items():
        clusters = ["|".join(sorted((a[0], b[0]))) for a, b in queries]
        results[field] = dict(queries=len(queries), distinct_topic_pairs=len(set(clusters)), methods={})
        if not queries:
            continue
        for medium, scores in rankings.items():
            variants = {}
            for depth in DEPTHS:
                truncated = {topic: dict(list(ranks.items())[:depth]) for topic, ranks in scores.items()}
                measurements = [BASE.evaluate(catalogue, subjects, atoms, gold, truncated, union,
                                              depth=depth, joint_nominator=nominator) for atoms in queries]
                variants[str(depth)] = BASE.summarize(measurements, clusters)
            results[field]["methods"][medium] = variants
    write_json(args.output, dict(
        sources=len(catalogue.ids), projection_sha256=checksum, topic_terms=len(topics),
        manifest_sha256=digest(args.manifest), script_sha256=digest(Path(__file__)),
        embedding_checkpoint=checkpoint, panels=results,
    ))


if __name__ == "__main__":
    main()
