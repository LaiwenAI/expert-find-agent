"""Transfer matched-read first-query routes from eLife to PLOS."""

import argparse
import json
from pathlib import Path

from experiments.credit import role_first_query_equal_reads as live
from experiments.credit import role_first_query_factorial as factorial
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json


COMPARISONS = (
    ("person_fixed", "topic_fixed"), ("person_adaptive", "topic_adaptive"),
    ("person_global_fixed", "person_fixed"),
)


def contrasts(cells, groups):
    output = {}
    for candidate, control in COMPARISONS:
        delta = {}
        for field in ("grounded", "off_label"):
            values = [(a[field] - b[field]) / pilot.TOP
                      for a, b in zip(cells[candidate], cells[control], strict=True)]
            interval = pilot.clustered_paired_interval(values, groups)
            interval["sampling_unit"] = "whole topic; query-weighted mean"
            delta[field] = interval
        output[f"{candidate}_vs_{control}"] = delta
    return output


def reporting(rows, queries):
    groups = [topic for topic, _, _ in queries]
    return dict(queries=len(groups), topics=len(set(groups)),
                arms=live.summarize(rows), contrasts=contrasts(rows, groups))


def run(projection, manifest, reference, embedding, reranker, offset):
    receipt = json.loads(manifest.read_bytes())
    prior = json.loads(reference.read_bytes())
    if digest(projection) != receipt["projection_sha256"]:
        raise ValueError("Unexpected PLOS projection")
    for directory, key in ((embedding, "embedding_sha256"),
                           (reranker / factorial.Verifier.name, "reranker_sha256")):
        if {name: digest(directory / name) for name in prior[key]} != prior[key]:
            raise ValueError(f"Unexpected checkpoint files: {directory}")

    source = json.loads(projection.read_bytes())
    catalog = pilot.Catalog(source["sources"], factorial.Encoder(embedding), factorial.Verifier(reranker))
    topics = sorted(topic for topic, articles in catalog.topics.items() if len(articles) >= 4)
    queries = list(pilot.query_panel(catalog, topics))
    original = set(source["original_topics"])
    novel = [i for i, query in enumerate(queries) if query[0] not in original]
    result = {}
    for mask, visible in catalog.visible.items():
        topic, _, pair = queries[0]
        for arm in live.ARMS:
            live.replay(catalog, topic, pair, visible, prior["cutoff"], arm)
        cells = {arm: [] for arm in live.ARMS}
        for i, (topic, _, pair) in enumerate(queries):
            shift = (offset + i) % len(live.ARMS)
            for arm in live.ARMS[shift:] + live.ARMS[:shift]:
                cells[arm].append(live.replay(catalog, topic, pair, visible, prior["cutoff"], arm))
        novel_cells = {arm: [rows[i] for i in novel] for arm, rows in cells.items()}
        result[mask] = dict(indexed=len(visible), all=reporting(cells, queries),
                            novel=reporting(novel_cells, [queries[i] for i in novel]))
    return dict(
        source_count=len(catalog.articles), projection_sha256=digest(projection),
        manifest_sha256=digest(manifest), reference_sha256=digest(reference),
        cutoff=prior["cutoff"], order_offset=offset, masks=result,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("projection", "manifest", "reference", "embedding", "reranker", "output"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    report = run(args.projection, args.manifest, args.reference, args.embedding, args.reranker,
                 args.order_offset)
    write_json(args.output, dict(
        script_sha256=digest(Path(__file__)), policy_runner_sha256=digest(Path(live.__file__)), **report,
    ))
