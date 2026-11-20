"""Cross first-round role nomination with local and global evidence rescue."""

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json


BUDGETS = ((2, 4), (4, 8), (6, 12))
RESCUE_CAP = 2
ARMS = ("topic_local", "person_local", "topic_global", "person_global")


class Encoder:
    def __init__(self, path):
        from fastembed import TextEmbedding

        self.model = TextEmbedding("BAAI/bge-small-en-v1.5", specific_model_path=str(path),
                                   threads=1, providers=["CPUExecutionProvider"])

    def encode(self, texts, query=False):
        method = self.model.query_embed if query else self.model.embed
        return np.asarray(list(method(texts)), dtype=np.float32)


class Verifier:
    name = "ms-marco-MiniLM-L-12-v2"

    def __init__(self, path):
        from flashrank import Ranker
        from flashrank.Config import model_file_map
        import onnxruntime as ort

        self.ranker = Ranker(model_name=self.name, cache_dir=str(path),
                             max_length=512, log_level="WARNING")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        model_path = self.ranker.model_dir / model_file_map[self.name]
        self.ranker.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
        )

    def score_pairs(self, pairs, task):
        from flashrank import RerankRequest

        query = pairs[0][0]
        if any(q != query for q, _ in pairs):
            raise ValueError("Expected a single need per verifier batch")
        candidates = [{"id": i, "text": text} for i, (_, text) in enumerate(pairs)]
        out = self.ranker.rerank(RerankRequest(query=query, passages=candidates))
        return np.array([row["score"] for row in sorted(out, key=lambda row: row["id"])]), False


def panel_queries(catalog):
    panels = {}
    for field, minimum in (("topics", 8), ("keywords", 4)):
        lookup = defaultdict(set)
        for i, article in enumerate(catalog.articles):
            for topic in article[field]:
                lookup[topic].add(i)
        catalog.topics = lookup
        keys = sorted(topic for topic, articles in lookup.items() if len(articles) >= minimum)
        panels[field] = (lookup, list(pilot.query_panel(catalog, keys)))
    return panels


def simulate(catalog, topic, pair, ranked, visible, arm, budget, pool, scores, cutoff):
    prefilter = arm.startswith("person")
    compatible = [i for i in ranked if i not in visible or i in catalog.person[pair]]
    initial = (compatible if prefilter else ranked)[:budget]

    def assess(sources):
        return nomination.evaluate(catalog, topic, pair, sources, scores(sources), cutoff)

    first = assess(initial)
    rescue = []
    # Compare actions under the same trigger and cap. Read only unseen sources.
    if first["issued"] < pilot.TOP:
        baseline_pool = set(ranked[:pool])
        allow_global = arm.endswith("global")
        seen = set(initial)
        rescue = [i for i in compatible if i not in seen and (allow_global or i in baseline_pool)][:RESCUE_CAP]
    final = assess(initial + rescue) if rescue else first
    return dict(first=first["grounded"], final=final["grounded"],
                off_label=final["cited_off_label"], issued=final["issued"],
                first_reads=len(initial), rescue_reads=len(rescue),
                initial_outside_pool=len(set(initial) - set(ranked[:pool])),
                rescue_outside_pool=len(set(rescue) - set(ranked[:pool])))


def calibrate(catalog, panels, rankings, verifier_scores, development):
    negatives = {}
    for lookup, queries in panels.values():
        for topic, _, _ in queries:
            if topic not in development:
                continue
            sources = rankings[topic][:8]
            for source, score in zip(sources, verifier_scores(topic, sources), strict=True):
                if source not in lookup[topic]:
                    negatives[topic, source] = score
    if not negatives:
        raise ValueError("Empty cutoff calibration")
    return float(np.quantile(list(negatives.values()), .90, method="higher")), len(negatives)


def run(projection, manifest, embedding, reranker):
    frozen = json.loads(manifest.read_bytes())
    if digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Changed publisher projection")
    articles = json.loads(projection.read_bytes())["sources"]
    catalog = pilot.Catalog(articles, Encoder(embedding), Verifier(reranker))
    complete = {i for i, article in enumerate(articles) if article["declaration_complete"]}
    panels = panel_queries(catalog)
    all_topics = {topic for _, queries in panels.values() for topic, _, _ in queries}
    development = {topic for topic in all_topics if int.from_bytes(
        pilot.stable("factorial-calibration-v1", topic)[:8], "big") % 3 == 0}
    rankings = {topic: catalog.rank(topic) for topic in sorted(all_topics)}
    cache = defaultdict(dict)

    def scores(topic, sources):
        known = cache[topic]
        missing = [source for source in sources if source not in known]
        if missing:
            known.update(zip(missing, catalog.verify(topic, missing), strict=True))
        return [known[source] for source in sources]

    cutoff, negatives = calibrate(catalog, panels, rankings, scores, development)
    results = {}
    for field, (lookup, all_queries) in panels.items():
        catalog.topics = lookup
        queries = [(topic, pair) for topic, _, pair in all_queries if topic not in development]
        groups = [topic for topic, _ in queries]
        field_result = dict(queries=len(queries), topics=len(set(groups)), masks={})
        for mask in pilot.CACHE:
            masking = catalog.visible[mask] & complete
            measured = {}
            for budget, pool in BUDGETS:
                cells = {arm: [] for arm in ARMS}
                for topic, pair in queries:
                    for arm in ARMS:
                        cells[arm].append(simulate(catalog, topic, pair, rankings[topic], masking,
                                                  arm, budget, pool,
                                                  lambda selected: scores(topic, selected), cutoff))
                measured[budget] = summarize(cells, groups)
            field_result["masks"][mask] = measured
        results[field] = field_result

    return dict(
        projection_sha256=digest(projection), manifest_sha256=digest(manifest),
        packages={name: importlib.metadata.version(name)
                  for name in ("fastembed", "flashrank", "onnxruntime", "numpy")},
        embedding_sha256={p.name: digest(p) for p in sorted(embedding.iterdir()) if p.is_file()},
        reranker_sha256={p.name: digest(p) for p in sorted(
            (reranker / Verifier.name).iterdir()) if p.is_file()},
        development_topics=len(development), cutoff=cutoff, development_off_label=negatives,
        visible_counts={mask: len(known & complete) for mask, known in catalog.visible.items()},
        source_count=len(articles), results=results,
    )


def summarize(cells, groups):
    def contrast(values):
        differences = np.asarray(values, dtype=float) / pilot.TOP
        interval = clustered_paired_interval(differences, groups)
        interval["sampling_unit"] = "whole topic; query-weighted mean"
        return interval

    rows = list(zip(*(cells[name] for name in ARMS), strict=True))
    return dict(
        arms={arm: {key: sum(row[key] for row in records) for key in records[0]}
              for arm, records in cells.items()},
        early_at_local=contrast([pl["final"] - tl["final"] for tl, pl, _, _ in rows]),
        global_at_topic=contrast([tg["final"] - tl["final"] for tl, _, tg, _ in rows]),
        global_at_person=contrast([pg["final"] - pl["final"] for _, pl, _, pg in rows]),
        interaction=contrast([pg["final"] - pl["final"] - tg["final"] + tl["final"]
                              for tl, pl, tg, pg in rows]),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    result = run(args.projection, args.manifest, args.embedding, args.reranker)
    write_json(args.output, dict(script_sha256=digest(Path(__file__)), **result))
