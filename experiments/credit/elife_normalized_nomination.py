"""Ablate role-aware nomination against a shared-normalization search baseline."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import elife_role_normalization_search as search
from experiments.credit import study_role_normalization as normalize
from experiments.credit import study_role_normalization_batch as batch
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge

ARMS = ("topic", "bounded", "global")
LIMIT = 16


def choose(ranked, visible, allowed, arm):
    base = ranked[:search.READS]
    if arm == "topic":
        return base
    scope = ranked[:LIMIT] if arm == "bounded" else ranked
    selected = [i for i in scope if i in allowed or i not in visible][:search.READS]
    required = {i for i in base if i in allowed or i not in visible}
    if not required.issubset(selected):
        raise ValueError("Route lost a potentially viable initial candidate")
    if len(selected) < search.READS:
        selected_set = set(selected)
        selected.extend(i for i in base if i not in selected_set)
    return selected[:search.READS]


def prepare_queries(catalog, complete):
    topics = {topic: posts & complete for topic, posts in catalog.topics.items()}
    person = {pair: posts & complete for pair, posts in catalog.person.items()}
    return list(search.query_set(SimpleNamespace(topics=topics, person=person)))


def query(catalog, judge, row, visible, arm):
    topic, phrases, expected, outside = row
    started = perf_counter()
    ordered = catalog.rank(topic)
    accepted = batch.normalize(judge, phrases, "one_batch")[1]
    fields = {search.NATIVE[value] for value in accepted if value in search.NATIVE}
    if len(fields) != 2:
        return [], perf_counter() - started, 0, 0

    role_pair = tuple(sorted(fields))
    chosen = choose(ordered, visible, catalog.person[role_pair], arm)
    if len(chosen) != search.READS or len(set(chosen)) != search.READS:
        raise ValueError("Each issued query must read the same number of distinct sources")
    scores = catalog.verify(topic, chosen)
    selected = []
    seen = set()
    for article_id, value in sorted(zip(chosen, scores, strict=True), key=lambda item: -item[1]):
        if value <= holdout.CUTOFF:
            continue
        article = catalog.articles[article_id]
        for author in article["authors"]:
            name = author["person"] or f'{article["doi"]}#{author["ordinal"]}'
            declared = set(author["roles"])
            if name not in seen and fields <= declared:
                seen.add(name)
                correct = not outside and expected <= declared
                tagged = correct and article_id in catalog.topics[topic]
                selected.append((name, article_id, correct, tagged))
                if len(selected) == search.SEATS:
                    break
        if len(selected) == search.SEATS:
            break
    elapsed = perf_counter() - started
    return selected, elapsed, len(chosen), len(set(chosen) - set(ordered[:search.READS]))


def evaluate(catalog, judge, complete, offset):
    rows = prepare_queries(catalog, complete)
    for arm in ARMS:
        query(catalog, judge, rows[0], complete, arm)
    panels = {}
    for mask, declared in catalog.visible.items():
        visible = declared & complete
        count = {panel: {arm: Counter() for arm in ARMS} for panel in ("covered", "outside")}
        timing = defaultdict(list)
        delta = {arm: [] for arm in ARMS}
        labels = []
        for i, row in enumerate(rows):
            panel = "outside" if row[3] else "covered"
            shift = (offset + mask + i) % len(ARMS)
            outcomes = {arm: query(catalog, judge, row, visible, arm)
                        for arm in ARMS[shift:] + ARMS[:shift]}
            baseline = outcomes["topic"][0]
            if panel == "covered":
                labels.append(row[0])
            for arm, (witnesses, elapsed, n, new_sources) in outcomes.items():
                common = min(len(baseline), len(witnesses))
                matched = sum(w[3] for w in witnesses[:common]) - sum(w[3] for w in baseline[:common])
                count[panel][arm].update(
                    queries=1, reads=n, injected_sources=new_sources, issued=len(witnesses),
                    tag_backed=sum(w[3] for w in witnesses),
                    wrong_role=sum(not w[2] for w in witnesses),
                    matched_issued=common, matched_tag_delta=matched,
                    different_lists=int(witnesses != baseline),
                )
                if panel == "covered":
                    delta[arm].append(matched / search.SEATS)
                timing[panel, arm].append(elapsed * 1000)
        panels[mask] = dict(indexed=len(visible), results={
            panel: {arm: dict(metrics, p95_ms=float(np.quantile(timing[panel, arm], .95)),
                              **({"matched_tag_p5_ci": holdout.interval(delta[arm], labels)["percentile_95"]}
                                 if panel == "covered" else {}))
                    for arm, metrics in arms.items()}
            for panel, arms in count.items()
        })
    return panels


def main():
    import torch

    from experiments.followir.core17_verifier import Reranker

    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("projection", "manifest", "reference", "output"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    pinned = json.loads(args.manifest.read_bytes())
    reference = json.loads(args.reference.read_bytes())["run"]
    if digest(args.projection) != pinned["projection_sha256"]:
        raise ValueError("Unexpected eLife projection")
    embedding = Path("artifacts/models/qwen3-embedding-0.6b")
    reranker = Path("artifacts/models/qwen3-reranker-0.6b")
    encoder_info = pilot.verify_checkpoint(embedding)
    enc = QwenTorchEmbedding(embedding)
    enc.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"]) != (reference["embedding"], reference["reranker"]):
        raise ValueError("Unexpected topical checkpoint")
    judge = Judge()
    normalize.extended_ids(judge)
    articles = [dict(article, topics=article["keywords"])
                for article in json.loads(args.projection.read_bytes())["sources"]]
    catalog = pilot.Catalog(articles, enc, reader)
    complete = {i for i, article in enumerate(articles) if article["declaration_complete"]}
    panels = evaluate(catalog, judge, complete, args.order_offset)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
        projection_sha256=digest(args.projection), reference_sha256=digest(args.reference),
        decision_model_sha256=judge.sha, documents=len(articles),
        complete_declarations=len(complete), panels=panels,
    ))


if __name__ == "__main__":
    main()
