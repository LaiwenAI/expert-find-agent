"""Test shared duty normalization in an end-to-end structured expert query."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import study_role_normalization as queries
from experiments.credit import study_role_normalization_batch as batch
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json
from experiments.transfer.constraint_counterfactual import Judge

NATIVE = dict(design="Methodology", execution="Investigation", analysis="Formal analysis",
              writing="Writing – original draft", resources="Resources")
ARMS = ("direct_person", "normalize_per_source", "normalize_per_query")
SEATS = 5
READS = 8


def query_set(catalog):
    person = {pair: posts for pair, posts in catalog.person.items() if set(pair) <= set(NATIVE.values())}
    reverse = {name: role for role, name in NATIVE.items()}
    selected = sorted(topic for topic, articles in catalog.topics.items() if len(articles) >= 4)
    index = SimpleNamespace(person=person, topics=catalog.topics)
    outside_seen = set()
    for topic, frequency, roles in pilot.query_panel(index, selected):
        seed = pilot.stable("elife-native-query-normalization-v1", topic, frequency, roles)
        canonical = [reverse[role] for role in roles]
        phrases = [queries.PHRASES[role][seed[i] % len(queries.PHRASES[role])]
                   for i, role in enumerate(canonical)]
        candidates = queries.PHRASES["outside"]
        offset = seed[2] % len(candidates)
        outside = next(phrase for phrase in candidates[offset:] + candidates[:offset]
                       if (topic, phrases[0], phrase) not in outside_seen)
        outside_seen.add((topic, phrases[0], outside))
        yield topic, phrases, set(roles), False
        yield topic, [phrases[0], outside], set(), True


def direct(judge, phrases, declared):
    task = ("Check whether both requested duties are established for this author by the declared "
            "fields. Use only the five allowed fields. A duty outside those fields is not established. "
            "Answer 1 for yes and 0 for no.")
    context = dict(requested_duties=phrases, allowed_fields=NATIVE, declared_fields=sorted(declared))
    return judge(task, context, (False, True))[0]


def bind(accepted, declared):
    names = {NATIVE[role] for role in accepted if role in NATIVE}
    return len(names) == 2 and names <= declared


def run_query(catalog, judge, row, arm):
    topic, phrases, expected, outside = row
    started = perf_counter()
    ranked = catalog.rank(topic)[:READS]
    values = catalog.verify(topic, ranked)
    candidates = sorted(zip(ranked, values, strict=True), key=lambda item: -item[1])
    passed = [i for i, score in candidates if score > holdout.CUTOFF]
    accepted = []
    if passed and arm == "normalize_per_query":
        accepted = batch.normalize(judge, phrases, "one_batch")[1]

    selected = []
    seen = set()
    role_calls = int(bool(passed) and arm == "normalize_per_query")
    reads_with_owners = 0
    for source in passed:
        if arm == "normalize_per_source":
            accepted = batch.normalize(judge, phrases, "one_batch")[1]
            role_calls += 1
        reads_with_owners += 1
        article = catalog.articles[source]
        for author in article["authors"]:
            owner = author["person"] or f'{article["doi"]}#{author["ordinal"]}'
            if owner in seen:
                continue
            declared = set(author["roles"])
            if arm == "direct_person":
                eligible = direct(judge, phrases, declared)
                role_calls += 1
            else:
                eligible = bind(accepted, declared)
            if eligible:
                selected.append((owner, source, not outside and expected <= declared))
                seen.add(owner)
                if len(selected) == SEATS:
                    break
        if len(selected) == SEATS:
            break
    elapsed = (perf_counter() - started) * 1000
    topics = catalog.topics[topic]
    return dict(
        elapsed_ms=elapsed, role_model_calls=role_calls, ranked_sources=len(ranked),
        passed_sources=len(passed), considered_with_owners=reads_with_owners,
        issued=len(selected), wrong_role=sum(not supports for _, _, supports in selected),
        tag_backed=sum(supports and source in topics for _, source, supports in selected),
        query_with_unsupported=int(any(not supports for _, _, supports in selected)),
        delivered=tuple((owner, source) for owner, source, _ in selected),
    )


def evaluate(catalog, judge, offset):
    rows = list(query_set(catalog))
    for arm in ARMS:
        run_query(catalog, judge, rows[0], arm)
    counts = {condition: {arm: Counter() for arm in ARMS} for condition in ("covered", "outside")}
    times = {condition: defaultdict(list) for condition in counts}
    for i, row in enumerate(rows):
        condition = "outside" if row[3] else "covered"
        shift = (i + offset) % len(ARMS)
        outcomes = {a: run_query(catalog, judge, row, a) for a in ARMS[shift:] + ARMS[:shift]}
        baseline = outcomes["normalize_per_source"]["delivered"]
        for arm, metrics in outcomes.items():
            counts[condition][arm].update(
                queries=1, differs_from_per_source=int(metrics["delivered"] != baseline),
                **{k: v for k, v in metrics.items() if k not in {"elapsed_ms", "delivered"}},
            )
            times[condition][arm].append(metrics["elapsed_ms"])
    return {condition: {arm: dict(values, p95_ms=float(np.quantile(times[condition][arm], .95)))
                        for arm, values in outcomes.items()}
            for condition, outcomes in counts.items()}


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
    queries.extended_ids(judge)
    articles = [dict(row, topics=row["keywords"]) for row in json.loads(args.projection.read_bytes())["sources"]
                if row["declaration_complete"]]
    catalog = pilot.Catalog(articles, enc, reader)
    report = evaluate(catalog, judge, args.order_offset)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        order_offset=args.order_offset, script_sha256=digest(Path(__file__)),
        projection_sha256=digest(args.projection), topical_reference_sha256=digest(args.reference),
        model_provenance_sha256=judge.sha, articles=len(articles), results=report,
    ))


if __name__ == "__main__":
    main()
