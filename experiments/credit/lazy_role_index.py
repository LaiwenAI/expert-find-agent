"""Compare equivalent resident and SQLite paths for same-person source nomination."""

import argparse
from collections import Counter
from itertools import cycle
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_holdout as holdout
from experiments.credit import topic_role_postings as pilot


ARMS = ("resident", "sqlite_postings", "sqlite_wave4", "sqlite_wave8")


def compile_postings(db, person):
    db.execute("CREATE TABLE postings (pair TEXT, article INTEGER, PRIMARY KEY(pair,article)) WITHOUT ROWID")
    db.executemany("INSERT INTO postings VALUES (?,?)", (
        (pair_key(pair), article)
        for pair, articles in sorted(person.items()) for article in sorted(articles)
    ))
    db.commit()


def pair_key(pair):
    return json.dumps(pair, separators=(",", ":"), ensure_ascii=True)


def select(db, ranked, pair, complete, person, arm):
    counts = Counter(sql_calls=0, ids_probed=0, rows_returned=0, prefix_considered=0)
    picked = []
    code = pair_key(pair)
    if arm == "resident":
        waves = (ranked,)
        positives = person.get(pair, set())
    elif arm == "sqlite_postings":
        rows = list(db.execute("SELECT article FROM postings WHERE pair=?", (code,)))
        counts.update(sql_calls=1, rows_returned=len(rows))
        positives = {article for article, in rows}
        waves = (ranked,)
    else:
        width = {"sqlite_wave4": 4, "sqlite_wave8": 8}[arm]
        waves = (ranked[start:start + width] for start in range(0, len(ranked), width))
        positives = None

    for wave in waves:
        support = positives
        if support is None:
            known = [article for article in wave if article in complete]
            support = set()
            if known:
                placeholders = ",".join("?" for _ in known)
                rows = db.execute(f"SELECT article FROM postings WHERE pair=? AND article IN ({placeholders})",
                                  (code, *known))
                support = {article for article, in rows}
                counts.update(sql_calls=1, ids_probed=len(known), rows_returned=len(support))
        counts["prefix_considered"] += len(wave) if positives is None else 0
        for article in wave:
            if positives is not None:
                counts["prefix_considered"] += 1
            if article not in complete or article in support:
                picked.append(article)
                if len(picked) == 4:
                    return picked, counts
    return picked, counts


def timed_queries(catalog, db, repeats, expected):
    queries = batches.queries(catalog)
    summaries = {}
    for pct in pilot.CACHE:
        observations = {arm: dict(counts=Counter(), selection_us=[], request_ms=[]) for arm in ARMS}
        complete = catalog.visible[pct]
        rotations = cycle(range(len(ARMS)))
        for _ in range(repeats):
            for topic, _, pair in queries:
                shift = next(rotations)
                oracle = None
                for arm in ARMS[shift:] + ARMS[:shift]:
                    start = perf_counter()
                    ranked = catalog.rank(topic)
                    selecting = perf_counter()
                    selected, counts = select(db, ranked, pair, complete, catalog.person, arm)
                    selection_us = (perf_counter() - selecting) * 1e6
                    scores = catalog.verify(topic, selected)
                    result = nomination.evaluate(catalog, topic, pair, selected, scores, holdout.CUTOFF)
                    request_ms = (perf_counter() - start) * 1e3
                    for key in ("grounded", "cited_off_label", "issued"):
                        counts[key] = result[key]
                    counts["reads"] = len(selected)
                    trace = ranked, selected, list(scores), result["witnesses"]
                    if oracle is None:
                        oracle = trace
                    elif trace != oracle:
                        raise ValueError("Equivalent routes must preserve ranking, selected sources, scores and citations")
                    observed = observations[arm]
                    observed["counts"].update(counts)
                    observed["selection_us"].append(selection_us)
                    observed["request_ms"].append(request_ms)
        original = expected["holdout"][str(pct)]["person_four"]["vs_topic_four"]
        results = {}
        for arm, seen in observations.items():
            for key in ("reads", "grounded", "cited_off_label", "issued"):
                if seen["counts"][key] != original[key] * repeats:
                    raise ValueError(f"Frozen outcome differs for {pct}, {arm}, {key}")
            results[arm] = dict(
                counts=seen["counts"],
                selection_p50_us=float(np.quantile(seen["selection_us"], .5)),
                selection_p95_us=float(np.quantile(seen["selection_us"], .95)),
                request_p50_ms=float(np.quantile(seen["request_ms"], .5)),
                request_p95_ms=float(np.quantile(seen["request_ms"], .95)),
            )
        summaries[pct] = dict(indexed=len(complete), requests_per_arm=len(queries) * repeats, arms=results)
    return summaries


def run(projection, manifest, embedding, reranker, reference, output, repeats):
    import torch

    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or repeats < 1:
        raise ValueError("Require fresh output, positive repeats and Slurm CUDA")
    frozen = json.loads(manifest.read_bytes())
    expected = json.loads(reference.read_bytes())
    if pilot.digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Projection differs from the frozen catalog")
    encoder_info = pilot.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    if (encoder_info["revision"], reader.provenance["revision"], pilot.digest(manifest)) != (
            expected["embedding"], expected["reranker"], expected["manifest_sha256"]):
        raise ValueError("Model or manifest differs from the reference")
    catalog = pilot.Catalog(json.loads(projection.read_bytes())["sources"], encoder, reader)
    topic, _, pair = batches.queries(catalog)[0]
    with TemporaryDirectory() as scratch:
        path = Path(scratch) / "postings.sqlite"
        db = sqlite3.connect(path)
        try:
            compile_postings(db, catalog.person)
            rows = db.execute("SELECT count(*) FROM postings").fetchone()[0]
            ranked = catalog.rank(topic)
            for arm in ARMS:
                selected, _ = select(db, ranked, pair, catalog.visible[100], catalog.person, arm)
                catalog.verify(topic, selected)
            outcomes = timed_queries(catalog, db, repeats, expected)
        finally:
            db.close()
        index_bytes = path.stat().st_size
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=pilot.digest(Path(__file__)), manifest_sha256=pilot.digest(manifest),
        reference_sha256=pilot.digest(reference), embedding=encoder_info["revision"],
        reranker=reader.provenance["revision"], sqlite=sqlite3.sqlite_version,
        index_bytes=index_bytes, posting_rows=rows, repeats=repeats, results=outcomes,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "embedding", "reranker", "reference", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--repeats", type=int, default=3)
    run(**vars(cli.parse_args()))
