"""Check source expansion using lexical anchors and agreement across clue fragments."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from experiments.credit import scoped_source_confirmation as reference
from experiments.credit import scoped_credit_search as discovery
from experiments.credit.batched_source_role_search import SourceIndex


PANEL = "c621d1d985ac7a5768b289b342068991de9aef20df3bcfa4f3db8578404087bc"
FEATURES = ("dense_rank", "full_similarity", "lexical_coverage", "fragment_similarity", "fragment_gap")
FRAGMENT = {"fragment_similarity", "fragment_gap"}


def prompts(texts):
    return [f"Instruct: {discovery.DENSE_TASK}\nQuery: {text}" for text in texts]


def halves(text):
    words = text.split()
    middle = (len(words) + 1) // 2
    if middle == len(words):
        raise ValueError("A clue needs at least two words for fragment scoring")
    return [" ".join(words[:middle]), " ".join(words[middle:])]


def lexical_coverage(clue, document, idf):
    words = set(tokens(clue))
    total = sum(idf[word] for word in words)
    return sum(idf[word] for word in words & document) / total if total else 0.0


def evaluate(index, corpus_terms, idf, clues, removed):
    start = perf_counter()
    embeddings = index.encoder.encode(prompts(clues))
    dense, fused = index.rank_sources(clues, embeddings, removed)
    simple = index.rank(dense, fused, "linked_top4")
    expanded = index.rank(dense, fused, "linked_expand")
    elapsed = (perf_counter() - start) * 1000
    outcome = dict(top4=simple, expanded=expanded, base_ms=elapsed, lexical_ms=0.0,
                   full_ms=0.0, fragment_ms=0.0)
    if simple is not None or expanded is None:
        return outcome
    ids = [index.ids[doc] for doc in expanded]
    start = perf_counter()
    outcome["dense_rank"] = -max(order.index(expected) + 1 for order, expected in zip(dense, ids, strict=True))
    outcome["full_similarity"] = min(float(index.vectors[doc] @ vector)
                                     for doc, vector in zip(ids, embeddings, strict=True))
    outcome["full_ms"] = (perf_counter() - start) * 1000
    start = perf_counter()
    outcome["lexical_coverage"] = min(lexical_coverage(clue, corpus_terms[doi], idf)
                                      for clue, doi in zip(clues, expanded, strict=True))
    outcome["lexical_ms"] = (perf_counter() - start) * 1000
    start = perf_counter()
    fragment_vectors = index.encoder.encode(prompts([part for clue in clues for part in halves(clue)]))
    values = index.vectors @ fragment_vectors.T
    if removed:
        values[[index.ids[doi] for doi in removed], :] = -np.inf
    similarities, gaps = [], []
    for i, doc in enumerate(ids):
        for j in (2 * i, 2 * i + 1):
            expected = float(values[doc, j])
            values[doc, j] = -np.inf
            similarities.append(expected)
            gaps.append(expected - float(np.max(values[:, j])))
    outcome.update(fragment_similarity=min(similarities), fragment_gap=min(gaps),
                   fragment_ms=(perf_counter() - start) * 1000)
    return outcome


def choose(row, method, threshold):
    if method == "expanded":
        return row["expanded"], row["base_ms"], 0
    if row["top4"] is not None or row["expanded"] is None or method == "top4":
        return row["top4"], row["base_ms"], 0
    if method in FRAGMENT:
        elapsed = row["fragment_ms"]
    elif method == "lexical_coverage":
        elapsed = row["lexical_ms"]
    else:
        elapsed = row["full_ms"]
    return (row["expanded"] if row[method] > threshold else None, row["base_ms"] + elapsed,
            int(method in FRAGMENT))


def cross_fit(records):
    components = sorted({row["component"] for row in records})
    thresholds = {}
    for component in components:
        for feature in FEATURES:
            negatives = [row[feature] for row in records if row["component"] != component and feature in row
                         and row["truth"] != row["expanded"]]
            if not negatives:
                raise ValueError("No independent incorrect expansion candidates in calibration")
            thresholds[component, feature] = max(negatives)
    reports = {}
    for subset in ("all", *discovery.VIEWS):
        rows = records if subset == "all" else [row for row in records if row["view"] == subset]
        reports[subset] = {}
        for method in ("top4", "expanded", *FEATURES):
            counts = Counter()
            times = []
            for row in rows:
                selected, ms, extra = choose(row, method, thresholds.get((row["component"], method)))
                times.append(ms)
                counts.update(dict(queries=1, extra_forwards=extra, correct_present=row["truth"] is not None and selected == row["truth"],
                                   wrong_present=row["truth"] is not None and selected not in (None, row["truth"]),
                                   wrong_missing=row["truth"] is None and selected is not None,
                                   added_correct=row["top4"] is None and selected is not None and selected == row["truth"],
                                   added_wrong=row["top4"] is None and selected is not None and selected != row["truth"]))
            reports[subset][method] = dict(counts, factored_p95_ms=float(np.quantile(times, .95)))
    return reports


def run(projection, cases, checkpoint, old_file, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    if discovery.digest(projection) != discovery.PROJECTION or discovery.digest(cases) != PANEL:
        raise ValueError("Changed source corpus or clue panel")
    prior = json.loads(old_file.read_text())
    if prior["panel_sha256"] != PANEL or prior["frozen_sha256"] != discovery.PROJECTION:
        raise ValueError("Expected the original source-confirmation comparison")
    payload = json.loads(projection.read_text())
    panel = json.loads(cases.read_text())
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.litsearch.dense_cache import verify_checkpoint

    provenance = verify_checkpoint(checkpoint)
    if provenance["revision"] != prior["embedding_revision"]:
        raise ValueError("Changed embedding revision")
    encoder = QwenTorchEmbedding(checkpoint, batch_size=24)
    encoder.model.config.use_cache = False
    corpus = payload["documents"]
    corpus_terms = {doi: set(tokens(row["index"])) for doi, row in corpus.items()}
    counts = Counter(term for words in corpus_terms.values() for term in words)
    idf = defaultdict(lambda: float(np.log1p(len(corpus))))
    idf.update({term: float(np.log((1 + len(corpus)) / (1 + n))) for term, n in counts.items()})
    records = []
    with TemporaryDirectory() as directory:
        index = SourceIndex(payload, encoder, Path(directory))
        for i, (a, b, component) in enumerate([panel["pairs"][0], *panel["pairs"]]):
            for view in discovery.VIEWS:
                clues = [panel["clues"][doi][view] for doi in (a, b)]
                for condition in reference.MISSING:
                    removed = {"missing_left": (a,), "missing_right": (b,)}.get(condition, ())
                    result = evaluate(index, corpus_terms, idf, clues, removed)
                    if i:
                        records.append(result | dict(component=component, view=view, truth=None if removed else (a, b)))
    report = cross_fit(records)
    for scope in report:
        for method in ("top4", "expanded"):
            previous = prior["results"][scope][method]
            observed = report[scope][method]
            for old, new in (("present_exact", "correct_present"), ("present_wrong_accepts", "wrong_present"),
                             ("missing_false_accepts", "wrong_missing")):
                if previous.get(old, 0) != observed[new]:
                    raise ValueError("Frozen source ranking did not reproduce the published control")
    candidates = [row for row in records if row["top4"] is None and row["expanded"] is not None]
    truth = {view: Counter("correct" if row["truth"] == row["expanded"] else "wrong"
                           for row in candidates if row["view"] == view) for view in discovery.VIEWS}
    discovery.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                      script_sha256=discovery.digest(Path(__file__)), panel_sha256=PANEL,
                                      corpus_sha256=discovery.PROJECTION, prior_sha256=discovery.digest(old_file),
                                      embedding_revision=provenance["revision"], components=prior["components"],
                                      additional_candidates=truth, results=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "cases", "checkpoint", "old_file", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
