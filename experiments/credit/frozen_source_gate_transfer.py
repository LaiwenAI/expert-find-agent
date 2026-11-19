"""Freeze lexical expansion risk on one source panel and transfer it to another."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.retrieval import tokens
from experiments.credit import scoped_source_confirmation as panel_tools
from experiments.credit import source_clue_agreement as agreement


def lexical_statistics(corpus):
    terms = {doi: set(tokens(row["index"])) for doi, row in corpus.items()}
    counts = Counter(term for paper in terms.values() for term in paper)
    idf = {term: float(np.log((1 + len(corpus)) / (1 + count))) for term, count in counts.items()}
    return terms, idf, float(np.log1p(len(corpus)))


def inspect(index, terms, idf, absent, clues, removed):
    start = perf_counter()
    vectors = index.encoder.encode(agreement.prompts(clues))
    ranks = index.rank_sources(clues, vectors, removed)
    narrow = index.rank(*ranks, "linked_top4")
    expanded = index.rank(*ranks, "linked_expand")
    elapsed = (perf_counter() - start) * 1000
    if narrow is not None or expanded is None:
        return narrow, expanded, None, elapsed, 0.0
    start = perf_counter()
    weights = {word: idf.get(word, absent) for clue in clues for word in tokens(clue)}
    score = min(agreement.lexical_coverage(clue, terms[doi], weights)
                for clue, doi in zip(clues, expanded, strict=True))
    return narrow, expanded, score, elapsed, (perf_counter() - start) * 1000


def run(mode, projection, panel_path, checkpoint, output, frozen=None):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.credit.batched_source_role_search import SourceIndex
    from experiments.litsearch.dense_cache import verify_checkpoint

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    projection_hash = agreement.discovery.digest(projection)
    panel_hash = agreement.discovery.digest(panel_path)
    payload, panel = [json.loads(p.read_text()) for p in (projection, panel_path)]
    corpus = payload["documents"]
    if panel["frozen_sha256"] != projection_hash:
        raise ValueError("Mismatched panel and corpus")
    if panel["pairs"] != [list(row) for row in panel_tools.graph(corpus)]:
        raise ValueError("Changed linked source graph")
    provenance = verify_checkpoint(checkpoint)
    encoder = QwenTorchEmbedding(checkpoint, batch_size=24)
    encoder.model.config.use_cache = False
    if mode == "freeze":
        if panel_hash != agreement.PANEL or projection_hash != agreement.discovery.PROJECTION or frozen:
            raise ValueError("Only the original development panel can freeze the gate")
    else:
        lock = json.loads(frozen.read_text())
        if (lock["mode"] != "freeze" or lock["embedding_revision"] != provenance["revision"]
                or lock["panel_sha256"] == panel_hash or lock["corpus_sha256"] == projection_hash):
            raise ValueError("Test requires an independent panel and frozen threshold")
    stats = lexical_statistics(corpus)
    decisions = []
    with TemporaryDirectory() as directory:
        index = SourceIndex(payload, encoder, Path(directory))
        for i, (left, right, component) in enumerate([panel["pairs"][0], *panel["pairs"]]):
            for view in agreement.discovery.VIEWS:
                clues = [panel["clues"][doi][view] for doi in (left, right)]
                for condition in panel_tools.MISSING:
                    removed = {"missing_left": (left,), "missing_right": (right,)}.get(condition, ())
                    observation = inspect(index, *stats, clues, removed)
                    if i:
                        decisions.append((view, None if removed else (left, right), observation))
    report = {}
    metadata = dict(mode=mode, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                    script_sha256=agreement.discovery.digest(Path(__file__)), corpus_sha256=projection_hash,
                    panel_sha256=panel_hash, pairs=len(panel["pairs"]),
                    components=len({r[2] for r in panel["pairs"]}), embedding_revision=provenance["revision"])
    if mode == "freeze":
        wrong = [obs[2] for _, gold, obs in decisions if obs[2] is not None and obs[1] != gold]
        if not wrong:
            raise ValueError("Cannot freeze without incorrect development expansion proposals")
        report = dict(threshold=max(wrong), false_development_expansions=len(wrong))
    else:
        cutoff = lock["threshold"]
        for view in ("all", *agreement.discovery.VIEWS):
            report[view] = {}
            data = decisions if view == "all" else [entry for entry in decisions if entry[0] == view]
            for method in ("top4", "expanded", "lexical"):
                counts, timings = Counter(), []
                for _, gold, (narrow, expanded, score, base_ms, extra_ms) in data:
                    if method == "expanded":
                        answer = expanded
                    elif method == "lexical" and score is not None and score > cutoff:
                        answer = expanded
                    else:
                        answer = narrow
                    counts.update(dict(present=gold is not None, missing=gold is None,
                                       correct_present=gold is not None and answer == gold,
                                       wrong_present=gold is not None and answer not in (None, gold),
                                       wrong_missing=gold is None and answer is not None,
                                       added_correct=score is not None and answer is not None and answer == gold,
                                       added_wrong=score is not None and answer not in (None, gold)))
                    timings.append(base_ms + (extra_ms if method == "lexical" else 0.0))
                report[view][method] = dict(counts, factored_p95_ms=float(np.quantile(timings, .95)))
        metadata["threshold"] = cutoff
        metadata["frozen_sha256"] = agreement.discovery.digest(frozen)
        metadata["calibration_corpus_sha256"] = lock["corpus_sha256"]
    agreement.discovery.write_json(output, metadata | report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "test"))
    for name in ("projection", "panel_path", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--frozen", type=Path)
    run(**vars(parser.parse_args()))
