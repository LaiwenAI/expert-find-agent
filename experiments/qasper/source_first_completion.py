"""Measure source selection and passage allocation on the same QASPER collections."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.adaptive_evidence import digest, evidence_sets, query_input, read_data
from experiments.qasper.completion_retrieval import dense_order
from experiments.qasper.completion_selector import FEATURE_NAMES
from experiments.qasper.evidence_gap_audit import choose
from experiments.qasper.predicted_completion import (
    DEPTH, POOL, WEIGHTS_SHA256, WINDOW, cue_title, prepare, score_next,
)


def title_tokens(text):
    return tuple(re.findall(r"[\w]+", text.casefold()))


def metadata_index(papers, encoder):
    keys = list(papers)
    titles = [title_tokens(papers[pid]["title"]) for pid in keys]
    postings = defaultdict(set)
    for pid, title in zip(keys, titles, strict=True):
        for width in (1, 2, 3):
            for offset in range(max(0, len(title) - width + 1)):
                postings[title[offset:offset + width]].add(pid)
    docs = [papers[pid]["title"] + "\n" + papers[pid]["abstract"] for pid in keys]
    return keys, encoder.encode(docs), postings


def passage_order(method, vector, meta, bank, sources, cue):
    if method == "flat":
        scores = bank @ vector
        order = np.argsort(-scores, kind="stable")[:WINDOW].tolist()
        return order, scores, len(bank)

    keys, metadata, postings = sources
    metadata_scores = metadata @ vector
    parents = sorted(range(len(keys)), key=lambda i: (-metadata_scores[i], i))
    ranked_pids = [keys[i] for i in parents if meta[keys[i]][3] > meta[keys[i]][2]]
    if method == "title_gate":
        matches = postings.get(title_tokens(cue), set())
        if len(matches) == 1:
            match = next(iter(matches))
            if match in ranked_pids:
                ranked_pids = [match]
    ranked_pids = ranked_pids[:3 if method == "balanced" else 1]
    ranked = []
    scores = np.zeros(len(bank), dtype=np.float32)
    scored = 0
    for pid in ranked_pids:
        _, _, left, right = meta[pid]
        scores[left:right] = bank[left:right] @ vector
        ranked.append(dense_order(scores, range(left, right)))
        scored += right - left
    if method == "balanced":
        order = [run[rank] for rank in range(max(map(len, ranked))) for run in ranked if rank < len(run)]
    else:
        order = ranked[0]
    return order[:WINDOW], scores, len(keys) + scored


def observe(context, encoder, weights, papers, data, sources, pid, qa, phase):
    meta, bank, _, _, _, parents = data
    paragraphs, _, left, _ = meta[pid]
    ids = {p: left + i for i, p in enumerate(paragraphs)}
    refs = [set(ids[text] for text in ref) for ref in evidence_sets(qa, paragraphs)]
    question = qa["question"]
    cue = ""
    methods = ["flat", "source_one", "balanced"]
    if context == "cued":
        cue = cue_title(papers[pid]["title"], qa["question_id"])
        question += f"\nPaper cue: {cue}"
        methods.append("title_gate")

    begin = perf_counter()
    vector = encoder.encode([query_input(question)])[0]
    encode_ms = (perf_counter() - begin) * 1000
    results = {}
    for offset in range(len(methods)):
        method = methods[(phase + offset) % len(methods)]
        begin = perf_counter()
        order, scores, comparisons = passage_order(method, vector, meta, bank, sources, cue)
        ranking_ms = (perf_counter() - begin) * 1000
        if not order:
            raise ValueError("Expected at least one ranked paragraph")
        seed = order[0]
        window = set(order)
        by_selector = {}
        for selector in ("dense", "learned_complement"):
            begin = perf_counter()
            shown = [seed]
            for _ in range(DEPTH - 1):
                pool = [i for i in order if i not in shown][:POOL]
                if pool:
                    shown.append(score_next(selector, None, weights, question, data, scores, shown, pool))
            selection_ms = (perf_counter() - begin) * 1000
            by_selector[selector] = {
                "complete": any(ref <= set(shown) for ref in refs),
                "overall_ms": encode_ms + ranking_ms + selection_ms,
            }
        results[method] = dict(
            comparisons=comparisons, seed_source=parents[seed] == pid,
            window_source=any(parents[idx] == pid for idx in order),
            seed_evidence=any(seed in ref for ref in refs),
            window_complete=any(ref <= window for ref in refs),
            attainable=any(ref <= window and len(ref | {seed}) <= DEPTH for ref in refs),
            ranking_ms=ranking_ms, selectors=by_selector,
        )
    return results


def summarize(rows):
    methods = {}
    for method in rows[0]:
        views = [row[method] for row in rows]
        flat = [row["flat"] for row in rows]
        metrics = ("seed_source", "window_source", "seed_evidence", "window_complete", "attainable")
        methods[method] = dict(
            **{name: sum(row[name] for row in views) for name in metrics},
            mean_vector_comparisons=float(np.mean([row["comparisons"] for row in views])),
            ranking_p95_ms=float(np.quantile([row["ranking_ms"] for row in views], .95)),
            paired_attainable_delta=paired_interval([
                int(a["attainable"]) - int(b["attainable"]) for a, b in zip(views, flat, strict=True)
            ]),
            selectors={},
        )
        for selector in ("dense", "learned_complement"):
            current = [row["selectors"][selector] for row in views]
            reference = [row["selectors"][selector] for row in flat]
            methods[method]["selectors"][selector] = dict(
                completed=sum(row["complete"] for row in current),
                component_p95_ms=float(np.quantile([row["overall_ms"] for row in current], .95)),
                paired_delta=paired_interval([
                    int(a["complete"]) - int(b["complete"]) for a, b in zip(current, reference, strict=True)
                ]),
            )
    return methods


def run(data, checkpoint, learned, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    if digest(learned) != WEIGHTS_SHA256:
        raise ValueError("Use the published selector artifact")
    model = json.loads(learned.read_bytes())
    if model["feature_names"] != list(FEATURE_NAMES) or verify_checkpoint(checkpoint)["revision"] != model["embedding"]:
        raise ValueError("Selector features or embedding revision differ")
    weights = {"learned_complement": np.asarray(model["weights"]["complement"]["weights"])}
    encoder = QwenTorchEmbedding(checkpoint)
    encoder.model.config.use_cache = False
    encoder.encode([query_input("Warm up")])
    results = {}
    for split in ("dev", "test"):
        papers = read_data(data, split)
        panel = choose(papers, multi=True)
        indexed = prepare(papers, encoder)
        start = perf_counter()
        sources = metadata_index(papers, encoder)
        metadata_build_seconds = perf_counter() - start
        print(f"{split}: indexed {len(papers)} papers", flush=True)
        results[split] = dict(
            questions=len(panel), index=indexed[2], metadata_build_seconds=metadata_build_seconds,
            metadata_vector_bytes=sources[1].nbytes,
            **{context: summarize([observe(context, encoder, weights, papers, indexed, sources, pid, qa, phase)
                                   for phase, (pid, qa) in enumerate(panel)])
               for context in ("raw", "cued")},
        )
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=digest(Path(__file__)), weights_sha256=digest(learned), results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "checkpoint", "learned", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
