"""Measure retrieval of reviewer-agreed PDF pages and isolate within-page reading."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.calibrated_topic_evidence import margins
from experiments.credit.crosspage_condition_search import PAGE_CHARS, TASK
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


WORDS = re.compile(r"[^\W_]+")
STOP = frozenset("and a after an by for from in of on or the to with within".split())
WINDOW_STEP = PAGE_CHARS // 2
ARMS = {
    "dense": ("dense", 1, None), "lexical": ("lexical", 1, None),
    "dense4_prefix": ("dense", 4, "prefix"), "dense4_focus": ("dense", 4, "focus"),
    "union4_prefix": ("union", 4, "prefix"), "union4_focus": ("union", 4, "focus"),
    "dense2_focus": ("dense", 2, "focus"), "union2_focus": ("union", 2, "focus"),
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def tokens(text):
    return WORDS.findall(text.casefold())


def prepare(documents, encoder):
    index = {}
    for doc, item in sorted(documents.items()):
        pages = {i: text for i, text in enumerate(item["pages"], 1) if len(text) >= 120}
        prefixes = [page[:PAGE_CHARS] for page in pages.values()]
        windows = {
            i: [(offset, Counter(tokens(text[offset:offset + PAGE_CHARS])))
                for offset in range(0, max(1, len(text) - WINDOW_STEP), WINDOW_STEP)]
            for i, text in pages.items()
        }
        frequency = Counter(token for text in pages.values() for token in set(tokens(text)))
        index[doc] = pages, prefixes, windows, frequency, encoder.encode(prefixes)
    return index


def lexical_pages(condition, pages, windows, frequency):
    words = tokens(condition)
    key = " ".join(words)
    weights = {token: math.log1p((len(pages) - frequency[token] + .5) / (frequency[token] + .5))
               for token in dict.fromkeys(words) if token not in STOP}
    focus, relevance = {}, {}
    for page, options in windows.items():
        def score(window):
            offset, counts = window
            coverage = sum(weight for token, weight in weights.items() if counts[token])
            frequency_score = sum(weight * counts[token] / (counts[token] + 1)
                                  for token, weight in weights.items())
            return coverage, frequency_score

        offset, counts = max(options, key=score)
        focus[page] = pages[page][offset:offset + PAGE_CHARS]
        coverage, frequency_score = score((offset, counts))
        phrase = key and f" {key} " in f" {' '.join(tokens(focus[page]))} "
        relevance[page] = coverage, bool(phrase), frequency_score
    return sorted(pages, key=relevance.get, reverse=True), focus


def search(row, index, encoder, reader, arm):
    route, limit, reader_style = ARMS[arm]
    pages, prefixes, windows, frequency, embeddings = index[row["doc"]]
    condition = row["condition"]
    prompt = f"The report substantively covers {condition}."
    dense, lexical, focus = [], [], {}
    if route in {"dense", "union"}:
        scores = encoder.encode([prompt], query=True)[0] @ embeddings.T
        ids = list(pages)
        dense = [ids[i] for i in np.argsort(-scores, kind="stable")]
    if route in {"lexical", "union"} or reader_style == "focus":
        lexical, focus = lexical_pages(condition, pages, windows, frequency)
    if route == "union":
        ranked = list(dict.fromkeys(page for pair in zip(dense, lexical, strict=True) for page in pair))
    else:
        ranked = dense if route == "dense" else lexical
    slate = ranked[:limit]
    if reader_style is None:
        return slate, slate[0], None, 0, 0
    text = dict(zip(pages, prefixes, strict=True)) if reader_style == "prefix" else focus
    pairs = [(prompt, text[page]) for page in slate]
    values, clipped = margins(reader, pairs, TASK)
    best = int(np.argmax(values))
    return slate, slate[best], float(values[best]), len(pairs), clipped


def evaluate(rows, index, encoder, reader, offset):
    records = defaultdict(list)
    names = tuple(ARMS)
    for i, row in enumerate([rows[0], *rows]):
        shift = (i + offset) % len(names)
        for arm in names[shift:] + names[:shift]:
            begin = perf_counter()
            slate, page, margin, reads, clipped = search(row, index, encoder, reader, arm)
            elapsed = (perf_counter() - begin) * 1000
            if i:
                gold = set(row["accepted_pages"])
                records[arm].append(dict(
                    case=row["case"], criterion=row["condition_index"], page=page, slate=slate,
                    hit=page in gold, reachable=bool(gold.intersection(slate)),
                    issued=None if margin is None else margin > 0, reads=reads, clipped=clipped,
                    elapsed=elapsed,
                ))
    result = {}
    for arm, outcomes in records.items():
        counts = Counter(hit=sum(row["hit"] for row in outcomes),
                         reachable=sum(row["reachable"] for row in outcomes),
                         reads=sum(row["reads"] for row in outcomes),
                         clipped=sum(row["clipped"] for row in outcomes))
        if ARMS[arm][2] is not None:
            counts["agreed_issued"] = sum(row["hit"] and row["issued"] for row in outcomes)
            counts["other_issued"] = sum(not row["hit"] and row["issued"] for row in outcomes)
        timing = np.quantile([row["elapsed"] for row in outcomes], [.5, .95]).tolist()
        evidence = [{key: value for key, value in row.items() if key != "elapsed"} for row in outcomes]
        result[arm] = dict(counts, p50_p95_ms=timing, evidence=evidence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("panel", "manifest", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    panel_hash = digest(args.panel)
    if json.loads(args.manifest.read_bytes())["panel_sha256"] != panel_hash:
        raise ValueError("Panel differs from manifest")
    panel = json.loads(args.panel.read_bytes())
    rows = panel["rows"]
    revision = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    index = prepare(panel["documents"], encoder)
    result = dict(
        conditions=len(rows), cases=len({r["case"] for r in rows}), families=len({r["family"] for r in rows}),
        documents=len(index), indexed_pages=sum(len(data[0]) for data in index.values()),
        reachable_at_any_depth=sum(bool(set(row["accepted_pages"]) & index[row["doc"]][0].keys()) for row in rows),
        gpu=torch.cuda.get_device_name(), embedding_revision=revision["revision"],
        reranker_revision=reader.provenance["revision"], panel_sha256=panel_hash,
        script_sha256=digest(Path(__file__)), offset=args.offset,
        results=evaluate(rows, index, encoder, reader, args.offset),
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
