"""Compare joint relevance and per-condition evidence on reconstructed PDFs."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.calibrated_topic_evidence import margins
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("whole_dense", "independent_dense", "distinct_dense",
        "whole_reader", "independent_reader", "distinct_reader",
        "distinct_body_dense", "distinct_body_reader")
PAGE_CHARS = 1800
CONTENTS = re.compile(r"^(?:.{0,100}?\btable of contents\b|contents\b)", re.IGNORECASE)
TASK = (
    "Decide whether this page provides substantive evidence for the requested condition. "
    "A title, heading, or table-of-contents mention alone is insufficient."
)


def assignment(scores, distinct):
    arity = len(scores)
    if not distinct:
        return tuple(map(int, scores.argmax(axis=1)))

    def quality(pages):
        values = [scores[i, page] for i, page in enumerate(pages)]
        return min(values), sum(values), tuple(-page for page in pages)

    # A better unused page must exist among the first arity pages of a
    # facet whenever an optimal assignment would use a lower-ranked one.
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :arity].tolist()
    return max((pages for pages in product(*ranked) if len(set(pages)) == arity), key=quality)


def is_contents(page, text):
    return page <= 12 and (bool(CONTENTS.search(text[:180])) or len(re.findall(r"\.{4,}\s*\d", text)) >= 4)


def page_index(documents, encoder):
    texts = {}
    physical_pages = {}
    for doc, row in sorted(documents.items()):
        pages = [(i + 1, page[:PAGE_CHARS]) for i, page in enumerate(row["pages"]) if len(page) >= 120]
        physical_pages[doc] = [i for i, _ in pages]
        texts[doc] = [content for _, content in pages]
    unique = {text: i for i, text in enumerate(dict.fromkeys(page for pages in texts.values() for page in pages))}
    embeddings = encoder.encode(list(unique))
    masks = {
        doc: [is_contents(i, full) for i, full in enumerate(documents[doc]["pages"], 1) if len(full) >= 120]
        for doc in texts
    }
    return (texts, {doc: embeddings[[unique[text] for text in pages]] for doc, pages in texts.items()},
            physical_pages, masks)


def score_held(query, texts, embeddings, pages, contents, encoder, reader, arm):
    whole = arm.startswith("whole")
    use_reader = arm.endswith("reader")
    condition_text = [f"The report substantively covers {condition}." for condition in query["conditions"]]
    search_text = [query["query"]] if whole else condition_text
    vectors = encoder.encode(search_text, query=True)
    selected = {}
    scores = {}
    for doc in sorted(query["grades"]):
        similarity = vectors @ embeddings[doc].T
        if whole:
            order = np.argsort(-similarity[0], kind="stable")
            chosen = tuple(order[:query["arity"]] if use_reader else order[:1])
        else:
            if "_body_" in arm and len(contents[doc]) - sum(contents[doc]) >= query["arity"]:
                similarity[:, contents[doc]] = -np.inf
            chosen = assignment(similarity, arm.startswith("distinct"))
        selected[doc] = chosen
        values = (similarity[0, list(chosen)] if whole
                  else [similarity[k, page] for k, page in enumerate(chosen)])
        scores[doc] = ((max(values), sum(values)) if whole else (min(values), sum(values)))
    clipped = checked = 0
    if use_reader:
        pairs, slots = [], []
        for doc, chosen in selected.items():
            for i, page in enumerate(chosen):
                pairs.append((search_text[0 if whole else i], texts[doc][page]))
                slots.append(doc)
        readings, clipped = margins(reader, pairs, TASK)
        by_doc = defaultdict(list)
        for doc, value in zip(slots, readings, strict=True):
            by_doc[doc].append(value)
        scores = {doc: ((max(values), sum(values)) if whole else (min(values), sum(values)))
                  for doc, values in by_doc.items()}
        checked = len(pairs)
    ordered = sorted(scores, key=lambda doc: (-scores[doc][0], -scores[doc][1], doc))
    top = ordered[0]
    json.dumps({"document": top, "pages": [pages[top][i] for i in selected[top]]})
    return ordered, scores[top][0], selected, clipped, checked


def evaluate(queries, texts, embeddings, pages, contents, encoder, reader, offset):
    records = defaultdict(list)
    for row_number, query in enumerate([queries[0], *queries]):
        shift = (row_number + offset) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            begin = perf_counter()
            ranked, score, selections, clipped, checked = score_held(query, texts, embeddings, pages, contents, encoder, reader, arm)
            elapsed = (perf_counter() - begin) * 1000
            if row_number == 0:
                continue
            grade = query["grades"]
            full = query["arity"]
            rank = next(i for i, doc in enumerate(ranked, 1) if grade[doc] == full)
            chosen = ranked[0]
            row = dict(query=query["query_id"], arity=full, win=rank == 1, rank=rank,
                       document=chosen, pages=[pages[chosen][i] for i in selections[chosen]],
                       checked=checked, clipped=clipped, ms=elapsed,
                       reuse_selected=sum(len(set(sel)) < len(sel) for sel in selections.values()),
                       contents_selected=sum(contents[doc][page] for doc, picked in selections.items() for page in picked))
            if arm.endswith("reader"):
                for threshold in (0., -4.):
                    row[f"full_{threshold}"] = rank == 1 and score > threshold
                    row[f"partial_{threshold}"] = rank != 1 and score > threshold
            records[arm].append(row)
    summary = {}
    baseline = records["whole_dense"]
    for arm, rows in records.items():
        counts = Counter()
        for row in rows:
            counts.update({key: val for key, val in row.items() if key in ("win", "checked", "clipped", "reuse_selected", "contents_selected")
                           or key.startswith(("full_", "partial_"))})
        summary[arm] = dict(
            **{key: int(value) for key, value in counts.items()}, mrr=float(np.mean([1 / row["rank"] for row in rows])),
            by_arity={str(n): {"count": sum(r["arity"] == n for r in rows),
                              "wins": sum(r["win"] and r["arity"] == n for r in rows)}
                      for n in sorted({q["arity"] for q in queries})},
            p50_p95_ms=np.quantile([row["ms"] for row in rows], [.5, .95]).tolist(),
            changes_vs_whole=dict(
                gains=sum(a["win"] and not b["win"] for a, b in zip(rows, baseline, strict=True)),
                losses=sum(b["win"] and not a["win"] for a, b in zip(rows, baseline, strict=True)),
            ),
            evidence=[{key: row[key] for key in ("query", "document", "pages", "rank")} for row in rows],
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "manifest", "embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["panel_sha256"] != digest(args.panel):
        raise ValueError("Changed PDF panel")
    panel = json.loads(args.panel.read_bytes())
    if any(len(row["conditions"]) != row["arity"] or len(row["grades"]) != 4 for row in panel["queries"]):
        raise ValueError("Unexpected arity or candidate pool")
    revision = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    texts, embeddings, pages, contents = page_index(panel["documents"], encoder)
    results = evaluate(panel["queries"], texts, embeddings, pages, contents, encoder, reader, args.order_offset)
    write_json(args.output, dict(
        queries=len(panel["queries"]), documents=len(pages), indexed_pages=sum(map(len,pages.values())), contents_pages=sum(map(sum, contents.values())),
        gpu=torch.cuda.get_device_name(), embedding_revision=revision["revision"],
        reranker_revision=reader.provenance["revision"], order_offset=args.order_offset,
        panel_sha256=digest(args.panel), script_sha256=digest(Path(__file__)), results=results,
    ))


if __name__ == "__main__":
    main()
