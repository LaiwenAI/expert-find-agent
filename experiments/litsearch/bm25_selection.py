"""Measure exact top-k selection from resident BM25 arrays under source masking."""

import argparse
import gzip
from heapq import nsmallest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


def array_topk(index, query, ids, positions, hidden, k):
    scores, active = index.document_arrays(query)
    if hidden:
        active[[positions[doc] for doc in hidden]] = False
    eligible = np.flatnonzero(active)
    if len(eligible) > k:
        ranked_values = scores[eligible]
        cutoff = np.partition(ranked_values, -k)[-k]
        eligible = eligible[ranked_values >= cutoff]
    selected = eligible[np.lexsort((ids[eligible], -scores[eligible]))[:k]]
    return ids[selected].tolist()


def heap_topk(index, query, ids, hidden, k):
    scores = index.document_scores(query)
    selected = nsmallest(k, (doc for doc in scores if ids[int(doc)] not in hidden),
                         key=lambda doc: (-scores[doc], ids[int(doc)]))
    return [ids[int(doc)] for doc in selected]


def run(projection, output):
    if output.exists():
        raise FileExistsError(output)
    with gzip.open(projection, "rb") as data:
        dataset = json.load(data)
    docs = dataset["papers"]
    ids = np.asarray([doc["corpusid"] for doc in docs], dtype=np.int64)
    doc_ids = ids.tolist()
    positions = {doc: i for i, doc in enumerate(doc_ids)}
    mapping = {}
    for i, doc in enumerate(docs):
        text = f"{doc['title'] or ''}\n{doc['abstract'] or ''}"
        while text in mapping:
            text += "\n"
        mapping[text] = i
    with TemporaryDirectory(dir=projection.parent) as temporary:
        index = build_lexical(mapping, Path(temporary))
    questions = [q for q in dataset["questions"] if q["query_set"].startswith("manual_")]
    rows = []
    for question_number, q in enumerate(questions):
        for available in (True, False):
            hidden = set() if available else set(q["corpusids"])
            for k in (6, 24):
                methods = (
                    ("heap", lambda: heap_topk(index, q["query"], doc_ids, hidden, k)),
                    ("array", lambda: array_topk(index, q["query"], ids, positions, hidden, k)),
                )
                times, answers = {}, {}
                offset = (question_number + int(available) + k // 6) % len(methods)
                for method, compute in methods[offset:] + methods[:offset]:
                    start = perf_counter()
                    answers[method] = compute()
                    times[method] = (perf_counter() - start) * 1000
                if answers["heap"] != answers["array"]:
                    raise AssertionError(f"BM25 result changed at question {question_number}, k={k}")
                if set(answers["array"]) & hidden:
                    raise AssertionError("Hidden candidate returned")
                rows.append(dict(question=question_number, available=available, k=k, ms=times))
    report = {}
    for k in (6, 24):
        subset = [row for row in rows if row["k"] == k]
        report[str(k)] = {name: list(map(float, np.quantile([row["ms"][name] for row in subset], [.5,.95])))
                          for name in ("heap", "array")}
    write_json(output, dict(projection_sha256=digest(projection), corpus=len(docs), exact_comparisons=len(rows),
                            report=report, rows=rows, script_sha256=digest(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.projection, args.output)
