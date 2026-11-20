"""Replay a frozen MuSiQue decision head on HotpotQA and QASPER."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import paired_interval
from musique_policy import DEPTHS, Index, METHODS, digest_ids, panels
from musique_search import Lexical, QwenTorchEmbedding, choices, document, key, pct, sha256_file
from musique_graph_control import diffusion_edges

HOT_SHA = "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"
QASPER_SHA = "6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5"


def read_pinned(path, expected):
    with path.open("rb") as handle:
        if sha256_file(handle) != expected:
            raise ValueError("Changed transfer dataset")
    return json.loads(path.read_bytes())


def paragraph(title, body):
    return {"title": title, "paragraph_text": body}


def hotpot(path):
    rows = read_pinned(path, HOT_SHA)
    rows.sort(key=lambda row: key("musique-transfer:hotpot:" + row["_id"]))
    samples = []
    for row in rows[:768]:
        items = {title: paragraph(title, "".join(text)) for title, text in row["context"]}
        sentence_count = {title: len(text) for title, text in row["context"]}
        if len(items) != len(row["context"]) or any(
            not 0 <= num < sentence_count[title] for title, num in row["supporting_facts"]
        ):
            raise ValueError("Invalid HotpotQA support pointers")
        gold = {document(items[title]) for title, _ in row["supporting_facts"]}
        samples.append({
            "id": row["_id"], "question": row["question"], "paragraphs": list(items.values()),
            "evidence": [gold], "group": row["type"],
        })
    return samples, samples


def qasper(path):
    papers = read_pinned(path, QASPER_SHA)
    catalog, samples = [], []
    for paper in papers.values():
        title = paper["title"]
        paragraphs = dict.fromkeys(
            text for section in paper["full_text"] for text in section["paragraphs"] if text.strip()
        )
        present = set(paragraphs)
        items = [paragraph(title, text) for text in paragraphs]
        catalog.append({"paragraphs": items})
        eligible = []
        for query in paper["qas"]:
            alternatives = []
            for annotation in query["answers"]:
                answer = annotation["answer"]
                evidence = {text for text in answer["evidence"] if "FLOAT SELECTED" not in text}
                if evidence and not answer["unanswerable"] and evidence <= present:
                    alternatives.append({(title, text) for text in evidence})
            if not alternatives or not 2 <= min(map(len, alternatives)) <= 4:
                continue
            eligible.append({
                "id": query["question_id"], "question": query["question"],
                "paper_title": title, "evidence": alternatives,
                "group": str(min(map(len, alternatives))),
            })
        if eligible:
            samples.append(min(eligible, key=lambda item: key("musique-transfer:qasper:" + item["id"])))
    return samples, catalog


def load_heads(path, checkpoint):
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads((checkpoint / "checkpoint-provenance.json").read_text())
        if str(bundle["embedding_revision"]) != metadata["revision"]:
            raise ValueError("The head requires the original embedding checkpoint")

        def make_scorer(name):
            center, sigma, w1, b1, w2, b2 = (
                bundle[f"{name}_{field}"] for field in ("center", "sigma", "w1", "b1", "w2", "b2")
            )
            def predict(features):
                hidden = ((features - center) / sigma) @ w1.T + b1
                return (np.maximum(hidden, 0) @ w2.T + b2).ravel()
            return predict

        heads = {name: make_scorer(name) for name in ("learned_static", "learned_adaptive")}
        signature = str(bundle["training_ids_sha256"])
    return heads, signature


def paper_indexes(index, titles):
    # The title is given in QASPER. A unique catalog title identifies one paper.
    ids_by_title = {title: [] for title in titles}
    for doc, title in enumerate(index.titles):
        if title in ids_by_title:
            ids_by_title[title].append(doc)
    result = {}
    for title, indices in ids_by_title.items():
        # The graph implementation has a fixed sixteen-seed prior.
        if len(indices) < 16:
            continue
        local = object.__new__(Index)
        local.docs = [index.docs[doc] for doc in indices]
        local.ids = {doc: offset for offset, doc in enumerate(local.docs)}
        local.titles = [title] * len(indices)
        local.terms = [index.terms[doc] for doc in indices]
        local.title_terms = [index.title_terms[doc] for doc in indices]
        local.links = [set() for _ in indices]
        local.edges = diffusion_edges(local.links)
        local.lexical = Lexical(local.docs)
        local.dense = index.dense[indices]
        frequencies = Counter(word for terms in local.terms for word in terms)
        local.idf = {word: float(np.log1p((len(indices) - count + 0.5) / (count + 0.5)))
                     for word, count in frequencies.items()}
        result[title] = local
    return result


def score(rows, index, encoder, models, with_title=False, by_title=None):
    counts = {name: [] for name in METHODS}
    times = {name: [] for name in METHODS}
    shared, reachable = [], []
    for ordinal, row in enumerate(rows):
        source = index if by_title is None else by_title[row["paper_title"]]
        evidence = [{source.ids[doc] for doc in refs} for refs in row["evidence"]]
        begin = perf_counter()
        search_row = row if not with_title else {"question": f"{row['question']}\nPaper: {row['paper_title']}"}
        state = source.query(search_row, encoder)
        shared.append(1000 * (perf_counter() - begin))
        pool = set(state.pool)
        reachable.append(any(items <= pool for items in evidence))
        for offset in range(len(METHODS)):
            name = METHODS[(ordinal + offset) % len(METHODS)]
            begin = perf_counter()
            if name == "learned_adaptive":
                order = state.select_incremental(models[name])
            elif name == "learned_static":
                order = state.select(models[name], adaptive=False)
            elif name == "linked":
                order = choices(name, state.ranks, source.titles, source.links, len(source.docs), max(DEPTHS))
            else:
                order = state.ranks[name][:max(DEPTHS)].tolist()
            times[name].append(1000 * (perf_counter() - begin))
            counts[name].append([
                any(items <= set(order[:depth]) for items in evidence) for depth in DEPTHS
            ])
    matrices = {name: np.asarray(items, dtype=np.int8) for name, items in counts.items()}
    groups = sorted({row["group"] for row in rows})
    summary = {
        "queries": len(rows), "docs": len(index.docs), "ids_sha256": digest_ids(rows),
        "pool_reachable": sum(reachable), "shared_ms": pct(shared),
        "group_sizes": dict(Counter(row["group"] for row in rows)), "depths": {},
        "postranking_ms": {name: pct(values) for name, values in times.items()},
    }
    for column, depth in enumerate(DEPTHS):
        adaptive = matrices["learned_adaptive"][:, column].astype(float)
        summary["depths"][str(depth)] = {
            "complete": {name: int(matrix[:, column].sum()) for name, matrix in matrices.items()},
            "complete_by_group": {
                group: {name: sum(row["group"] == group and bool(value) for row, value in zip(
                    rows, matrix[:, column], strict=True,
                )) for name, matrix in matrices.items()}
                for group in groups
            },
            "adaptive_paired": {
                name: paired_interval(adaptive - matrices[name][:, column])
                for name in ("dense", "diffusion", "learned_static")
            },
        }
    return summary


def main(args):
    models, signature = load_heads(args.heads, args.embedding)
    encoder = QwenTorchEmbedding(args.embedding, max_length=256, batch_size=64)
    result = {"heads_sha256": sha256(args.heads.read_bytes()).hexdigest(),
              "training_ids_sha256": signature}
    if args.dataset == "hotpot":
        samples, data = hotpot(args.input)
    elif args.dataset == "qasper":
        samples, data = qasper(args.input)
    else:
        _, data = panels(args.input, 768, 768)
        samples = [
            {**row, "evidence": [{document(doc) for doc in row["paragraphs"] if doc["is_supporting"]}],
             "group": row["id"].split("__", 1)[0]}
            for row in data
        ]
    index = Index(data, encoder)
    encoder.encode(["Warm up the query encoder."])
    if args.bound_qasper:
        if args.dataset != "qasper":
            raise ValueError("Passage binding requires the QASPER source")
        paper_views = paper_indexes(index, {row["paper_title"] for row in samples})
        eligible = [row for row in samples if row["paper_title"] in paper_views]
        result["excluded_short_papers"] = len(samples) - len(eligible)
        result["raw"] = score(eligible, index, encoder, models)
        result["cued"] = score(eligible, index, encoder, models, with_title=True)
        result["bound"] = score(eligible, index, encoder, models, by_title=paper_views)
    else:
        result["raw"] = score(samples, index, encoder, models)
        if args.dataset == "qasper":
            result["cued"] = score(samples, index, encoder, models, with_title=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset", choices=("hotpot", "qasper", "replay"))
    for arg in ("input", "embedding", "heads", "output"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("--bound-qasper", action="store_true")
    main(cli.parse_args())
