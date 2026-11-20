"""Train and compare static and reading-conditioned policies on disjoint MuSiQue panels."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from expert_search.statistics import paired_interval
from musique_graph_control import diffusion_edges
from musique_search import (
    ARCHIVE_SHA, DEPTHS, TASK, Lexical, QwenTorchEmbedding, Reranker, candidate_ranks, choices,
    document, joined, key, link_index, load_pairs, make_negative, pct, sha256_file, tokens,
)

SEED = 20261121
WIDTH = 32
POOL_PER_RANKER = 28
BASES = ("dense", "lexical", "hybrid", "diffusion")
METHODS = ("dense", "hybrid", "diffusion", "static_blend", "linked", "learned_static", "learned_adaptive")


def digest_ids(rows):
    return sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()


def panels(archive, train_cap, held_cap):
    with archive.open("rb") as handle:
        if sha256_file(handle) != ARCHIVE_SHA:
            raise ValueError("Changed upstream archive")
    already_used = {row["id"] for row, _, _ in load_pairs(archive)[:464]}
    with ZipFile(archive) as bundle:
        def load(split, limit):
            member = f"data/musique_ans_v1.0_{split}.jsonl"
            with bundle.open(member) as handle:
                rows = [row for row in map(json.loads, handle) if row["id"] not in already_used]
            rows.sort(key=lambda row: key(f"musique-policy-v1:{split}:{row['id']}"))
            return rows[:limit]

        train, held = load("train", train_cap), load("dev", held_cap)
    if len(train) != train_cap or len(held) != held_cap:
        raise ValueError("Inadequate independent data")
    if {r["question"] for r in train} & {r["question"] for r in held}:
        raise ValueError("Training and held-out questions overlap")
    return train, held


def positions(order):
    output = np.empty(len(order), dtype=np.int32)
    output[order] = np.arange(len(order))
    return output


def propagation(order, edges):
    row, col, mass = edges
    p = np.zeros(len(order))
    p[order[:16]] = 1 / np.arange(1, 17)
    p /= p.sum()
    value = p
    for _ in range(3):
        value = 0.4 * p + 0.6 * np.bincount(col, mass * value[row], minlength=len(order))
    return value


def lean_order(scores):
    return np.argsort(-scores, kind="stable")


class Index:
    def __init__(self, rows, encoder):
        start = perf_counter()
        self.docs = sorted({document(p) for row in rows for p in row["paragraphs"]})
        self.ids = {doc: n for n, doc in enumerate(self.docs)}
        self.titles = [doc[0] for doc in self.docs]
        self.title_terms = [set(tokens(title)) for title in self.titles]
        self.terms = [set(tokens(title) + tokens(text)) for title, text in self.docs]
        self.links = link_index(self.docs)
        self.edges = diffusion_edges(self.links)
        self.lexical = Lexical(self.docs)
        self.dense = torch.from_numpy(encoder.encode(
            [title + "\n" + text for title, text in self.docs])).cuda()
        freqs = Counter(word for terms in self.terms for word in terms)
        self.idf = {word: float(np.log1p((len(self.docs) - count + 0.5) / (count + 0.5)))
                    for word, count in freqs.items()}
        self.seconds = perf_counter() - start

    def query(self, row, encoder):
        q = row["question"]
        baseline = candidate_ranks(q, encoder, self.dense, self.lexical)
        spread = propagation(baseline["hybrid"], self.edges)
        baseline["diffusion"] = lean_order(spread)
        all_positions = {name: positions(baseline[name]) for name in BASES}
        base_mix = 1 / (40 + all_positions["diffusion"]) + 1 / (40 + all_positions["dense"])
        baseline["static_blend"] = lean_order(base_mix)
        pool = sorted({int(doc) for name in (*BASES, "static_blend") for doc in baseline[name][:POOL_PER_RANKER]})
        initial = np.asarray([
            [1 / np.log2(all_positions[name][doc] + 2) for name in BASES]
            for doc in pool
        ], dtype=np.float32)
        question = {term: self.idf.get(term, 0.0) for term in set(tokens(q))}
        total = sum(question.values()) or 1.0
        query_spans = [self.terms[doc] & question.keys() for doc in pool]
        extras = np.asarray([
            (sum(question[word] for word in self.title_terms[doc] & question.keys()) / total,
             sum(question[word] for word in matched) / total,
             min(1., np.log1p(len(self.links[doc])) / 5),
             min(1., np.log1p(len(self.terms[doc])) / 6),
             min(1., 8 * np.sqrt(spread[doc])))
            for doc, matched in zip(pool, query_spans, strict=True)
        ], dtype=np.float32)
        return State(self, baseline, pool, np.column_stack((initial, extras)), query_spans, question, total)


class State:
    def __init__(self, index, ranks, pool, base, spans, question, weight):
        self.index, self.ranks, self.pool, self.base = index, ranks, pool, base
        self.spans, self.question, self.weight = spans, question, weight

    def history(self, selected):
        if not selected:
            return np.column_stack((self.base, np.zeros((len(self.pool), 7), dtype=np.float32)))
        index = self.index
        covered = set().union(*(index.terms[doc] for doc in selected))
        seen_titles = Counter(index.titles[doc] for doc in selected)
        selected_set = set(selected)
        last = selected[-1]

        def row(doc, matched):
            neighbor_count = len(index.links[doc] & selected_set)
            seen = seen_titles[index.titles[doc]]
            return (
                bool(seen), min(seen, 4) / 4, bool(neighbor_count),
                min(neighbor_count, 4) / 4, doc in index.links[last],
                sum(self.question[word] for word in matched - covered) / self.weight,
                len(selected) / max(DEPTHS),
            )
        features = np.asarray([row(doc, matched) for doc, matched in zip(self.pool, self.spans, strict=True)],
                              dtype=np.float32)
        return np.column_stack((self.base, features))

    def select(self, learned, adaptive):
        seen = set()
        selected = []
        scores = learned(self.base) if not adaptive else None
        for _ in range(max(DEPTHS)):
            if adaptive:
                scores = learned(self.history(selected))
            winner = max((i for i, doc in enumerate(self.pool) if doc not in seen), key=lambda i: (scores[i], -i))
            doc = self.pool[winner]
            seen.add(doc)
            selected.append(doc)
        return selected


    def select_incremental(self, learned):
        # Update only the selected paragraph's contribution to the current state.
        docs = self.pool
        pool_size = len(docs)
        terms = sorted(self.question)
        weights = np.asarray([self.question[term] for term in terms])
        match = np.asarray([[term in span for term in terms] for span in self.spans])
        title = np.asarray([self.index.titles[doc] for doc in docs])
        covered = np.zeros(len(terms), dtype=bool)
        repeated = np.zeros(pool_size, dtype=np.int32)
        linked = np.zeros(pool_size, dtype=np.int32)
        last_neighbor = np.zeros(pool_size, dtype=bool)
        available = np.ones(pool_size, dtype=bool)
        features = np.column_stack((self.base, np.zeros((pool_size, 7), dtype=np.float32)))
        selected = []
        for depth in range(max(DEPTHS)):
            if depth:
                features[:, -7] = repeated > 0
                features[:, -6] = np.minimum(repeated, 4) / 4
                features[:, -5] = linked > 0
                features[:, -4] = np.minimum(linked, 4) / 4
                features[:, -3] = last_neighbor
                features[:, -2] = (match * (weights * ~covered)).sum(axis=1) / self.weight
                features[:, -1] = depth / max(DEPTHS)
            winner = int(np.argmax(np.where(available, learned(features), -np.inf)))
            selected.append(docs[winner])
            available[winner] = False
            covered |= match[winner]
            repeated += title == title[winner]
            neighbors = self.index.links[docs[winner]]
            last_neighbor = np.asarray([doc in neighbors for doc in docs])
            linked += last_neighbor
        return selected


def append_training(target, features, pool, prefix, gold):
    if not (gold - set(prefix)).intersection(pool):
        return
    ready = np.asarray([doc not in prefix for doc in pool], dtype=bool)
    labels = np.asarray([doc in gold for doc in pool], dtype=np.float32)[ready]
    if labels.min() == labels.max():
        return
    # Each state carries equal positive and negative weight.
    weights = np.where(labels, 0.5 / labels.sum(), 0.5 / (len(labels) - labels.sum()))
    target.append((features[ready], labels, weights.astype(np.float32)))


def fit(blocks, rng):
    inputs, label, weight = map(np.concatenate, zip(*blocks, strict=True))
    center, sigma = inputs.mean(axis=0), np.maximum(inputs.std(axis=0), 0.05)
    train_x = torch.from_numpy((inputs - center) / sigma)
    train_y, train_w = torch.from_numpy(label), torch.from_numpy(weight)
    layers = torch.nn.Sequential(torch.nn.Linear(inputs.shape[1], WIDTH), torch.nn.ReLU(), torch.nn.Linear(WIDTH, 1))
    optimizer = torch.optim.AdamW(layers.parameters(), lr=0.003, weight_decay=0.03)
    for _ in range(8):
        for batch in np.array_split(rng.permutation(len(inputs)), max(1, len(inputs) // 4096)):
            scores = layers(train_x[batch]).flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, train_y[batch], reduction="none")
            objective = (loss * train_w[batch]).sum() / train_w[batch].sum()
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

    w1, b1, w2, b2 = [tensor.detach().numpy().copy() for tensor in layers.parameters()]
    def predict(features):
        result = ((features - center) / sigma) @ w1.T + b1
        return (np.maximum(result, 0) @ w2.T + b2).ravel()
    return predict, len(inputs)


def training(rows, index, encoder):
    static, adaptive = [], []
    for row in rows:
        gold = {index.ids[document(doc)] for doc in row["paragraphs"] if doc["is_supporting"]}
        state = index.query(row, encoder)
        schedules = [state.ranks[arm][:max(DEPTHS)].tolist() for arm in ("static_blend", "diffusion", "dense")]
        for depth in (0, 1, 3, 5, 8, 11):
            seen = set()
            for schedule in schedules:
                prefix = tuple(schedule[:depth])
                if prefix in seen:
                    continue
                seen.add(prefix)
                append_training(static, state.base, state.pool, prefix, gold)
                append_training(adaptive, state.history(prefix), state.pool, prefix, gold)
    return static, adaptive


def audit(rows, index, encoder, models, reader=None, contrast=None):
    results, timing = {name: [] for name in METHODS}, {name: [] for name in METHODS}
    reachable = 0
    queries_ms = []
    incremental_ms = []
    incremental_difference = 0
    calls = ("diffusion", "learned_static", "learned_adaptive", "contrast")
    reader_rows = {name: [] for name in calls}
    for i, row in enumerate(rows):
        gold = {index.ids[document(doc)] for doc in row["paragraphs"] if doc["is_supporting"]}
        gold_titles = {index.titles[doc] for doc in gold}
        start = perf_counter()
        state = index.query(row, encoder)
        queries_ms.append((perf_counter() - start) * 1000)
        reachable += gold <= set(state.pool)
        if i % 2:
            start = perf_counter()
            incremental = state.select_incremental(models["learned_adaptive"])
            incremental_ms.append((perf_counter() - start) * 1000)
        reader_choices = {}
        for offset in range(len(METHODS)):
            name = METHODS[(i + offset) % len(METHODS)]
            start = perf_counter()
            if name == "linked":
                picked = choices(name, state.ranks, index.titles, index.links, len(index.docs), max(DEPTHS))
            elif name in models:
                picked = state.select(models[name], adaptive=name == "learned_adaptive")
            else:
                picked = state.ranks[name][:max(DEPTHS)].tolist()
            if name == "learned_adaptive":
                adaptive_reference = picked
            timing[name].append((perf_counter() - start) * 1000)
            if reader is not None and name in calls:
                reader_choices[name] = [index.docs[doc] for doc in picked[:4]]
            results[name].append([
                (gold.issubset(picked[:depth]), gold_titles.issubset(index.titles[doc] for doc in picked[:depth]),
                 len(gold.intersection(picked[:depth])))
                for depth in DEPTHS
            ])
        if not i % 2:
            start = perf_counter()
            incremental = state.select_incremental(models["learned_adaptive"])
            incremental_ms.append((perf_counter() - start) * 1000)
        incremental_difference += incremental != adaptive_reference
        if reader is not None:
            label_docs = {index.docs[doc] for doc in gold}
            neg = contrast[row["id"]]
            reader_choices["contrast"] = make_negative(neg, label_docs, index.lexical, index.ids, 4)
            for offset in range(len(calls)):
                name = calls[(i + offset) % len(calls)]
                context = reader_choices[name]
                start = perf_counter()
                score, clipped = reader.score_pairs([(row["question"], joined(context))], task=TASK)
                reader_rows[name].append({
                    "complete": label_docs <= set(context),
                    "accepted": score[0] > 3.375 and not clipped,
                    "clipped": bool(clipped),
                    "ms": (perf_counter() - start) * 1000,
                })
    profiles = {
        "query_ms": pct(queries_ms), "incremental_ms": pct(incremental_ms),
        "incremental_sequence_differences": incremental_difference,
    }
    return results, timing, profiles, reachable, reader_rows


def summarize(results):
    converted = {name: np.asarray(rows, dtype=np.int32) for name, rows in results.items()}
    summary = {}
    stronger = {
        "learned_adaptive": ("learned_static", "diffusion", "static_blend"),
        "learned_static": ("diffusion", "static_blend"),
    }
    for method, data in converted.items():
        split = {}
        for i, depth in enumerate(DEPTHS):
            complete, all_titles, supports = map(int, data[:, i].sum(axis=0))
            split[str(depth)] = {
                "complete": complete, "all_titles": all_titles, "support_count": supports,
                "comparisons": {
                    reference: paired_interval(data[:, i, 0] - converted[reference][:, i, 0])
                    for reference in stronger.get(method, ())
                },
            }
        summary[method] = split
    return summary


def reader_summary(outcomes):
    references = outcomes["learned_static"]
    negatives = outcomes["contrast"]
    if any(row["complete"] for row in negatives):
        raise ValueError("Lost the missing-evidence contrast")
    result = {}
    for name, rows in outcomes.items():
        accepted = [row["complete"] and row["accepted"] for row in rows]
        result[name] = {
            "complete_accepted": sum(accepted),
            "unclipped_complete": sum(row["complete"] and not row["clipped"] for row in rows),
            "unclipped_incomplete": sum(not row["complete"] and not row["clipped"] for row in rows),
            "incomplete_accepted": sum(not row["complete"] and row["accepted"] for row in rows),
            "clipped": sum(row["clipped"] for row in rows),
            "reader_ms": pct([row["ms"] for row in rows]),
        }
        if name != "contrast":
            result[name]["paired_complete_accept_and_contrast_reject"] = sum(
                passed and not neg["accepted"] and not neg["clipped"]
                for passed, neg in zip(accepted, negatives, strict=True)
            )
            result[name]["complete_accepted_diff_vs_learned_static"] = paired_interval([
                int(passed) - int(ref["complete"] and ref["accepted"])
                for passed, ref in zip(accepted, references, strict=True)
            ])
    return result


def main(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    train, held = panels(args.archive, args.train, args.held)
    encoder = QwenTorchEmbedding(args.embedding, max_length=256, batch_size=64)
    train_index = Index(train, encoder)
    train_data = training(train, train_index, encoder)
    models, counts = {}, {}
    fit_start = perf_counter()
    for name, blocks in zip(METHODS[-2:], train_data, strict=True):
        models[name], counts[name] = fit(blocks, rng)
    fit_seconds = perf_counter() - fit_start
    torch.cuda.empty_cache()
    held_index = Index(held, encoder)
    encoder.encode(["Warm up the query encoder."])
    reader = Reranker(args.reranker) if args.reranker else None
    if reader is not None:
        contrast = {pos["id"]: neg for pos, neg, _ in load_pairs(args.archive)}
        reader.score_pairs([(held[0]["question"], joined(held_index.docs[:4]))], task=TASK)
    else:
        contrast = None
    decisions, timing, shared, reachable, readout = audit(held, held_index, encoder, models, reader, contrast)
    provenance = {
        "train": {"n": len(train), "docs": len(train_index.docs), "ids_sha256": digest_ids(train)},
        "held": {"n": len(held), "docs": len(held_index.docs), "ids_sha256": digest_ids(held)},
        "catalog_overlap": len(set(train_index.docs) & set(held_index.docs)),
        "training_rows": counts,
        "index_build_seconds": {"train": train_index.seconds, "held": held_index.seconds},
        "fit_seconds": fit_seconds,
        "pool_reachable": reachable,
        "shared_query_ms": shared["query_ms"],
        "adaptive_incremental_ms": shared["incremental_ms"],
        "adaptive_incremental_sequence_differences": shared["incremental_sequence_differences"],
        "postranking_ms": {name: pct(values) for name, values in timing.items()},
        "coverage": summarize(decisions),
    }
    if reader is not None:
        provenance["first_pass_reader"] = reader_summary(readout)
    args.output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "embedding", "output"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--reranker", type=Path)
    cli.add_argument("--train", type=int, default=768)
    cli.add_argument("--held", type=int, default=768)
    main(cli.parse_args())
