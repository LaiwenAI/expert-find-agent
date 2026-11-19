"""Learn which additional evidence block completes a multi-paragraph answer."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.qasper import certified_reranking as base
from experiments.qasper.evidence_gap_audit import choose, score
from experiments.qasper.evidence_shortlist import online_scores, paper_indexes, terms
from experiments.qasper.set_repair import read_train


HEAD = 8
MAX_BLOCKS = 4
THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
ROUTES = ("fixed", "lexical_gap", "learned")


def shortlist(question, dense, sections, vectors, stats, prefix_scored, blocks):
    """Describe a proposed block using only the visible prefix and index metadata."""
    counts, frequency = stats
    anchors = [i for _, i in sorted(prefix_scored, key=lambda x: (-x[0], x[1]))[:3]]
    query = set(terms(question))
    gap = query.difference(*(counts[i].keys() for i in anchors))
    weights = {word: 1.0 / (1 + frequency[word]) for word in query}
    total = max(sum(weights.values()), 1e-9)
    ref_sections = {sections[i] for i in anchors}
    rows = []
    for block in blocks:
        found = set().union(*(counts[i].keys() for i in block))
        past = vectors[block] @ vectors[anchors].T
        novelty = 1 - past.max(axis=1)
        missing = sum(weights[w] for w in sorted(gap & found)) / total
        all_hit = sum(weights[w] for w in sorted(query & found)) / total
        rows.append([
            float(np.max(dense[block])), float(np.mean(dense[block])),
            missing, all_hit,
            float(novelty.max()), float(novelty.mean()),
            sum(sections[i] not in ref_sections for i in block) / len(block),
            sum(any(abs(i - a) <= 2 and sections[i] == sections[a] for a in anchors)
                for i in block) / len(block),
            float(np.max(dense[block])) * missing,
            float(novelty.mean()) * missing,
        ])
    return np.asarray(rows, dtype=float)


def evidence_outcome(paragraphs, refs, prefix, extension):
    combined = prefix + extension
    selected = [i for _, i in sorted(combined, key=lambda x: (-x[0], x[1]))[:base.TOP]]
    read = {paragraphs[i] for _, i in combined}
    final = {paragraphs[i] for i in selected}
    return selected, any(ref <= read for ref in refs), any(ref <= final for ref in refs)


def observe(encoder, reader, qa, index):
    question = qa["question"]
    paragraphs, sections, vectors, stats = index
    begin = perf_counter()
    dense = online_scores(encoder, question, vectors)
    order = np.argsort(-dense, kind="stable").tolist()[:HEAD * MAX_BLOCKS]
    context = [f"{section}\n{paragraph}" for section, paragraph in zip(sections, paragraphs, strict=True)]
    prefix, _ = score(reader, question, context, dense, order[:HEAD])
    blocks = [order[offset:offset + HEAD] for offset in range(HEAD, len(order), HEAD)]
    feature = shortlist(question, dense, sections, vectors, stats, prefix, blocks)
    shared_ms = (perf_counter() - begin) * 1000
    refs = base.evidence_sets(qa, paragraphs)
    values = []
    calls = []
    for block in blocks:
        begin = perf_counter()
        scored, _ = score(reader, question, context, dense, block)
        elapsed = (perf_counter() - begin) * 1000
        values.append(evidence_outcome(paragraphs, refs, prefix, scored))
        calls.append(elapsed)
    if not values:
        values = [evidence_outcome(paragraphs, refs, prefix, [])]
        calls = [0.0]
        feature = np.zeros((1, 10))
    return dict(features=feature, values=values, shared_ms=shared_ms, block_ms=calls,
                checks=[len(order[:HEAD]) + len(block) for block in blocks] or [len(order)])


def train(rows):
    differences = []
    for row in rows:
        labels = np.asarray([value[2] for value in row["values"]], dtype=bool)
        if labels.any() and (~labels).any():
            x = row["features"]
            differences.extend(x[a] - x[b] for a in np.flatnonzero(labels) for b in np.flatnonzero(~labels))
    feats = np.asarray(differences)
    if len(feats) < 12:
        raise ValueError("Not enough training actions with differing outcomes")
    scale = np.maximum(np.sqrt(np.mean(feats**2, axis=0)), 1e-8)
    z = feats / scale
    return np.linalg.solve(z.T @ z + 32 * np.eye(z.shape[1]), z.sum(axis=0)) / scale, len(feats)


def pick(features, weights, threshold, method):
    if method == "fixed" or len(features) <= 1:
        return 0
    if method == "lexical_gap":
        return int(np.argmax(features[:, 2]))
    estimates = (features - features[0]) @ weights
    best = int(np.argmax(estimates))
    return best if estimates[best] > threshold else 0


def calibration(rows, weights):
    settings = []
    for cutoff in THRESHOLDS:
        chosen = [pick(row["features"], weights, cutoff, "learned") for row in rows]
        hits = sum(row["values"][i][2] for i, row in zip(chosen, rows, strict=True))
        settings.append(dict(cutoff=cutoff, complete=hits, changes=sum(i != 0 for i in chosen)))
    baseline = sum(row["values"][0][2] for row in rows)
    selected = max([dict(cutoff=None, complete=baseline, changes=0), *settings],
                   key=lambda x: (x["complete"], -x["changes"]))
    return dict(selected=selected, candidates=settings, fixed=baseline)


def assess(rows, weights, cutoff):
    result = {}
    fixed = [row["values"][0][2] for row in rows]
    for route in (*ROUTES, "forced_learned"):
        use = "learned" if route == "forced_learned" else route
        threshold = 0. if route == "forced_learned" else cutoff
        choices = [0 if threshold is None and use == "learned" else
                   pick(row["features"], weights, threshold, use) for row in rows]
        current = [row["values"][idx] for row, idx in zip(rows, choices, strict=True)]
        delta = [int(value[2]) - int(base_value) for value, base_value in zip(current, fixed, strict=True)]
        result[route] = dict(
            shortlist_complete=sum(value[1] for value in current), output_complete=sum(value[2] for value in current),
            gains=delta.count(1), losses=delta.count(-1), delta_ci=paired_interval(delta),
            changed=sum(idx != 0 for idx in choices),
            mean_reads=float(np.mean([row["checks"][idx] for row, idx in zip(rows, choices, strict=True)])),
            factored_p95_ms=float(np.quantile([row["shared_ms"] + row["block_ms"][idx]
                                              for row, idx in zip(rows, choices, strict=True)], .95)),
        )
    return result


def run(dataset, archive, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    partitions = {"train": read_train(archive), "dev": base.read_data(dataset, "dev"),
                  "test": base.read_data(dataset, "test")}
    if sum(map(len, partitions.values())) != len(set().union(*partitions.values())):
        raise ValueError("Overlapping official paper splits")
    provenance = base.verify_checkpoint(embedding)
    encoder = base.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = base.Reranker(reranker)
    inventory, cache = {}, {}
    for split, papers in partitions.items():
        panel = choose(papers, multi=True)
        indexes, offline = paper_indexes(papers, panel, encoder)
        observe(encoder, reader, panel[0][1], indexes[panel[0][0]])
        rows = [observe(encoder, reader, qa, indexes[pid]) for pid, qa in panel]
        inventory[split] = dict(
            papers=len(rows), offline=offline,
            panel_sha256=sha256("\n".join(qa["question_id"] for _, qa in panel).encode()).hexdigest(),
            observed_fixed=sum(row["values"][0][2] for row in rows),
            output_ceiling=sum(any(v[2] for v in row["values"]) for row in rows),
            shortlist_ceiling=sum(any(v[1] for v in row["values"]) for row in rows),
            comparable=sum(len({v[2] for v in row["values"]}) > 1 for row in rows),
        )
        cache[split] = rows
        print(split, inventory[split], flush=True)
        del indexes
    weights, fitted = train(cache["train"])
    settings = calibration(cache["dev"], weights)
    base.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 script_sha256=base.digest(Path(__file__)), archive_sha256=base.digest(archive),
                                 embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
                                 inventory=inventory, fitted_action_pairs=fitted, weights=weights.tolist(),
                                 calibration=settings, test=assess(cache["test"], weights, settings["selected"]["cutoff"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "archive", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
