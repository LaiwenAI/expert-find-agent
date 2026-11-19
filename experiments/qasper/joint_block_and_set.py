"""Separate the value of next-block routing from joint evidence selection."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.qasper import next_evidence_block as blocks
from experiments.qasper import set_repair as repair


def action(question, passages, sections, dense, vectors, stats, scored, refs):
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
    fused, indices = zip(*ordered, strict=True)
    start = perf_counter()
    candidates, features, swaps = repair.features(question, indices, np.asarray(fused), dense, sections, vectors, stats)
    selector_ms = 1000 * (perf_counter() - start)
    return dict(view=features, labels=repair.gold_sets(passages, indices, candidates, refs), one_swap=swaps,
                reachable=any(ref <= {passages[j] for j in indices} for ref in refs), selector_ms=selector_ms)


def observe(encoder, reader, qa, index):
    question = qa["question"]
    paragraphs, sections, vectors, stats = index
    start = perf_counter()
    dense = blocks.online_scores(encoder, question, vectors)
    order = np.argsort(-dense, kind="stable").tolist()[:blocks.HEAD * blocks.MAX_BLOCKS]
    context = [f"{section}\n{paragraph}" for section, paragraph in zip(sections, paragraphs, strict=True)]
    first, _ = blocks.score(reader, question, context, dense, order[:blocks.HEAD])
    prefix_ms = 1000 * (perf_counter() - start)
    candidates = [order[offset:offset + blocks.HEAD] for offset in range(blocks.HEAD, len(order), blocks.HEAD)]
    start = perf_counter()
    route_features = blocks.shortlist(question, dense, sections, vectors, stats, first, candidates)
    route_ms = 1000 * (perf_counter() - start)
    refs = blocks.base.evidence_sets(qa, paragraphs)
    actions, read_ms, reads = [], [], []
    for slot in candidates or [[]]:
        start = perf_counter()
        second, _ = blocks.score(reader, question, context, dense, slot)
        read_ms.append(1000 * (perf_counter() - start))
        reads.append(len(first) + len(second))
        actions.append(action(question, paragraphs, sections, dense, vectors, stats, first + second, refs))
    if not candidates:
        route_features = np.zeros((1, 10))
    return dict(actions=actions, route_features=route_features, prefix_ms=prefix_ms,
                route_ms=route_ms, read_ms=read_ms, reads=reads)


def labelled(rows, model=None, threshold=None):
    output = []
    for row in rows:
        values, selectors = [], []
        for item in row["actions"]:
            start = perf_counter()
            proposed = (0, 0.0) if threshold is None else repair.preference(item, model, "joint_two")
            complete, edited = repair.outcome(item, *proposed, threshold)
            selectors.append((item["selector_ms"] if threshold is not None else 0.0) + 1000 * (perf_counter() - start))
            values.append((None, item["reachable"], complete, edited))
        output.append(dict(row, features=row["route_features"], values=values, selection_ms=selectors))
    return output


def report(rows, weights, cutoff):
    summary = {}
    fixed = [row["values"][0][2] for row in rows]
    for method in blocks.ROUTES:
        decisions, times, reads = [], [], []
        for row in rows:
            start = perf_counter()
            uses_router = method != "fixed" and not (method == "learned" and cutoff is None)
            choice = blocks.pick(row["features"], weights, cutoff, method) if uses_router else 0
            route_time = (1000 * (perf_counter() - start) + row["route_ms"]) if uses_router else 0.0
            decisions.append(row["values"][choice])
            times.append(row["prefix_ms"] + row["read_ms"][choice] + row["selection_ms"][choice] + route_time)
            reads.append(row["reads"][choice])
        changes = [int(x[2]) - int(y) for x, y in zip(decisions, fixed, strict=True)]
        summary[method] = dict(
            shortlist_complete=sum(x[1] for x in decisions), output_complete=sum(x[2] for x in decisions),
            edited_sets=sum(x[3] for x in decisions), gain=changes.count(1), loss=changes.count(-1),
            paired_delta_vs_fixed=paired_interval(changes), mean_reads=float(np.mean(reads)),
            factored_p95_ms=float(np.quantile(times, .95)),
        )
    return dict(
        routes=summary,
        best_action_ceiling=sum(any(x[2] for x in row["values"]) for row in rows),
        comparable_queries=sum(len({x[2] for x in row["values"]}) > 1 for row in rows),
        shortlist_ceiling=sum(any(x[1] for x in row["values"]) for row in rows),
    )


def run(dataset, archive, embedding, reranker, route_reference, set_reference, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("A fresh output and Slurm CUDA are required")
    route_prior = json.loads(route_reference.read_text())
    set_prior = json.loads(set_reference.read_text())
    if blocks.base.digest(archive) != route_prior["archive_sha256"]:
        raise ValueError("Changed training archive")
    train_dev_test = {"train": repair.read_train(archive), "dev": blocks.base.read_data(dataset, "dev"),
                      "test": blocks.base.read_data(dataset, "test")}
    if sum(map(len, train_dev_test.values())) != len(set().union(*train_dev_test.values())):
        raise ValueError("Overlapping paper splits")
    provenance = blocks.base.verify_checkpoint(embedding)
    encoder = blocks.base.QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader = blocks.base.Reranker(reranker)
    for prior in (route_prior, set_prior):
        if (provenance["revision"] != prior["embedding_revision"]
                or reader.provenance["revision"] != prior["reranker_revision"]):
            raise ValueError("Changed model checkpoints")
    samples, statistics = {}, {}
    for name, papers in train_dev_test.items():
        panel = blocks.choose(papers, multi=True)
        panel_hash = sha256("\n".join(qa["question_id"] for _, qa in panel).encode()).hexdigest()
        if panel_hash != route_prior["inventory"][name]["panel_sha256"] or panel_hash != set_prior["panels"][name]["panel_sha256"]:
            raise ValueError("Changed paper cohort")
        indexes, offline = blocks.paper_indexes(papers, panel, encoder)
        observe(encoder, reader, panel[0][1], indexes[panel[0][0]])
        samples[name] = [observe(encoder, reader, qa, indexes[pid]) for pid, qa in panel]
        statistics[name] = dict(questions=len(panel), panel_sha256=panel_hash, offline=offline)
        del indexes
        print(name, "prepared", flush=True)

    fixed_train = [row["actions"][0] for row in samples["train"]]
    fixed_dev = [row["actions"][0] for row in samples["dev"]]
    selector, comparable = repair.fit(fixed_train, fixed_train[0]["view"].shape[1])
    selection_gate = repair.calibrate(fixed_dev, {"joint_two": selector})["joint_two"]
    if comparable != set_prior["fit_comparable_questions"][1] or selection_gate != set_prior["calibration"]["joint_two"]:
        raise ValueError("The fixed-block set selector did not reproduce its calibrated reference")
    results, fitted = {}, {}
    for mode in ("pointwise", "joint_two"):
        threshold = selection_gate["threshold"] if mode == "joint_two" else None
        trace = {name: labelled(rows, selector, threshold) for name, rows in samples.items()}
        route, comparisons = blocks.train(trace["train"])
        routing = blocks.calibration(trace["dev"], route)
        fitted[mode] = dict(comparisons=comparisons, calibration=routing, weights=route.tolist())
        results[mode] = report(trace["test"], route, routing["selected"]["cutoff"])
        expected = route_prior["test"]["fixed"]["output_complete"] if mode == "pointwise" else set_prior["results"]["joint_two"]["complete"]
        if results[mode]["routes"]["fixed"]["output_complete"] != expected:
            raise ValueError("The fixed path did not reproduce prior exact evidence counts")
    blocks.base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=blocks.base.digest(Path(__file__)),
        inputs=[blocks.base.digest(p) for p in (archive, route_reference, set_reference)],
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        cohorts=statistics, selector_gate=selection_gate, selected_routes=fitted, result=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "archive", "embedding", "reranker", "route_reference", "set_reference", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
