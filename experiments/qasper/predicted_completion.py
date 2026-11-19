"""Evaluate a frozen complementary selector with predicted evidence and source distractors."""

import argparse
from hashlib import sha256
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
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.adaptive_evidence import digest, evidence_sets, offline, query_input, read_data
from experiments.qasper.completion_retrieval import dense_order
from experiments.qasper.completion_selector import FEATURE_NAMES, features, select
from experiments.qasper.evidence_gap_audit import choose
from experiments.qasper.remaining_evidence import TASK, excerpt


WEIGHTS_SHA256 = "74058d91b9035f72dde12796d9e3fb7e535fe3216cab45ecfce2213fcbac2a2a"
ROUTES = ("dense", "frozen", "learned_question", "learned_complement")
SCOPES = ("known", "open_raw", "open_cued")
DEPTH = 3
POOL = 4
WINDOW = DEPTH - 1 + POOL


def cue_title(title, identity):
    words = re.findall(r"\S+", title)
    offset = int.from_bytes(sha256(identity.encode()).digest()[:2], "big") % max(1, len(words) - 2)
    return " ".join(words[offset:offset + 3])


def prepare(papers, encoder):
    meta, _, bank, index = offline(papers, encoder)
    paragraphs, sections, parents = [], [], []
    for pid, (chunks, labels, start, stop) in meta.items():
        assert start == len(paragraphs) and stop == start + len(chunks)
        paragraphs.extend(chunks)
        sections.extend(f"{pid}:{section}" for section in labels)
        parents.extend([pid] * len(chunks))
    return meta, bank, index, paragraphs, sections, parents


def score_next(route, reader, weights, question, data, scores, shown, pool):
    meta, bank, _, paragraphs, sections, parents = data
    if route == "dense":
        return pool[0]
    if route == "frozen":
        anchor = "\n".join(excerpt(reader.tokenizer, paragraphs[i], 128) for i in shown)
        prompt = question + "\nEvidence already available:\n" + anchor
        docs = [(prompt, f"{sections[p].partition(':')[2]}\n{paragraphs[p]}") for p in pool]
        margins, _ = reader.score_pairs(docs, task=TASK)
        return pool[select(margins, pool)]
    local = []
    for pid in dict.fromkeys(parents[i] for i in [*shown, *pool]):
        _, _, start, stop = meta[pid]
        local.extend(range(start, stop))
    mapping = {idx: offset for offset, idx in enumerate(local)}
    x = features(question, [paragraphs[i] for i in local], [sections[i] for i in local],
                 bank[local], scores[local], [mapping[i] for i in shown], [mapping[i] for i in pool])
    model = weights[route]
    return pool[select(x[:, :len(model)] @ model, pool)]


def observe(scope, encoder, reader, weights, papers, data, pid, qa, phase):
    meta, bank, _, _, _, parents = data
    paragraphs, _, left, right = meta[pid]
    ids = {p: left + i for i, p in enumerate(paragraphs)}
    refs = [set(ids[text] for text in ref) for ref in evidence_sets(qa, paragraphs)]
    question = qa["question"]
    if scope == "open_cued":
        question = f'{question}\nPaper cue: {cue_title(papers[pid]["title"], qa["question_id"])}'

    begin = perf_counter()
    vector = encoder.encode([query_input(question)])[0]
    if scope == "known":
        scores = np.zeros(len(bank), dtype=np.float32)
        scores[left:right] = bank[left:right] @ vector
        ranked = dense_order(scores, range(left, right))
    else:
        scores = bank @ vector
        ranked = np.argsort(-scores, kind="stable").tolist()[:WINDOW]
    shared_ms = (perf_counter() - begin) * 1000
    seed = ranked[0]
    result = {}
    for offset in range(len(ROUTES)):
        route = ROUTES[(offset + phase) % len(ROUTES)]
        begin = perf_counter()
        selected = [seed]
        for _ in range(DEPTH - 1):
            pool = [i for i in ranked if i not in selected][:POOL]
            if pool:
                selected.append(score_next(route, reader, weights, question, data, scores, selected, pool))
        elapsed = (perf_counter() - begin) * 1000
        result[route] = dict(complete=any(ref <= set(selected) for ref in refs),
                             overlap=max(len(ref.intersection(selected)) for ref in refs),
                             all_target=all(parents[i] == pid for i in selected),
                             ms=shared_ms + elapsed)
    first_window = set(ranked[:WINDOW])
    return dict(
        seed_source=parents[seed] == pid, seed_witness=any(seed in ref for ref in refs),
        seed_completion_possible=any(len(ref | {seed}) <= DEPTH for ref in refs),
        window_completion_possible=any(ref <= first_window and len(ref | {seed}) <= DEPTH for ref in refs),
        arms=result,
    )


def summarize(rows):
    counts = {name: sum(row[name] for row in rows)
              for name in ("seed_source", "seed_witness", "seed_completion_possible", "window_completion_possible")}
    arms = {}
    for name in ROUTES:
        values = [row["arms"][name] for row in rows]
        comparisons = {}
        for baseline in ("dense", "frozen", "learned_question"):
            paired = [int(value["complete"]) - int(row["arms"][baseline]["complete"])
                      for value, row in zip(values, rows, strict=True)]
            comparisons[baseline] = paired_interval(paired)
        arms[name] = dict(
            complete=sum(value["complete"] for value in values),
            overlap=sum(value["overlap"] for value in values),
            all_target=sum(value["all_target"] for value in values),
            component_p95_ms=float(np.quantile([value["ms"] for value in values], .95)),
            comparisons=comparisons,
        )
    return dict(questions=len(rows), **counts, arms=arms)


def run(data, checkpoint, reranker, learned, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    if digest(learned) != WEIGHTS_SHA256:
        raise ValueError("Use the published selector artifact")
    model = json.loads(learned.read_bytes())
    if model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Selector features differ")
    embed = verify_checkpoint(checkpoint)
    reader = Reranker(reranker)
    if (embed["revision"] != model["embedding"] or reader.provenance["revision"] != model["reranker"]):
        raise ValueError("Selector checkpoint revisions differ")
    weights = {f"learned_{name}": np.asarray(item["weights"]) for name, item in model["weights"].items()}
    encoder = QwenTorchEmbedding(checkpoint)
    encoder.model.config.use_cache = False
    reader.score_pairs([("Warm up", "A passage")] * 4, task=TASK)
    encoder.encode([query_input("Warm up")])
    results = {}
    for split in ("dev", "test"):
        papers = read_data(data, split)
        panel = choose(papers, multi=True)
        indexed = prepare(papers, encoder)
        outcomes = {scope: [] for scope in SCOPES}
        print(f"{split}: indexed {len(papers)} papers", flush=True)
        for phase, (pid, qa) in enumerate(panel):
            for offset in range(len(SCOPES)):
                scope = SCOPES[(offset + phase) % len(SCOPES)]
                outcomes[scope].append(observe(scope, encoder, reader, weights, papers, indexed, pid, qa, phase))
            if (phase + 1) % 25 == 0:
                print(f"{split}: {phase + 1}/{len(panel)}", flush=True)
        results[split] = dict(papers=len(papers), index=indexed[2],
                              **{scope: summarize(rows) for scope, rows in outcomes.items()})
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=digest(Path(__file__)), weights_sha256=digest(learned), results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "checkpoint", "reranker", "learned", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
