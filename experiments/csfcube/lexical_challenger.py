"""Spend one of two verification slots on a lexical challenger in judged facet pools."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.csfcube.frozen_confirmation import JudgedPoolEndpoint, load
from experiments.csfcube.grouped_calibration import BUDGETS, VIEWS, cross_validate, partition, summarize
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import paper_text, verify_checkpoint
from experiments.litsearch.learned_confirmation import observation
from experiments.litsearch.source_withholding import TASK
from experiments.retrieval.lrbench_frozen_baselines import build_lexical


SHORTLISTS = ("dense_two", "lexical_challenger", "dense_six")
AGREEMENT = "lexical_challenger_agreement"
CONTROLS = (*SHORTLISTS, AGREEMENT)


def measure(endpoint, question, view, i):
    start = perf_counter()
    pool, positions, _ = endpoint.retrieve(question, view)
    retrieval_ms = (perf_counter() - start) * 1000
    if len(pool) != 6 or (view == "withheld" and set(question["corpusids"]).intersection(pool)):
        raise ValueError("Invalid shared six-document pool")
    query = question["query"]
    begin_lexical = perf_counter()
    sparse, _ = endpoint.lexical.document_arrays(query.partition("\n")[2])
    lexical_index = int(np.argmax(sparse[positions]))
    lexical = pool[lexical_index]
    lexical_ms = (perf_counter() - begin_lexical) * 1000
    challenger = [pool[0], lexical if lexical_index else pool[1]]
    selections = dict(dense_two=pool[:2], lexical_challenger=challenger, dense_six=pool)
    rows = {}
    grades = question["grades"]
    offset = (i + int(view == "withheld")) % len(SHORTLISTS)
    for method in SHORTLISTS[offset:] + SHORTLISTS[:offset]:
        selected = selections[method]
        start = perf_counter()
        margins, truncated = endpoint.reranker.score(query, endpoint.docs, selected, task=TASK)
        if truncated or not np.isfinite(margins).all():
            raise ValueError("Invalid reranker scores")
        scored, _ = observation(selected, margins)
        elapsed = (perf_counter() - start) * 1000
        rows[method] = dict(scored=scored, grade=grades[scored[0]], rows=len(selected),
                            ms=retrieval_ms + elapsed + (lexical_ms if method == "lexical_challenger" else 0),
                            strong_exposed=any(grades[doc] >= 2 for doc in selected))
    diverse = rows["lexical_challenger"]
    agreement = diverse["scored"] if diverse["scored"][0] == lexical else (None, 0., 0.)
    rows[AGREEMENT] = dict(diverse, scored=agreement)
    diagnostics = dict(lexical_rank=lexical_index + 1, lexical_grade=grades[lexical],
                       lexical_ms=lexical_ms, changed=challenger != pool[:2], agreement=agreement[0] is not None)
    return rows, diagnostics


def report(observations, questions, diagnostics):
    folds = partition([(str(q["query_id"]), q["grades"]) for q in questions])
    groups = [q["query_id"] for q in questions]
    configs = {}
    for budget in BUDGETS:
        evaluated = {}
        for method, rows in observations.items():
            predictions, thresholds = cross_validate(rows, folds, budget)
            evaluated[method] = dict(summary=summarize(predictions, rows), thresholds=thresholds)
            if method == "dense_two":
                baseline = predictions
            paired = [int((r["present"] or 0) >= 2) - int((b["present"] or 0) >= 2)
                      for r, b in zip(predictions, baseline, strict=True)]
            evaluated[method]["paired_strong_delta_vs_dense_two"] = clustered_paired_interval(paired, groups)
        configs[str(budget)] = evaluated
    slot = {}
    for view in VIEWS:
        items = diagnostics[view]
        base = [r[view] for r in observations["dense_two"]]
        alternate = [r[view] for r in observations["lexical_challenger"]]
        slot[view] = dict(
            changed=sum(x["changed"] for x in items),
            agreement=sum(x["agreement"] for x in items),
            lexical_ranks=dict(Counter(x["lexical_rank"] for x in items)),
            lexical_ms_p95=float(np.quantile([x["lexical_ms"] for x in items], .95)),
            strong_exposure_gains=sum(a["strong_exposed"] and not b["strong_exposed"]
                                      for b, a in zip(base, alternate, strict=True)),
            strong_exposure_losses=sum(b["strong_exposed"] and not a["strong_exposed"]
                                       for b, a in zip(base, alternate, strict=True)),
            strong_before_agreement_veto=sum(a["grade"] >= 2 for a in alternate),
            strong_rejected_by_agreement_veto=sum(a["grade"] >= 2 and not x["agreement"]
                                                  for a, x in zip(alternate, items, strict=True)))
        for method, rows in observations.items():
            items = [r[view] for r in rows]
            slot[view][method] = dict(model_pairs=sum(x["rows"] for x in items),
                                      strong_exposure=sum(x["strong_exposed"] for x in items),
                                      top_one_grades=[sum(x["grade"] == g and x["scored"][0] is not None
                                                          for x in items) for g in range(4)])
    return dict(queries=len(questions), query_papers=len(set(groups)), reports=configs, observations=slot)


def run(data, reference, embedding, reranker_path, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    frozen = json.loads(reference.read_bytes())
    papers, questions, hashes = load(data)
    provenance = verify_checkpoint(embedding)
    reranker = Reranker(reranker_path)
    if hashes != frozen["source_hashes"] or provenance != frozen["embedding"] or reranker.provenance != frozen["reranker"]:
        raise ValueError("Dataset or checkpoint differs from the pinned CSFCube benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(embedding, batch_size=32)
    encoder.model.config.use_cache = False
    texts = [paper_text(p) for p in papers]
    vectors = torch.from_numpy(encoder.encode(texts)).cuda()
    with TemporaryDirectory() as directory:
        documents = {}
        for i, text in enumerate(texts):
            while text in documents:
                text += "\n"
            documents[text] = i
        lexical = build_lexical(documents, Path(directory))
        endpoint = JudgedPoolEndpoint(papers, vectors, encoder, lexical, reranker)
        if endpoint.lexical.ids != tuple(map(str, range(len(papers)))):
            raise ValueError("Lexical row order differs from the dense index")
    measure(endpoint, questions[0], "present", 0)
    rows = {method: [] for method in CONTROLS}
    diagnostics = {view: [] for view in VIEWS}
    for i, question in enumerate(questions):
        local = {}
        for view in VIEWS[i % 2:] + VIEWS[:i % 2]:
            local[view], notes = measure(endpoint, question, view, i)
            diagnostics[view].append(notes)
        for method in CONTROLS:
            rows[method].append({view: local[view][method] for view in VIEWS})
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            reference_sha256=digest(reference), code_sha256=digest(Path(__file__)),
                            embedding=provenance["revision"], reranker=reranker.provenance["revision"],
                            **report(rows, questions, diagnostics)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "reference", "embedding", "reranker_path", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
