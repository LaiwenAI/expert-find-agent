"""Cross source confirmation with role agreement on the same expert proposals."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from expert_search.qwen_embedding import TASK as EXPERT_TASK
from expert_search.retrieval import tokens
from experiments.credit import joint_source_selection as joint
from experiments.credit import role_prototypes as role
from experiments.credit import source_clue_agreement as clue
from experiments.credit.scoped_credit_search import DENSE_TASK, PROJECTION, VIEWS
from experiments.followir.core17_verifier import Reranker


COMPARISON = "5ed20882f101bb84228f1db55df8ebb1b0a67dc1da01b918bf292fb4ed06de2b"
SOURCE_GATES = ("none", "dense", "lexical", "cross_encoder")
ROLE_GATES = ("shared", "agreement")


def removed_sources(pair, condition):
    return {"missing_left": pair[:1], "missing_right": pair[1:]}.get(condition, ())


def source_scores(index, vectors, texts, pair, corpus_terms, reranker):
    start = perf_counter()
    dense = min(float(index.vectors[index.ids[doi]] @ vector) for doi, vector in zip(pair, vectors, strict=True))
    dense_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    query_terms = [set(tokens(text)) for text in texts]
    lexical = min(len(terms & corpus_terms[doi]) / max(1, len(terms))
                  for doi, terms in zip(pair, query_terms, strict=True))
    lexical_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    values, truncations = reranker.score_pairs([(text, index.texts[doi])
                                               for text, doi in zip(texts, pair, strict=True)], task=DENSE_TASK)
    if truncations or not all(np.isfinite(values)):
        raise ValueError("Unexpected source verifier input or output")
    return dict(dense=dense, lexical=lexical, cross_encoder=min(values)), dict(
        none=0., dense=dense_ms, lexical=lexical_ms, cross_encoder=(perf_counter() - start) * 1000)


def build_index(payload, encoder, directory):
    index = joint.SourceIndex(payload, encoder, directory)
    index.texts = {doi: row["index"] for doi, row in payload["documents"].items()}
    terms = {doi: set(tokens(text)) for doi, text in index.texts.items()}
    return index, terms


def freeze(payload, panel, encoder, reranker):
    records = {}
    with TemporaryDirectory() as directory:
        index, terms = build_index(payload, encoder, Path(directory))
        for left, right, _ in panel["pairs"]:
            intended = left, right
            for view in VIEWS:
                texts = [panel["clues"][doi][view] for doi in intended]
                vectors = encoder.encode(clue.prompts(texts))
                for condition in joint.CONDITIONS:
                    removed = removed_sources(intended, condition)
                    options = joint.pairs(index, index.rank_sources(texts, vectors, removed)[1])[:2]
                    for _, a, b in options:
                        pair = a, b
                        key = intended, view, pair
                        if key not in records:
                            records[key] = source_scores(index, vectors, texts, pair, terms, reranker)[0]
    negative = [row for (gold, _, got), row in records.items() if gold != got]
    positive = [row for (gold, _, got), row in records.items() if gold == got]
    if not positive or len(negative) < 8:
        raise ValueError("Insufficient independent calibration proposals")
    thresholds = {name: max(row[name] for row in negative) for name in SOURCE_GATES[1:]}
    return thresholds, dict(correct=len(positive), wrong=len(negative),
                            correct_retained={name: sum(row[name] > threshold for row in positive)
                                              for name, threshold in thresholds.items()})


def evaluate(index, terms, evidence, tasks, texts, words, removed, encoder,
             banks, role_threshold, source_thresholds, reranker):
    started = perf_counter()
    prompts = clue.prompts(texts) + [f"Instruct: {EXPERT_TASK}\nQuery:{word}" for word in words]
    vectors = encoder.encode(prompts)
    options = joint.pairs(index, index.rank_sources(texts, vectors[:2], removed)[1])
    pair, candidates, _, _ = joint.select(options, evidence, tasks, vectors[2:] @ banks["shared"].T,
                                           role_threshold, "source_backfill")
    baseline_ms = (perf_counter() - started) * 1000
    if pair is None:
        return {f"{sg}/{rg}": (pair, candidates, baseline_ms) for sg in SOURCE_GATES for rg in ROLE_GATES}
    scores, overhead = source_scores(index, vectors[:2], texts, pair, terms, reranker)
    started = perf_counter()
    second = role.encode_tasks(encoder, words) @ banks["dedicated"].T
    agreed = candidates & joint.fitting_owners(evidence, pair, tasks, second, 0.)[0]
    role_ms = (perf_counter() - started) * 1000
    evaluations = {}
    for source_gate in SOURCE_GATES:
        accept = source_gate == "none" or scores[source_gate] > source_thresholds[source_gate]
        evaluations[f"{source_gate}/shared"] = (
            pair, candidates if accept else set(), baseline_ms + overhead[source_gate]
        )
        evaluations[f"{source_gate}/agreement"] = (
            pair, agreed if accept else set(), baseline_ms + overhead[source_gate] + role_ms * accept
        )
    return evaluations


def run(development, calibration_panel, projection, panel, glossary, boundary_words,
        training, prior, comparison, embedding, reranker_path, output):
    files = dict(projection=projection, panel=panel, glossary=glossary, boundary_words=boundary_words, prior=prior)
    expected = {**joint.FROZEN, "development": PROJECTION, "calibration_panel": clue.PANEL, "comparison": COMPARISON}
    inputs = {name: joint.gate.replay.digest(path) for name, path in
              dict(files, development=development, calibration_panel=calibration_panel, comparison=comparison).items()}
    if inputs != expected or output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require pinned inputs and fresh results in a Slurm GPU allocation")
    corpus, held, inventory, boundary, lock = [json.loads(path.read_text()) for path in files.values()]
    dev, cal = [json.loads(path.read_text()) for path in (development, calibration_panel)]
    if dev["documents"].keys() & corpus["documents"].keys():
        raise ValueError("Calibration source overlap")
    revision = joint.verify_checkpoint(embedding)["revision"]
    if revision != lock["embedding_revision"] or joint.gate.replay.digest(training) != lock["training_sha256"]:
        raise ValueError("Changed role mapping provenance")
    encoder = joint.gate.replay.QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    reranker = Reranker(reranker_path)
    thresholds, calibration = freeze(dev, cal, encoder, reranker)
    shared = joint.gate.build_banks(encoder, inventory, training)[0]
    dedicated, _ = role.train_banks(training, SimpleNamespace(embedder=encoder, vectors=shared["definition"]),
                                    inventory["examples"])
    banks = dict(shared=shared["ridge"], dedicated=dedicated["ridge"])
    role_threshold = lock["thresholds"]["ridge"]["caps"]["10"]["threshold"]
    evidence = joint.EvidenceIndex(corpus["documents"])
    examples = defaultdict(set)
    for row in inventory["examples"]:
        examples[row["role"]].add(row["text"])
    counts, timing = defaultdict(Counter), defaultdict(list)
    cases, _ = joint.gate.replay.panel(corpus["documents"])
    with TemporaryDirectory() as directory:
        index, terms = build_index(corpus, encoder, Path(directory))
        for intended, family, _, tasks, gold in cases:
            for wording, bank in (("niso_examples", examples), ("external_wording", boundary)):
                words = [min(bank[label], key=lambda p: joint.gate.replay.stable(family, label, p)) for _, label in tasks]
                for view in VIEWS:
                    texts = [held["clues"][doi][view] for doi in intended]
                    for condition in joint.CONDITIONS:
                        removed = removed_sources(intended, condition)
                        key = "/".join((wording, view, condition))
                        results = evaluate(index, terms, evidence, tasks, texts, words, removed, encoder,
                                           banks, role_threshold, thresholds, reranker)
                        for method, (pair, returned, ms) in results.items():
                            correct_pair = pair == intended and not removed
                            supported = returned & gold if correct_pair else set()
                            errors = returned - supported
                            counts[key, method].update(queries=1, supported_queries=bool(supported),
                                                       unsupported_queries=bool(errors),
                                                       wrong_source_queries=bool(returned and not correct_pair),
                                                       wrong_author_queries=bool(errors and correct_pair))
                            timing[key, method].append(ms)

    prior_report = json.loads(comparison.read_text())["report"]
    report = defaultdict(dict)
    for (setting, method), values in counts.items():
        if method == "none/shared":
            previous = prior_report[setting]["arms"]["source_backfill"]
            if any(previous[key] != value for key, value in values.items()):
                raise ValueError(f"Changed two-pair backfill outcomes in {setting}")
        report[setting][method] = dict(values, p95_ms=float(np.quantile(timing[setting, method], .95)))
    joint.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                  script_sha256=joint.gate.replay.digest(Path(__file__)), inputs=inputs,
                                  embedding_revision=revision, reranker=reranker.provenance,
                                  thresholds=thresholds, calibration=calibration, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "calibration_panel", "projection", "panel", "glossary", "boundary_words",
                 "training", "prior", "comparison", "embedding", "reranker_path", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
