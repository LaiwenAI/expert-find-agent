"""Move scope binding and source embeddings to index time for publisher notes."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import sqrt
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit.claim_bound_attribution import judge, operating_point
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.paired_reader import load_reader


REFERENCE = "eebb08023052791243ff15fe7dea7791afd8a67998cb436d979148a1e8a27e21"
TASK = "Identify the individual research task described in this publisher contribution statement."
GRAMMAR = re.compile(r"(?:^|(?<=\. ))([^.:\n]{2,110}): ((?:[A-Z]{2,8} )*[A-Z]{2,8})\.(?= |$)")
ARMS = ("local_character", "absolute_semantic", "margin_semantic")


@dataclass(frozen=True)
class Clause:
    action: str
    owners: frozenset
    start: int
    end: int


def compile_note(text):
    """Bind each action head to the exact initials in its statement tail."""
    return tuple(Clause(m[1].strip(), frozenset(m[2].split()), m.start(), m.end()) for m in GRAMMAR.finditer(text))


def char_grams(text):
    s = " " + " ".join(re.findall(r"[a-z0-9]+", text.casefold())) + " "
    return Counter(s[i:i + n] for n in (3, 4, 5) for i in range(len(s) + 1 - n))


def gram_norm(counts):
    return sqrt(sum(n * n for n in counts.values()))


def grams_score(query, cached):
    norm = gram_norm(query)
    return np.asarray([
        sum(count * other.get(key, 0) for key, count in query.items()) / (norm * length) if norm and length else 0.
        for other, length in cached
    ])


def weakest(scores, clauses, owner, margin, start):
    owned = np.asarray([owner in clause.owners for clause in clauses], dtype=bool)
    if not owned.any():
        return -2., ()
    accepted = scores[:, owned].max(axis=1)
    rival = scores[:, ~owned].max(axis=1) if (~owned).any() else np.full(len(scores), start)
    strengths = accepted - rival if margin else accepted
    ids = np.flatnonzero(owned)[np.argmax(scores[:, owned], axis=1)]
    return float(strengths.min()), tuple(int(i) for i in ids)


def catalog(panels, encoder):
    started = perf_counter()
    records = {doi: compile_note(text) for rows, _ in panels.values() for doi, text, *_ in rows}
    heads = sorted({clause.action for clauses in records.values() for clause in clauses})
    if not heads:
        raise ValueError("No compilable author statements")
    vectors = dict(zip(heads, encoder.encode(heads), strict=True))
    grams = {action: char_grams(action) for action in heads}
    local = {doi: dict(
        clauses=clauses,
        vectors=np.stack([vectors[c.action] for c in clauses]) if clauses else np.zeros((0, len(next(iter(vectors.values()))))),
        grams=[(grams[c.action], gram_norm(grams[c.action])) for c in clauses],
    ) for doi, clauses in records.items()}
    return local, dict(ms=(perf_counter() - started) * 1000, papers=len(local), distinct_action_heads=len(heads),
                       clauses=sum(len(clauses) for clauses in records.values()),
                       embedding_bytes=sum(row["vectors"].nbytes for row in local.values()))


def inspect(rows, index, variant, encoder):
    outcomes = defaultdict(list)
    for doi, text, tasks, good, other in rows:
        entry = index[doi]
        clauses = entry["clauses"]
        requirements = [legacy.PHRASES[t][variant] for t in tasks]
        for arm in ARMS:
            started = perf_counter()
            if arm == "local_character":
                scores = np.stack([grams_score(char_grams(phrase), entry["grams"]) for phrase in requirements])
            else:
                query = encoder.encode([f"Instruct: {TASK}\nQuery: {text}" for text in requirements])
                scores = query @ entry["vectors"].T
            verdicts = [weakest(scores, clauses, owner, arm != "absolute_semantic",
                                0. if arm == "local_character" else -1.) for owner in (good, other)]
            elapsed = (perf_counter() - started) * 1000
            # Confirm the two model-scoped snippets point back to the note and the
            # requested author. Labels never enter source lookup or this check.
            for owner, (_, selected) in zip((good, other), verdicts, strict=True):
                for idx in selected:
                    clause = clauses[idx]
                    cite = text[clause.start:clause.end]
                    if not cite.startswith(clause.action + ": ") or owner not in clause.owners:
                        raise ValueError("Untraceable selected evidence")
            outcomes[arm].append((verdicts[0][0], verdicts[1][0], elapsed))
    return outcomes


def compare(calibration, evaluation, reference):
    result = {}
    for method, pairs in evaluation.items():
        cal_negatives = sorted((b for _, b, _ in calibration[method]), reverse=True)
        threshold = cal_negatives[int(.10 * len(cal_negatives))]
        result[method] = dict(
            correctly_ordered=sum(a > b for a, b, _ in pairs),
            ties=sum(a == b for a, b, _ in pairs), threshold=threshold,
            calibrated_false=sum(b > threshold for b in cal_negatives),
            held_threshold=operating_point([(a, b) for a, b, _ in pairs], threshold),
            order_delta=paired_interval([int(a > b) - int(x > y)
                                         for (a, b, _), (x, y, _) in zip(pairs, reference, strict=True)]),
            p95_ms=float(np.quantile([t for _, _, t in pairs], .95)),
        )
    return result


def run(snapshot, embedding, reranker_path, reader_path, reference, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or digest(reference) != REFERENCE):
        raise ValueError("Require a pinned earlier result and new output in Slurm")
    old = json.loads(reference.read_text())
    if digest(snapshot / "capture.json") != old["capture_sha256"]:
        raise ValueError("Changed real publisher cohort")
    data = legacy.prepare(snapshot)
    cal, test = data["calibration"][0], data["test"][0]
    if any([row[0] for row in rows] != old["panels"][name]["included_dois"]
           for name, (rows, _) in data.items()):
        raise ValueError("Publication order changed")
    metadata = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, batch_size=8)
    encoder.model.config.use_cache = False
    encoder.encode(["research action"])
    index, build = catalog(data, encoder)
    inspect(cal[:1], index, 0, encoder)

    reranker = Reranker(reranker_path)
    tokenizer, reader, reader_info = load_reader(reader_path)
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("1", "0")]
    if any(len(item) != 1 for item in ids) or ids[0] == ids[1]:
        raise ValueError("Expected two distinct single-token decisions")
    if reranker.provenance["revision"] != old["reranker_revision"] or reader_info["revision"] != old["reader_revision"]:
        raise ValueError("Changed verifier checkpoints")
    engines = (reranker, judge, reader, tokenizer, [item[0] for item in ids])
    legacy.measure(cal[:1], 0, *engines)
    base_cal, base_test = legacy.measure(cal, 0, *engines), legacy.measure(test, 1, *engines)
    base = compare(base_cal, base_test, base_test["reader_4b"])
    for name, row in base.items():
        previous = old["report"][name]
        if (row["correctly_ordered"] != previous["correct_order"] or row["threshold"] != previous["threshold"]
                or row["held_threshold"] != previous["frozen"]):
            raise ValueError(f"Unable to reproduce {name} verifier outcomes")
    methods = compare(inspect(cal, index, 0, encoder), inspect(test, index, 1, encoder), base_test["reader_4b"])
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        reference_sha256=REFERENCE, embedding_revision=metadata["revision"], compilation=build,
        calibration_papers=len(cal), test_papers=len(test), baselines=base, indexed=methods,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("snapshot", "embedding", "reranker_path", "reader_path", "reference", "output"):
        parser.add_argument(item, type=Path)
    run(**vars(parser.parse_args()))
