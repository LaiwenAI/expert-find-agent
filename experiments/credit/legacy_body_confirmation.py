"""Try source-local body evidence only when the confidence gate abstains."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np
import torch

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit import legacy_open_attribution as base
from experiments.followir.core17_verifier import Reranker


REFERENCE = "24a51646181a4eae0e244a3902bfde444c532f42950a9b0a98bf90c5c996e103"
ARMS = ("dense", "lexical_body", "reader_prefix", "reader_body")
READER_TASK = ("Determine whether this text comes from the specific study described by the query. "
               "Shared topic alone is insufficient. Look for matching method, setting or reported findings.")


@dataclass(frozen=True)
class Section:
    text: str
    words: frozenset


def shingles(text):
    words = tokens(text)
    return set(zip(words, words[1:], words[2:], words[3:]))


def body_windows(xml, held):
    body = ET.fromstring(xml).find("./body")
    if body is None:
        return []
    anchors = [shingles(line) for line in base.sentences(held) if len(tokens(line)) >= 8]
    windows = []
    for p in body.findall(".//p"):
        text = base.whitespace(" ".join(p.itertext()))
        sentences = []
        for s in base.sentences(text):
            ngrams = shingles(s)
            if (len(ngrams) < 5 or len(tokens(s)) > 220
                    or any(len(ngrams & gold) / max(1, min(len(ngrams), len(gold))) >= .70 for gold in anchors)):
                continue
            sentences.append(s)
        for i in range(0, len(sentences), 3):
            chunk = " ".join(sentences[i:i+3])
            if len(tokens(chunk)) > 220:
                windows.extend(Section(s, frozenset(tokens(s))) for s in sentences[i:i+3])
            else:
                windows.append(Section(chunk, frozenset(tokens(chunk))))
    return windows


def compile_body(old, snapshot, index):
    result = {}
    prior = json.loads((old / "capture.json").read_text())
    recent = json.loads((snapshot / "capture.json").read_text())
    collections = [(old / split, part["checksums"]) for split, part in prior.items()]
    collections.append((snapshot, recent["checksums"]))
    for path, checks in collections:
        for doi, sha in checks.items():
            if doi in index:
                result[doi] = body_windows(base.transfer.checked_xml(path, doi, sha), index[doi].result)
    if result.keys() != index.keys():
        raise ValueError("Candidate bodies do not match the source index")
    return result


def select_windows(clue, body, idf):
    words = set(tokens(clue))
    weights = {word: idf.get(word, 1.) for word in words}
    total = sum(weights.values())
    ranked = []
    for i, section in enumerate(body):
        match = sum(weights[w] for w in sorted(section.words & words))
        if match and total:
            ranked.append((match / total, -i))
    ranked.sort(reverse=True)
    return (ranked[0][0] if ranked else None), [body[-i].text for _, i in ranked[:2]]


def confirm(verify, clue, documents):
    if not documents:
        return None, 0.
    started = perf_counter()
    margins, truncated = verify.score_pairs([(clue, doc) for doc in documents], task=READER_TASK)
    if truncated or not all(np.isfinite(margins)):
        raise ValueError("Unexpected input length or reader margin")
    return max(margins), 1e3 * (perf_counter() - started)


def observe(index, bodies, verify, doi, form, removed, dense_cutoff):
    clue = index.clue(doi, form)
    start = perf_counter()
    selected, dense = index.find(clue, removed)
    base_ms = 1e3 * (perf_counter() - start)
    correct = removed is None and selected == doi
    dense_ok = dense > dense_cutoff
    scores, costs = {}, {}
    if not dense_ok:
        begin = perf_counter()
        lexical, docs = select_windows(clue, bodies[selected], index.idf)
        selection = 1e3 * (perf_counter() - begin)
        scores["lexical_body"], costs["lexical_body"] = lexical, selection
        scores["reader_prefix"], costs["reader_prefix"] = confirm(verify, clue, [index.notes[selected].indexed_text])
        scores["reader_body"], model_ms = confirm(verify, clue, docs)
        costs["reader_body"] = selection + model_ms
    return dict(candidate=selected, correct=correct, dense_accept=dense_ok, base_ms=base_ms, scores=scores,
                costs=costs)


def calibration(index, bodies, verify, calibration_rows, dense_thresholds, reference):
    result = {}
    for form in base.FORMS:
        data = [observe(index, bodies, verify, doi, form, removed, dense_thresholds[form])
                for doi, *_ in calibration_rows for removed in (None, doi)]
        previous = reference["calibration"][form]
        if (sum(o["correct"] for o in data) != previous["exact"]
                or sum(o["correct"] and o["dense_accept"] for o in data) != previous["exact_after_threshold"]
                or any(o["dense_accept"] and not o["correct"] for o in data)):
            raise ValueError("Source baseline disagrees with the pinned calibration")
        thresholds = {}
        for arm in ARMS[1:]:
            negatives = [row["scores"][arm] for row in data if not row["correct"] and row["scores"].get(arm) is not None]
            if not negatives:
                raise ValueError("No incorrect rescues available for calibration")
            cutoff = max(negatives)
            thresholds[arm] = dict(threshold=cutoff, calibration_negatives=len(negatives),
                                   calibration_extra_exact=sum(row["correct"] and row["scores"].get(arm) is not None
                                                               and row["scores"][arm] > cutoff for row in data))
        result[form] = thresholds
    return result


def evaluate(index, bodies, verifier, authors, rows, groups, thresholds, prior, gate, author_cutoff):
    results = {}
    for form in base.FORMS:
        for state in ("present", "withheld"):
            outputs = {arm: [] for arm in ARMS}
            for doi, _, tasks, *_ in rows:
                obs = observe(index, bodies, verifier, doi, form, None if state == "present" else doi,
                              prior["thresholds"][form])
                viable = {"dense": obs["dense_accept"]}
                for arm in ARMS[1:]:
                    value = obs["scores"].get(arm)
                    viable[arm] = obs["dense_accept"] or (value is not None and value > thresholds[form][arm]["threshold"])
                candidates, author_ms, calls = [], 0., 0
                if any(viable.values()):
                    start = perf_counter()
                    candidates, calls = base.verify(index.notes[obs["candidate"]], obs["candidate"],
                                                   tasks, True, authors, gate, author_cutoff)
                    author_ms = 1e3 * (perf_counter() - start)
                truth = (set.intersection(*(index.notes[doi].memberships[task] for task in tasks))
                         if obs["correct"] else set())
                for arm, accept in viable.items():
                    returned = set(candidates) if accept else set()
                    outputs[arm].append(dict(
                        correct_source=accept and obs["correct"], wrong_source=accept and not obs["correct"],
                        source_selected=accept, supported=bool(truth & returned),
                        wrong=bool(returned - truth), wrong_source_recommendation=bool(returned and not obs["correct"]),
                        examined_body=arm in obs["costs"] and arm.endswith("body"),
                        author_calls=calls if accept else 0,
                        ms=obs["base_ms"] + obs["costs"].get(arm, 0.) + (author_ms if accept else 0.),
                    ))
            summary = {}
            controls = outputs["dense"]
            for arm, evaluations in outputs.items():
                tallies = {key: sum(item[key] for item in evaluations) for key in evaluations[0] if key != "ms"}
                summary[arm] = dict(tallies, factored_p95_ms=float(np.quantile([r["ms"] for r in evaluations], .95)),
                                    support_delta=clustered_paired_interval(
                                        [int(x["supported"]) - int(y["supported"])
                                         for x, y in zip(evaluations, controls, strict=True)], groups))
            old = prior["report"][f"{form}/{state}"]["gated_character"]
            if (summary["dense"]["correct_source"] != old["correct_source_emitted"]
                    or summary["dense"]["supported"] != old["supported_queries"]
                    or summary["dense"]["wrong"] != old["unsupported_queries"]):
                raise ValueError("Source or author control changed from the prior benchmark")
            results[f"{form}/{state}"] = summary
    return results


def run(old, snapshot, embedding, reranker, reader, reference, frozen, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or base.indexed.digest(reference) != REFERENCE or base.indexed.digest(frozen) != base.transfer.REFERENCE):
        raise ValueError("Require pinned inputs and fresh Slurm CUDA output")
    prior = json.loads(reference.read_text())
    freeze = json.loads(frozen.read_text())
    notes, _ = base.inventory(old, snapshot)
    cal = base.legacy.prepare(old)["calibration"][0]
    test, names, _ = base.transfer.cohort(old, snapshot)
    if len(notes) != prior["sources"] or len(cal) != prior["calibration_queries"] or len(test) != prior["test_queries"]:
        raise ValueError("Changed source, development, or test rows")
    if not all(row[0] in notes for row in (*cal, *test)):
        raise ValueError("Source missing from the fixed inventory")
    bodies = compile_body(old, snapshot, notes)
    manifest = base.indexed.verify_checkpoint(embedding)
    encoder = base.indexed.QwenTorchEmbedding(embedding, batch_size=12)
    encoder.model.config.use_cache = False
    selector = base.Search(notes, encoder)
    cross = Reranker(reranker)
    tokenizer, model, reader_info = base.indexed.load_reader(reader)
    if manifest["revision"] != freeze["embedding_revision"] or reader_info["revision"] != freeze["reader_revision"]:
        raise ValueError("Changed model revisions")
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("1", "0")]
    if any(len(token) != 1 for token in ids) or ids[0] == ids[1]:
        raise ValueError("Expected distinct single-token output scores")
    engines = model, tokenizer, [token[0] for token in ids]
    gate = base.transfer.load_gate(freeze["routes"]["character"]["selected"]["gate"], base.cascade.Gate)
    if gate is None or gate.features != (0,):
        raise ValueError("Changed character-gate format")
    cross.score_pairs([("What did the study report?", "The study reported a significant increase.")], task=READER_TASK)
    base.cascade.reader_pair(cal[0], 1, engines)
    thresholds = calibration(selector, bodies, cross, cal, prior["thresholds"], prior)
    report = evaluate(selector, bodies, cross, engines, test, base.transfer.clusters(names), thresholds,
                      prior, gate, freeze["reader_cutoff"])
    base.indexed.write_json(output, dict(
        script_sha256=base.indexed.digest(Path(__file__)), reference_sha256=REFERENCE, frozen_sha256=base.transfer.REFERENCE,
        reranker_revision=cross.provenance["revision"], job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        candidate_sources=len(notes), body_sources=sum(bool(windows) for windows in bodies.values()),
        body_windows=sum(len(windows) for windows in bodies.values()), test_queries=len(test), thresholds=thresholds,
        report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("old", "snapshot", "embedding", "reranker", "reader", "reference", "frozen", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
