"""Combine open source retrieval and author-scoped verification on real notes."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np
import torch

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit import compile_legacy_notes as indexed
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit import legacy_note_cascade as cascade
from experiments.credit.conjunctive_source_retrieval import sentences


COLD_RESULTS = "8f9b302970fa702cd6b1a9958e1b09ed1e3702ace013c5ada426ee407d2468ab"
TASK = "Recover the exact scientific publication described by this reported result."
FORMS = ("result", "keywords")
ARMS = ("always_4b", "gated_4b", "always_character", "gated_character")


@dataclass(frozen=True)
class Article:
    indexed_text: str
    result: str
    note: str
    authors: tuple
    memberships: dict
    entry: dict


def whitespace(text):
    return " ".join(text.split())


def publication(raw, doi):
    value = legacy.note(raw, doi)
    if value is None:
        return None
    _, text, labels, _ = value
    node = ET.fromstring(raw).find("./front/article-meta")
    title = node.find("./title-group/article-title")
    abstract = node.find("./abstract")
    if title is None or abstract is None:
        return None
    stem = whitespace(" ".join(title.itertext()))
    parts = sentences(whitespace(" ".join(abstract.itertext())))
    if len(parts) < 3 or not stem:
        return None
    cut = len(parts) - 2
    prefix = stem + "\n" + " ".join(parts[:cut])
    result = " ".join(parts[cut:])
    if result.casefold() in prefix.casefold():
        return None
    names = node.findall("./contrib-group/contrib[@contrib-type='author']/name")
    counts = Counter(
        "".join(part[0] for part in legacy.NAME_PARTS.findall(
            f"{name.findtext('given-names') or ''} {name.findtext('surname') or ''}")).upper()
        for name in names
    )
    clauses = indexed.compile_note(text)
    named = {name for row in clauses for name in row.owners if counts[name] == 1}
    if len(named) < 2:
        return None
    grams = [indexed.char_grams(row.action) for row in clauses]
    return Article(prefix, result, text, tuple(sorted(named)), labels,
                   dict(clauses=clauses, grams=[(g, indexed.gram_norm(g)) for g in grams]))


def inventory(old, current):
    notes, omissions = {}, Counter()
    previous = json.loads((old / "capture.json").read_text())
    recent = json.loads((current / "capture.json").read_text())
    catalogs = [(old / split, part["checksums"]) for split, part in previous.items()]
    catalogs.append((current, recent["checksums"]))
    for folder, checks in catalogs:
        for doi, digest in checks.items():
            value = publication(transfer.checked_xml(folder, doi, digest), doi)
            if value is None:
                omissions[folder.name] += 1
                continue
            if doi in notes:
                raise ValueError("Duplicate publication in searchable archive")
            notes[doi] = value
    return notes, dict(omissions)


class Search:
    def __init__(self, notes, encoder):
        self.notes = notes
        self.keys = tuple(sorted(notes))
        self.pos = {key: i for i, key in enumerate(self.keys)}
        self.index = [set(tokens(notes[key].indexed_text)) for key in self.keys]
        self.encoder = encoder
        self.vectors = encoder.encode([notes[key].indexed_text for key in self.keys])
        counts = Counter(term for words in self.index for term in words)
        self.idf = {word: float(np.log1p((len(notes) - n + .5) / (n + .5))) for word, n in counts.items()}

    def clue(self, doi, form):
        result = self.notes[doi].result
        if form == "result":
            return result
        words = sorted({w for w in tokens(result) if len(w) >= 5 and not w.isdecimal()},
                       key=lambda w: (w not in self.idf, -self.idf.get(w, 0.), w))
        return " ".join(islice(words, 8))

    def find(self, clue, removed):
        vector = self.encoder.encode([f"Instruct: {TASK}\nQuery: {clue}"])[0]
        dense = self.vectors @ vector
        clue_terms = set(tokens(clue))
        overlap = [sum(self.idf[w] for w in sorted(row & clue_terms)) for row in self.index]
        missing = self.pos.get(removed)
        order = [idx for idx in np.argsort(-dense, kind="stable") if idx != missing]
        scores = {idx: 1 / (60 + rank) for rank, idx in enumerate(order, 1)}
        lexical = sorted((idx for idx in order if overlap[idx] > 0), key=lambda idx: (-overlap[idx], idx))
        for rank, idx in enumerate(lexical, 1):
            scores[idx] += 1 / (60 + rank)
        best = max(scores, key=scores.get)
        return self.keys[best], float(dense[best])


def source_calibration(index, calibration):
    thresholds, report = {}, {}
    for form in FORMS:
        correct, incorrect = [], []
        for doi, *_ in calibration:
            clue = index.clue(doi, form)
            for removed in (None, doi):
                hit, confidence = index.find(clue, removed)
                (correct if hit == doi and removed is None else incorrect).append(confidence)
        if not incorrect or not correct:
            raise ValueError("Insufficient source calibration evidence")
        thresholds[form] = max(incorrect)
        report[form] = dict(exact=len(correct), wrong=len(incorrect),
                            exact_after_threshold=sum(score > thresholds[form] for score in correct))
    return thresholds, report


def verify(article, doi, tasks, use_character, engines, gate, cutoff):
    supported, calls = [], 0
    for i in range(0, len(article.authors), 2):
        authors = article.authors[i:i+2]
        row = (doi, article.note, tasks, authors[0], authors[-1])
        if use_character:
            values = cascade.score_pair(row, article.entry, 1, None, gate.features)
            verified, pending = cascade.first_stage(values, gate)
        else:
            verified, pending = np.zeros(2, dtype=bool), np.ones(2, dtype=bool)
        if pending[:len(authors)].any():
            verified |= pending & (cascade.reader_pair(row, 1, engines) > cutoff)
            calls += 1
        supported.extend(author for author, yes in zip(authors, verified[:len(authors)], strict=True) if yes)
    return supported, calls


def measure(index, rows, engines, thresholds, gate, cutoff, groups):
    results = {}
    for form in FORMS:
        for state in ("present", "withheld"):
            results[form, state] = {arm: (Counter(), [], []) for arm in ARMS}
    for i, (doi, _, tasks, *_) in enumerate(rows):
        expected = set.intersection(*(index.notes[doi].memberships[t] for t in tasks))
        if not expected or not expected <= set(index.notes[doi].authors):
            raise ValueError("Source oracle names are missing from the candidate pool")
        for form in FORMS:
            query = index.clue(doi, form)
            for state in ("present", "withheld"):
                removed = doi if state == "withheld" else None
                offset = i % len(ARMS)
                for arm in ARMS[offset:] + ARMS[:offset]:
                    start = perf_counter()
                    selected, margin = index.find(query, removed)
                    exact = selected == doi and not removed
                    if arm.startswith("gated") and margin <= thresholds[form]:
                        selected = None
                    candidates, calls = verify(index.notes[selected], selected, tasks, arm.endswith("character"),
                                               engines, gate, cutoff) if selected else ([], 0)
                    payload = json.dumps(dict(source=selected, authors=candidates), separators=(",", ":"))
                    elapsed = 1000 * (perf_counter() - start)
                    outcome = json.loads(payload)["authors"]
                    known = expected if selected == doi else set()
                    counts, times, decisions = results[form, state][arm]
                    good, errors = bool(set(outcome) & known), bool(set(outcome) - known)
                    decisions.append((good, errors))
                    counts.update(queries=1, exact_rank_before_gate=bool(exact), source_emitted=selected is not None,
                                  correct_source_emitted=selected == doi, supported_queries=good, unsupported_queries=errors,
                                  wrong_source_queries=bool(outcome and selected != doi),
                                  wrong_author_queries=bool(selected == doi and (set(outcome) - known)),
                                  supported_authors=len(set(outcome) & known), wrong_authors=len(set(outcome) - known),
                                  model_calls=calls)
                    times.append(elapsed)
    report = {}
    for key, routes in results.items():
        methods = {}
        for arm, (counts, times, decisions) in routes.items():
            effects = {}
            for reference in ("always_4b", "gated_4b"):
                base = routes[reference][2]
                effects[reference] = {}
                for i, field in enumerate(("supported", "unsupported")):
                    delta = [int(v[i]) - int(b[i]) for v, b in zip(decisions, base, strict=True)]
                    effects[reference][field] = dict(gains=delta.count(1), losses=delta.count(-1),
                                                     interval=clustered_paired_interval(delta, groups))
            methods[arm] = dict(counts, complete_p95_ms=float(np.quantile(times, .95)), paired=effects)
        report["/".join(key)] = methods
    return report


def run(old, snapshot, embedding, reader, transfer_results, frozen, output):
    if (output.exists() or indexed.digest(transfer_results) != COLD_RESULTS or indexed.digest(frozen) != transfer.REFERENCE
            or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise ValueError("Require pinned prior results and fresh output inside a Slurm CUDA job")
    prior = json.loads(transfer_results.read_text())
    freeze = json.loads(frozen.read_text())
    calibration = legacy.prepare(old)["calibration"][0]
    rows, names, _ = transfer.cohort(old, snapshot)
    groups_by_doi = dict(zip((row[0] for row in rows), transfer.clusters(names), strict=True))
    if [row[0] for row in rows] != prior["doi_inventory"]:
        raise ValueError("Changed prior held-out cohort")
    notes, omitted = inventory(old, snapshot)
    calibration, rows = [[row for row in panel if row[0] in notes and all(
        r in notes[row[0]].memberships for r in row[2]) and set.intersection(
        *(notes[row[0]].memberships[r] for r in row[2])) <= set(notes[row[0]].authors)]
        for panel in (calibration, rows)]
    if min(len(calibration), len(rows)) < 20:
        raise ValueError("Insufficient compatible source notes")
    provenance = indexed.verify_checkpoint(embedding)
    encoder = indexed.QwenTorchEmbedding(embedding, batch_size=12)
    encoder.model.config.use_cache = False
    tokenizer, model, reader_info = indexed.load_reader(reader)
    if provenance["revision"] != freeze["embedding_revision"] or reader_info["revision"] != freeze["reader_revision"]:
        raise ValueError("Changed checkpoint revisions")
    digits = [tokenizer.encode(k, add_special_tokens=False) for k in ("1", "0")]
    if any(len(item) != 1 for item in digits) or digits[0] == digits[1]:
        raise ValueError("Unexpected label tokens")
    engines = model, tokenizer, [x[0] for x in digits]
    gate = transfer.load_gate(freeze["routes"]["character"]["selected"]["gate"], cascade.Gate)
    if gate is None or gate.features != (0,):
        raise ValueError("Changed character gate")

    source = Search(notes, encoder)
    thresholds, fit = source_calibration(source, calibration)
    measure(source, calibration[:1], engines, thresholds, gate, freeze["reader_cutoff"], [0])
    groups = [groups_by_doi[row[0]] for row in rows]
    report = measure(source, rows, engines, thresholds, gate, freeze["reader_cutoff"], groups)
    indexed.write_json(output, dict(
        script_sha256=indexed.digest(Path(__file__)), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        sources=len(notes), calibration_queries=len(calibration), test_queries=len(rows), clusters=len(set(groups)),
        omitted=omitted,
        transfer_sha256=COLD_RESULTS, gate_sha256=transfer.REFERENCE, thresholds=thresholds, calibration=fit, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "snapshot", "embedding", "reader", "transfer_results", "frozen", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
