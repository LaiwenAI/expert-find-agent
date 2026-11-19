"""Compare equal-read contribution search over complete publisher articles."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import log1p
import os
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

import numpy as np

from expert_search.retrieval import tokens
from expert_search.statistics import clustered_paired_interval
from experiments.credit import anchored_clause_rescue as anchored
from experiments.credit import legacy_cascade_transfer as transfer
from experiments.credit import legacy_contribution_text as legacy
from experiments.credit.compile_legacy_notes import compile_note
from experiments.followir.core17_reranking import digest, write_json

ARMS = ("lexical", "owner_lexical", "section_lexical", "owner_postings")
WORD_CAP = 96


@dataclass
class Article:
    text: list
    postings: dict
    section: tuple
    owners: dict
    original: dict

    def lexical_scores(self, phrase):
        scores = defaultdict(float)
        for term in sorted(set(tokens(phrase))):
            posting = self.postings.get(term, ())
            weight = log1p((len(self.text) - len(posting) + .5) / (len(posting) + .5))
            for i, frequency in posting:
                scores[i] += weight * frequency
        return scores


def article_index(raw, note):
    root = ET.fromstring(raw)
    body = root.find("./body")
    blocks = []
    if body is not None:
        for paragraph in body.findall(".//p"):
            if paragraph.findall(".//p"):
                continue
            words = " ".join(paragraph.itertext()).split()
            blocks.extend(" ".join(words[i:i + WORD_CAP]) for i in range(0, len(words), WORD_CAP))
    note_blocks = [text for text in legacy.SENTENCE.split(note) if text.strip()]
    if not blocks or not note_blocks or max(map(lambda s: len(s.split()), note_blocks)) > WORD_CAP:
        raise ValueError("Expected body and bounded original declaration clauses")
    section = tuple(range(len(blocks), len(blocks) + len(note_blocks)))
    blocks.extend(note_blocks)
    postings = defaultdict(list)
    owners = defaultdict(list)
    original = {}
    for i, text in enumerate(blocks):
        counts = Counter(tokens(text))
        norm = .35 + .65 * len(text.split()) / 45
        for token, count in counts.items():
            postings[token].append((i, count * 2.2 / (count + 1.2 * norm)))
        clauses = compile_note(text)
        # Index a local text pattern anywhere, without using JATS section labels.
        if len(clauses) == 1 and clauses[0].start == 0 and clauses[0].end == len(text):
            clause = clauses[0]
            for owner in sorted(clause.owners):
                owners[owner].append(i)
            if i in section:
                original[i] = (legacy.LABELS.get(clause.action.casefold()), clause.owners)
    return Article(blocks, postings, section, owners, original)


def rank(article, phrase, names, arm):
    topic = article.lexical_scores(phrase)
    if arm == "section_lexical":
        choice = max(article.section, key=lambda i: (topic[i], -i))
        return [choice] * len(names)
    if arm == "owner_postings":
        fallback = max(topic, key=lambda i: (topic[i], -i), default=0)
        return [max(article.owners.get(owner, ()), key=lambda i: (topic[i], -i), default=fallback)
                for owner in names]
    selected = []
    owner_weight = .75 if arm == "owner_lexical" else 0.
    for name in names:
        scores = dict(topic)
        if owner_weight:
            for block, frequency in article.postings.get(name.casefold(), ()):
                scores[block] = scores.get(block, 0.) + owner_weight * frequency
        selected.append(max(scores, key=lambda i: (scores[i], -i), default=0))
    return selected


def cases(rows, clusters, folder, checks, phrase_id):
    for (doi, note, duties, good, other), cluster in zip(rows, clusters, strict=True):
        common = anchored.make_case((doi, note, duties, good, other), phrase_id)
        if common is None:
            continue
        article = article_index(transfer.checked_xml(folder, doi, checks[doi]), note)
        if not any(role == common["missing"] and good in owners for role, owners in article.original.values()):
            raise ValueError("Case lost its original positive declaration")
        if any(role == common["missing"] and other in owners for role, owners in article.original.values()):
            raise ValueError("Malformed contrasting author case")
        yield article, common["phrase"], common["missing"], (good, other), cluster


def collect(inputs, reader):
    outcomes = []
    for j, (article, phrase, role, names, cluster) in enumerate(inputs):
        results = {}
        offset = j % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            started = perf_counter()
            selected = rank(article, phrase, names, arm)
            windows = [article.text[i] for i in selected]
            margins, clipped = reader.score_pairs(
                [(f"Author initials: {name}. Requested activity: {phrase}", text)
                 for name, text in zip(names, windows, strict=True)], task=anchored.PERSONAL,
            )
            elapsed = (perf_counter() - started) * 1000
            if clipped or not np.isfinite(margins).all():
                raise ValueError("Invalid reader outputs")
            labeled = [article.original.get(i, (None, frozenset())) for i in selected]
            valid = [activity == role and name in owners
                     for name, (activity, owners) in zip(names, labeled, strict=True)]
            results[arm] = dict(ids=selected, scores=margins, good=valid, ms=elapsed,
                                words=sum(len(text.split()) for text in windows))
        outcomes.append((cluster, results))
    return outcomes


def threshold(rows, arm):
    items = [case[arm] for _, case in rows]
    cuts = sorted({value for row in items for score in row["scores"]
                   for value in (float(np.nextafter(score, -np.inf)), score)})
    choices = []
    for cut in cuts:
        accepted = [[score > cut for score in row["scores"]] for row in items]
        risk = sum(any(allowed and not valid for allowed, valid in zip(flags, row["good"], strict=True))
                   for flags, row in zip(accepted, items, strict=True))
        good = sum(flags[0] and row["good"][0] for flags, row in zip(accepted, items, strict=True))
        if risk <= len(items) // 10:
            choices.append((good, -risk, cut))
    return max(choices)[-1]


def summarize(observations, cuts):
    groups = [group for group, _ in observations]
    rows = {}
    for arm in ARMS:
        items = [row[arm] for _, row in observations]
        cut = cuts[arm]
        flags = [[score > cut for score in item["scores"]] for item in items]
        valid = [flag[0] and item["good"][0] for flag, item in zip(flags, items, strict=True)]
        bad = [any(accepted and not good for accepted, good in zip(flag, item["good"], strict=True))
               for flag, item in zip(flags, items, strict=True)]
        clean = [good and not wrong for good, wrong in zip(valid, bad, strict=True)]
        rows[arm] = (clean, dict(
            cases=len(items), reader_pairs=2 * len(items), window_words=sum(item["words"] for item in items),
            changed_vs_lexical=sum(row[arm]["ids"] != row["lexical"]["ids"] for _, row in observations),
            retrieved_correct_clause=sum(row["good"][0] for row in items), accepted_correct=sum(valid),
            accepted_partial_author=sum(flag[1] for flag in flags), adverse_papers=sum(bad),
            clean_papers=sum(clean), p95_ms=float(np.quantile([item["ms"] for item in items], .95)),
        ))
    return {arm: values | {"clean_delta_vs_lexical": clustered_paired_interval(
        [int(a) - int(b) for a, b in zip(clean, rows["lexical"][0], strict=True)], groups)}
        for arm, (clean, values) in rows.items()}


def run(old, external, checkpoint, output):
    import torch
    from experiments.followir.core17_verifier import Reranker

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    panels = legacy.prepare(old)
    later, names, _ = transfer.cohort(old, external)
    receipts = json.loads((old / "capture.json").read_bytes())
    sources = [
        (split, list(cases(rows, range(len(rows)), old / split, receipts[split]["checksums"], phrase)))
        for (split, (rows, _)), phrase in zip(panels.items(), (0, 1), strict=True)
    ]
    checks = json.loads((external / "capture.json").read_bytes())["checksums"]
    sources.append(("transfer", list(cases(later, transfer.clusters(names), external, checks, 1))))
    if any(not inputs for _, inputs in sources):
        raise ValueError("Missing article panel")

    reader = Reranker(checkpoint)
    collect(sources[0][1][:3], reader)
    calibration = collect(sources[0][1], reader)
    cuts = {arm: threshold(calibration, arm) for arm in ARMS}
    results = {name: summarize(collect(inputs, reader), cuts) for name, inputs in sources[1:]}
    index_summary = {
        name: dict(median_blocks=float(np.median([len(article.text) for article, *_ in inputs])),
                   median_words=float(np.median([sum(len(s.split()) for s in article.text)
                                                for article, *_ in inputs])))
        for name, inputs in sources
    }
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=reader.provenance["revision"], code_sha256=digest(Path(__file__)),
        original_capture_sha256=digest(old / "capture.json"), transfer_capture_sha256=digest(external / "capture.json"),
        calibration_papers=len(calibration), thresholds=cuts,
        calibration=summarize(calibration, cuts), index=index_summary, panels=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for input_name in ("old", "external", "checkpoint", "output"):
        cli.add_argument(input_name, type=Path)
    run(**vars(cli.parse_args()))
