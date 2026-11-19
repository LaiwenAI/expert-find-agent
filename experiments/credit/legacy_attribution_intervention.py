"""Compare full and author-scoped contribution verification at fixed source decisions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit import legacy_body_confirmation as body
from experiments.credit import legacy_body_transfer as transfer
from experiments.credit import legacy_open_attribution as base
from experiments.credit import legacy_source_expansion as expansion


REFERENCE = "90c322e2dd1f30ae6adb72e073e19cccf21ab688633fced67191fca016c66352"
DEPTHS = ("top_one", "strict_four")
ARMS = ("character", "full", "owner", "veto")


def scoped_pair(article, row, engine):
    contexts, flipped = base.cascade.reader_contexts(row, 1)
    owners = row[3:][::-1] if flipped else row[3:]
    docs = []
    for context, owner in zip(contexts, owners, strict=True):
        statements = [article.note[c.start:c.end] for c in article.entry["clauses"] if owner in c.owners]
        if not statements:
            raise ValueError("A nominated byline has no matching source clause")
        prompt, sep, original = context.partition("\n\n")
        if not sep or original != article.note:
            raise ValueError("Unexpected contribution note prompt")
        docs.append(prompt + sep + " ".join(statements))
    reader, tokenizer, digits = engine
    values = base.indexed.judge(reader, tokenizer, digits, "Apply the decision separately to this author.", docs)
    return np.asarray(values[::-1] if flipped else values)


def measure(article, doi, duties, gate, frozen_cut, engine):
    start = perf_counter()
    baseline, calls = base.verify(article, doi, duties, True, engine, gate, frozen_cut)
    observations = dict(baseline=set(baseline), full={}, owner={}, full_ms=0., owner_ms=0.,
                        veto_ms=0., pairs=0, rechecked_pairs=0, char_calls=calls,
                        char_ms=1e3 * (perf_counter() - start))
    for position in range(0, len(article.authors), 2):
        authors = article.authors[position:position+2]
        row = (doi, article.note, duties, authors[0], authors[-1])
        start = perf_counter()
        full = base.cascade.reader_pair(row, 1, engine)
        observations["full_ms"] += 1e3 * (perf_counter() - start)
        start = perf_counter()
        scoped = scoped_pair(article, row, engine)
        spent = 1e3 * (perf_counter() - start)
        observations["owner_ms"] += spent
        observations["pairs"] += 1
        if observations["baseline"].intersection(authors):
            observations["veto_ms"] += spent
            observations["rechecked_pairs"] += 1
        for owner, full_score, scoped_score in zip(authors, full[:len(authors)], scoped[:len(authors)], strict=True):
            observations["full"][owner] = float(full_score)
            observations["owner"][owner] = float(scoped_score)
    return observations


def decision(arm, scores, cutoff):
    if arm == "character":
        return scores["baseline"], scores["char_ms"], scores["char_calls"]
    values = scores["full" if arm == "full" else "owner"]
    returned = {author for author, score in values.items() if score > cutoff}
    if arm == "veto":
        return (returned & scores["baseline"], scores["char_ms"] + scores["veto_ms"],
                scores["char_calls"] + scores["rechecked_pairs"])
    return returned, scores[f"{arm}_ms"], scores["pairs"]


def calibrate(notes, rows, gate, cutoff, engine):
    cases = []
    for doi, _, duties, *_ in rows:
        article = notes[doi]
        expected = set.intersection(*(article.memberships[t] for t in duties))
        cases.append((expected, measure(article, doi, duties, gate, cutoff, engine)))
    risk = sum(bool(item["baseline"] - expected) for expected, item in cases)
    thresholds = {}
    for arm in ARMS[1:]:
        universe = set()
        for expected, item in cases:
            candidates = item["full" if arm == "full" else "owner"]
            eligible = (set(candidates) & item["baseline"]) if arm == "veto" else set(candidates)
            universe.update(candidates[a] for a in eligible - expected)
        candidates = sorted(universe | {-1000000.0})
        threshold = next(t for t in candidates if sum(bool(decision(arm, item, t)[0] - expected)
                                                      for expected, item in cases) <= risk)
        thresholds[arm] = threshold
    return dict(cases=len(cases), baseline_wrong_author_queries=risk, thresholds=thresholds)


def evaluate(index, bodies, questions, groups, settings, calibration, gate, author_cut, engine):
    report, cache = {}, {}
    for form in base.FORMS:
        dense, lexical, expansion_cut = settings[form]
        for view in ("present", "withheld"):
            withheld = view == "withheld"
            rows = {depth: {arm: [] for arm in ARMS} for depth in DEPTHS}
            for doi, duties in questions:
                candidates, fixed, scores, rank_ms = expansion.inspect(index, bodies, doi, form, withheld, dense, lexical)
                for depth in DEPTHS:
                    selected, _, source_ms = expansion.select(depth, candidates, fixed, scores, expansion_cut, lexical)
                    correct = selected is not None and not withheld and selected == doi
                    expected = (set.intersection(*(index.notes[doi].memberships[t] for t in duties))
                                if correct else set())
                    if selected is not None and (selected, duties) not in cache:
                        cache[selected, duties] = measure(index.notes[selected], selected, duties, gate, author_cut, engine)
                    for arm in ARMS:
                        if selected is None:
                            returned, verifier_ms, calls = set(), 0., 0
                        else:
                            returned, verifier_ms, calls = decision(
                                arm, cache[selected, duties], calibration["thresholds"].get(arm, author_cut))
                        rows[depth][arm].append(dict(
                            correct_source=int(correct), supported=int(bool(expected & returned)),
                            clean=int(bool(returned) and returned <= expected),
                            wrong_author_on_correct_source=int(correct and bool(returned - expected)),
                            wrong_source_recommendation=int(bool(returned) and not correct),
                            calls=calls, ms=rank_ms + source_ms + verifier_ms,
                        ))
            summary = {}
            for depth, by_arm in rows.items():
                summary[depth] = {}
                for arm, items in by_arm.items():
                    totals = Counter()
                    for item in items:
                        totals.update({key: value for key, value in item.items() if key != "ms"})
                    delta = {}
                    for metric in ("clean", "wrong_author_on_correct_source", "wrong_source_recommendation"):
                        observations = [item[metric] - control[metric] for item, control in
                                        zip(items, by_arm["character"], strict=True)]
                        delta[metric] = clustered_paired_interval(observations, groups)["percentile_95"]
                    summary[depth][arm] = dict(totals, factored_p95_ms=float(np.quantile([v["ms"] for v in items], .95)),
                                               paired_95_interval=delta)
            report[f"{form}/{view}"] = summary
    return report


def run(old, first, fresh, embedding, reader, open_policy, body_policy, author_policy, reference, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh CUDA Slurm run")
    for path, expected in ((open_policy, body.REFERENCE), (body_policy, transfer.BODY_RESULTS),
                           (author_policy, base.transfer.REFERENCE), (reference, REFERENCE)):
        if base.indexed.digest(path) != expected:
            raise ValueError("Unexpected frozen input")
    opened, lexical, frozen, prior = [json.loads(p.read_bytes()) for p in
                                      (open_policy, body_policy, author_policy, reference)]
    notes, bodies, questions, groups, _ = transfer.prepare(old, first, fresh)
    cal_rows = base.legacy.prepare(old)["calibration"][0]
    encoder = base.indexed.QwenTorchEmbedding(embedding, batch_size=12)
    encoder.model.config.use_cache = False
    index = base.Search(notes, encoder)
    tokenizer, reader_model, meta = base.indexed.load_reader(reader)
    if (base.indexed.verify_checkpoint(embedding)["revision"] != frozen["embedding_revision"]
            or meta["revision"] != frozen["reader_revision"]):
        raise ValueError("Changed reader or embedding checkpoint")
    digits = [tokenizer.encode(x, add_special_tokens=False) for x in ("1", "0")]
    if any(len(token) != 1 for token in digits) or digits[0] == digits[1]:
        raise ValueError("Unexpected verifier outputs")
    engine = reader_model, tokenizer, [x[0] for x in digits]
    cutoff = frozen["reader_cutoff"]
    gate = base.transfer.load_gate(frozen["routes"]["character"]["selected"]["gate"], base.cascade.Gate)
    warm, duties = questions[0]
    measure(notes[warm], warm, duties, gate, cutoff, engine)
    calibration = calibrate(notes, cal_rows, gate, cutoff, engine)
    settings = {form: (opened["thresholds"][form], lexical["thresholds"][form]["lexical_body"]["threshold"],
                       prior["thresholds"][form]) for form in base.FORMS}
    report = evaluate(index, bodies, questions, groups, settings, calibration, gate, cutoff, engine)
    pairs = (("clean", "clean_support"), ("calls", "author_calls")) + tuple(
        (m, m) for m in ("correct_source", "supported", "wrong_author_on_correct_source", "wrong_source_recommendation"))
    for view, summary in report.items():
        for depth in DEPTHS:
            baseline = prior["report"][view]["results"][depth]
            if any(summary[depth]["character"][a] != baseline[b] for a, b in pairs):
                raise ValueError("Changed source or character-gate decisions")
    base.indexed.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), reference_sha256=REFERENCE,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), queries=len(questions),
        clusters=len(set(groups)), calibration=calibration, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "first", "fresh", "embedding", "reader", "open_policy", "body_policy",
                 "author_policy", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
