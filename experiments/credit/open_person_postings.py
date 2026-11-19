"""Compare source-first and author-constrained fallback in an open JATS catalog."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.retrieval import tokens
from expert_search.statistics import paired_interval
from experiments.credit.conjunctive_source_retrieval import sentences
from experiments.credit.legacy_body_confirmation import body_windows, select_windows
from experiments.credit.legacy_contribution_text import stable
from experiments.credit.legacy_open_attribution import Search, TASK, whitespace
from experiments.credit.plos_role_inventory import declared_roles
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


PRIOR = "8196801a0f0d83de28cc5c14a0cb1bf7231fdb6ffa776ac5f896e743e5789e87"
EXTERNAL = "239f69b91899114f98335daa6f3b6a47b508dae88810e36849fe3b3e06c74939"
ARMS = ("source_top1", "source_top4", "owner_postings", "body_top1", "body_top4", "owner_body")
FORMS = ("result", "keywords")
STATES = ("present", "withheld")


def load(prior, external, folders):
    if (digest(prior), digest(external)) != (PRIOR, EXTERNAL):
        raise ValueError("Changed publisher manifests")
    manifests = [json.loads(p.read_bytes()) for p in (prior, external)]
    captures = {}
    for folder in folders:
        for path in sorted(folder.glob("*.xml")):
            captures[path.stem, digest(path)] = path
    catalog = {}
    bodies = {}
    for row in (*manifests[0]["input_sources"], *manifests[1]["inventory"]):
        doi = row["doi"]
        path = captures[doi.rsplit(".", 1)[-1], row["source_sha256"]]
        raw = path.read_bytes()
        roles = declared_roles(raw, doi)
        meta = ET.fromstring(raw).find("./front/article-meta")
        title = whitespace(" ".join(meta.find("./title-group/article-title").itertext()))
        abstract = meta.find("abstract")
        parts = sentences(whitespace(" ".join(abstract.itertext()))) if abstract is not None else []
        if len(parts) < 3 or not title:
            continue
        result = " ".join(parts[-2:])
        prefix = title + "\n" + " ".join(parts[:-2])
        if result.casefold() in prefix.casefold():
            continue
        people = defaultdict(set)
        for ordinal, labels in enumerate(roles, 1):
            for pair in combinations(sorted(labels), 2):
                people[pair].add(ordinal)
        article = SimpleNamespace(indexed_text=prefix, result=result, people=dict(people), roles=roles)
        if doi not in catalog:
            bodies[doi] = body_windows(raw, result)
        earlier = catalog.setdefault(doi, article)
        if earlier != article:
            raise ValueError(f"Conflicting title, abstract or contribution declarations for {doi}")
    test = [row["doi"] for row in manifests[1]["sources"] if row["doi"] in catalog]
    for row in manifests[1]["sources"]:
        if row["doi"] in catalog:
            expected = tuple(tuple(values) for values in row["roles_by_author"])
            if tuple(catalog[row["doi"]].roles) != expected:
                raise ValueError(f"Changed evaluation labels for {row['doi']}")
    development = [row["doi"] for row in manifests[0]["input_sources"] if row["doi"] in catalog]
    development = sorted(development, key=lambda doi: stable("source-calibration", doi))[:len(test)]
    if set(development) & set(test):
        raise ValueError("Overlapping source calibration and test publications")
    return catalog, bodies, development, test


def candidates(index, clue, withheld):
    start = perf_counter()
    vector = index.encoder.encode([f"Instruct: {TASK}\nQuery: {clue}"])[0]
    dense = index.vectors @ vector
    words = set(tokens(clue))
    overlap = [sum(index.idf[w] for w in sorted(row & words)) for row in index.index]
    missing = index.pos.get(withheld)
    order = [i for i in np.argsort(-dense, kind="stable") if i != missing]
    scores = {i: 1 / (60 + r) for r, i in enumerate(order, 1)}
    lexical = sorted((i for i in order if overlap[i] > 0), key=lambda i: (-overlap[i], i))
    for rank, i in enumerate(lexical, 1):
        scores[i] += 1 / (60 + rank)
    ranked = [index.keys[i] for i in sorted(scores, key=scores.get, reverse=True)]
    return ranked, dict(zip(index.keys, map(float, dense), strict=True)), (perf_counter() - start) * 1000


def select_pairs(doi, roles, people):
    if not roles or not all(roles):
        return {}
    distinct = set().union(*map(set, roles))
    eligible = dict(
        supported=[pair for pair, owners in people.items() if len(owners) < len(roles)],
        split_between_authors=[pair for pair in combinations(sorted(distinct), 2) if pair not in people],
    )
    return {name: min(pairs, key=lambda p: stable("duty-pair", doi, p)) for name, pairs in eligible.items() if pairs}


def body_score(index, bodies, clue, doi):
    score, _ = select_windows(clue, bodies[doi], index.idf)
    return score if score is not None else 0.


def calibrate(index, bodies, development, form):
    scores, cases = [], []
    for doi in development:
        clue = index.clue(doi, form)
        ranked, confidence, _ = candidates(index, clue, doi)
        scores.append(confidence[ranked[0]])
        cases.append((clue, ranked[:4], confidence))
    if not scores:
        raise ValueError("Need held-source calibration")
    dense_cut = max(scores)
    wrong = [body_score(index, bodies, clue, candidate)
             for clue, ranked, confidence in cases for candidate in ranked if confidence[candidate] <= dense_cut]
    return dense_cut, max(wrong)


def route(arm, ranked, confidence, pair, index, postings, cuts, bodies, clue):
    top = ranked[0]
    dense_cut, body_cut = cuts
    using_body = arm.startswith("body_") or arm == "owner_body"
    body_checks = 0

    def confirmed(doi):
        nonlocal body_checks
        if confidence[doi] > dense_cut:
            return True
        if using_body:
            body_checks += 1
            return body_score(index, bodies, clue, doi) > body_cut
        return False

    # If the strongest source is identified, check that study even if no author qualifies.
    if confirmed(top):
        return top, index.notes[top].people.get(pair, set()), 1, body_checks
    if arm in ("source_top1", "body_top1"):
        return None, set(), 1, body_checks
    if arm in ("source_top4", "body_top4"):
        followups = ranked[1:4]
    else:
        # The owner posting is the set of documents with one author credited for both duties.
        eligible = postings.get(pair, set())
        followups = [doi for doi in ranked[1:] if doi in eligible][:3]
    for examined, doi in enumerate(followups, 2):
        if confirmed(doi):
            return doi, index.notes[doi].people.get(pair, set()), examined, body_checks
    return None, set(), 1 + len(followups), body_checks


def collect(index, test, postings, form, threshold, bodies):
    rows = defaultdict(list)
    for rotation, doi in enumerate(test):
        article = index.notes[doi]
        pairs = select_pairs(doi, article.roles, article.people)
        if not pairs:
            continue
        clue = index.clue(doi, form)
        for state in STATES:
            goal = doi if state == "present" else None
            ranked, confidence, retrieval = candidates(index, clue, None if goal else doi)
            for condition, pair in pairs.items():
                record = {}
                shift = rotation % len(ARMS)
                for arm in ARMS[shift:] + ARMS[:shift]:
                    start = perf_counter()
                    selected, owners, checked, body_checks = route(
                        arm, ranked, confidence, pair, index, postings, threshold, bodies, clue,
                    )
                    cost = retrieval + (perf_counter() - start) * 1000
                    truth = article.people.get(pair, set()) if goal else set()
                    record[arm] = dict(
                        clean=int(bool(owners) and selected == goal and owners == truth),
                        wrong_source=int(bool(owners) and selected != goal),
                        wrong_author=int(bool(owners) and selected == goal and owners != truth),
                        returned=int(bool(owners)), exact_nomination=int(selected is not None and selected == goal),
                        inspected=checked, body_checks=body_checks, ms=cost,
                    )
                rows[state, condition].append(record)
    return rows


def summarize(rows):
    result = {}
    for arm in ARMS:
        measures = [row[arm] for row in rows]
        totals = {name: sum(value[name] for value in measures)
                  for name in ("returned", "clean", "wrong_source", "wrong_author", "exact_nomination", "inspected", "body_checks")}
        result[arm] = dict(totals, component_p95_ms=float(np.quantile([r["ms"] for r in measures], .95)),
                           clean_delta_vs_top1=paired_interval([r[arm]["clean"] - r["source_top1"]["clean"] for r in rows]))
    return dict(requests=len(rows), arms=result)


def run(prior, external, embedding, output, folders):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    provenance = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    catalog, bodies, development, test = load(prior, external, folders)
    index = Search(catalog, encoder)
    postings = defaultdict(set)
    for doi, article in catalog.items():
        for pair in article.people:
            postings[pair].add(doi)
    panels = {}
    for form in FORMS:
        cutoff = calibrate(index, bodies, development, form)
        panels[form] = dict(dense_cutoff=cutoff[0], body_cutoff=cutoff[1], cases={
            f"{stage}/{kind}": summarize(rows)
            for (stage, kind), rows in collect(index, test, postings, form, cutoff, bodies).items()
        })
    write_json(output, dict(
        sources=len(catalog), calibration_sources=len(development), evaluation_sources=len(test),
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        embedding_revision=provenance["revision"], panels=panels,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prior", "external", "embedding", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    run(**vars(parser.parse_args()))
