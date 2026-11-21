"""Calibrate source existence and compositional role evidence on disjoint data."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.engineering_cross_source_panel import gold, requests
from experiments.credit.engineering_cross_source_search import (
    AVAILABILITY, KEYWORD_STOP, TOPIC_TASK, Catalog, candidates, phrase_bank, required_roles, select,
)
from experiments.credit.engineering_role_banks import TASK, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint


PLOS_SHA256 = "1b9d9b1fae5397b790be93df66b42d8295105f4a57ade1ca3618e6ee40d2c322"
SOURCE_POLICIES = ("linked", "anchored", "gated_linked", "oracle_sources")
ROLE_POLICIES = ("oracle_roles", "ridge_band", "prototype_band", "ridge_mass", "prototype_mass")
SUPPORT = {role: i for i, role in enumerate(ROLES)}


def open_calibration(path):
    if digest(path) != PLOS_SHA256:
        raise ValueError("Changed PLOS calibration projection")
    source = json.loads(path.read_bytes())
    articles, index = {}, defaultdict(set)
    for doc, entry in source["documents"].items():
        roles = {owner["owner"]: owner["roles"] for owner in entry["owners"]}
        articles[doc] = dict(indexed=entry["index"], roles=roles)
        for owner in roles:
            if not owner.startswith("other-"):
                index[owner].add(doc)
    links = [(pair, owner) for owner, docs in index.items() for pair in combinations(sorted(docs), 2)]
    paired = {(row["family"], row["variant"]): row for row in requests(articles, links)}
    # Add same-source role replacements. The rival role is declared on the
    # right-hand document, but no linked contributor meets the whole request.
    # Keep at most six such families per article pair, irrespective of owners.
    edits = defaultdict(dict)
    for pair, owner in links:
        left, right = [articles[doc]["roles"] for doc in pair]
        declared = set().union(*map(set, right.values()))
        for a, b in product(left[owner], right[owner]):
            if a == b:
                continue
            for replacement in sorted(declared - set(right[owner]) - {a}):
                rival = (a, replacement)
                if gold(articles, pair, rival):
                    continue
                stable = digest_text(pair, (a, b), rival)
                edits[pair][stable] = ((a, b), rival)
    for pair, options in sorted(edits.items()):
        for key, (positive, negative) in sorted(options.items())[:6]:
            for variant, roles in (("present", positive), ("replacement", negative)):
                paired[key, variant] = dict(
                    family=key, papers=pair, roles=roles, variant=variant, gold=gold(articles, pair, roles),
                )
    return articles, source["clues"], list(paired.values())


def digest_text(*values):
    from hashlib import sha256
    return sha256(json.dumps(values).encode()).hexdigest()


def topic_prompts(texts):
    return [f"Instruct: {TOPIC_TASK}\nQuery:{clue}" for clue in texts]


def features(catalog, topic, text, doc):
    lexical = set(normalize(text).split()) - KEYWORD_STOP
    evidence = set(normalize(catalog.articles[doc]["indexed"]).split())
    denom = sum(catalog.idf.get(token, 0) for token in sorted(lexical))
    overlap = sum(catalog.idf.get(token, 0) for token in sorted(lexical & evidence)) / max(denom, .001)
    cosine = topic["cosine"][doc]
    leaders = sorted(topic["cosine"].values(), reverse=True)[:2]
    return cosine, cosine - topic["best"], overlap, leaders[0] - leaders[1], 1 / (1 + topic["fused"].index(doc))


def fit_source(catalog, clues, encoder, kind):
    x, y, absent_cases = [], [], []
    for paper, row in sorted(clues.items()):
        clue = row["result"] if kind == "excerpt" else catalog.keywords(row["result"])
        encoded = encoder.encode(topic_prompts([clue]))
        for withheld in (None, paper):
            top = catalog.rank(encoded, [clue], withheld)[0]
            report = []
            for doc in top["fused"]:
                vector = features(catalog, top, clue, doc)
                x.append(vector)
                y.append(int(doc == paper))
                report.append(vector)
            if withheld:
                absent_cases.append(report)
    x, y = np.asarray(x), np.asarray(y)
    if not y.any() or y.all():
        raise ValueError("Source validation requires true and substitute candidates")
    center, scale = x.mean(axis=0), x.std(axis=0).clip(.0001)
    design = np.column_stack(((x - center) / scale, np.ones(len(x))))
    weights = np.where(y == 1, .5 / y.sum(), .5 / (1 - y).sum())
    matrix = design.T @ (design * weights[:, None]) + .05 * np.eye(design.shape[1])
    coef = np.linalg.solve(matrix, design.T @ (weights * y))
    model = dict(center=center, scale=scale, coef=coef)
    worst_absent = sorted((max(source_score(model, f) for f in case) for case in absent_cases), reverse=True)
    model["threshold"] = worst_absent[len(worst_absent) // 20]
    return model, dict(
        queries=len(clues), training_positives=int(y.sum()), absent_queries=len(absent_cases),
        accepted_absent=sum(score > model["threshold"] for score in worst_absent),
        threshold=model["threshold"], coefficient=coef.tolist(),
    )


def source_score(model, values):
    return float(((np.asarray(values) - model["center"]) / model["scale"]) @ model["coef"][:-1] + model["coef"][-1])


def gated_pair(catalog, topic, clues, model):
    pools = [[doc for doc in side["fused"]
              if source_score(model, features(catalog, side, clue, doc)) > model["threshold"]]
             for side, clue in zip(topic, clues, strict=True)]
    eligible = [(a, b) for a, b in product(*pools)
                if a != b and catalog.articles[a]["roles"].keys() & catalog.articles[b]["roles"].keys()]
    return min(eligible, key=lambda pair: (sum(t["fused"].index(p) for t,p in zip(topic,pair,strict=True)), pair),
               default=None)


def masses(scores, temperature):
    standardized = (scores - scores.max(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True).clip(.01)
    scaled = np.exp(standardized / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def witnesses(articles, pair, probability):
    left, right = [articles[key]["roles"] for key in pair]
    result = {}
    for owner in sorted(left.keys() & right.keys()):
        indices = [sorted(SUPPORT[role] for role in fields[owner]) for fields in (left, right)]
        if not all(indices):
            continue
        marginals = [side[ids] for side, ids in zip(probability, indices, strict=True)]
        score = float(marginals[0].sum() * marginals[1].sum())
        roles = tuple(ROLES[ids[int(side.argmax())]] for ids, side in zip(indices, marginals, strict=True))
        result[owner] = (score, roles)
    return result


def judge(row, candidates_by_owner, cutoff):
    offered = {owner: roles for owner, (score, roles) in candidates_by_owner.items() if score > cutoff}
    owners = set(offered)
    gold = set(row["gold"])
    exact = bool(gold) and owners == gold
    evidence = exact and all(roles == tuple(row["roles"]) for roles in offered.values())
    adverse = bool(owners - gold) or any(roles != tuple(row["roles"]) for roles in offered.values())
    return dict(issued=bool(owners), exact=exact, exact_proof=evidence, adverse=adverse)


def fit_roles(pairs, articles, encoder, banks):
    gloss = encoder.encode([f"Instruct: {TASK}\nQuery:{phrase}" for phrase in VIEWS["original"]])
    roles = {bank: gloss @ banks[bank].T for bank in ("ridge", "prototype")}
    chosen = {}
    for name, matrix in roles.items():
        best = None
        for temperature in (.25, .5, 1., 2.):
            rows = [witnesses(articles, q["papers"], masses(matrix[[SUPPORT[r] for r in q["roles"]]], temperature))
                    for q in pairs]
            values = {0.} | {v for row in rows for v, _ in row.values()}
            for threshold in sorted(values):
                counts = Counter()
                for q, by_owner in zip(pairs, rows, strict=True):
                    counts.update(judge(q, by_owner, threshold))
                if counts["adverse"] > len(pairs) // 20:
                    continue
                key = counts["exact_proof"], -counts["adverse"], threshold, -temperature
                if best is None or key > best[0]:
                    best = (key, dict(temperature=temperature, threshold=threshold, source=dict(counts)))
        if best is None:
            raise AssertionError("Empty output must fit the source risk threshold")
        chosen[name] = best[1]
    return chosen


def run(data, encoder, banks, wording, bands, source_policy, role_policy, offset):
    articles = data["articles"]
    catalog = Catalog(articles, encoder)
    phrases = phrase_bank(wording)
    result, ms = defaultdict(Counter), defaultdict(list)
    cases = sorted(data["queries"], key=lambda row: (row["family"], row["variant"]), reverse=bool(offset))
    for index, query in enumerate([cases[0], *cases]):
        descriptions = [phrases[r][int(query["family"][:16], 16) % len(phrases[r])] for r in query["roles"]]
        excerpt = [articles[p]["clue"] for p in query["papers"]]
        for label, clues in (("excerpt", excerpt), ("keywords", [catalog.keywords(clue) for clue in excerpt])):
            start = perf_counter()
            enc = encoder.encode(topic_prompts(clues) + [f"Instruct: {TASK}\nQuery:{text}" for text in descriptions])
            scores = {bank: enc[2:] @ banks[bank].T for bank in bands}
            encode_ms = (perf_counter() - start) * 1000
            for status in AVAILABILITY:
                removed = None if status == "all" else query["papers"][0 if status == "left_missing" else 1]
                start = perf_counter()
                topic = catalog.rank(enc[:2], clues, removed)
                topic_ms = (perf_counter() - start) * 1000
                for mode in SOURCE_POLICIES:
                    start = perf_counter()
                    if mode == "gated_linked":
                        pair = gated_pair(catalog, topic, clues, source_policy[label])
                    elif mode == "anchored":
                        # Anchor on the first-ranked hybrid source. Do not apply
                        # role labels before resolving the source-pair choice.
                        pools = product(*(side["fused"] for side in topic))
                        pairs = [(a,b) for a,b in pools if a != b and
                                 (a == topic[0]["fused"][0] or b == topic[1]["fused"][0]) and
                                 all(t["best"] - t["cosine"][p] <= .04 for t,p in zip(topic,(a,b),strict=True)) and
                                 candidates(catalog,(a,b),[set(),set()])]
                        pair = min(pairs, key=lambda p: (sum(t["fused"].index(s) for t,s in zip(topic,p,strict=True)),p), default=None)
                    else:
                        pair, _ = select(catalog, topic, query, [set(), set()], mode, removed)
                    pair_ms = (perf_counter() - start) * 1000
                    correct_source = pair == tuple(query["papers"])
                    label_row = query if correct_source else (query | {"gold": []})
                    for arm in ROLE_POLICIES:
                        start = perf_counter()
                        if pair is None:
                            offers, cutoff = {}, 0.
                        elif arm.endswith("_mass"):
                            bank = arm.split("_")[0]
                            policy = role_policy[bank]
                            offers = witnesses(articles, pair, masses(scores[bank], policy["temperature"]))
                            cutoff = policy["threshold"]
                        else:
                            required = required_roles(query, scores, arm, bands)
                            owners = candidates(catalog, pair, required)
                            roles = tuple(query["roles"]) if all(role in options for role, options in zip(query["roles"], required, strict=True)) else ("", "")
                            offers, cutoff = {owner: (1., roles) for owner in owners}, 0.
                        checked = judge(label_row, offers, cutoff)
                        json.dumps(checked)
                        elapsed = encode_ms + topic_ms + pair_ms + (perf_counter() - start) * 1000
                        if index:
                            key = "|".join((label,status,mode,arm))
                            result[key].update(checked, requests=1, sources_correct=correct_source,
                                               substitutions=checked["issued"] and not correct_source,
                                               swap_issued=query["variant"]=="swap" and checked["issued"])
                            ms[key].append(elapsed)
    return {key:dict(counts)|{"p95_ms":float(np.quantile(ms[key],.95))} for key,counts in sorted(result.items())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("eng_panel", "plos", "niso", "training", "published", "embedding", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output")
    plos, clues, pairs = open_calibration(args.plos)
    if not pairs:
        raise ValueError("No source-independent role conjunctions")
    glossary, phrases, wording = queries(args.niso, args.training)
    checkpoint = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    banks = projections(encoder, glossary, phrases)
    catalog = Catalog(plos, encoder)
    fit = {name:fit_source(catalog, clues, encoder, name) for name in ("excerpt","keywords")}
    role = fit_roles(pairs, plos, encoder, banks)
    bands = json.loads(args.published.read_bytes())["runs"][0]["bands"]
    report = run(json.loads(args.eng_panel.read_bytes()), encoder, banks, wording, bands,
                 {key:val[0] for key,val in fit.items()}, role, args.offset)
    write_json(args.output, dict(
        target_sha256=digest(args.eng_panel), calibration_sha256=PLOS_SHA256, script_sha256=digest(Path(__file__)),
        role_calibration_cases=len(pairs), role_calibration_pairs=len({tuple(q["papers"]) for q in pairs}),
        topic_fit={key:v[1] for key,v in fit.items()}, role_fit=role, gpu=torch.cuda.get_device_name(),
        embedding_revision=checkpoint["revision"], offset=args.offset, results=report,
    ))
