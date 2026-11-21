"""Test source-pair confirmation before an indexed contributor proof."""

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import engineering_joint_calibration as search
from experiments.credit.engineering_cross_source_panel import SENTENCE
from experiments.credit.engineering_contribution_verify import stable
from experiments.credit.engineering_role_banks import TASK as ROLE_TASK, encode, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, fit

INPUTS = (
    "ee6715866d53d973a437d13c679f8ef2efa8a995d721782843c5b1944859f2a3",
    "8ab52c04301c4b3583ed294d0d62f2aacd652a6d44bd51f1a47286b0d2516fa2",
)
ARMS = ("linked", "anchored", "individual", "paired")


def partition(text):
    title, _, abstract = text.partition("\n")
    parts = SENTENCE.split(abstract)
    if len(parts) > 1:
        cut = len(parts) // 2
        first, last = " ".join(parts[:cut]), " ".join(parts[cut:])
    else:
        words = abstract.split()
        cut = len(words) // 2
        first, last = " ".join(words[:cut]), " ".join(words[cut:])
    if not title or not first or not last:
        raise ValueError("Cannot split the publication clue and index text")
    return title + "\n" + first, last


def load_catalog(paths):
    articles = {}
    for path, expected in zip(paths, INPUTS, strict=True):
        if digest(path) != expected:
            raise ValueError("Changed publisher projection")
        for row in json.loads(path.read_bytes())["sources"]:
            indexed, clue = partition(row["text"])
            roles = {a["person"]: set(a["roles"]) for a in row["authors"]
                     if row["declaration_complete"] and a["person"] and a["roles"]}
            article = dict(indexed=indexed, clue=clue, roles=roles)
            if articles.setdefault(row["doi"], article) != article:
                raise ValueError("Overlapping DOI carries different source content")
    return articles


def valid_pair(catalog, a, b):
    return a != b and bool(catalog.articles[a]["roles"].keys() & catalog.articles[b]["roles"].keys())


def rank_candidates(catalog, vectors, clues, papers, withheld, arm=None):
    topic = catalog.rank(vectors, clues, withheld)
    pools = [side["fused"] for side in topic]
    options = {(a, b): None for a, b in product(*pools) if valid_pair(catalog, a, b)}
    source_features = []
    if arm not in ("linked", "anchored"):
        source_features = [{paper: np.asarray(search.features(catalog, side, clue, paper)) for paper in side["fused"]}
                           for side, clue in zip(topic, clues, strict=True)]
    if arm in (None, "paired"):
        for pair in options:
            sources = [source_features[i][doc] for i, doc in enumerate(pair)]
            crossed = [topic[i]["cosine"][doc] - topic[1-i]["cosine"][doc] for i, doc in enumerate(pair)]
            options[pair] = np.r_[np.minimum(*sources), np.maximum(*sources), min(crossed), max(crossed)]
    return dict(papers=papers, withheld=withheld, topic=topic, pools=pools, features=source_features, options=options)


def rows(catalog, panel, encoder, mode, reverse):
    prepared = {}
    groups = list(panel["pairs"])
    for row in (reversed(groups) if reverse else groups):
        papers = tuple(row["papers"])
        clues = [catalog.articles[p]["clue"] for p in papers]
        if mode == "keywords":
            clues = [catalog.keywords(clue) for clue in clues]
        vectors = encoder.encode(search.topic_prompts(clues))
        prepared[papers] = tuple(rank_candidates(catalog, vectors, clues, papers, withheld) for withheld in (None, *papers))
    return prepared


def train(training):
    source, source_labels, proof, proof_labels = [], [], [], []
    for papers, trials in sorted(training.items()):
        for row in trials:
            for i, scope in enumerate(row["features"]):
                for doc, x in scope.items():
                    source.append(x)
                    source_labels.append(doc == papers[i])
            for pair, x in row["options"].items():
                proof.append(x)
                proof_labels.append(pair == papers)
    return fit(np.array(source), np.array(source_labels)), fit(np.array(proof), np.array(proof_labels))


def dispatch(row, models, arms=ARMS):
    topic = row["topic"]
    ordered = sorted(row["options"], key=lambda p: (sum(row["pools"][i].index(doc) for i, doc in enumerate(p)), p))
    choices = dict.fromkeys(arms, (None, None))
    if not ordered:
        return choices
    if "linked" in arms:
        choices["linked"] = ordered[0], 1.
    if "anchored" in arms:
        first = row["pools"][0][0], row["pools"][1][0]
        eligible = [pair for pair in ordered if (pair[0] == first[0] or pair[1] == first[1]) and all(
            topic[i]["best"] - topic[i]["cosine"][doc] <= .04 for i, doc in enumerate(pair)
        )]
        if eligible:
            choices["anchored"] = eligible[0], 1.
    if "individual" in arms:
        source = [{doc: float(value) for doc, value in zip(scope, estimate(models[0], np.array(list(scope.values()))))}
                  for scope in row["features"]]
        scores = {pair: min(source[i][doc] for i, doc in enumerate(pair)) for pair in ordered}
        choices["individual"] = max(scores.items(), key=lambda item: item[1])
    if "paired" in arms:
        values = estimate(models[1], np.array([row["options"][pair] for pair in ordered]))
        choices["paired"] = max(zip(ordered, values, strict=True), key=lambda item: item[1])
    return choices


def calibrate(calibration, models):
    worst = {arm: [] for arm in ARMS[2:]}
    for papers, cases in calibration.items():
        bad = {arm: [] for arm in worst}
        for row in cases:
            for arm, (pair, score) in dispatch(row, models).items():
                if arm in bad and pair is not None and pair != papers:
                    bad[arm].append(float(score))
        for arm, scores in bad.items():
            worst[arm].append(max(scores, default=-1.))
    return {arm: sorted(scores, reverse=True)[len(scores)//20] for arm, scores in worst.items()}


def phrase_cases(panel, wording):
    phrases = defaultdict(list)
    for index, _, text in wording["niso"]:
        phrases[ROLES[index]].append(text)
    planned = {}
    for group in panel["pairs"]:
        papers = tuple(group["papers"])
        cases = []
        for i, positive in enumerate(group["bindings"]):
            family = stable(papers, i)
            for variant, roles in (("positive", positive), ("swap", positive[::-1])):
                text = [min(phrases[role], key=lambda phrase: stable(family, role, phrase)) for role in roles]
                cases.append((variant, roles, text))
        planned[papers] = cases
    return planned


def known_owners(articles, papers, roles):
    a, b = (articles[p]["roles"] for p in papers)
    return {person for person in a.keys() & b.keys() if roles[0] in a[person] and roles[1] in b[person]}


def candidates_with_roles(articles, pair, scores, consensus=False):
    if pair is None:
        return {}
    left, right = (articles[p]["roles"] for p in pair)
    result = {}
    for person in left.keys() & right.keys():
        supports, rendered, agreement = [], [], True
        for (ridge, definition), owner_roles in zip(scores, (left[person], right[person]), strict=True):
            mask = np.array([role in owner_roles for role in ROLES])
            choices, bands = [], []
            for values in (ridge, definition):
                choices.append(ROLES[int(np.argmax(np.where(mask, values, -np.inf)))])
                bands.append(float(np.clip(np.max(values[mask], initial=-np.inf)
                                           - np.max(values[~mask], initial=-np.inf), -100, 100)))
            rendered.append(choices[0])
            agreement &= choices[0] == choices[1]
            if consensus:
                supports.append(min(bands[0] / max(.01, float(ridge.std())),
                                    bands[1] / max(.01, float(definition.std()))))
            else:
                supports.append(bands[0])
        if agreement or not consensus:
            result[person] = min(supports), tuple(rendered)
    return result


def issued_proof(articles, pair, scores, threshold, consensus=False):
    return {owner: labels for owner, (value, labels) in candidates_with_roles(
        articles, pair, scores, consensus
    ).items() if value > threshold}


def embed_roles(encoder, banks, plans):
    unique = sorted({phrase for plan in plans for cases in plan.values() for _, _, texts in cases for phrase in texts})
    vectors = encode(encoder, unique)
    return dict(zip(unique, zip(vectors @ banks["ridge"].T, vectors @ banks["definition"].T, strict=True), strict=True))


def calibrate_roles(articles, plan, role_scores, consensus):
    worst = []
    for papers, cases in plan.items():
        values = []
        for _, roles, phrases in cases:
            reference = known_owners(articles, papers, roles)
            rows = candidates_with_roles(articles, papers, [role_scores[p] for p in phrases], consensus)
            values.extend(score for person, (score, quoted) in rows.items()
                          if person not in reference or quoted != tuple(roles))
        worst.append(max(values, default=-1000000.))
    return sorted(worst, reverse=True)[len(worst)//20]


def quality(catalog, held, role_scores, models, thresholds, role_cut, plan, role_ablation=None):
    metrics = defaultdict(Counter)
    for papers, cases in held.items():
        for trial in cases:
            missing = trial["withheld"] is not None
            track = "missing" if missing else "present"
            choices = dispatch(trial, models)
            if role_ablation:
                candidate, value = choices["linked"]
                choices = {arm: (candidate, value) for arm in role_ablation}
            for arm, (candidate, score) in choices.items():
                found = candidate if score is not None and score > thresholds.get(arm, 0.) else None
                cutoff, consensus = (role_cut, False) if not role_ablation else role_ablation[arm]
                measurements = metrics[track, arm]
                measurements.update(groups=1, source_reachable=papers in trial["options"],
                                    source_found=found == papers, source_substitution=found is not None and found != papers)
                for variant, roles, texts in plan[papers]:
                    reference = known_owners(catalog.articles, papers, roles) if not missing else set()
                    answer = issued_proof(catalog.articles, found, [role_scores[t] for t in texts], cutoff, consensus)
                    correct_source = found == papers
                    proved = {person for person, evidence in answer.items()
                              if person in reference and evidence == tuple(roles)}
                    measurements.update(
                        requests=1, positives=bool(reference),
                        owner_correct=correct_source and bool(answer.keys() & reference),
                        proof_correct=correct_source and bool(proved),
                        proof_wrong=bool(answer) and (not correct_source or answer.keys() != proved),
                        oracle_role_correct=correct_source and bool(reference),
                        issued=bool(answer),
                        swapped_wrong=variant == "swap" and bool(answer),
                    )
    names = role_ablation or ARMS
    return {split: {arm: dict(metrics[split, arm]) for arm in names} for split in ("present", "missing")}


def live_timing(catalog, encoder, mode, models, threshold, bank, role_cut, plan, reverse):
    times, counts = defaultdict(list), defaultdict(Counter)
    groups = list(plan)
    for index, papers in enumerate(reversed(groups) if reverse else groups):
        clues = [catalog.articles[p]["clue"] for p in papers]
        if mode == "keywords":
            clues = [catalog.keywords(text) for text in clues]
        _, roles, phrases = plan[papers][0]
        inputs = search.topic_prompts(clues) + [f"Instruct: {ROLE_TASK}\nQuery:{text}" for text in phrases]
        offset = (index + int(reverse)) % len(ARMS)
        for arm in ARMS[offset:] + ARMS[:offset]:
            began = perf_counter()
            vectors = encoder.encode(inputs)
            candidates = rank_candidates(catalog, vectors[:2], clues, papers, None, arm=arm)
            pair, score = dispatch(candidates, models, (arm,))[arm]
            accepted = pair if score is not None and score > threshold.get(arm, 0.) else None
            score_rows = list(zip(vectors[2:] @ bank.T, vectors[2:] @ bank.T, strict=True))
            result = issued_proof(catalog.articles, accepted, score_rows, role_cut)
            json.dumps((accepted, result), sort_keys=True)
            times[arm].append((perf_counter() - began) * 1000)
            reference = known_owners(catalog.articles, papers, roles)
            counts[arm].update(found=accepted == papers,
                               strict=accepted == papers and any(
                                   owner in reference and label == tuple(roles) for owner, label in result.items()
                               ))
    return {arm: dict(counts[arm], p95_ms=float(np.quantile(values, .95)), requests=len(values))
            for arm, values in times.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("confirmation","extended","frozen","thresholds","glossary","training","embedding","output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require GPU Slurm and fresh output")
    archive = json.loads(args.thresholds.read_bytes())
    if archive["source_manifest_sha256"] != digest(args.frozen):
        raise ValueError("Changed source group manifest")
    frozen = json.loads(args.frozen.read_bytes())["panels"]
    articles = load_catalog((args.confirmation, args.extended))
    info = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    catalog = search.Catalog(articles, encoder)
    definitions, phrases, wording = queries(args.glossary, args.training)
    banks = projections(encoder, definitions, phrases)
    bank = banks["ridge"]
    role_cut = archive["cutoffs"]["group_calibrated"]["role_index"]
    plan = phrase_cases(frozen["held"], wording)
    dev_plan = phrase_cases(frozen["calibration"], wording)
    role_scores = embed_roles(encoder, banks, (plan, dev_plan))
    role_ablation = {
        "role_index": (role_cut, False),
        "tight_role_index": (calibrate_roles(articles, dev_plan, role_scores, False), False),
        "bank_agreement": (calibrate_roles(articles, dev_plan, role_scores, True), True),
    }
    results = {}
    for mode in ("excerpt", "keywords"):
        dev = rows(catalog, frozen["calibration"], encoder, mode, args.reverse)
        order = sorted(dev, key=lambda pair: stable("open-proof", pair))
        model = train({pair: dev[pair] for pair in order[:18]})
        thresholds = calibrate({pair: dev[pair] for pair in order[18:]}, model)
        hold = rows(catalog, frozen["held"], encoder, mode, args.reverse)
        dev_cut = {pair: dev[pair] for pair in order[18:]}
        results[mode] = dict(
            cutoff=thresholds,
            tested=quality(catalog, hold, role_scores, model, thresholds, role_cut, plan),
            calibration=quality(catalog, dev_cut, role_scores, model, thresholds, role_cut, dev_plan),
            role_ablation=quality(catalog, hold, role_scores, model, thresholds, role_cut, plan, role_ablation),
            timing=live_timing(catalog, encoder, mode, model, thresholds, bank, role_cut, plan, args.reverse),
        )
        print(mode, results[mode]["tested"], flush=True)
    write_json(args.output, dict(
        catalog_articles=len(articles), results=results, role_cutoff=role_cut, role_ablation_cutoffs=role_ablation,
        group_sha256=digest(args.frozen),
        inputs=list(INPUTS), embedding=info["revision"], script_sha256=digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse
    ))


if __name__ == "__main__":
    main()
