"""Test candidate-scoped and indexed role checks against original publisher statements."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import engineering_contribution_verify as previous
from experiments.credit.engineering_contribution_panel import article_panel, normalize
from experiments.credit.engineering_cross_source_panel import owner_key
from experiments.credit.engineering_role_banks import encode, project, projections, queries
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.trec_enterprise.nomination_coverage_scheduler import estimate, fit

PANEL = "6ce6bc2afc342a07c768d740169d6dac44e8935f468610e883e177a231947466"
LINKS = "53f4cda52f54d2982ff748c0e9fbbff627cbc1c9eeef024cd35e8539422137fb"
ARMS = ("shared", "scoped", "role_index", "fusion")
BATCH = 8


def load(path, checksum):
    if digest(path) != checksum:
        raise ValueError(f"Changed input: {path}")
    return json.loads(path.read_bytes())


def publisher_views(seed, linked, manifest):
    views = dict(seed)
    source = json.loads((linked / "sources.json").read_bytes())
    for pmc, row in source["records"].items():
        raw = (linked / "xml" / f"{pmc}.xml").read_bytes()
        record = manifest["records"][pmc]
        if digest(linked / "xml" / f"{pmc}.xml") != row["sha256"] or row["sha256"] != record["sha256"]:
            raise ValueError("Publisher proof changed")
        views[pmc] = row | article_panel(raw, row)
    return views


def linked_units(views, linked, link_panel, wording):
    phrases = defaultdict(list)
    for role_i, _, phrase in wording["niso"]:
        phrases[ROLES[role_i]].append(phrase)
    required = {article for row in link_panel["queries"] for article in row["papers"]}
    ordinals = {}
    for pmc in sorted(required):
        structure = link_panel["articles"][pmc]["roles"]
        raw_view = views[pmc]
        slots = {owner: int(i) for owner, i in zip(structure, raw_view["roles"], strict=True)}
        for owner, slot in slots.items():
            declared = set(raw_view["roles"][str(slot)] if str(slot) in raw_view["roles"] else raw_view["roles"][slot])
            if declared != set(structure[owner]):
                raise ValueError("Raw and source-scope publisher labels disagree")
        # Exact source-order mapping is encoded by the source projection.
        # Recover indices independently from XML for the additional papers.
        file = linked / "xml" / f"{pmc}.xml"
        if file.exists():
            authors = ET.fromstring(file.read_bytes()).findall(
                "./front/article-meta/contrib-group/contrib[@contrib-type='author']"
            )
            if slots != {owner_key(pmc, i, author): i for i, author in enumerate(authors)}:
                raise ValueError("Author ordinals or source projection changed")
        ordinals[pmc] = slots
    for request in link_panel["queries"]:
        papers = request["papers"]
        total_owners = sorted(set(ordinals[papers[0]]) & ordinals[papers[1]].keys())
        if not total_owners:
            raise ValueError("No cross-source identity")
        role_phrases = [min(phrases[role], key=lambda phrase: previous.stable(request["family"], role, phrase))
                        for role in request["roles"]]
        for owner in total_owners:
            for side, (pmc, role, phrase) in enumerate(zip(papers, request["roles"], role_phrases, strict=True)):
                slot = ordinals[pmc][owner]
                document = link_panel["articles"][pmc]
                yield (pmc, slot, role, phrase, role in document["roles"][owner],
                       (request["family"], request["variant"], owner, side))


def atomic_units(seed):
    for journal, groups in previous.cases(seed).items():
        for case in groups:
            for owner, label in zip(case["candidates"], case["known"], strict=True):
                yield case["article"], owner, case["role"], case["described"], label, (journal,)


def withhold_link_phrases(atoms, linked, wording):
    held = {normalize(row[3]) for row in linked}
    choices = defaultdict(list)
    for role, _, text in wording["niso"]:
        if normalize(text) not in held:
            choices[ROLES[role]].append(text)
    return [(pmc, author, role, min(choices[role], key=lambda p: previous.stable(pmc, role, p)), label, source)
            for pmc, author, role, _, label, source in atoms if choices[role]]


def margin_rows(units, views, reader, encoder, role_banks, reverse):
    raw = {}
    role = {}
    for pmc, author, _, phrase, _, _ in units:
        key = pmc, author, phrase
        if key in raw:
            continue
        doc = views[pmc]
        alias = doc["aliases"].get(str(author), doc["aliases"].get(author))
        scoped = [row for row in doc["blocks"] if author in row["owners"]]
        if not alias or not scoped:
            raise ValueError("Original owner phrase is unavailable")
        # Role membership is parsed from the visible statement, independently
        # of the structured publisher tags used to label test outcomes.
        visible = {name for block in scoped for name in block["roles"]}
        prompt = f"Target author: {alias}. Requested contribution: {phrase}."
        raw[key] = [(prompt, "\n".join(row["text"] for row in selected)) for selected in (doc["blocks"], scoped)]
        role[key] = np.array([[name in visible for name in ROLES]])
    unique = sorted({key[2] for key in raw})
    vectors = dict(zip(unique, encode(encoder, unique), strict=True))
    rows = {}
    keys = sorted(raw)
    batches = range(0, len(keys), BATCH)
    for start in (reversed(batches) if reverse else batches):
        group = keys[start:start+BATCH]
        packed = [pair for key in group for pair in raw[key]]
        cross_scores, clipped = reader.score_pairs(packed, task=previous.TASK)
        if clipped or len(cross_scores) != len(packed) or not np.isfinite(cross_scores).all():
            raise ValueError("Truncated or missing raw-text verifier scores")
        for i, key in enumerate(group):
            _, options = project(vectors[key[2]], role_banks, role[key])
            rows[key] = (*cross_scores[2*i:2*i+2], float(options["ridge"][0]), float(options["definition"][0]))
    return rows


def matrix(units, values):
    return np.asarray([values[pmc, owner, phrase] for pmc, owner, _, phrase, *_ in units])


def routes(values, fusion):
    return dict(shared=values[:, 0], scoped=values[:, 1], role_index=values[:, 2],
                fusion=estimate(fusion, values[:, 1:]))


def calibrated(scores, labels):
    negatives = sorted(scores[~labels], reverse=True)
    return float(negatives[len(negatives) // 20])


def atom_metrics(units, scores, threshold):
    labels = np.array([row[4] for row in units])
    return {arm: dict(credited=int(labels.sum()), uncredited=int((~labels).sum()),
                      accepted_credited=int((labels & (values > threshold[arm])).sum()),
                      accepted_uncredited=int((~labels & (values > threshold[arm])).sum()))
            for arm, values in scores.items()}


def proofs(units, scores, thresholds, truth):
    report = {}
    for arm in ARMS:
        tested = defaultdict(dict)
        for row, score in zip(units, scores[arm], strict=True):
            family, variant, owner, side = row[-1]
            tested[family, variant, owner][side] = bool(score > thresholds[arm])
        answer = {key: set() for key in truth}
        for (family, variant, owner), sides in tested.items():
            if set(sides) != {0, 1}:
                raise ValueError("Missing side decision")
            if all(sides.values()):
                answer[family, variant].add(owner)
        owner_scores = defaultdict(list)
        for row, score in zip(units, scores[arm], strict=True):
            family, variant, owner, _ = row[-1]
            owner_scores[family, variant, owner].append(float(score))
        contrasts = []
        for (family, variant), expected in truth.items():
            if variant != "present":
                continue
            for owner in expected:
                positive = min(owner_scores[family, variant, owner])
                negative = min(owner_scores[family, "swap", owner])
                contrasts.append((positive > negative, positive == negative))
        result = Counter()
        for (family, variant), expected in truth.items():
            got = answer[family, variant]
            result.update(queries=1, positive=bool(expected), known_hit=bool(got & expected),
                          exact_positive=bool(expected) and got == expected,
                          unsupported=bool(got - expected), negative_issued=not expected and bool(got),
                          exact=got == expected, issued=bool(got))
        report[arm] = dict(result, ordered_contrasts=sum(row[0] for row in contrasts),
                           tied_contrasts=sum(row[1] for row in contrasts), contrasts=len(contrasts), signature=sha256(json.dumps([
            sorted(answer[key]) for key in sorted(answer)
        ]).encode()).hexdigest())
    return report


def evaluate(atoms, linked, values, truth):
    frontiers = [row for row in atoms if row[-1] == ("frontiers",)]
    holdout = [row for row in atoms if row[-1] == ("sensors",)]
    articles = sorted({unit[0] for unit in frontiers}, key=lambda key: previous.stable("raw-conjunction", key))
    train = set(articles[:2*len(articles)//3])
    fit_rows = [row for row in frontiers if row[0] in train]
    cal_rows = [row for row in frontiers if row[0] not in train]
    model = fit(matrix(fit_rows, values)[:, 1:], np.array([row[4] for row in fit_rows]))
    calibration = routes(matrix(cal_rows, values), model)
    labels = np.array([row[4] for row in cal_rows])
    thresholds = {arm: calibrated(score, labels) for arm, score in calibration.items()}
    return dict(
        linked=proofs(linked, routes(matrix(linked, values), model), thresholds, truth),
        sensors=atom_metrics(holdout, routes(matrix(holdout, values), model), thresholds),
        calibration=atom_metrics(cal_rows, calibration, thresholds),
        thresholds=thresholds, train_articles=len(train), calibration_articles=len(articles)-len(train),
        calibrated_roles=len({row[2] for row in cal_rows}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("seed", "linked", "link_panel", "manifest", "glossary", "training", "embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a Slurm GPU and fresh output")
    seed = load(args.seed, PANEL)
    pairs = load(args.link_panel, LINKS)
    manifest = json.loads(args.manifest.read_bytes())
    if manifest["panel_sha256"] != LINKS:
        raise ValueError("Unexpected cross-source panel digest")
    raw = publisher_views(seed, args.linked, manifest)
    definitions, phrases, wording = queries(args.glossary, args.training)
    encoder_info = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    reader = Reranker(args.reranker)
    bank = projections(encoder, definitions, phrases)
    linked = list(linked_units(raw, args.linked, pairs, wording))
    atoms = list(atomic_units(seed))
    alternates = withhold_link_phrases(atoms, linked, wording)
    values = margin_rows([*atoms, *alternates, *linked], raw, reader, encoder, bank, args.reverse)
    expected = {(q["family"], q["variant"]): set(q["gold"]) for q in pairs["queries"]}
    output = dict(
        authored_calibration=evaluate(atoms, linked, values, expected),
        matched_register_calibration=evaluate(alternates, linked, values, expected),
        source_pairs=len({tuple(q["papers"]) for q in pairs["queries"]}),
        linked_atomic_checks=len(linked), inputs=dict(seed=PANEL, cross_source=LINKS, manifest=digest(args.manifest),
                                                   glossary=digest(args.glossary), training=digest(args.training)),
        script_sha256=digest(Path(__file__)), job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
        gpu=torch.cuda.get_device_name(), embedding=encoder_info["revision"], reranker=reader.provenance["revision"],
    )
    write_json(args.output, output)


if __name__ == "__main__":
    main()
