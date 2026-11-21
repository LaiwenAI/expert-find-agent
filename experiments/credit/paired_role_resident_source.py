"""Verify both proposed sources from resident features before one 4B role read."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from experiments.credit.engineering_joint_calibration import features, open_calibration, topic_prompts
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.independent_source_confirmation import VERIFY
from experiments.credit.keyword_author_rescue import PostingCatalog
from experiments.credit.paired_role_mixed_verifier import availability
from experiments.credit.paired_role_source_postings import pairs_for
from experiments.credit.paired_role_stage_audit import query_state, signatures
from experiments.credit.paired_role_verification import alternative_roles
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_confirmation_transfer import verify
from experiments.credit.source_role_factorial_diagnosis import payload
from experiments.followir.core17_reranking import digest, write_json


def observations(service, catalog, clues, view):
    records = []
    ordered = sorted(clues)
    texts = [clues[doc]["result"] for doc in ordered]
    if view == "keywords":
        texts = [catalog.keywords(text) for text in texts]
    vectors = service.encoder.encode(topic_prompts(texts))
    for source, text, vector in zip(ordered, texts, vectors, strict=True):
        if not text:
            continue
        proposals = {}
        for withheld in (None, source):
            topic = catalog.full_rank(vector[None, :], [text], withheld)[0]
            for doc in topic["fused"]:
                proposals[withheld is not None, doc] = features(catalog, topic, text, doc)
        docs = sorted({doc for _, doc in proposals})
        teacher = {}
        for i in range(0, len(docs), 32):
            batch = docs[i:i + 32]
            scores, clipped = service.source.score_pairs(
                [(text, catalog.articles[doc]["indexed"]) for doc in batch], task=VERIFY)
            if clipped:
                raise ValueError("Calibration source was clipped")
            teacher.update(zip(batch, scores, strict=True))
        fold = int.from_bytes(sha256(source.encode()).digest()[:4], "big") % 4
        records.extend((fold, missing, doc == source, values, teacher[doc])
                       for (missing, doc), values in proposals.items())
    return records


def train(x, gold, target):
    center, scale = x.mean(axis=0), x.std(axis=0).clip(.0001)
    design = np.column_stack(((x - center) / scale, np.ones(len(x))))
    weight = np.where(gold == 1, .5 / gold.sum(), .5 / (1 - gold).sum())
    coef = np.linalg.solve(design.T @ (design * weight[:, None]) + .05 * np.eye(design.shape[1]),
                           design.T @ (weight * target))
    return center, scale, coef


def score(model, x):
    center, scale, coef = model
    return ((np.asarray(x) - center) / scale) @ coef[:-1] + coef[-1]


def calibrate(service, plos):
    articles, clues, _ = open_calibration(plos)
    catalog = PostingCatalog(articles, service.encoder)
    gates, report = {}, {}
    for view in ("keywords", "excerpt"):
        rows = observations(service, catalog, clues, view)
        folds, missing, gold, x, teacher = zip(*rows, strict=True)
        x, gold, folds, missing, teacher = map(np.asarray, (x, gold, folds, missing, teacher))
        strength = 1 / (1 + np.exp(np.clip((service.cutoffs["source"] - teacher) / 3, -40, 40)))
        report[view], gates[view] = {}, {}
        for name, target in (("gold_features", gold), ("distilled_features", .5 * gold + .5 * strength)):
            held = np.zeros(len(x))
            for fold in sorted(set(folds)):
                included = folds != fold
                if len(set(gold[included])) != 2:
                    raise ValueError("Expected both classes in each source-training split")
                held[~included] = score(train(x[included], gold[included], target[included]), x[~included])
            threshold = float(held[missing].max())
            gates[view][name] = (train(x, gold, target), threshold)
            report[view][name] = dict(
                cutoff=threshold, target_rows=int(gold.sum()), missing_source_rows=int(missing.sum()),
                target_pass=int((gold & (held > threshold)).sum()),
                missing_pass=int((missing & (held > threshold)).sum()),
                wrong_source_present_pass=int((~missing & ~gold & (held > threshold)).sum()),
                training_rows=len(x),
            )
    return gates, report


def resident_checks(service, pair, pools, clues, phrases, ranking, cutoff, models):
    values = [features(service.index, side, clue, doc)
              for side, clue, doc in zip(ranking, clues, pair, strict=True)]
    margins = {name: min(float(score(model, x)) for x in values) - threshold
               for name, (model, threshold) in models.items()}
    passes = {name: margin > 0. for name, margin in margins.items()}
    if not any(passes.values()):
        return {name: payload(service.index, None, []) for name in passes}, passes, margins
    roles, gap, tokens = service.role.batched(phrases, pools)
    role_ok = cutoff is not None and gap > cutoff and None not in roles
    selected = [{role} for role in roles] if role_ok else []
    return {name: payload(service.index, pair if allowed and role_ok else None, selected,
                          role_calls=int(allowed), tokens=tokens if allowed else 0)
            for name, allowed in passes.items()}, passes, margins


def clue_view(clues):
    if all(clue == normalize(clue) and len(clue.split()) <= 8 for clue in clues):
        return "keywords"
    return "excerpt"


def matched_source_fallback(service, ranking, scores, clues, phrases, cutoff, models):
    pools = alternative_roles(scores)
    candidates = pairs_for(service.index, ranking, [{pool[0]} for pool in pools])
    if not candidates:
        return payload(service.index, None, [])
    name = "gold_matched"
    outputs, _, _ = resident_checks(service, candidates[0], pools, clues, phrases, ranking, cutoff,
                                    {name: models[clue_view(clues)][name]})
    return outputs[name]


def run_resident(service, labels, cutoff, lightweight, plos, reference, output, reverse, *, matched=None):
    models, fitted = calibrate(service, plos)
    matched_report = None
    if matched is not None:
        from experiments.credit.paired_role_matched_source import calibrate_pairs

        extra, matched_report = calibrate_pairs(service, models, *matched)
        for view, heads in extra.items():
            models[view].update(heads)
    names = tuple(models["keywords"])
    arms = ("cascade", "dense_joint", *names)
    requests = list(labels)
    counts = defaultdict(lambda: defaultdict(Counter))
    diagnostics = defaultdict(lambda: defaultdict(Counter))
    missing = defaultdict(lambda: defaultdict(Counter))
    diagnostic_margins = defaultdict(lambda: defaultdict(list))
    recorded = defaultdict(dict)
    for request in (requests[::-1] if reverse else requests):
        label = labels[request]
        view = label["view"]
        cascade, plan = availability(service, request, cutoff, lightweight)
        response = dict.fromkeys(arms, cascade)
        if plan is not None:
            pair, pools, required, clues, phrases, _, ranking = plan
            allowed = verify(service.source, service.index, clues, pair, (0, 1)) > service.cutoffs["source"]
            response["dense_joint"] = payload(service.index, pair if allowed else None, required, source_calls=1)
            checked, passes, margins = resident_checks(service, pair, pools, clues, phrases, ranking, cutoff, models[view])
            response.update(checked)
            correct_source = pair == tuple(label["papers"])
            for arm, margin in margins.items():
                diagnostic_margins[view][arm].append((correct_source, margin))
            for arm, eligible in passes.items():
                diagnostics[view][arm].update(fallbacks=1, source_accepted=eligible,
                                               intended_pair=eligible and pair == tuple(label["papers"]))
        for name, out in response.items():
            if out["calls"] > 1 or out["calls"] != out["source_calls"] + out["role_calls"]:
                raise ValueError("Exceeded the one-forward cap")
            counts[view][name].update(evidence(out, label))
            recorded[name][request] = out
        for withheld in label["papers"]:
            state, ranking, scores, clues, phrases = query_state(service, request, withheld)
            for name in names:
                missing[view][name].update(requests=1)
            if state["pair"] is not None or state["response"]["calls"]:
                continue
            pools = alternative_roles(scores)
            options = pairs_for(service.index, ranking, [{pool[0]} for pool in pools])
            if not options:
                continue
            pair = options[0]
            if withheld in pair:
                raise ValueError("A missing source was proposed")
            checked, passes, margins = resident_checks(service, pair, pools, clues, phrases, ranking, cutoff, models[view])
            for name, margin in margins.items():
                diagnostic_margins[view][name].append((False, margin))
            for name, out in checked.items():
                missing[view][name].update(fallbacks=1, source_accepted=passes[name],
                                           substitutes=bool(out["owners"]), role_calls=out["role_calls"])
    proof = signatures(recorded, requests)
    expected = json.loads(reference.read_bytes())["response_signatures"]
    if proof["cascade"] != expected["student_then_teacher"] or proof["dense_joint"] != expected["dense_joint"]:
        raise ValueError("Original proof signatures changed")
    write_json(output, dict(counts=counts, calibration=fitted, fallback=diagnostics, missing_source=missing,
                            diagnostic_margins={view: {name: sorted(items) for name, items in by_arm.items()}
                                                for view, by_arm in diagnostic_margins.items()},
                            signatures=proof, requests=len(requests), reverse=reverse,
                            reference_sha256=digest(reference), script_sha256=digest(Path(__file__)),
                            **({"matched_calibration": matched_report} if matched is not None else {})))
