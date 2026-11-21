"""Role confirmation from resident embeddings or the source reranker."""

from collections import Counter
from types import SimpleNamespace

import numpy as np

from experiments.credit.budgeted_source_role_search import role_check
from experiments.credit.distilled_role_recheck import fit_heads, options
from experiments.credit.engineering_all_author_gate import VIEWS
from experiments.credit.engineering_cross_source_search import candidates, required_roles
from experiments.credit.engineering_joint_calibration import SUPPORT, open_calibration
from experiments.credit.engineering_role_banks import encode
from experiments.credit.paired_role_verification import alternative_roles, apply
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.credit.source_role_factorial_diagnosis import payload


METHODS = ("reserve_student", "reserve_reranker")
TASK = "Does this CRediT contributor role describe the research activity in the query?"


def observation(method, service, vectors, phrases, pools, heads, definitions):
    if method == "reserve_student":
        weights, bias = heads["distilled_head"]
        decisions = [options(weights @ vector + bias, pool) for vector, pool in zip(vectors, pools, strict=True)]
        selected, gaps = zip(*decisions, strict=True)
        return list(selected), min(gaps), 0, 0
    pairs = [(phrase, definitions[SUPPORT[role]])
             for phrase, roles in zip(phrases, pools, strict=True) for role in roles]
    values, clipped, tokens = service.source.score_pairs(pairs, task=TASK, count_tokens=True)
    if clipped:
        raise ValueError("Short role definitions unexpectedly clipped")
    margins, selected = [], []
    for scores, pool in zip(np.asarray(values).reshape(2, 3), pools, strict=True):
        winner, runner_up = np.argsort(-scores, kind="stable")[:2]
        selected.append(pool[winner])
        margins.append(min(scores[winner], scores[winner] - scores[runner_up]))
    return selected, float(min(margins)), 1, tokens


def decide(method, service, vectors, phrases, scores, pair, base, heads, definitions, cutoff):
    pools = alternative_roles(scores)
    expected = [pool[0] for pool in pools]
    if cutoff is None or not candidates(service.index, pair, [{role} for role in expected]):
        return payload(service.index, pair, base)
    selected, margin, forwards, tokens = observation(method, service, vectors, phrases, pools, heads, definitions)
    value = dict(choice=selected if selected == expected else None,
                 margin=margin, forwards=forwards, tokens=tokens)
    return apply(service.index, pair, base, value, cutoff)


def fit_lightweight(path, service, definitions, training):
    model, training_report = fit_heads(training, encode(service.encoder, training), service.banks["ridge"],
                                       service.bands["ridge"], service.role)
    articles, _, cases = open_calibration(path)
    index = SimpleNamespace(articles=articles)
    calibration_service = SimpleNamespace(index=index, source=service.source)
    vectors = encode(service.encoder, VIEWS["original"])
    variants = {method: [] for method in METHODS}
    for row in cases:
        pair = tuple(row["papers"])
        positions = [SUPPORT[role] for role in row["roles"]]
        subset = vectors[positions]
        phrases = [VIEWS["original"][i] for i in positions]
        scores = {name: subset @ service.banks[name].T for name in service.bands}
        initial = required_roles(row, scores, "ridge_band", service.bands)
        if candidates(index, pair, initial):
            base = payload(index, pair, initial)
            applicable = False
        else:
            repaired, calls, tokens, _ = role_check(index, pair, scores, service.bands, service.role,
                                                     phrases, service.cutoffs["role"])
            base = payload(index, pair, repaired, role_calls=calls, tokens=tokens)
            applicable = not calls and not base["owners"]
        pools = alternative_roles(scores)
        expected = [pool[0] for pool in pools]
        feasible = applicable and candidates(index, pair, [{role} for role in expected])
        for method in METHODS:
            choice = dict(choice=None, margin=0., forwards=0, tokens=0)
            if feasible:
                labels, margin, calls, tokens = observation(method, calibration_service, subset, phrases,
                                                             pools, model, definitions)
                choice = dict(choice=labels if labels == expected else None,
                              margin=margin, forwards=calls, tokens=tokens)
            variants[method].append((row, pair, initial, base, choice))
    fit = {}
    for method, records in variants.items():
        cutoffs = sorted({0., *(obs["margin"] for *_, obs in records if obs["margin"] > 0.)})
        best = None
        for cutoff in cutoffs:
            counts = Counter()
            for row, pair, initial, base, choice in records:
                if choice["choice"] is not None and choice["margin"] > cutoff:
                    response = apply(index, pair, initial, choice, cutoff)
                else:
                    response = base
                counts.update(evidence(response, row))
            score = (counts["strict"], -counts["adverse"], cutoff)
            if counts["adverse"] <= len(cases) // 20 and (best is None or score > best[0]):
                best = score, dict(cutoff=cutoff, counts=dict(counts))
        fit[method] = best[1] if best else dict(cutoff=None)
    return model, training_report, fit
