"""Train and calibrate a one-batch candidate substitution policy."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import automatic_role_entity_index as auto
from experiments.docred import dispatch_recovery_actions as dispatch
from experiments.docred import global_role_search as index
from experiments.docred import learned_role_candidate_rescue as proposal
from experiments.docred import noisy_role_source_routing as routing
from experiments.docred import nomination_evidence_windows as windows
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as evidence
from experiments.docred import syntax_role_evidence as syntax


ARMS = ("single_batch", "syntax_window", "syntax_joint", "pointwise", "contrasts")
ROLES = tuple(sorted(index.ROLES))
REVIEW_INCREASE_LIMIT = .05


def candidates(query, sources, documents, profiles, spans, links, uncertain):
    features, details, mentions = {}, {}, {}
    for rank, doc in enumerate(sources):
        _, works, positions = profiles[doc]
        anchors = works.get(query["work"], set())
        cue_terms = index.words(index.ROLES[query["role"]])
        cues = [position for term in cue_terms for position in positions.get(term, ())]
        for name, occurrences, origins in proposal.pool(documents[doc], query, profiles[doc], spans[doc]):
            attached = syntax.role_sentences(query["role"], occurrences, links[doc])
            distance = min((abs(s - w) for s in attached for w in anchors), default=99)
            nearest_cue = min((min(abs(a - pos), abs(b - 1 - pos)) for s, a, b in occurrences
                               for sent, pos in cues if sent == s), default=99)
            competing = sum(bool(syntax.role_sentences(role, occurrences, links[doc]))
                            for role in links[doc] if role != query["role"])
            pair = doc, name
            strength = 2 if distance <= 1 else int(bool(attached) and not anchors)
            details[pair] = strength, -rank, len(occurrences)
            features[pair] = [
                *map(float, origins), len(name) / 4, math.log1p(len(occurrences)),
                1 / (1 + rank), 1 / (1 + nearest_cue), bool(attached),
                math.log1p(len(attached)), float(distance == 0), float(distance == 1),
                float(bool(anchors)), float(not uncertain), math.log1p(competing),
                *(float(query["role"] == role) for role in ROLES),
            ]
            mentions[pair] = occurrences
    return features, details, mentions


def collect(queries, route, pool, identities):
    inputs, labels, pairs, records = [], [], [], []
    by_person = defaultdict(lambda: (set(), set()))
    for query in queries:
        sources, _, uncertain = route(query)
        features, _, _ = pool(query, sources, uncertain)
        by_source = defaultdict(lambda: ([], []))
        for (doc, name), vector in features.items():
            match = doc, identities[doc].get(name, name)
            positive = match in query["positive"]
            if not positive and match not in query["rivals"]:
                continue
            i = len(inputs)
            inputs.append(vector)
            labels.append(int(positive))
            records.append(dict(source=doc, work=query["work"], name=name, role=query["role"], label=int(positive)))
            by_source[doc][int(positive)].append(i)
            by_person[(query["work"], match)][int(positive)].add(i)
        for negatives, positives in by_source.values():
            pairs.extend((positive, negative, "person_swap") for positive in positives for negative in negatives)
    for negatives, positives in by_person.values():
        pairs.extend((positive, negative, "role_swap") for positive in sorted(positives)
                     for negative in sorted(negatives) if records[positive]["role"] != records[negative]["role"])
    if len(inputs) < 100 or not pairs:
        raise ValueError("Insufficient known training contrasts")
    return np.asarray(inputs, dtype=np.float32), labels, pairs, records


def train(inputs, labels, contrasts):
    import torch
    from torch import nn
    from torch.nn import functional as F

    x = torch.as_tensor(inputs, device="cuda")
    y = torch.as_tensor(labels, device="cuda", dtype=torch.float32)
    stats, scorers = {}, {}
    by_kind = {kind: torch.as_tensor([(p, n) for p, n, label in contrasts if label == kind], device="cuda")
               for kind in ("person_swap", "role_swap")}
    if any(not len(pair) for pair in by_kind.values()):
        raise ValueError("Both contrast types must be available")
    for method in ("pointwise", "contrasts"):
        torch.manual_seed(20261120)
        model = nn.Linear(x.shape[1], 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.035, weight_decay=.02)
        began = perf_counter()
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).flatten()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if method == "contrasts":
                loss += .5 * sum(F.softplus(logits[pairs[:, 1]] - logits[pairs[:, 0]]).mean() for pairs in by_kind.values())
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        params = model.weight.detach().cpu().numpy().ravel().copy()
        bias = float(model.bias.detach().item())
        scorers[method] = params
        stats[method] = dict(parameters=len(params) + 1, gpu_seconds=perf_counter() - began,
                             final_loss=float(loss.detach().item()), weights=params.tolist(), bias=bias)
    return scorers, stats


def nominate(prior, features, details, weights, threshold):
    if not prior:
        return prior, 0.
    incumbent = prior[-1]
    eligible = sorted(pair for pair in details.keys() - set(prior) if details[pair][0] > details[incumbent][0])
    if not eligible:
        return prior, 0.
    scores = np.asarray([features[pair] for pair in [incumbent, *eligible]], dtype=np.float32) @ weights
    best = int(np.argmax(scores[1:])) + 1
    margin = float(scores[best] - scores[0])
    return (prior[:-1] + [eligible[best - 1]] if margin > threshold else prior), margin


def execute(query, method, threshold, route, get_prior, pool, links, scorers, reader, context):
    began = perf_counter()
    sources, _, uncertain = route(query)
    prior = get_prior(query, sources)
    features, details, mentions = pool(query, sources, uncertain)
    chosen, margin = nominate(prior, features, details, scorers[method], threshold)
    positions, prompts = [], []
    work, role = " ".join(query["work"]), evidence.ROLES[query["role"]]
    for doc, person in chosen:
        text, profile, raw_names = context[doc]
        attached = syntax.role_sentences(query["role"], raw_names.get(person, mentions.get((doc, person), ())),
                                         links[doc])
        window = syntax.choose_window(query, person, profile, attached)
        positions.append(window)
        prompts.append((f"Was {' '.join(person)} credited in the role of {role} for {work}?",
                        "\n".join(text[i] for i in window)))
    scores, clipped = reader(prompts) if prompts else ([], 0)
    if clipped or not np.isfinite(scores).all() or len(chosen) != len(prior):
        raise ValueError("Invalid or unequal verification work")
    result = list(zip(chosen, positions, scores, strict=True)), [], int(bool(chosen)), (perf_counter() - began) * 1000
    return result, margin, chosen != prior


def evaluate(query, method, threshold, run, gold, identities):
    outcome, margin, replaced = run(query, method, threshold)
    return dispatch.assess(query, outcome, gold, identities) | {"replaced": int(replaced)}, margin


def calibrate(queries, run, gold, identities):
    rows = {m: [] for m in ("pointwise", "contrasts")}
    for query in queries:
        fixed, _ = evaluate(query, "syntax_window", 0., run, gold, identities)
        for method in rows:
            result, margin = evaluate(query, method, 0., run, gold, identities)
            if margin > 0:
                rows[method].append((margin, *(result[key] - fixed[key] for key in (
                    "all_annotated", "rivals", "unjudged"))))
    limit = int(REVIEW_INCREASE_LIMIT * len(queries))
    gates = {}
    for method, observations in rows.items():
        choices = []
        for threshold in sorted({0., *(row[0] for row in observations)}):
            moves = [row for row in observations if row[0] > threshold]
            quality, rivals, review = (sum(row[col] for row in moves) for col in (1, 2, 3))
            if rivals <= 0 and review <= limit:
                choices.append((quality, -review, -len(moves), threshold, rivals))
        quality, review, count, threshold, rivals = max(choices)
        gates[method] = dict(threshold=threshold, swaps=-count, covered_delta=quality,
                             rival_delta=rivals, unjudged_delta=-review, review_increase_limit=limit)
    return gates


def run(snapshot, holdout, sha256, checkpoint, output):
    import torch

    if output.exists() or "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    if index.digest(holdout) != sha256:
        raise ValueError("Changed holdout snapshot")
    old = auto.load_documents(snapshot)
    known = {index.words(doc["title"]) for doc in old}
    docs = old + [doc for doc in json.loads(holdout.read_bytes()) if index.words(doc["title"]) not in known]
    spans, links = syntax.extract(docs)
    _, raw = index.mentions_index(docs)
    identities = auto.person_alias_judgments(docs)
    gold = windows.annotated_sentences(docs)
    training = index.study_queries(old[:500], 0)
    calibration = index.study_queries(old, 500)
    training_works = {q["work"] for q in training}
    calibration = [q for q in calibration if q["work"] not in training_works]
    prior_works = {q["work"] for q in index.study_queries(old, 0)}
    test = [q for q in index.study_queries(docs, len(old)) if q["work"] not in prior_works]

    def setup(size):
        aliases, profiles = auto.auto_index(docs[:size], spans[:size], raw[:size])
        lookup, postings = routing.AliasLookup(aliases), index.bm25_index(docs[:size])

        def route(query):
            return routing.route(query, postings, lookup, "swap_then_guard")

        def pool(query, sources, uncertain):
            return candidates(query, sources, docs, profiles, spans, links, uncertain)

        def shortlist(query, sources):
            return index.rank_people(query, sources, profiles)[0]

        def bind(scorers, reader):
            context = []
            for doc, (people, _, _) in zip(docs[:size], profiles, strict=True):
                text = [" ".join(sentence) for sentence in doc["sents"]]
                context.append((text, evidence.profile(text), dict(people)))

            def run(query, method, threshold):
                if method in scorers:
                    return execute(query, method, threshold, route, shortlist, pool, links, scorers, reader, context)
                outcome, replaced = syntax.measure(query, method, lookup, postings, profiles, docs, spans, links,
                                                   reader, context)
                return outcome, 0., replaced
            return run
        return route, pool, bind

    train_route, train_pool, _ = setup(500)
    x, labels, contrasts, records = collect(training, train_route, train_pool, identities)
    scorers, training_stats = train(x, labels, contrasts)
    print(f"Trained with {len(records)} judgments and {len(contrasts)} contrasts", flush=True)
    tokenizer, model, info = oracle.load_reader(checkpoint)
    reader = oracle.reader_scorer(tokenizer, model)
    _, _, bind_cal = setup(len(old))
    evaluate_calibration = bind_cal(scorers, reader)
    cal_views = calibration + [q for q in routing.query_styles(calibration, old) if q["style"] == "token_drop"]
    gates = calibrate(cal_views, evaluate_calibration, gold, identities)
    print(f"Calibrated on {len(cal_views)} queries: {gates}", flush=True)
    _, _, bind = setup(len(docs))
    evaluate_test = bind(scorers, reader)
    views = [dict(q, style="exact") for q in test] + [
        q for q in routing.query_styles(test, docs) if q["style"] == "token_drop"
    ]
    measured = defaultdict(lambda: defaultdict(list))
    for method in ARMS:
        evaluate_test(views[0], method, gates.get(method, {}).get("threshold", 0.))
    for i, query in enumerate(views):
        shift = i % len(ARMS)
        for method in ARMS[shift:] + ARMS[:shift]:
            data, _ = evaluate(query, method, gates.get(method, {}).get("threshold", 0.), evaluate_test,
                               gold, identities)
            measured[query["style"]][method].append(data)
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{len(views)} test queries", flush=True)
    aggregate = dispatch.summarize(measured, reference="single_batch")
    for style, panel in measured.items():
        for method in ("pointwise", "contrasts"):
            rows = panel[method]
            fixed = panel["syntax_joint"]
            delta = [(a["all_annotated"] - b["all_annotated"]) / index.TOP
                     for a, b in zip(rows, fixed, strict=True)]
            aggregate[style][method]["covered_p5_change_vs_syntax_joint"] = clustered_paired_interval(
                delta, [row["group"] for row in rows]
            )
    index.write_json(output.with_name("training-contrasts.json"), dict(
        inputs=index.HASHES, example_rows=records, pairs=contrasts))
    index.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=info["revision"], inputs=index.HASHES, holdout_sha256=sha256,
        training_records=len(records), training_contrasts_by_kind={kind: sum(row[2] == kind for row in contrasts)
            for kind in ("person_swap", "role_swap")}, training=training_stats,
        calibration_queries=len(cal_views), calibration=gates, code_sha256=index.digest(Path(__file__)),
        panels=aggregate))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "holdout", "checkpoint", "output"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("sha256")
    run(**vars(cli.parse_args()))
