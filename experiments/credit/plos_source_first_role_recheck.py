"""Check independently proposed sources before revisiting low-confidence roles."""

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.credit import contrastive_rescue_identity as shared
from experiments.credit.paired_role_verification import PairedReader
from experiments.credit.plos_role_anchor_audit import top_candidate

route, base = shared.route, shared.base
METHODS = ("fused_ridge", "fused_4b", "source_score_4b", "independent_4b")


def source_proposal(topic, clues, footprints, model, cutoff):
    pair = tuple(lane["fused"][0] for lane in topic)
    if len(set(pair)) < 2:
        return None, False, False
    a, b = (route.source_features(topic, clues, footprints, doc, side) for side, doc in enumerate(pair))
    source_value = float(base.estimate(model, np.array([np.r_[np.minimum(a, b), np.maximum(a, b)]]))[0])
    independent = all(
        doc == topic[side]["dense"][0]
        and topic[side]["cosine"][doc] > topic[1 - side]["cosine"][doc]
        for side, doc in enumerate(pair)
    )
    return pair, source_value > cutoff, independent


def measured_recheck(reader, phrases, labels, uncertain):
    clauses = [phrases[i] for i in uncertain]
    torch.cuda.synchronize()
    start = perf_counter()
    result, margin, tokens = reader.batched(clauses, [base.ROLES] * len(clauses))
    torch.cuda.synchronize()
    elapsed = (perf_counter() - start) * 1000
    roles = list(labels)
    for i, value in zip(uncertain, result, strict=True):
        roles[i] = value
    return tuple(roles), margin, tokens, elapsed


def observation(articles, intended, gold, pair, roles, gap, tokens=0, elapsed=0.):
    issued = pair is not None and None not in roles and route.applicable(articles, pair, roles)
    exact = issued and route.result_is_correct(articles, intended, pair, gold, roles)
    reason = "none"
    if issued:
        reason = "source" if pair != intended else "role" if not exact else "exact"
    return dict(issued=bool(issued), exact=bool(exact), gap=gap, tokens=tokens, ms=elapsed, reason=reason)


def gather(index, encoder, panel, plans, entrance, predictions, role_cut, footprints, model, cutoff, reader, reverse):
    articles = index.catalog.articles
    outputs = []
    for group in sorted(panel["pairs"], key=lambda row: row["papers"], reverse=reverse):
        intended = tuple(group["papers"])
        clues = [articles[doc]["clue"] for doc in intended]
        if entrance == "keywords":
            clues = [index.catalog.keywords(clue) for clue in clues]
        embedded = encoder.encode(base.search.topic_prompts(clues))
        for absent in (None, *intended):
            topic = index.catalog.rank(embedded, clues, absent, depth=route.DEEP)
            cases = list(route.choices(index, topic, clues, absent, plans[intended], predictions,
                                       role_cut, footprints, False))
            pair, passes_score, independent = source_proposal(topic, clues, footprints, model, cutoff)
            for (_, _, phrases), (gold, labels, initial, routes) in zip(plans[intended], cases, strict=True):
                prior_options = routes.get("author_role", ({}, 0, 0))[0]
                baseline = initial or top_candidate(prior_options, model, cutoff)
                before = observation(articles, intended, gold, baseline, labels, 0.)
                uncertain = [i for i, phrase in enumerate(phrases) if predictions[phrase][1] <= role_cut]
                ready = pair is not None and not before["issued"] and bool(uncertain)
                compare = {}
                if ready:
                    revised, confidence, tokens, elapsed = measured_recheck(reader, phrases, labels, uncertain)
                    compare["fused_ridge"] = observation(articles, intended, gold, pair, labels, float("inf"))
                    compare["fused_4b"] = observation(articles, intended, gold, pair, revised, confidence, tokens, elapsed)
                    if passes_score:
                        compare["source_score_4b"] = compare["fused_4b"]
                    if independent:
                        compare["independent_4b"] = compare["fused_4b"]
                outputs.append((intended, absent, before, compare))
    return outputs


def cuts(rows, method):
    if method == "fused_ridge":
        return 0.
    return max((row["gap"] for _, _, _, values in rows if (row := values.get(method))
                and row["issued"] and not row["exact"]), default=0.)


def evaluate(rows, method, cutoff):
    stats = defaultdict(Counter)
    latency = defaultdict(list)
    affected = defaultdict(set)
    for group, absent, baseline, alternatives in rows:
        context = "present" if absent is None else "withdrawn"
        row = alternatives.get(method)
        final = baseline
        if row:
            if row["tokens"]:
                stats[context].update(role_calls=1, role_tokens=row["tokens"])
                latency[context].append(row["ms"])
            if row["issued"] and row["gap"] > cutoff:
                final = row
        extra = final is not baseline
        stats[context].update(requests=1, correct=int(final["exact"]),
                              wrong=int(final["issued"] and not final["exact"]),
                              new_correct=int(extra and final["exact"]),
                              new_wrong_sources=int(extra and final["reason"] == "source"),
                              new_wrong_roles=int(extra and final["reason"] == "role"))
        if extra and not final["exact"]:
            affected[context].add(group)
    return {name: dict(count, adverse_added_groups=len(affected[name]),
                       recheck_p95_ms=float(np.quantile(latency[name], .95)) if latency[name] else None)
            for name, count in stats.items()}


def run(args):
    definitions, _, _ = base.queries(args.glossary, args.training)
    reader = PairedReader(args.reader, definitions)
    reference = shared.role.read(args.previous)["outcomes"]
    report = {}
    for entrance, _, index, encoder, panel, plans, predictions, role_cut, footprints, model, threshold in shared.prepared(args):
        output = {split: gather(index, encoder, panel[split], plans[split], entrance, predictions,
                                role_cut, footprints, model, threshold["author_role"], reader, args.reverse)
                  for split in ("calibration", "test")}
        test = output["test"]
        baseline = evaluate(test, "baseline", 0.)
        control = reference[entrance]["tested"]["elife"]["natural"]
        for frozen_name, current in (("present", "present"), ("missing", "withdrawn")):
            expected = control[frozen_name]["author_role"]
            if (expected["strict"], expected["wrong"]) != (baseline[current]["correct"], baseline[current]["wrong"]):
                raise ValueError("Archived source-first controls did not match")
        report[entrance] = {"baseline": baseline}
        for method in METHODS:
            cutoff = cuts(output["calibration"], method)
            report[entrance][method] = dict(
                cutoff=cutoff, open=evaluate(test, method, 0.), calibrated=evaluate(test, method, cutoff),
                calibration=evaluate(output["calibration"], method, cutoff),
            )
    base.write_json(args.output, dict(
        outcomes=report, reader=reader.provenance["revision"],
        helper_sha256=base.digest(Path(shared.__file__)), script_sha256=base.digest(Path(__file__)),
        manifest_sha256=base.digest(args.prospective_plos), previous_sha256=base.digest(args.previous),
        gpu=torch.cuda.get_device_name(), reverse=args.reverse,
    ))


if __name__ == "__main__":
    parser = shared.command(__doc__)
    parser.add_argument("--reader", type=Path, required=True)
    run(parser.parse_args())
