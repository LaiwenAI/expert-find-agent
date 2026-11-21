"""Test selective original-text review under matched proof-check budgets."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import conjunction_group_calibration as base
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint

raw = base.raw
ROUTERS = ("hash", "margin", "disagreement", "learned_gain")


def signature(units):
    return sha256(json.dumps([list(unit[:2]) + [unit[3]] for unit in units]).encode()).hexdigest()


def request_index(units):
    requests = defaultdict(list)
    for unit in units:
        requests[unit[-1][:2]].append(unit)
    return requests


def owners(units, sides):
    values = defaultdict(list)
    for unit, row in zip(units, sides, strict=True):
        values[unit[-1][2]].append(tuple(row))
    if any(len(rows) != 2 for rows in values.values()):
        raise ValueError("Expected two clauses for every linked author")
    return values


def cheap_features(values, cut, definition_cut):
    array = np.asarray(list(values.values()))
    proof = array.min(axis=1)
    index = proof[:, 0]
    definition = proof[:, 1]
    margin = float(np.min(np.abs(index - cut)))
    signs = index > cut
    disagreement = int(np.sum(signs != (definition > definition_cut)))
    current = int(signs.sum())
    features = [
        index.max() - cut, margin, definition.max() - definition_cut,
        disagreement, current, len(index), float(np.mean(np.abs(array[:, 0, 0] - array[:, 1, 0]))),
    ]
    return np.asarray(features)


def summarize(decisions, truth, groups):
    counts = Counter()
    adverse = set()
    for key, predicted in decisions.items():
        target = truth[key]
        wrong = bool(predicted - target)
        counts.update(requests=1, supported=bool(predicted & target), wrong=wrong,
                      wrong_negative=not target and bool(predicted), issued=bool(predicted))
        if wrong:
            adverse.add(groups[key])
    return dict(counts, adverse_groups=len(adverse))


def outcome_delta(original, reviewed, target):
    gain = int(bool(reviewed & target)) - int(bool(original & target))
    repaired = int(bool(original - target)) - int(bool(reviewed - target))
    return gain + 2 * repaired


def learn(features, original, reviewed, panel):
    keys = list(features)
    matrix = np.stack([features[key] for key in keys])
    target = np.array([outcome_delta(original[key], reviewed[key], panel["truth"][key]) for key in keys])
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(keys)), (matrix - center) / scale))
    group_sizes = Counter(panel["groups"].values())
    weights = np.array([len(keys) / len(group_sizes) / group_sizes[panel["groups"][key]] for key in keys])
    penalty = np.diag([0.] + [10.] * matrix.shape[1])
    coefficients = np.linalg.solve((design.T * weights) @ design + penalty, design.T @ (weights * target))
    return center, scale, coefficients


def priorities(vector, model):
    center, scale, coefficients = model
    score = float(coefficients[0] + ((vector - center) / scale) @ coefficients[1:])
    return (0., -float(vector[1]), float(vector[3]), score)


def strata(requests, features, policy):
    groups = defaultdict(list)
    for key, rows in requests.items():
        groups[len(rows)].append((features[key][policy], signature(rows), key))
    return groups


def quota(n):
    # Integer rounding is fixed independently of scores and labels.
    return n // 4


def select_matched(requests, scores, policy):
    groups = strata(requests, scores, policy)
    return {row[2] for entries in groups.values() for row in sorted(entries, reverse=True)[:quota(len(entries))]}


def freeze_policy(requests, scores, policy):
    limits = {}
    for cost, entries in strata(requests, scores, policy).items():
        if keep := quota(len(entries)):
            limits[cost] = sorted(entries, reverse=True)[keep - 1][:2]
    return limits


def uses_reader(rows, rank, cutoff):
    limit = cutoff.get(len(rows))
    return limit is not None and (rank, signature(rows)) >= tuple(limit)


def replay(requests, index, fusion, panel, reviewed):
    output = {key: fusion[key] if key in reviewed else index[key] for key in requests}
    return dict(
        summarize(output, panel["truth"], panel["groups"]),
        reviewed=len(reviewed), reranker_checks=sum(len(requests[key]) for key in reviewed),
        changed=len([key for key in output if output[key] != index[key]]),
    )


def decision_scores(requests, values, thresholds, fusion_model):
    indexed, fused, features = {}, {}, {}
    for key, units in requests.items():
        mat = raw.matrix(units, values)
        proof_margins = owners(units, mat[:, 2:4])
        fused_margins = owners(units, raw.estimate(fusion_model, mat[:, 1:])[:, None])
        indexed[key] = {person for person, rows in proof_margins.items() if min(row[0] for row in rows) > thresholds["role_index"]}
        fused[key] = {person for person, rows in fused_margins.items() if min(row[0] for row in rows) > thresholds["fusion"]}
        features[key] = proof_margins
    return indexed, fused, features


def suite(panels, values, thresholds, model):
    requests = {key: request_index(panel["units"]) for key, panel in panels.items()}
    scored = {key: decision_scores(req, values[key], thresholds, model) for key, req in requests.items()}
    calibration = panels["calibration"]
    dev_index, dev_fusion, dev_raw = scored["calibration"]
    # On development, give the definition-only score its own group risk threshold.
    defin = {"definition": {key: {person: min(value[1] for value in rows) for person, rows in by_owner.items()}
                            for key, by_owner in dev_raw.items()}}
    definition_cut = base.risk_cut(defin, calibration)["definition"]
    feature_rows = {
        key: {request: cheap_features(values, thresholds["role_index"], definition_cut)
              for request, values in members.items()}
        for key, (_, _, members) in scored.items()
    }
    router = learn(feature_rows["calibration"], dev_index, dev_fusion, calibration)
    scores = {key: {request: dict(zip(ROUTERS, priorities(row, router), strict=True))
                    for request, row in matrix.items()} for key, matrix in feature_rows.items()}
    limits = {arm: freeze_policy(requests["calibration"], scores["calibration"], arm) for arm in ROUTERS}
    results = {}
    for name, panel in panels.items():
        req = requests[name]
        index, fusion, _ = scored[name]
        matched = {arm: select_matched(req, scores[name], arm) for arm in ROUTERS}
        frozen = {arm: {key for key, rows in req.items() if uses_reader(rows, scores[name][key][arm], limits[arm])}
                  for arm in ROUTERS}
        matched_rows = {arm: replay(req, index, fusion, panel, selected) for arm, selected in matched.items()}
        if len({(row["reviewed"], row["reranker_checks"]) for row in matched_rows.values()}) != 1:
            raise ValueError("Unmatched reranker costs")
        results[name] = dict(
            index=replay(req, index, fusion, panel, set()), fusion=replay(req, index, fusion, panel, req.keys()),
            matched=matched_rows,
            frozen={arm: replay(req, index, fusion, panel, selected) for arm, selected in frozen.items()},
        )
    return results, limits, router, definition_cut


def timed(panel, views, encoder, reader, banks, model, thresholds, limits, router, definition_cut, reverse):
    requests = request_index(panel["units"])
    timings, checks, recorded, decisions = defaultdict(list), Counter(), defaultdict(dict), defaultdict(set)
    for i, key in enumerate(reversed(requests) if reverse else requests):
        rows = requests[key]
        phrases = list(dict.fromkeys(row[3] for row in rows))
        contexts = base.compile_request(rows, views)
        member = np.array([row[2] for row in contexts])
        positions = {phrase: j for j, phrase in enumerate(phrases)}
        arms = ("index", "margin", "disagreement", "disagreement_veto", "learned_gain", "fusion")
        index = (i + int(reverse)) % len(arms)
        for arm in arms[index:] + arms[:index]:
            started = perf_counter()
            vectors = raw.encode(encoder, phrases)
            projected = []
            for j, row in enumerate(rows):
                _, components = raw.project(vectors[positions[row[3]]], banks, member[j:j+1])
                projected.append((components["ridge"][0], components["definition"][0]))
            projected = np.asarray(projected)
            indexed = owners(rows, projected)
            indexed_answer = {person for person, sides in indexed.items()
                              if min(side[0] for side in sides) > thresholds["role_index"]}
            route = arm.removesuffix("_veto")
            veto = route != arm
            use = arm == "fusion"
            if route in limits:
                features = cheap_features(indexed, thresholds["role_index"], definition_cut)
                priority = priorities(features, router)[ROUTERS.index(route)]
                use = uses_reader(rows, priority, limits[route]) and (not veto or bool(indexed_answer))
            if use:
                checked, clipped = reader.score_pairs([row[:2] for row in contexts], task=raw.previous.TASK)
                if clipped or len(checked) != len(rows):
                    raise ValueError("Incomplete online review")
                values = raw.estimate(model, np.column_stack((checked, projected)))
                checks[arm] += len(rows)
                decisions[arm].add(key)
                answer = {person for person, sides in owners(rows, values[:, None]).items()
                          if min(side[0] for side in sides) > thresholds["fusion"]}
                if veto:
                    answer &= indexed_answer
            else:
                answer = indexed_answer
            json.dumps(sorted(answer))
            timings[arm].append((perf_counter() - started) * 1000)
            recorded[arm][key] = answer
    return {arm: dict(
        summarize(outcomes, panel["truth"], panel["groups"]),
        reviewed=len(decisions[arm]), reranker_checks=checks[arm], p95_ms=float(np.quantile(timings[arm], .95)),
        p50_ms=float(np.quantile(timings[arm], .50)),
        signature=sha256(json.dumps([[key, sorted(outcomes[key])] for key in sorted(outcomes)]).encode()).hexdigest(),
    ) for arm, outcomes in recorded.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "linked", "link_panel", "manifest", "glossary", "training", "embedding", "reranker",
                 "frozen", "notes", "atomic_archive", "group_archive", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    for option in ("external-manifest", "external-notes", "router-archive"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    external_paths = (args.external_manifest, args.external_notes, args.router_archive)
    if any(external_paths) and not all(external_paths):
        parser.error("Provide the external manifest, notes, and frozen router archive together")
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a Slurm GPU and fresh output")
    seed, links = raw.load(args.seed, raw.PANEL), raw.load(args.link_panel, raw.LINKS)
    atomic, archive, files, frozen, notes = (json.loads(p.read_bytes()) for p in (
        args.atomic_archive, args.group_archive, args.manifest, args.frozen, args.notes,
    ))
    if digest(args.frozen) != archive["source_manifest_sha256"] or digest(args.notes) != archive["raw_notes_sha256"]:
        raise ValueError("Changed frozen groups or original notes")
    original = raw.publisher_views(seed, args.linked, files)
    definitions, phrases, wording = raw.queries(args.glossary, args.training)
    sensors = list(raw.linked_units(original, args.linked, links, wording))
    panel, projected = base.prepare_elife(frozen, notes, wording, sensors)
    embed = verify_checkpoint(args.embedding)
    encoder, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    if embed["revision"] != archive["embedding"] or reader.provenance["revision"] != archive["reranker"]:
        raise ValueError("Changed model artifacts")
    banks = raw.projections(encoder, definitions, phrases)
    atoms = list(raw.atomic_units(seed))
    alternates = raw.withhold_link_phrases(atoms, sensors, wording)
    sensor_values = raw.margin_rows([*atoms, *alternates, *sensors], original, reader, encoder, banks, args.reverse)
    sensor_truth = {(row["family"], row["variant"]): set(row["gold"]) for row in links["queries"]}
    fusion, _, _ = base.atomic_baseline(atoms, sensors, sensor_values, atomic | {"wording": wording}, sensor_truth)
    elife_units = [unit for item in panel.values() for unit in item["units"]]
    elife_values = raw.margin_rows(elife_units, projected, reader, encoder, banks, args.reverse)
    panel["sensors"] = dict(units=sensors, truth=sensor_truth,
        groups={(row["family"], row["variant"]): tuple(row["papers"]) for row in links["queries"]})
    thresholds = archive["cutoffs"]["group_calibrated"]
    measurements = dict(calibration=elife_values, held=elife_values, sensors=sensor_values)
    live_panels = (("held", projected), ("sensors", original))
    if args.external_manifest is not None:
        external = json.loads(args.external_manifest.read_bytes())
        external_notes = json.loads(args.external_notes.read_bytes())
        indexed_ids = {item["engineering_key"] for source in notes.values() for item in source.values()}
        indexed_ids.update(person for paper in links["articles"].values() for person in paper["roles"])
        if indexed_ids & {item["engineering_key"] for source in external_notes.values() for item in source.values()}:
            raise ValueError("An external author overlaps the prior proof corpus")
        new_panels, new_views = base.prepare_elife(external, external_notes, wording, sensors)
        if len(new_panels) != 1 or new_panels.keys() & panel.keys():
            raise ValueError("External manifest must add one new panel")
        name, extra = next(iter(new_panels.items()))
        external_units = extra["units"]
        measurements[name] = raw.margin_rows(external_units, new_views, reader, encoder, banks, args.reverse)
        panel.update(new_panels)
        live_panels = ((name, new_views),)
    reports, limits, router, definition = suite(panel, measurements, thresholds, fusion)
    if args.router_archive is not None:
        previous = json.loads(args.router_archive.read_bytes())
        if (json.loads(json.dumps(limits)) != previous["router_limits"]
                or router[2].tolist() != previous["router_weights"] or definition != previous["definition_cutoff"]
                or any(reports[key] != values for key, values in previous["reports"].items())):
            raise ValueError("Previously frozen controls did not reproduce")
    for key in archive["results"]:
        result = reports[key]
        for arm, route in (("index", "role_index"), ("fusion", "fusion")):
            gold = archive["results"][key]["group_calibrated"][route]
            summary = result[arm]
            if (summary["supported"], summary["wrong"], summary["adverse_groups"]) != (
                gold["correct"], gold["wrong"], gold["adverse_groups"]
            ):
                raise ValueError("Changed archived full-route result")
    live = {key: timed(panel[key], views, encoder, reader, banks, fusion, thresholds,
                      limits, router, definition, args.reverse)
            for key, views in live_panels}
    external_inputs = {item.name: digest(item) for item in external_paths if item is not None}
    write_json(args.output, dict(reports=reports, live=live, router_limits=limits, definition_cutoff=definition,
        external_inputs=external_inputs,
        router_weights=router[2].tolist(), archive_sha256=digest(args.group_archive),
        script_sha256=digest(Path(__file__)), job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        reverse=args.reverse))


if __name__ == "__main__":
    main()
