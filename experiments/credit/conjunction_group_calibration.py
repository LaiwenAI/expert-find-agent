"""Calibrate complete same-author proofs on source-disjoint publication groups."""

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
from experiments.credit import raw_conjunction_audit as raw
from experiments.credit.engineering_contribution_panel import normalize
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint

ARMS = raw.ARMS


def prepare_elife(manifest, notes, wording, sensors):
    phrases = defaultdict(list)
    held_words = {normalize(row[3]) for row in sensors}
    for role, _, phrase in wording["niso"]:
        if normalize(phrase) not in held_words:
            phrases[ROLES[role]].append(phrase)
    all_articles, all_owners = set(), set()
    panels, views = {}, {}
    for name, panel in manifest["panels"].items():
        for doi in sorted(panel["articles"]):
            article = notes[doi]
            owners = list(article)
            aliases = {i: article[owner]["alias"] for i, owner in enumerate(owners)}
            blocks = [dict(text=article[owner]["text"], owners=[i], roles=article[owner]["roles"])
                      for i, owner in enumerate(owners)]
            views[doi] = dict(aliases=aliases, blocks=blocks, slots={owner: i for i, owner in enumerate(owners)})
        truth, groups, units = {}, {}, []
        for set_id, item in enumerate(panel["pairs"]):
            a, b = item["papers"]
            found = set(notes[a]) | set(notes[b])
            if {a, b} & all_articles or found & all_owners:
                raise ValueError("Test and calibration groups are not owner-disjoint")
            all_articles.update((a, b))
            all_owners.update(found)
            common = sorted(set(notes[a]) & set(notes[b]))
            if not common:
                raise ValueError("Frozen linked group lost its cross-paper identity")
            contrasts = item["contrasts"] if "contrasts" in item else [(roles, roles[::-1]) for roles in item["bindings"]]
            for pos, (binding, negative) in enumerate(contrasts):
                family = f"{name}-{set_id}-{pos}"
                for variant, roles in (("present", binding), ("swap", negative)):
                    key = family, variant
                    truth[key] = set()
                    groups[key] = (a, b)
                    for owner in common:
                        labels = [role in notes[paper][owner]["roles"]
                                  for paper, role in zip((a, b), roles, strict=True)]
                        if all(labels):
                            truth[key].add(owner)
                        for i, (paper, role, label) in enumerate(zip((a, b), roles, labels, strict=True)):
                            bank = phrases[role] or [raw.previous.DESCRIPTIONS[ROLES.index(role)]]
                            phrase = min(bank, key=lambda text: raw.previous.stable(family, role, text))
                            units.append((paper, views[paper]["slots"][owner], role, phrase, label, (*key, owner, i)))
                    if bool(truth[key]) != (variant == "present"):
                        raise ValueError("Raw labels differ from frozen construction")
        panels[name] = dict(units=units, truth=truth, groups=groups)
    if {normalize(row[3]) for panel in panels.values() for row in panel["units"]} & held_words:
        raise ValueError("Calibration contains an exact Sensors test phrase")
    return panels, views


def join_scores(units, scores):
    owners = {arm: defaultdict(dict) for arm in ARMS}
    for arm in ARMS:
        for row, score in zip(units, scores[arm], strict=True):
            family, variant, owner, side = row[-1]
            key = family, variant
            owners[arm][key].setdefault(owner, {})[side] = float(score)
    joined = {}
    for arm, requests in owners.items():
        joined[arm] = {}
        for key, people in requests.items():
            if any(set(sides) != {0, 1} for sides in people.values()):
                raise ValueError("Incomplete proof score")
            joined[arm][key] = {owner: min(sides.values()) for owner, sides in people.items()}
    return joined


def risk_cut(scores, panel):
    cut = {}
    for arm, cases in scores.items():
        by_group = defaultdict(list)
        for request, owners in cases.items():
            group = panel["groups"][request]
            by_group[group].extend(score for owner, score in owners.items() if owner not in panel["truth"][request])
        if any(not values for values in by_group.values()):
            raise ValueError("Each calibration group needs one unsupported proof")
        worst = sorted((max(values) for values in by_group.values()), reverse=True)
        cut[arm] = worst[len(worst) // 20]
    return cut


def evaluate(scores, panel, thresholds):
    result = {}
    for arm, cases in scores.items():
        counts, adverse = Counter(), set()
        for key, options in cases.items():
            actual = {owner for owner, score in options.items() if score > thresholds[arm]}
            truth = panel["truth"][key]
            wrong = bool(actual - truth)
            counts.update(queries=1, positives=bool(truth), correct=bool(actual & truth),
                          exact_positive=bool(truth) and actual == truth, issued=bool(actual), wrong=wrong,
                          wrong_negative=not truth and bool(actual))
            if wrong:
                adverse.add(panel["groups"][key])
        result[arm] = dict(counts, groups=len(set(panel["groups"].values())), adverse_groups=len(adverse))
    return result


def compile_request(units, views):
    contexts = []
    for paper, author, _, phrase, _, _ in units:
        article = views[paper]
        alias = article["aliases"].get(str(author), article["aliases"].get(author))
        scopes = [block for block in article["blocks"] if author in block["owners"]]
        visible = {role for block in scopes for role in block["roles"]}
        prompt = f"Target author: {alias}. Requested contribution: {phrase}."
        contexts.append((prompt, "\n".join(block["text"] for block in scopes), [role in visible for role in ROLES]))
    return contexts


def live_replay(panel, units, views, encoder, reader, banks, model, thresholds, reference, reverse):
    requests = defaultdict(list)
    for unit in units:
        requests[unit[-1][:2]].append(unit)
    recorded = {arm: {} for arm in ("role_index", "fusion")}
    timings, counts = defaultdict(list), Counter()
    for i, key in enumerate(reversed(requests) if reverse else requests):
        clauses = requests[key]
        phrase_list = list(dict.fromkeys(row[3] for row in clauses))
        contexts = compile_request(clauses, views)
        members = np.array([row[2] for row in contexts])
        encoded_index = {phrase: j for j, phrase in enumerate(phrase_list)}
        decision_order = tuple(recorded) if (i + int(reverse)) % 2 == 0 else tuple(reversed(recorded))
        for arm in decision_order:
            started = perf_counter()
            vectors = raw.encode(encoder, phrase_list)
            projected = []
            for j, row in enumerate(clauses):
                _, options = raw.project(vectors[encoded_index[row[3]]], banks, members[j:j+1])
                projected.append((options["ridge"][0], options["definition"][0]))
            if arm == "fusion":
                inspected, clipped = reader.score_pairs([row[:2] for row in contexts], task=raw.previous.TASK)
                if clipped or len(inspected) != len(clauses):
                    raise ValueError("Incomplete live verification")
                values = raw.estimate(model, np.column_stack((inspected, projected)))
                counts[arm] += len(inspected)
            else:
                values = np.asarray(projected)[:, 0]
            scores = defaultdict(list)
            for clause, score in zip(clauses, values, strict=True):
                scores[clause[-1][2]].append(float(score))
            answer = {owner for owner, sides in scores.items() if min(sides) > thresholds[arm]}
            json.dumps(sorted(answer))
            timings[arm].append((perf_counter()-started)*1000)
            recorded[arm][key] = answer
    return {arm: dict(
        p50_ms=float(np.quantile(timings[arm], .5)), p95_ms=float(np.quantile(timings[arm], .95)),
        correct=sum(bool(answer & panel["truth"][key]) for key, answer in answers.items()),
        wrong=sum(bool(answer - panel["truth"][key]) for key, answer in answers.items()),
        changes=sum(answer != {owner for owner,score in reference[arm][key].items() if score > thresholds[arm]}
                    for key, answer in answers.items()),
        requests=len(answers), reranker_checks=counts[arm],
    ) for arm, answers in recorded.items()}


def atomic_baseline(rows, sensors, values, archived, truth):
    dev = raw.withhold_link_phrases(rows, sensors, archived["wording"])
    fronts = [row for row in dev if row[-1] == ("frontiers",)]
    keys = sorted({row[0] for row in fronts}, key=lambda p: raw.previous.stable("raw-conjunction", p))
    fit_keys = set(keys[:2*len(keys)//3])
    train = [row for row in fronts if row[0] in fit_keys]
    cal = [row for row in fronts if row[0] not in fit_keys]
    model = raw.fit(raw.matrix(train, values)[:, 1:], np.array([row[4] for row in train]))
    cal_scores = raw.routes(raw.matrix(cal, values), model)
    gold = np.array([row[4] for row in cal])
    thresholds = {arm: raw.calibrated(score, gold) for arm, score in cal_scores.items()}
    baseline = archived["matched_register_calibration"]
    sensor_scores = raw.routes(raw.matrix(sensors, values), model)
    if thresholds != baseline["thresholds"] or raw.proofs(sensors, sensor_scores, thresholds, truth) != baseline["linked"]:
        raise ValueError("Atomic baseline did not match the previous frozen run")
    return model, thresholds, sensor_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "linked", "link_panel", "links_manifest", "glossary", "training", "embedding",
                 "reranker", "frozen", "notes", "baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--component-timing", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a Slurm GPU and fresh output")
    seed, links = raw.load(args.seed, raw.PANEL), raw.load(args.link_panel, raw.LINKS)
    files = json.loads(args.links_manifest.read_bytes())
    notes, frozen, archive = (json.loads(p.read_bytes()) for p in (args.notes, args.frozen, args.baseline))
    raw_views = raw.publisher_views(seed, args.linked, files)
    definitions, phrases, wording = raw.queries(args.glossary, args.training)
    sensors = list(raw.linked_units(raw_views, args.linked, links, wording))
    relevant = {paper for request in links["queries"] for paper in request["papers"]}
    sensor_keys = {person for paper in relevant for person in links["articles"][paper]["roles"]}
    if sensor_keys & {row["engineering_key"] for article in notes.values() for row in article.values()}:
        raise ValueError("The Sensors test intersects an eLife source group by verified identity")
    sensor_truth = {(q["family"], q["variant"]): set(q["gold"]) for q in links["queries"]}
    sensor_groups = {(q["family"], q["variant"]): tuple(q["papers"]) for q in links["queries"]}
    panels, views = prepare_elife(frozen, notes, wording, sensors)
    embed = verify_checkpoint(args.embedding)
    encoder, reader = QwenTorchEmbedding(args.embedding), Reranker(args.reranker)
    banks = raw.projections(encoder, definitions, phrases)
    atoms = list(raw.atomic_units(seed))
    alternates = raw.withhold_link_phrases(atoms, sensors, wording)
    values = raw.margin_rows([*atoms, *alternates, *sensors], raw_views, reader, encoder, banks, args.reverse)
    model, atomic, predictions = atomic_baseline(atoms, sensors, values, archive | {"wording": wording}, sensor_truth)
    rows = [row for panel in panels.values() for row in panel["units"]]
    margins = raw.margin_rows(rows, views, reader, encoder, banks, args.reverse)
    joined = {key: join_scores(panel["units"], raw.routes(raw.matrix(panel["units"], margins), model))
              for key, panel in panels.items()}
    cut = risk_cut(joined["calibration"], panels["calibration"])
    panels["sensors"] = dict(truth=sensor_truth, groups=sensor_groups)
    joined["sensors"] = join_scores(sensors, predictions)
    results = {
        key: {policy: evaluate(scores, panels[key], threshold)
              for policy, threshold in (("atomic", atomic), ("group_calibrated", cut))}
        for key, scores in joined.items()
    }
    timings = {}
    if args.component_timing:
        for key, units, sources in (("held", panels["held"]["units"], views), ("sensors", sensors, raw_views)):
            timings[key] = live_replay(panels[key], units, sources, encoder, reader, banks, model, cut,
                                       joined[key], args.reverse)
    write_json(args.output, dict(
        results=results, live_component=timings, cutoffs=dict(atomic=atomic, group_calibrated=cut),
        source_manifest_sha256=digest(args.frozen), raw_notes_sha256=digest(args.notes),
        baseline_sha256=digest(args.baseline), script_sha256=digest(Path(__file__)),
        embedding=embed["revision"], reranker=reader.provenance["revision"], gpu=torch.cuda.get_device_name(),
        job=os.environ["SLURM_JOB_ID"], reverse=args.reverse,
    ))


if __name__ == "__main__":
    main()
