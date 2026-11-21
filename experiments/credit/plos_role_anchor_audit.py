"""Attribute natural proof losses and test role-aware anchor allocation."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.credit import contrastive_rescue_identity as shared
from experiments.credit.engineering_contribution_panel import normalize

route, base, prior = shared.route, shared.base, shared.prior
ARMS = ("baseline", "role_first", "three_anchors")


def allocate(index, topic, clues, absent, labels, footprints, arm):
    articles = index.catalog.articles
    work, visited = 0, 0
    options = set()
    for side in range(2):
        menus = []
        terms = sorted(set(normalize(clues[1 - side]).split()) - route.KEYWORD_STOP)
        for position, anchor in enumerate(topic[side]["fused"][:3]):
            neighbors, reads = index.target(anchor, labels[side], labels[1 - side], "author_role")
            neighbors -= {anchor, absent}
            work += reads
            visited += len(neighbors)
            if not neighbors:
                continue
            ranked = route.rrf(index, topic[1 - side], terms, neighbors)
            menus.extend((rank + position, position, anchor, other) for rank, other in enumerate(ranked))
            if arm == "role_first":
                break
        for _, _, anchor, other in sorted(menus)[:route.DISCOVERY_CAP]:
            options.add((anchor, other) if side == 0 else (other, anchor))

    features = {}
    for pair in sorted(options):
        if route.applicable(articles, pair, labels):
            a, b = (route.source_features(topic, clues, footprints, doc, side)
                    for side, doc in enumerate(pair))
            features[pair] = np.r_[np.minimum(a, b), np.maximum(a, b)]
    return features, visited, work


def top_candidate(options, model, cutoff):
    return max(((pair, score) for pair, score in route.scored(options, model) if score > cutoff),
               key=lambda item: item[1], default=(None, None))[0]


def positive_stage(topic, pair, labels, gold, confidence, initial, old, model, threshold):
    if not confidence:
        return "role_below_cutoff"
    if labels != gold:
        return "role_wrong"
    if initial is not None:
        return "shallow_correct" if initial == pair else "shallow_wrong"
    if pair in old:
        return ("rescued" if top_candidate(old, model, threshold) == pair
                else "scored_without_exact_recovery")
    if any(doc == topic[side]["fused"][0] for side, doc in enumerate(pair)):
        return "top1_anchor_exact_pair_omitted"
    return "no_top1_exact_anchor"


def replay(index, encoder, panel, plan, entrance, predictions, role_cut, footprints, model, cutoff, reverse):
    stage = Counter()
    trial_set = {arm: [] for arm in ARMS}
    articles = index.catalog.articles
    for group in sorted(panel["pairs"], key=lambda row: row["papers"], reverse=reverse):
        pair = tuple(group["papers"])
        clues = [articles[doc]["clue"] for doc in pair]
        if entrance == "keywords":
            clues = [index.catalog.keywords(clue) for clue in clues]
        embedded = encoder.encode(base.search.topic_prompts(clues))
        for absent in (None, *pair):
            topic = index.catalog.rank(embedded, clues, absent, depth=route.DEEP)
            trials = list(route.choices(index, topic, clues, absent, plan[pair], predictions,
                                        role_cut, footprints, False))
            rows = {arm: [] for arm in ARMS}
            for (_, reference_roles, phrases), (gold, labels, initial, old_routes) in zip(plan[pair], trials, strict=True):
                positive = bool(base.known_owners(articles, pair, reference_roles))
                confidence = min(predictions[phrase][1] for phrase in phrases) > role_cut
                old = old_routes.get("author_role", ({}, 0, 0))[0]
                if absent is None and positive:
                    stage.update(positives=1, both_top6=int(all(doc in lane["fused"][:route.SHALLOW]
                                                                  for doc, lane in zip(pair, topic, strict=True))),
                                 both_top18=int(all(doc in lane["fused"] for doc, lane in zip(pair, topic, strict=True))))
                    stage["first_loss_" + positive_stage(topic, pair, labels, gold, confidence, initial,
                                                         old, model, cutoff)] += 1
                rows["baseline"].append((gold, labels, initial, old_routes))
                for arm in ARMS[1:]:
                    revised = old_routes
                    if "author_role" in old_routes:
                        option = allocate(index, topic, clues, absent, labels, footprints, arm)
                        revised = old_routes | {"author_role": option}
                    rows[arm].append((gold, labels, initial, revised))
            for arm, current in rows.items():
                trial_set[arm].append((pair, absent, current))
    if sum(value for key, value in stage.items() if key.startswith("first_loss_")) != stage["positives"]:
        raise ValueError("Positive loss categories must partition the input")
    return trial_set, dict(stage)


def run(args):
    report = {}
    archived = shared.role.read(args.previous)["outcomes"]
    for entrance, articles, index, encoder, panel, plan, predictions, role_cut, footprints, model, threshold in shared.prepared(args):
        options, audit = {}, {}
        for split in ("calibration", "test"):
            options[split], audit[split] = replay(
                index, encoder, panel[split], plan[split], entrance, predictions, role_cut,
                footprints, model, threshold["author_role"], args.reverse,
            )
        comparison = {}
        for arm in ARMS:
            rows = options["test"][arm]
            calibrated, calibration_info = shared.previous.calibration(options["calibration"][arm], model, articles)
            frozen = route.evaluate(rows, model, threshold, articles)
            tuned = route.evaluate(rows, model, threshold | calibrated, articles)
            if arm == "baseline" and frozen != archived[entrance]["tested"]["elife"]["natural"]:
                raise ValueError("Archived natural baseline did not reproduce")
            comparison[arm] = dict(
                frozen={key: value["author_role"] for key, value in frozen.items()},
                tuned={key: value["author_role"] for key, value in tuned.items()},
                candidates=shared.candidate_counts(rows, articles)["author_role"],
                changed_cutoff=calibrated.get("author_role"),
                calibration=calibration_info,
            )
        report[entrance] = dict(stages=audit, methods=comparison)
    base.write_json(args.output, dict(
        outcomes=report, script_sha256=base.digest(Path(__file__)),
        helper_sha256=base.digest(Path(shared.__file__)), previous_sha256=base.digest(args.previous),
        manifest_sha256=base.digest(args.prospective_plos), gpu=torch.cuda.get_device_name(),
        reverse=args.reverse,
    ))


if __name__ == "__main__":
    run(shared.command(__doc__).parse_args())
