"""Test whether a recovered paper is identifiable independently of its author."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import torch

from experiments.credit import plos_recoverable_source_calibration as previous

prior, route, base, role = previous.prior, previous.route, previous.base, previous.role
GUARDS = ("unfiltered", "top3_cross_cosine", "coverage_floor", "unique_coverage")
FLOOR = .5


class LexicalIdentity:
    def __init__(self, footprints, clues):
        postings = defaultdict(list)
        own = [footprints.footprint(clues[i], clues[1 - i])[0] for i in range(2)]
        relevant = set(own[0]) | set(own[1])
        for paper, (words, _) in footprints.index.items():
            for word in words & relevant:
                postings[word].append(paper)
        self.sides = []
        for weights in own:
            total = sum(weights.values())
            counts = defaultdict(float)
            if total:
                for word, value in weights.items():
                    for paper in postings[word]:
                        counts[paper] += value / total
            self.sides.append(counts)

    def filters(self, topic, absent):
        leaders = []
        for counts in self.sides:
            ranked = sorted(((value, paper) for paper, value in counts.items() if paper != absent),
                            key=lambda item: (-item[0], item[1]))
            leaders.append(ranked[1][0] if len(ranked) > 1 else 0.)

        def pass_pair(pair, name):
            if name == "top3_cross_cosine":
                return all(
                    doc in topic[side]["fused"][:3]
                    and topic[side]["cosine"][doc] > topic[1 - side]["cosine"][doc]
                    for side, doc in enumerate(pair)
                )
            threshold = FLOOR
            if any(scores.get(doc, 0.) < threshold for scores, doc in zip(self.sides, pair, strict=True)):
                return False
            return name == "coverage_floor" or all(
                scores.get(doc, 0.) > competitor + 1e-9
                for scores, doc, competitor in zip(self.sides, pair, leaders, strict=True)
            )

        return pass_pair


def filter_cases(cases, offer, guard):
    if guard == "unfiltered":
        return cases
    return [(gold, labels, initial, {
        arm: ({pair: vector for pair, vector in candidates.items() if offer(pair, guard)}, work, read)
        for arm, (candidates, work, read) in routes.items()
    }) for gold, labels, initial, routes in cases]


def gather(index, encoder, groups, plan, entrance, predictions, cut, footprints, reverse):
    collected = {guard: dict(natural=[], challenge=[]) for guard in GUARDS}
    for group in sorted(groups, key=lambda row: row["papers"], reverse=reverse):
        pair = tuple(group["papers"])
        clues = [index.catalog.articles[paper]["clue"] for paper in pair]
        if entrance == "keywords":
            clues = [index.catalog.keywords(text) for text in clues]
        embedding = encoder.encode(base.search.topic_prompts(clues))
        identity = LexicalIdentity(footprints, clues)

        def record(mode, absent, topic):
            cases = list(route.choices(index, topic, clues, absent, plan[pair], predictions, cut, footprints, False))
            offer = identity.filters(topic, absent)
            for guard in GUARDS:
                collected[guard][mode].append((pair, absent, filter_cases(cases, offer, guard)))

        natural = None
        for absent in (None, *pair):
            topic = index.catalog.rank(embedding, clues, absent, depth=route.DEEP)
            record("natural", absent, topic)
            if absent is None:
                natural = topic
        for side, paper in enumerate(pair):
            changed = previous.demote(natural, side, paper)
            if changed is not None:
                record("challenge", None, changed)
    return collected


def candidate_counts(rows, articles):
    result = defaultdict(Counter)
    for papers, absent, cases in rows:
        for gold, labels, _, options in cases:
            for arm, (pairs, _, _) in options.items():
                for pair in pairs:
                    correct = absent is None and route.result_is_correct(articles, papers, pair, gold, labels)
                    result[arm].update(correct=int(correct), wrong=int(not correct))
    return {arm: dict(result[arm]) for arm in route.METHODS}


def run(args):
    if args.output.exists() or not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and Slurm GPU")
    original = role.read(args.elife_groups)["panels"]
    role_ref, policy_ref = role.read(args.role_archive), role.read(args.elife_policy)
    expected = role.read(args.previous)["outcomes"]
    p_ref, panel = role.read(args.prospective_result), role.read(args.prospective_plos)
    meta = base.verify_checkpoint(args.embedding)
    if (p_ref["manifest_sha256"] != base.digest(args.prospective_plos)
            or p_ref["elife_policy_sha256"] != base.digest(args.elife_policy)
            or role_ref["embedding"] != meta["revision"]
            or base.digest(args.elife_groups) != role_ref["frozen_sha256"]
            or base.digest(args.elife_held) != policy_ref["panel_sha256"]
            or base.digest(Path(route.__file__)) != policy_ref["script_sha256"]):
        raise ValueError("Changed benchmark inputs")
    e_articles = base.load_catalog((args.confirmation, args.extended))
    p_articles = prior.restore(role.read(args.base_plos_manifest), panel, args.prospective_xml,
                              args.base_plos_projection, args.confirmation, args.extended)
    encoder = base.QwenTorchEmbedding(args.embedding)
    definitions, phrases, wording = base.queries(args.glossary, args.training)
    banks, _ = role.train(encoder, definitions, phrases)
    weights, cut = role_ref["training"]["weights"], role_ref["role_cutoffs"]["query_ridge_calibrated"]
    e_plan = tuple(base.phrase_cases(group, wording)
                   for group in (original["calibration"], original["held"], role.read(args.elife_held)))
    p_plan = {part: prior.transfer.plos_requests(p_articles, panel[part], wording)
              for part in ("calibration", "test")}
    _, e_roles = role.embed(encoder, banks, weights, e_plan)
    _, p_roles = role.embed(encoder, banks, weights, tuple(p_plan.values()))
    e_index, p_index = (route.PostingIndex(route.Catalog(articles, encoder)) for articles in (e_articles, p_articles))
    e_footprints, p_footprints = (route.nested.previous.Fingerprints(index.catalog) for index in (e_index, p_index))
    order = sorted(original["calibration"]["pairs"], key=lambda row: base.stable("open-proof", tuple(row["papers"])))
    report = {}
    for entrance in ("excerpt", "keywords"):
        training = route.data(e_index, encoder, order[:18], e_plan[0], entrance, e_roles["ridge"],
                              cut, e_footprints, args.reverse, training=True)
        tuning = route.data(e_index, encoder, order[18:] + original["held"]["pairs"], e_plan[0] | e_plan[1],
                            entrance, e_roles["ridge"], cut, e_footprints, args.reverse)
        model, samples = route.fit(training)
        original_cut, adverse = route.calibrate(tuning, model, e_articles)
        prior.transfer.check_calibration(samples, original_cut, adverse, policy_ref["outcomes"][entrance])
        calibration, held = (gather(p_index, encoder, panel[part]["pairs"], p_plan[part],
                                    entrance, p_roles["ridge"], cut, p_footprints, args.reverse)
                             for part in ("calibration", "test"))
        report[entrance] = {}
        for guard in GUARDS:
            tuned, tested = calibration[guard], held[guard]
            guard_cut, calibration_info = previous.calibration([*tuned["natural"], *tuned["challenge"]],
                                                                model, p_articles)
            frozen, calibrated = {}, {}
            for mode, requests in tested.items():
                frozen[mode] = route.evaluate(requests, model, original_cut, p_articles)
                calibrated[mode] = route.evaluate(requests, model, original_cut | guard_cut, p_articles)
            if guard == "unfiltered" and (
                frozen["natural"] != expected[entrance]["tested"]["elife"]["natural"]
                or calibrated["natural"] != expected[entrance]["tested"]["paired_plos"]["natural"]
                or calibrated["challenge"]["present"] != expected[entrance]["tested"]["paired_plos"]["shallow_source_demoted"]
            ):
                raise ValueError("Unfiltered benchmark control did not reproduce")
            report[entrance][guard] = dict(
                candidates={mode: candidate_counts(requests, p_articles) for mode, requests in tested.items()},
                frozen_elife=frozen, calibrated_plos=calibrated, calibration=calibration_info,
                calibration_cutoff=original_cut | guard_cut,
            )
    base.write_json(args.output, dict(
        outcomes=report, previous_sha256=base.digest(args.previous),
        manifest_sha256=base.digest(args.prospective_plos), script_sha256=base.digest(Path(__file__)),
        gpu=torch.cuda.get_device_name(), reverse=args.reverse, job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation", "extended", "elife_groups", "elife_held", "role_archive", "elife_policy",
                 "base_plos_manifest", "base_plos_projection", "prospective_plos", "prospective_xml",
                 "prospective_result", "glossary", "training", "embedding", "previous", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
