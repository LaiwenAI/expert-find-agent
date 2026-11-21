"""Compare source-identity guards on the enlarged eLife recovery diagnostic."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.credit.engineering_joint_calibration import features
from experiments.credit.paired_role_prospective_service import execute
from experiments.credit.paired_role_resident_source import clue_view, score
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.followir.core17_reranking import digest, write_json


def rules(index, candidate, clues, ranking, resident):
    routes = {}
    worst_rank = max(side["rank"][doc] for side, doc in zip(ranking, candidate, strict=True))
    for depth in (3, 6, 12):
        routes[f"both_top_{depth}"] = worst_rank < depth
    routes["own_clue_cosine"] = all(
        side["cosine"][own] > side["cosine"][other]
        for side, own, other in zip(ranking, candidate, reversed(candidate), strict=True)
    )
    model, _ = resident[clue_view(clues)]["gold_matched"]
    lex = []
    score_gap = []
    for side, clue, own in zip(ranking, clues, candidate, strict=True):
        own_features = features(index, side, clue, own)
        lex.append(own_features[2])
        other = max(float(score(model, features(index, side, clue, alternative)))
                    for alternative in side["fused"] if alternative != own)
        score_gap.append(float(score(model, own_features)) - other)
    for threshold in (.15, .30, .45):
        routes[f"idf_coverage_{threshold:.2f}"] = min(lex) >= threshold
    routes["independent_source_leader"] = min(score_gap) > 0.
    routes["leader_or_both_top_3"] = routes["independent_source_leader"] or routes["both_top_3"]
    routes["role_free_corroboration"] = routes["own_clue_cosine"] and routes["idf_coverage_0.30"]
    return routes


def evaluate(service, labels, cutoff, lightweight, resident, groups, output, reverse, metadata, *, prior):
    reference = json.loads(prior.read_bytes())
    if any(reference[field] != metadata[field] for field in ("manifest_sha256", "projection_sha256", "calibration")):
        raise ValueError("The diagnostic did not reuse the frozen expanded holdout")
    requests = list(labels)
    results = defaultdict(lambda: defaultdict(Counter))
    identity = defaultdict(lambda: defaultdict(Counter))
    keys = defaultdict(dict)
    for req in reversed(requests) if reverse else requests:
        target = labels[req]
        view = target["view"]
        for j, missing in enumerate((None, *target["papers"])):
            status = "one_withheld" if missing else "both_present"
            key = status, view
            choices = None
            proposed = None

            def observer(candidate, clues, ranking):
                nonlocal choices, proposed
                proposed = tuple(candidate)
                choices = rules(service.index, candidate, clues, ranking, resident)

            candidate = execute(service, req, "matched_resident", cutoff, lightweight, resident,
                                withheld=missing, inspect_source=observer)
            baseline = execute(service, req, "student_then_teacher", cutoff, lightweight, resident, withheld=missing)
            reference_measure = evidence(json.loads(baseline), target)
            candidate_measure = evidence(json.loads(candidate), target)
            keys["matched_resident"][req, j] = sha256(candidate.encode()).hexdigest()
            keys["student_then_teacher"][req, j] = sha256(baseline.encode()).hexdigest()
            results[key]["matched_resident"].update(candidate_measure)
            results[key]["student_then_teacher"].update(reference_measure)

            if choices is not None:
                case = "exact_pair" if proposed == tuple(target["papers"]) else "wrong_pair"
                identity[key]["unfiltered"].update(checked=1, **{case: 1})
                for rule, accept in choices.items():
                    identity[key][rule].update(checked=1, **{case: int(accept)})
                    delta = {k: int(candidate_measure[k]) - int(value) if accept else 0
                             for k, value in reference_measure.items()}
                    results[key][rule].update(delta)

    signatures = {name: sha256("".join(by_case[req, j] for req in requests for j in range(3)).encode()).hexdigest()
                  for name, by_case in keys.items()}
    if any(signatures[name] != reference["output_signatures"][name] for name in signatures):
        raise ValueError("Observed signatures differ from the archived replay")

    sizes = {}
    reports = {}
    for (status, view), by_arm in sorted(results.items()):
        sizes.setdefault(status, {})[view] = {name: dict(counts) for name, counts in identity[status, view].items()}
        complete = {}
        for name, counts in by_arm.items():
            if name in signatures:
                complete[name] = dict(counts)
                continue
            combined = by_arm["student_then_teacher"].copy()
            combined.update(counts)
            complete[name] = dict(combined)
        reports.setdefault(status, {})[view] = complete
    write_json(output, dict(code_sha256=digest(Path(__file__)), previous_sha256=digest(prior),
                            reverse=reverse, requests=len(requests), groups=len(set(groups.values())),
                            output_signatures=signatures, source_proposals=sizes, outcomes=reports, **metadata))
