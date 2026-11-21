"""Run frozen source recovery on a fresh publisher holdout and missing-evidence cases."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit.paired_role_prospective_service import MATCHED_SERVICE_ARMS, execute
from experiments.credit.prose_budgeted_source_role import evidence
from experiments.followir.core17_reranking import digest, write_json


def evaluate(service, labels, cutoff, lightweight, resident, groups, output, reverse, metadata):
    requests = list(labels)
    arms = MATCHED_SERVICE_ARMS[::-1] if reverse else MATCHED_SERVICE_ARMS
    results = defaultdict(lambda: defaultdict(Counter))
    timings = defaultdict(lambda: defaultdict(list))
    issued_groups = defaultdict(set)
    adverse_groups = defaultdict(set)
    signatures, exact = {}, defaultdict(dict)
    keys = {}
    for i, req in enumerate(reversed(requests) if reverse else requests):
        label = labels[req]
        view = label["view"]
        family = groups[label["family"]]
        rotation = i % len(arms)
        ordered = arms[rotation:] + arms[:rotation]
        for availability, missing in enumerate((None, *label["papers"])):
            condition = "both_present" if missing is None else "one_withheld"
            for arm in ordered:
                if missing is not None and arm == "baseline":
                    continue
                start = perf_counter()
                raw = execute(service, req, arm, cutoff, lightweight, resident, withheld=missing)
                elapsed = (perf_counter() - start) * 1000
                response = json.loads(raw)
                if response["calls"] > 1 or sum(response[k] for k in ("role_calls", "source_calls")) != response["calls"]:
                    raise ValueError("One-forward cap exceeded")
                if missing and missing in (response["papers"] or ()):
                    raise ValueError("A withheld paper appeared in the result")
                measure = evidence(response, label)
                results[condition, view][arm].update(measure)
                key = condition, view, arm
                if measure["issued"]:
                    issued_groups[key].add(family)
                if measure["adverse"]:
                    adverse_groups[key].add(family)
                timings[condition, view][arm].append(elapsed)
                keys[req, availability, arm] = sha256(raw.encode()).hexdigest()
                if missing is None:
                    exact[arm][req] = measure["strict"]
        if (i + 1) % 100 == 0:
            print(i + 1, "fresh requests and withholding controls evaluated", flush=True)

    report = {}
    for (condition, view), by_arm in sorted(results.items()):
        report.setdefault(condition, {})[view] = {
            arm: dict(totals=dict(counts), serial_p95_ms=float(np.quantile(timings[condition, view][arm], .95)),
                      groups_issued=len(issued_groups[condition, view, arm]),
                      groups_adverse=len(adverse_groups[condition, view, arm]))
            for arm, counts in by_arm.items()
        }
    for view in ("keywords", "excerpt"):
        positive = [req for req in requests if labels[req]["gold"] and labels[req]["view"] == view]
        group_ids = [groups[labels[req]["family"]] for req in positive]
        report["both_present"][view]["positives"] = len(positive)
        for policy in ("dense_joint", "matched_resident"):
            contrast = [int(exact[policy][req]) - int(exact["student_then_teacher"][req]) for req in positive]
            report["both_present"][view][policy]["paired_strict_delta_vs_cascade"] = clustered_paired_interval(
                contrast, group_ids,
            )

    for arm in arms:
        combined = [keys[req, case, arm] for req in requests for case in range(3 if arm != "baseline" else 1)]
        signatures[arm] = sha256("".join(combined).encode()).hexdigest()
    write_json(output, dict(
        code_sha256=digest(Path(__file__)), requests=len(requests), groups=len(set(groups.values())),
        reverse=reverse, output_signatures=signatures, evaluations=report, **metadata,
    ))
