"""Compare query-level role-evidence budgets on frozen enterprise development candidates."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_project_role_rescue as role
from experiments.retrieval.enterprise_routed_review import digest, write

ARMS = ("rank_round_robin", "static_bottleneck", "adaptive_gap")
BUDGETS = (1, 2, 4)


def prepare(panel, original):
    if panel["requests"] != 80 or len(panel["cases"]) != 330 or original["job"] != "43013":
        raise ValueError("Require the development panel and original development trace")
    frozen = review.index(original["results"], lambda row: row["id"])
    by_query = defaultdict(list)
    for case in panel["cases"]:
        by_query[case["query"]].append(case)
    if not set(by_query) <= frozen.keys():
        raise ValueError("Unrecognized request")
    queries = []
    for query, row in frozen.items():
        original_result = row["arms"]["whole"]
        ranking = original_result["ranked"]
        options = by_query[query]
        issued = bool(original_result["recommendations"])
        criteria = options[0]["criteria"] if options else []
        if any(item["criteria"] != criteria or item["issued"] != issued for item in options):
            raise ValueError("Inconsistent frozen request state")
        criterion_terms = list(map(role.terms, criteria))
        actions = []
        for item in options:
            expert = item["expert"]
            rank = ranking.index(expert)
            seen = set()
            views = []
            for pool in ("role_passage", "whole_source"):
                for view in item["options"][pool]:
                    span = view["id"], view["start"], view["end"]
                    if span in seen or not view["roles"]:
                        continue
                    seen.add(span)
                    words = role.terms(view["text"])
                    coverage = [len(words & terms) / max(1, len(terms)) for terms in criterion_terms]
                    static = .65 * min(coverage) + .35 * mean(coverage) + .10 / (rank + 1)
                    views.append(dict(expert=expert, rank=rank, span=span, view=view, static=static,
                                      coverage=coverage, action_type=pool))
            views.sort(key=lambda action: (-action["static"], action["span"]))
            actions.extend(dict(action, depth=depth) for depth, action in enumerate(views))
        queries.append(dict(query=query, issued=issued, criteria=criteria, actions=actions))
    return queries


def choose(actions, arm, supported, observed):
    def priority(action):
        person = action["expert"]
        rank, depth = action["rank"], action["depth"]
        if arm == "rank_round_robin":
            score = (-depth, -rank)
        else:
            utility = action["static"]
            if arm == "adaptive_gap" and person in observed:
                previous = observed[person]
                gaps = [i for i, margin in enumerate(previous) if margin <= 0]
                if gaps:
                    repair = min(action["coverage"][i] for i in gaps)
                    already_supported = 1 - len(gaps) / len(previous)
                    utility += .25 * already_supported * repair - .08 * (1 - already_supported)
            score = (utility, -rank)
        return person not in supported, *score, -depth

    return max(actions, key=priority)


def execute(query, arm, reader, task):
    actions = list(query["actions"])
    supported, observed = set(), {}
    criteria = query["criteria"]
    events = []
    started = perf_counter()
    for _ in range(min(len(actions), max(BUDGETS))):
        action = choose(actions, arm, supported, observed)
        actions.remove(action)
        expert = action["expert"]
        inputs = [(criterion, action["view"]["text"]) for criterion in criteria]
        scores, cuts = reader.score_pairs(inputs, task=task)
        if cuts or not np.isfinite(scores).all():
            raise ValueError("Clipped or nonfinite evidence")
        accepted = all(score > 0 for score in scores)
        if accepted:
            supported.add(expert)
        elif expert not in observed or min(scores) > min(observed[expert]):
            observed[expert] = scores
        events.append(dict(expert=expert, span=action["span"], accepted=accepted, criterion_pairs=len(inputs),
                           action_type=action["action_type"], milliseconds=(perf_counter() - started) * 1000))
    return events


def aggregate(queries, records):
    results = {}
    qid = {query["query"]: query for query in queries}
    for budget in BUDGETS:
        results[budget] = {}
        for arm in ARMS:
            counters = Counter()
            elapsed = []
            for row in records:
                current = row["arms"][arm][:budget]
                adds = {event["expert"] for event in current if event["accepted"]}
                counters.update(
                    source_window_checks=len(current),
                    model_pairs=sum(event["criterion_pairs"] for event in current),
                    provisional_experts=len(adds),
                    provisional_queries=bool(adds),
                    first_provisional_result=bool(adds) and not qid[row["query"]]["issued"],
                    queries_with_actions=bool(current),
                    role_passages_read=sum(event["action_type"] == "role_passage" for event in current),
                )
                if current:
                    elapsed.append(current[-1]["milliseconds"])
            results[budget][arm] = dict(counters, rescue_p95_ms=float(np.quantile(elapsed, .95)))
        paired_reads = [{k: len(row["arms"][k][:budget]) for k in ARMS} for row in records]
        if any(len(set(counts.values())) != 1 for counts in paired_reads):
            raise ValueError("Read budgets differ by strategy")
        if len({row["model_pairs"] for row in results[budget].values()}) != 1:
            raise ValueError("Criterion pair budgets differ")
    return results


def run(prepared, trace, checkpoint, output, private):
    import torch
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.enterprise_first_query import TASK

    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists() or private.exists():
        raise ValueError("Require Slurm CUDA and new output paths")
    panel = json.loads(prepared.read_bytes())
    if panel["trace_sha256"] != digest(trace):
        raise ValueError("Mismatched frozen trace")
    queries = prepare(panel, json.loads(trace.read_bytes()))
    reader = Reranker(checkpoint)
    reader.score_pairs([("Industrial testing", "Performed industrial testing")], task=TASK)
    records = []
    for i, query in enumerate(queries):
        offset = i % len(ARMS)
        arms = {arm: execute(query, arm, reader, TASK) for arm in ARMS[offset:] + ARMS[:offset]}
        records.append(dict(query=query["query"], arms=arms))
    fingerprints = dict(prepared_sha256=digest(prepared), code_sha256=digest(Path(__file__)))
    write(private, dict(records=records, **fingerprints))
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoint=reader.provenance["revision"], queries=len(queries), originally_issued=panel["original_issued"],
        action_pool=sum(len(row["actions"]) for row in queries), results=aggregate(queries, records), **fingerprints,
    ))


def packet(dataset, prepared, private, output):
    if output.exists():
        raise FileExistsError(output)
    panel = json.loads(prepared.read_bytes())
    trace = json.loads(private.read_bytes())
    manifest_path = dataset / "manifest.json"
    if panel["private_manifest_sha256"] != digest(manifest_path) or trace["prepared_sha256"] != digest(prepared):
        raise ValueError("Changed inputs")
    manifest = json.loads(manifest_path.read_bytes())
    queries, experts, corpus = (role.profile_subset(dataset, key, manifest) for key in ("queries", "experts", "corpus"))
    lookup = {}
    qualifications = {}
    for case in panel["cases"]:
        qualifications[case["query"]] = case["criteria"][1:]
        for options in case["options"].values():
            for option in options:
                lookup[case["query"], case["expert"], option["id"], option["start"], option["end"]] = option
    methods = {arm: {qid: [] for qid in queries} for arm in ARMS}
    for row in trace["records"]:
        qid = row["query"]
        for arm, events in row["arms"].items():
            best = {}
            for event in events:
                if not event["accepted"] or event["expert"] in best:
                    continue
                eid = event["expert"]
                source, start, end = event["span"]
                option = lookup[qid, eid, source, start, end]
                if not option["roles"] or corpus[source]["text"][start:end] != option["text"]:
                    raise ValueError("Evidence lost its original span or role marker")
                witness = [dict(source_id=source, start=start, end=end)]
                best[eid] = review.sources(corpus, eid, witness)
            methods[arm][qid].extend(best.items())
    forms, mapping, predictions, _ = review.packet_rows(queries, experts, methods)
    for form, item in zip(forms, mapping, strict=True):
        form["prior_qualifications"] = qualifications[item["query_id"]]
    output.mkdir(mode=0o700)
    blind, private_mapping = output / "blind-review.jsonl", output / "private-mapping.jsonl"
    write(blind, forms, lines=True)
    write(private_mapping, mapping, lines=True)
    files = [blind, private_mapping]
    for arm, values in predictions.items():
        path = output / f"{arm}-predictions.jsonl"
        write(path, values, lines=True)
        files.append(path)
    write(output / "manifest.json", dict(queries=len(queries), candidate_views=len(forms),
                                        outputs_sha256={p.name: digest(p) for p in files}))
    validation = review.check(output, blind)
    if any(set(counts) != {"missing"} for counts in validation["counts"].values()):
        raise ValueError("Expected a new unlabeled review")
    for duplicate in ("a", "b"):
        write(output / f"reviewer-{duplicate}.jsonl", forms, lines=True)
    print(json.dumps(dict(blind_cases=len(forms), human_judgments=0, packet_sha256=digest(output / "manifest.json"))))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subcommands = cli.add_subparsers(dest="command", required=True)
    for mode, args in {
        "run": ("prepared", "trace", "checkpoint", "output", "private"),
        "packet": ("dataset", "prepared", "private", "output"),
    }.items():
        parser = subcommands.add_parser(mode)
        for name in args:
            parser.add_argument(name, type=Path)
    values = vars(cli.parse_args())
    {"run": run, "packet": packet}[values.pop("command")](**values)
