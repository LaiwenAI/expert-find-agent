"""Compare role-first retrieval against deeper reads for frozen enterprise candidates."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np

from expert_search.corpus import FIELDS
from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval import enterprise_gap_budget_replay as previous
from experiments.retrieval import enterprise_project_role_rescue as role
from experiments.retrieval.enterprise_routed_review import digest, write

ARMS = ("frozen_pool", "deeper_owned", "global_roles")
CAP = 4
BIOGRAPHY_FIELDS = frozenset(FIELDS[2:4])
NAME_LEAD = re.compile(
    r"(?:\s*[,\uff0c:\uff1a]?\s*)(?:\u6559\u6388\s*)?(?:\u66fe|\u73b0\u8d1f\u8d23)?"
    r"(?:\u4e3b\u6301|\u8d1f\u8d23|\u7275\u5934|\u5f00\u53d1|\u7814\u53d1|\u8bbe\u8ba1)"
    r"|\s+(?:previously\s+)?(?:led|developed|designed|implemented|built|managed)\b",
    re.I,
)


def named_actions(source, author):
    if source.get("field") not in BIOGRAPHY_FIELDS or not author:
        return
    for match in role.BREAKS.finditer(source["text"]):
        sentence = match[0]
        trimmed = sentence.lstrip()
        if not 16 <= len(sentence) <= role.MAX_PASSAGE or not trimmed.casefold().startswith(author.casefold()):
            continue
        action = NAME_LEAD.match(trimmed[len(author):])
        if action and not role.NEGATION.search(sentence):
            yield dict(id=source["id"], start=match.start(), end=match.end(),
                       roles=["profile_named_action"], text=sentence)


def build_index(experts, corpus):
    windows = []
    postings = defaultdict(list)
    seen = set()
    for source in corpus.values():
        eid = source["expert_id"]
        author = experts[eid]["name"]
        views = list(role.passages(source, role.name_key(author)))
        views.extend(named_actions(source, author))
        for view in views:
            terms = role.terms(view["text"])
            signature = eid, view["text"]
            if not terms or signature in seen:
                continue
            seen.add(signature)
            index = len(windows)
            item = dict(expert=eid, view=view, terms=terms)
            windows.append(item)
            for term in sorted(terms):
                postings[term].append(index)
    return windows, postings


def select(candidates, criteria, ranks):
    terms = list(map(role.terms, criteria))
    scored = []
    for candidate in candidates:
        coverage = [len(words & candidate["terms"]) / max(1, len(words)) for words in terms]
        if not coverage[0]:
            continue
        expert = candidate["expert"]
        view = candidate["view"]
        score = .65 * min(coverage) + .35 * mean(coverage) + .10 / (ranks.get(expert, 10) + 1)
        scored.append((-score, expert, view["id"], view["start"], candidate))
    scored.sort(key=lambda row: row[:-1])
    seen = Counter()
    selected = []
    for *_, item in scored:
        expert = item["expert"]
        if seen[expert] < 2:
            selected.append({k: item[k] for k in ("expert", "view")})
            seen[expert] += 1
            if len(selected) == CAP:
                break
    return selected


def prepare(dataset, frozen, routed, output):
    info = dataset / "manifest.json"
    panel = json.loads(frozen.read_bytes())
    if panel["trace_sha256"] != digest(routed) or panel["private_manifest_sha256"] != digest(info):
        raise ValueError("Changed enterprise development inputs")
    old = json.loads(routed.read_bytes())
    baseline = previous.prepare(panel, old)
    manifest = json.loads(info.read_bytes())
    experts, corpus = (role.profile_subset(dataset, key, manifest) for key in ("experts", "corpus"))
    started = perf_counter()
    windows, postings = build_index(experts, corpus)
    index_ms = (perf_counter() - started) * 1000
    seed = review.index(old["results"], lambda row: row["id"])
    queries = []
    for row in baseline:
        qid = row["query"]
        whole = seed[qid]["arms"]["whole"]
        candidates = {item["expert"] for item in whole["recommendations"]}
        ranks = {expert: i for i, expert in enumerate(whole["ranked"])}
        criteria = row["criteria"]
        if not criteria:
            options = {mode: [] for mode in ARMS}
            measured = {mode: 0. for mode in ARMS}
        else:
            started = perf_counter()
            options = {"frozen_pool": [
                {key: candidate[key] for key in ("expert", "view")}
                for candidate in sorted(row["actions"], key=lambda a: (-a["static"], a["rank"], a["depth"]))[:CAP]
            ]}
            measured = {"frozen_pool": (perf_counter() - started) * 1000}
            for arm in ("deeper_owned", "global_roles"):
                tick = perf_counter()
                hits = set().union(*(postings[term] for term in role.terms(criteria[0]) if term in postings))
                pool = (windows[i] for i in sorted(hits) if windows[i]["expert"] not in candidates
                        and (arm == "global_roles" or windows[i]["expert"] in ranks))
                options[arm] = select(pool, criteria, ranks)
                measured[arm] = (perf_counter() - tick) * 1000
        queries.append(dict(query=qid, criteria=criteria, originally_issued=bool(candidates),
                            ranked=whole["ranked"], arms=options, retrieval_ms=measured))
    kinds = Counter(label for item in windows for label in item["view"]["roles"])
    write(output, dict(
        queries=queries, corpus_role_windows=len(windows), role_window_types=dict(kinds),
        indexed_experts=len({item["expert"] for item in windows}), index_ms=index_ms,
        private_manifest_sha256=digest(info), frozen_sha256=digest(frozen), code_sha256=digest(Path(__file__)),
    ))


def inference(query, options, reader, task):
    votes = []
    started = perf_counter()
    criteria = query["criteria"]
    for row in options:
        view = row["view"]
        scores, cuts = reader.score_pairs([(criterion, view["text"]) for criterion in criteria], task=task)
        if cuts or not np.isfinite(scores).all():
            raise ValueError("Invalid or clipped role verifier margin")
        votes.append(dict(expert=row["expert"], source=view["id"], start=view["start"], end=view["end"],
                          support=all(value > 0 for value in scores), calls=len(scores), roles=view["roles"],
                          cumulative_ms=(perf_counter() - started) * 1000))
    return votes


def summarize(queries, records):
    results = {}
    for condition in ("matched", "available"):
        arms = {}
        for arm in ARMS:
            counts = Counter()
            times = []
            for query, record in zip(queries, records, strict=True):
                available = record["arms"]
                limit = min(map(len, available.values())) if condition == "matched" else CAP
                events = available[arm][:limit]
                supports = {e["expert"] for e in events if e["support"]}
                counts.update(
                    windows=len(events), pairs=sum(e["calls"] for e in events), queries_with_windows=bool(events),
                    promoted=len(supports), queries_promoted=bool(supports),
                    first_provisional=bool(supports) and not query["originally_issued"],
                    outside_original_candidates=len(supports - set(query["ranked"])),
                    named_action_windows=sum("profile_named_action" in event["roles"] for event in events),
                )
                if events:
                    times.append(events[-1]["cumulative_ms"])
            arms[arm] = dict(counts, verifier_p95_ms=float(np.quantile(times, .95)))
        if condition == "matched" and len({(row["windows"], row["pairs"]) for row in arms.values()}) != 1:
            raise ValueError("Mismatched verifier budgets")
        results[condition] = arms
    return results


def run(panel, checkpoint, output, private):
    import torch
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.enterprise_first_query import TASK

    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ or output.exists() or private.exists():
        raise ValueError("Require Slurm CUDA and new output paths")
    prepared = json.loads(panel.read_bytes())
    if prepared["code_sha256"] != digest(Path(__file__)):
        raise ValueError("Changed retrieval policy")
    queries = prepared["queries"]
    reader = Reranker(checkpoint)
    reader.score_pairs([("Optical test experience", "Led an optical test")], task=TASK)
    records = []
    for i, query in enumerate(queries):
        order = i % len(ARMS)
        arms = {arm: inference(query, query["arms"][arm], reader, TASK) for arm in ARMS[order:] + ARMS[:order]}
        records.append(dict(query=query["query"], arms=arms))
    result = summarize(queries, records)
    metadata = {key: prepared[key] for key in ("corpus_role_windows", "role_window_types", "indexed_experts",
                                                "index_ms", "code_sha256", "frozen_sha256")}
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), checkpoint=reader.provenance["revision"],
        **metadata, queries=len(queries), prepared_sha256=digest(panel), results=result,
        retrieval_p95_ms={arm: float(np.quantile([q["retrieval_ms"][arm] for q in queries], .95)) for arm in ARMS},
    ))
    write(private, dict(records=records, panel_sha256=digest(panel)))


def packet(dataset, panel, private, output):
    if output.exists():
        raise FileExistsError(output)
    data, observed = (json.loads(path.read_bytes()) for path in (panel, private))
    info = dataset / "manifest.json"
    if observed["panel_sha256"] != digest(panel) or data["private_manifest_sha256"] != digest(info):
        raise ValueError("Changed inputs")
    manifest = json.loads(info.read_bytes())
    queries, experts, corpus = (role.profile_subset(dataset, key, manifest) for key in ("queries", "experts", "corpus"))
    methods = {arm: {qid: [] for qid in queries} for arm in ARMS}
    qualifications = {}
    for record, prep in zip(observed["records"], data["queries"], strict=True):
        qid = record["query"]
        if qid != prep["query"]:
            raise ValueError("Mismatched requests")
        qualifications[qid] = prep["criteria"][1:]
        for arm, events in record["arms"].items():
            decisions = {}
            for event, proposal in zip(events, prep["arms"][arm], strict=True):
                eid, view = proposal["expert"], proposal["view"]
                key = view["id"], view["start"], view["end"]
                if (event["expert"], event["source"], event["start"], event["end"]) != (eid, *key):
                    raise ValueError("Mismatched selected window")
                if event["support"] and eid not in decisions:
                    if corpus[key[0]]["text"][key[1]:key[2]] != view["text"]:
                        raise ValueError("Changed original quotation")
                    decisions[eid] = review.sources(corpus, eid, [
                        dict(source_id=key[0], start=key[1], end=key[2]),
                    ])
            methods[arm][qid].extend(decisions.items())
    forms, mapping, predictions, details = review.packet_rows(queries, experts, methods)
    for form, identifier in zip(forms, mapping, strict=True):
        form["prior_qualifications"] = qualifications[identifier["query_id"]]
    output.mkdir(mode=0o700)
    blind, mapping_path = output / "blind-review.jsonl", output / "private-mapping.jsonl"
    write(blind, forms, lines=True)
    write(mapping_path, mapping, lines=True)
    files = [blind, mapping_path]
    for arm, rows in predictions.items():
        path = output / f"{arm}-predictions.jsonl"
        write(path, rows, lines=True)
        files.append(path)
    write(output / "manifest.json", dict(queries=len(queries), candidate_views=len(forms),
                                        outputs_sha256={p.name: digest(p) for p in files}))
    validation = review.check(output, blind)
    if any(set(counts) != {"missing"} for counts in validation["counts"].values()):
        raise ValueError("Expected a new unlabeled packet")
    for subject in ("a", "b"):
        write(output / f"reviewer-{subject}.jsonl", forms, lines=True)
    print(json.dumps(dict(views=len(forms), human_labels=0, **details)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    tasks = {
        "prepare": (prepare, ("dataset", "frozen", "routed", "output")),
        "run": (run, ("panel", "checkpoint", "output", "private")),
        "packet": (packet, ("dataset", "panel", "private", "output")),
    }
    for command, (_, names) in tasks.items():
        parser = commands.add_parser(command)
        for name in names:
            parser.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    tasks[args.pop("command")][0](**args)
