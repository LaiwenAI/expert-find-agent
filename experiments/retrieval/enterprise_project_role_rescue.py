"""Compare ordinary source rechecks against project and patent role passages."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
import unicodedata

from expert_search.corpus import FIELDS
from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("whole_source", "role_passage")
CAP = 2
PREFIX = 800
CRITERION = 320
MAX_PASSAGE = 520
PROJ, PATENT = FIELDS[-2:]
BREAKS = re.compile(r"[^\r\n;\uff1b\u3002]+")
COLUMNS = re.compile(r"[,:\uff0c\uff1a\t|\uff08\uff09()\[\]]+")
LEAD = frozenset(("\u4e3b\u6301", "\u4e3b\u6301\u4eba", "\u8d1f\u8d23\u4eba",
                  "\u9879\u76ee\u8d1f\u8d23\u4eba", "\u8bfe\u9898\u8d1f\u8d23\u4eba",
                  "project lead", "principal investigator"))
PARTICIPATE = frozenset(("\u53c2\u4e0e", "\u53c2\u52a0", "\u9aa8\u5e72", "\u9879\u76ee\u9aa8\u5e72", "participant"))
NEGATION = re.compile(r"\u672a\u53c2\u4e0e|\u4e0d\u53c2\u4e0e|\u975e\u4e3b\u6301|not (?:a |the )?(?:project lead|participant)", re.I)
INVENTORS = re.compile(r"(?:^|[\uff0c,\s])(?:\u53d1\u660e\u4eba|inventors?)\s*[:\uff1a]\s*([^,\uff0c;\uff1b]+)", re.I)
HAN_RUN = re.compile(r"[\u3400-\u9fff]+")
LATIN = re.compile(r"[a-z0-9]+")


def terms(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = set(LATIN.findall(normalized))
    for run in HAN_RUN.findall(normalized):
        words.update(run[i:i+2] for i in range(len(run) - 1))
    return words


def name_key(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def roles(field, text, owner):
    if NEGATION.search(text):
        return ()
    if field == PROJ:
        columns = {x.strip().casefold() for x in COLUMNS.split(text)}
        matches = tuple(role for role, labels in (
            ("profile_project_lead", LEAD), ("profile_project_participant", PARTICIPATE)
        ) if columns & labels)
        return matches if len(matches) == 1 else ()
    if field == PATENT and owner:
        for finding in INVENTORS.finditer(text):
            names = re.split(r"[\u3001/|]|\band\b", finding[1], flags=re.I)
            if any(name_key(name) == owner for name in names):
                return ("profile_listed_inventor",)
    return ()


def passages(source, owner):
    if source.get("field") not in (PROJ, PATENT):
        return
    for match in BREAKS.finditer(source["text"]):
        text = match[0]
        if not 16 <= len(text) <= MAX_PASSAGE:
            continue
        declared = roles(source["field"], text, owner)
        if declared:
            yield dict(id=source["id"], start=match.start(), end=match.end(), roles=declared, text=text)


def baseline_view(source, owner):
    text = source["text"][:PREFIX]
    marked = [chunk for chunk in passages(source, owner) if chunk["end"] <= len(text)]
    return dict(id=source["id"], start=0, end=len(text), roles=sorted({r for x in marked for r in x["roles"]}),
                text=text)


def choose(pool, query_terms):
    def rank(view):
        words = terms(view["text"])
        overlap = len(words & query_terms)
        return -overlap / math.sqrt(len(words) or 1), view["id"], view["start"]

    return sorted(pool, key=rank)[:CAP]


def profile_subset(data, file, manifest):
    path = data / f"{file}.jsonl"
    if digest(path) != manifest["files_sha256"][path.name]:
        raise ValueError("Changed data snapshot")
    return review.index((row for row in rows(path) if row["split"] == "development"), lambda x: x["id"])


def prepare(dataset, trace, output):
    pinned = json.loads((dataset / "manifest.json").read_bytes())
    evidence_run = json.loads(trace.read_bytes())
    if evidence_run["job"] != "43013" or len(evidence_run["results"]) != 80:
        raise ValueError("Expected the frozen development comparison")
    inputs = {k: profile_subset(dataset, k, pinned) for k in ("queries", "experts", "corpus")}
    results = review.index(evidence_run["results"], lambda record: record["id"])
    if set(results) != set(inputs["queries"]):
        raise ValueError("Mismatched query inventory")
    owners = {candidate for record in results.values() for candidate in record["arms"]["whole"]["ranked"]}
    by_owner = defaultdict(list)
    for source in inputs["corpus"].values():
        if source["expert_id"] in owners and source["text"]:
            by_owner[source["expert_id"]].append(source)
    options = {}
    for owner in owners:
        name = name_key(inputs["experts"][owner]["name"])
        ordinary = [baseline_view(source, name) for source in by_owner[owner]]
        bound = [view for source in by_owner[owner] for view in passages(source, name)]
        options[owner] = ordinary, bound

    cases = []
    for qid, trace_query in results.items():
        raw = inputs["queries"][qid]["text"]
        original = trace_query["arms"]["whole"]
        criteria = [raw.strip().splitlines()[0][:CRITERION]]
        criteria.extend(raw[a:b] for a,b in original["qualifications"])
        for expert in original["ranked"]:
            existing = next((rec["citations"] for rec in original["recommendations"] if rec["expert"] == expert), None)
            # Only unresolved slots can be promoted in this experiment.
            if existing:
                continue
            query_terms = terms(raw)
            ordinary, bound = options[expert]
            cases.append(dict(query=qid, expert=expert, criteria=criteria, issued=bool(original["recommendations"]),
                              options={arm:choose(pool,query_terms) for arm,pool in zip(ARMS,(ordinary,bound),strict=True)}))
    write(output, dict(
        cases=cases, requests=len(results), original_issued=evidence_run["metrics"]["whole"]["issued"],
        inventory_candidates=len(owners), trace_sha256=digest(trace), private_manifest_sha256=digest(dataset/"manifest.json"),
        code_sha256=digest(Path(__file__)),
    ))


def run(prepared, checkpoint, output):
    import numpy as np
    import torch
    from experiments.followir.core17_verifier import Reranker
    from experiments.retrieval.enterprise_first_query import TASK

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh GPU result")
    panel = json.loads(prepared.read_bytes())
    if panel["code_sha256"] != digest(Path(__file__)):
        raise ValueError("Changed experimental policy")
    model = Reranker(checkpoint)
    model.score_pairs([("Find an expert with optical testing experience", "The study performed optical testing.")], task=TASK)
    results, timings = [], defaultdict(list)
    for i, case in enumerate(panel["cases"]):
        decisions = {}
        for arm in ARMS[i%2:]+ARMS[:i%2]:
            views = case["options"][arm]
            criteria = case["criteria"]
            votes = []
            start = perf_counter()
            for view in views:
                if not view["roles"] or any(len(c) > CRITERION for c in criteria):
                    continue
                # A catalog role is required in both arms. Technical fit is a separate check.
                scores, cuts = model.score_pairs([(criterion,view["text"]) for criterion in criteria],task=TASK)
                if not all(math.isfinite(value) for value in scores):
                    raise ValueError("Nonfinite support score")
                if cuts:
                    continue
                votes.append(dict(id=view["id"],start=view["start"],end=view["end"],
                                  roles=view["roles"],supported=all(value>0 for value in scores)))
            timings[arm].append((perf_counter()-start)*1000)
            decisions[arm] = votes
        results.append(dict(query=case["query"], expert=case["expert"], decisions=decisions))
    metrics = {}
    for arm in ARMS:
        n = Counter()
        proposed = set()
        first = set()
        for case,result in zip(panel["cases"],results,strict=True):
            selected = case["options"][arm]
            votes = result["decisions"][arm]
            n["candidate_slots_with_any_read"] += bool(selected)
            n["completed_role_span_checks"] += len(votes)
            n["scheduled_reads"] += len(selected)
            n["slots_with_role_markers"] += any(x["roles"] for x in selected)
            n["provisional_slots"] += any(row["supported"] for row in votes)
            if any(row["supported"] for row in votes):
                proposed.add(case["query"])
                if not case["issued"]:
                    first.add(case["query"])
        metrics[arm] = dict(n, requests_with_provisional_role_support=len(proposed),
                            previously_abstained_requests_with_provisional_support=len(first),
                            verifier_only_ms_p50_p95=np.quantile(timings[arm],[.5,.95]).tolist())
    payload = {key:value for key,value in panel.items() if key != "cases"}
    write(output, payload | dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                 tested_unresolved_slots=len(results), checkpoint=model.provenance["revision"],
                                 prepared_sha256=digest(prepared), metrics=metrics, private_results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, names in {"prepare":("dataset","trace","output"),"run":("prepared","checkpoint","output")}.items():
        child=commands.add_parser(command)
        for name in names:
            child.add_argument(name,type=Path)
    arguments = vars(parser.parse_args())
    {"prepare":prepare,"run":run}[arguments.pop("command")](**arguments)
