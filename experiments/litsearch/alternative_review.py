"""Build a blinded source-review packet and score adjudicated alternatives."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path

from experiments.followir.core17_reranking import digest, write_json


STATES = ("present", "withheld")
LABELS = ("supported", "unsupported", "insufficient")
CONTROL_COUNT = 12
SALT = "litsearch-alternative-review-v1"
RUBRIC = (
    "Judge whether the title and abstract establish that the paper addresses the specific question. "
    "Use supported when they directly establish the requested method, finding or relation. "
    "Use unsupported when they identify a materially different task or explicitly conflict with a requirement. "
    "Use insufficient when a relevant paper leaves a required condition unresolved. "
    "For supported, provide an exact contiguous quotation from the title or abstract. "
    "For every label, give a short reason grounded in the supplied text."
)


def fingerprint(*parts):
    return sha256(json.dumps((SALT, *parts), ensure_ascii=False).encode()).hexdigest()


def jsonl(path, rows):
    with path.open("x") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(projection, replay_path, output):
    with gzip.open(projection, "rb") as source:
        data = json.load(source)
    replay = json.loads(replay_path.read_bytes())
    questions = {sha256(q["query"].encode()).hexdigest(): q for q in data["questions"]
                 if q["query_set"].startswith("manual_")}
    papers = {paper["corpusid"]: paper for paper in data["papers"]}
    cases = {}
    bases = {}
    for budget, results in sorted(replay["raw"].items()):
        if {r["id"] for r in results} != questions.keys() or len(results) != len(questions):
            raise ValueError("Replay and query inventory differ")
        for row in results:
            key, targets = row["id"], row["targets"]
            if targets != questions[key]["corpusids"]:
                raise ValueError("Replay target labels differ")
            for state in STATES:
                for method, arm in row["views"][state].items():
                    route = budget, method, state
                    scores = bases.setdefault(route, Counter())
                    scores["requests"] += 1
                    doi = arm["accepted"]
                    if doi is None:
                        continue
                    if doi not in arm["pool"] or (state == "withheld" and doi in targets):
                        raise ValueError("Selected source outside the retrieved inventory")
                    known = doi in targets
                    scores["known" if known else "unjudged"] += 1
                    case = cases.setdefault((key, doi), dict(known_target=known, selections=[]))
                    case["selections"].append(dict(budget=budget, method=method, state=state))

    unknown = [pair for pair, item in cases.items() if not item["known_target"]]
    known = [pair for pair, item in cases.items() if item["known_target"]]
    reviewed_queries = {key for key, _ in unknown}
    known.sort(key=lambda pair: (pair[0] not in reviewed_queries, fingerprint(*pair, "control")))
    controls = known[:CONTROL_COUNT]
    assignments, mapping = [], []
    for key, doi in unknown + controls:
        paper = papers[doi]
        identifier = fingerprint(key, doi)[:20]
        assignments.append(dict(id=identifier, question=questions[key]["query"],
                                title=paper["title"] or "", abstract=paper["abstract"] or ""))
        mapping.append(dict(id=identifier, query_sha256=key, corpusid=doi, **cases[(key, doi)]))
    assignments.sort(key=lambda row: fingerprint(row["id"], "order"))
    if len({row["id"] for row in assignments}) != len(assignments):
        raise ValueError("Duplicate review IDs")

    output.mkdir(parents=True, exist_ok=False)
    review = output / "review"
    private = output / "private"
    review.mkdir()
    private.mkdir()
    jsonl(review / "cases.jsonl", assignments)
    jsonl(review / "judgments-template.jsonl", [
        dict(id=row["id"], label=None, quotation=None, reason=None) for row in assignments
    ])
    with (review / "README.md").open("x") as handle:
        handle.write("# Source relevance review\n\n" + RUBRIC + "\n\n")
        handle.write("Review independently. Use only the displayed question and paper text. Fill one row per case. "
                     "A second reviewer should use a separate copy. Resolve disagreements with a third reviewer "
                     "before consulting the analysis key.\n\n")
        for row in assignments:
            handle.write(f"## {row['id']}\n\n{row['question']}\n\n"
                         f"**{row['title']}**\n\n{row['abstract']}\n\n")
    metadata = dict(
        projection_sha256=digest(projection), replay_sha256=digest(replay_path),
        cases_sha256=digest(review / "cases.jsonl"), queries=len(questions),
        unique_unjudged=len(unknown), known_controls=len(controls), review_cases=len(assignments),
        unjudged_selections=sum(value["unjudged"] for value in bases.values()),
    )
    write_json(private / "key.json", dict(
        metadata=metadata, cases=mapping,
        routes=[dict(budget=b, method=m, state=s, requests=c["requests"],
                     known=c["known"], unjudged=c["unjudged"]) for (b, m, s), c in sorted(bases.items())],
    ))
    print(json.dumps(metadata, indent=2))


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case IDs")
    return rows


def normalize_judgment(label_record, case):
    label = label_record.get("label")
    quote = label_record.get("quotation") or ""
    reason = label_record.get("reason") or ""
    if label not in LABELS or not isinstance(reason, str) or not reason.strip():
        return "invalid"
    if label == "supported" and (not isinstance(quote, str) or not quote.strip() or not any(
            quote in case[field] for field in ("title", "abstract"))):
        return "invalid"
    return label


def score(packet, judgments, output, origin):
    key = json.loads((packet / "private/key.json").read_bytes())
    cases_path = packet / "review/cases.jsonl"
    if digest(cases_path) != key["metadata"]["cases_sha256"]:
        raise ValueError("Changed review sources")
    cases = {row["id"]: row for row in read_rows(cases_path)}
    answers = {row["id"]: row for row in read_rows(judgments)}
    if not answers.keys() <= cases.keys():
        raise ValueError("Unknown case IDs in completed judgments")
    outcomes = defaultdict(Counter)
    paired = defaultdict(lambda: [0, 0, 0])
    counts = {group: Counter() for group in ("unjudged", "known_controls")}
    for row in key["cases"]:
        identifier = row["id"]
        label = normalize_judgment(answers[identifier], cases[identifier]) if identifier in answers else "pending"
        if row["known_target"]:
            counts["known_controls"][label] += 1
        else:
            counts["unjudged"][label] += 1
            routes = {(s["budget"], s["method"], s["state"]) for s in row["selections"]}
            if len(routes) != len(row["selections"]):
                raise ValueError("Duplicate case selection for a route")
            for route in routes:
                outcomes[route][label] += 1
            for budget, state in {(b, v) for b, _, v in routes}:
                delta = int((budget, "gap", state) in routes) - int((budget, "fixed_two", state) in routes)
                interval = paired[(budget, state)]
                if delta == 0 and (budget, "gap", state) in routes:
                    interval[2] += 1
                if label == "unsupported":
                    interval[0] += delta
                    interval[1] += delta
                elif label != "supported":
                    interval[0] += min(delta, 0)
                    interval[1] += max(delta, 0)
    joined = []
    for item in key["routes"]:
        budget, method, state = (item[name] for name in ("budget", "method", "state"))
        labels = outcomes[(budget, method, state)]
        if sum(labels.values()) != item["unjudged"]:
            raise ValueError("Case and route counts disagree")
        lower = labels["unsupported"]
        upper = item["unjudged"] - labels["supported"]
        joined.append(dict(**item, labels=labels,
                           unsupported_if_review_is_correct=[lower, upper]))
    report = dict(
        metadata=key["metadata"], judgment_origin=origin, verdicts=counts, routes=joined,
        paired_gap_minus_fixed_two=[
            dict(budget=budget, state=state, unsupported_difference=[lower, upper], shared_alternatives=shared)
            for (budget, state), (lower, upper, shared) in sorted(paired.items())
        ],
    )
    write_json(output, report)
    print(json.dumps(counts, indent=2))


def pilot(cases_path, checkpoint, output):
    import torch

    from experiments.qasper.paired_reader import load_reader

    if output.exists():
        raise FileExistsError(output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    tokenizer, reader, provenance = load_reader(checkpoint)
    cases = read_rows(cases_path)
    predictions = []
    for case in cases:
        prompt = tokenizer.apply_chat_template([
            dict(role="system", content=RUBRIC + " Reply as one JSON object with keys label, quotation and reason. "
                 "The label must be supported, unsupported or insufficient. Use an empty quotation if unavailable."),
            dict(role="user", content=f"Question: {case['question']}\nTitle: {case['title']}\nAbstract: {case['abstract']}"),
        ], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            result = reader.generate(**tokens, do_sample=False, max_new_tokens=144, use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(result[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            start, end = response.index("{"), response.rindex("}")
            parsed = json.loads(response[start:end + 1])
            fields = {key: parsed.get(key) for key in ("label", "quotation", "reason")}
            if any(value is not None and not isinstance(value, str) for value in fields.values()):
                raise ValueError("Non-string label fields")
        except (ValueError, TypeError, AttributeError):
            fields = dict(label=None, quotation=None, reason=None)
        predictions.append(dict(id=case["id"], **fields))
    jsonl(output, predictions)
    write_json(output.with_suffix(".provenance.json"), dict(
        cases_sha256=digest(cases_path), judgments_sha256=digest(output), job=os.environ["SLURM_JOB_ID"],
        model=provenance, gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, arguments in (
        ("build", ("projection", "replay", "output")),
        ("score", ("packet", "judgments", "output")),
        ("pilot", ("cases", "checkpoint", "output")),
    ):
        command = sub.add_parser(name)
        for argument in arguments:
            command.add_argument(argument, type=Path)
        if name == "score":
            command.add_argument("--origin", choices=("open_model_pilot", "human_adjudication"), required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.projection, args.replay, args.output)
    elif args.command == "score":
        score(args.packet, args.judgments, args.output, args.origin)
    else:
        pilot(args.cases, args.checkpoint, args.output)
