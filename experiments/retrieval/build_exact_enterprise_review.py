"""Bind each development recommendation to the exact evidence shown to reviewers."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from experiments.retrieval.check_joint_review import FIELDS, check
from experiments.retrieval.score_enterprise_evidence_gate import records, sha, view_key


TOP = 5
PREFIX_LENGTH = 1200
SEED = 20261125


def index(rows, key):
    result = {}
    for row in rows:
        identifier = key(row)
        if identifier in result:
            raise ValueError("Duplicate input record")
        result[identifier] = row
    return result


def sources(corpus, expert, requested):
    views = []
    for citation in requested:
        identifier = citation["source_id"]
        source = corpus[identifier]
        start, end = citation["start"], citation["end"]
        if source["expert_id"] != expert or not 0 <= start < end <= len(source["text"]):
            raise ValueError("Citation differs from its candidate or corpus")
        text = source["text"][start:end]
        checksum = sha256(text.encode()).hexdigest()
        if "sha256" in citation and citation["sha256"] != checksum:
            raise ValueError("Citation text differs from its frozen snapshot")
        views.append(dict(id=identifier, start=start, end=end, sha256=checksum, text=text, kind=source["kind"]))
    view_key(views)
    return views


def packet_rows(queries, experts, methods):
    cases = {}
    roles = defaultdict(set)
    identities = defaultdict(set)
    for method, runs in methods.items():
        if set(runs) != set(queries):
            raise ValueError("Each method must cover the same development requests")
        for query, matches in runs.items():
            returned_experts = [expert for expert, _ in matches]
            if len(set(returned_experts)) != len(returned_experts) or len(returned_experts) > TOP:
                raise ValueError("Duplicate expert or excessive recommendations")
            for expert, views in matches:
                signature = tuple(sorted(view_key(views)))
                key = query, expert, signature
                cases.setdefault(key, views)
                roles[key].add(method)
                identities[query, expert].add(signature)

    rng = Random(SEED)
    active = sorted(cases)
    rng.shuffle(active)
    forms, mapping, aliases = [], [], {}
    for number, key in enumerate(active, 1):
        query, expert, _ = key
        case = f"case-{number:04d}"
        aliases[key] = case
        views = sorted(cases[key], key=lambda row: (row["id"], row["start"], row["end"]))
        rng.shuffle(views)
        form_sources, private_sources = [], []
        for i, view in enumerate(views, 1):
            alias = f"source-{i}"
            form_sources.append(dict(id=alias, kind=view["kind"], text=view["text"]))
            private_sources.append(dict(alias=alias, **{field: view[field] for field in ("id", "start", "end", "sha256")}))
        candidate = {field: experts[expert][field] for field in ("name", "institution", "college")}
        if not candidate["name"]:
            raise ValueError("Personal attribution requires an identified candidate")
        forms.append(dict(case_id=case, need=queries[query]["text"], candidate=candidate, sources=form_sources,
                          **dict.fromkeys(FIELDS), rationale=None, reviewer_id=None))
        mapping.append(dict(case_id=case, query_id=query, expert_id=expert, sources=private_sources))

    predictions = {}
    for method, runs in methods.items():
        predictions[method] = []
        for query, matches in sorted(runs.items()):
            recommendations = []
            for expert, views in matches:
                key = query, expert, tuple(sorted(view_key(views)))
                recommendations.append(dict(expert_id=expert, case_id=aliases[key], sources=[
                    {field: view[field] for field in ("id", "start", "end", "sha256")} for view in views
                ]))
            predictions[method].append(dict(query_id=query, status="issued" if recommendations else "abstained",
                                            recommendations=recommendations))
    counts = Counter(len(method_set) for method_set in roles.values())
    return forms, mapping, predictions, dict(
        shared_exact_views=sum(n for methods_per_view, n in counts.items() if methods_per_view > 1),
        same_expert_distinct_views=sum(len(signatures) > 1 for signatures in identities.values()),
        recommendation_slots={method: sum(len(matches) for matches in runs.values()) for method, runs in methods.items()},
    )


def run(dataset, lexical, output):
    if output.exists():
        raise FileExistsError(output)
    dataset_manifest = json.loads((dataset / "manifest.json").read_bytes())
    inputs = {name: sha(dataset / name) for name in (
        "queries.jsonl", "pools.jsonl", "snapshots.jsonl", "corpus.jsonl", "experts.jsonl",
    )}
    if any(checksum != dataset_manifest["files_sha256"][name] for name, checksum in inputs.items()):
        raise ValueError("Changed dataset source files")
    lexical_meta = json.loads((lexical / "summary.json").read_bytes())
    if (sha(lexical / "runs.jsonl") != lexical_meta["runs_sha256"]
            or any(sha(dataset / name) != checksum for name, checksum in lexical_meta["inputs_sha256"].items())):
        raise ValueError("Changed lexical run or underlying corpus")

    def development(name, key):
        return index((r for r in records(dataset / name) if r["split"] == "development"), key)

    queries = development("queries.jsonl", lambda r: r["id"])
    pools = development("pools.jsonl", lambda r: r["query_id"])
    snapshots = development("snapshots.jsonl", lambda r: (r["query_id"], r["expert_id"]))
    corpus = development("corpus.jsonl", lambda r: r["id"])
    experts = development("experts.jsonl", lambda r: r["id"])
    runs = index((r for r in records(lexical / "runs.jsonl") if r["variant"] == "full-demand"), lambda r: r["query_id"])
    dense = {query: [(expert, sources(corpus, expert, snapshots[query, expert]["sources"]))
                     for expert in row["dense_order"][:TOP]] for query, row in pools.items()}
    sparse = {}
    for query, row in runs.items():
        matches = []
        for hit in row["ranking"][:TOP]:
            expert = hit["expert_id"]
            requested = [dict(source_id=source, start=0, end=min(PREFIX_LENGTH, len(corpus[source]["text"])))
                         for source in hit["source_ids"]]
            matches.append((expert, sources(corpus, expert, requested)))
        sparse[query] = matches
    forms, mapping, predictions, counts = packet_rows(queries, experts, {"dense": dense, "lexical": sparse})
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    def write(name, data, *, lines=False):
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            path.chmod(0o600)
            if lines:
                for row in data:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                stream.write(json.dumps(data, indent=2) + "\n")
        return path

    form = write("blind-review.jsonl", forms, lines=True)
    private = write("private-mapping.jsonl", mapping, lines=True)
    manifest = write("manifest.json", dict(
        queries=len(queries), candidate_views=len(forms), human_judgments=0, source_views=sum(len(r["sources"]) for r in forms),
        outputs_sha256={path.name: sha(path) for path in (form, private)},
        inputs_sha256=inputs | {"lexical_runs": lexical_meta["runs_sha256"]},
    ))
    check(output, form)
    frame = write("frame.jsonl", [
        dict(id=query, group=query, text_sha256=sha256(row["text"].encode()).hexdigest())
        for query, row in sorted(queries.items())
    ], lines=True)
    for method, rows in predictions.items():
        protocol = write(f"{method}-protocol.json", dict(
            frame_sha256=sha(frame), policy=method, evaluation_role="Development wiring check with placeholder groups",
            top_k=TOP, required_fields=FIELDS, coverage_floor=.60, risk_ceiling=.05, confidence=.95,
        ))
        results = write(f"{method}-outputs.jsonl", rows, lines=True)
        write(f"{method}-execution.json", dict(
            outputs_sha256=sha(results), protocol_sha256=sha(protocol), packet_sha256=sha(manifest),
        ))
    summary = dict(
        queries=len(queries), private_review_cases=len(forms), **counts,
        source_views=sum(len(row["sources"]) for row in forms),
        human_judgments=0, packet_sha256=sha(manifest), script_sha256=sha(Path(__file__)),
    )
    write("summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "lexical", "output"):
        parser.add_argument(name, type=Path)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
