"""Test scoped and cross-publication hard-constraint stopping on CRediT declarations."""

import argparse
from collections import defaultdict
import json
import inspect
from pathlib import Path
from statistics import mean

from expert_search.constraint_verification import resolve_constraints
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json


METHODS = ("exhaustive", "short_circuit", "bundled_first", "certificate")
TOP = (1, 3)
SCOPES = ("specified_publication", "either_publication")
VIEWS = ("present", "left_withheld", "right_withheld")


def candidates(sources, query, view, scope):
    hidden = {"left_withheld": 0, "right_withheld": 1}.get(view)
    positions = defaultdict(dict)
    for slot, doi in enumerate(query["sources"]):
        if slot == hidden:
            continue
        for row in sources[doi]["authors"]:
            if not row["person"]:
                continue
            if slot in positions[row["person"]]:
                raise ValueError("Ambiguous publisher identifier")
            positions[row["person"]][slot] = row
    # Prior uses the best byline position, not the declared roles or the edit.
    order = sorted(positions, key=lambda expert: (
        min(r["ordinal"] for r in positions[expert].values()), expert))
    owners = []
    for expert in order:
        witnesses = []
        for slot, role in query["requirements"]:
            allowed = (slot,) if scope == "specified_publication" else (slot, 1-slot)
            witnesses.append(tuple(s for s in allowed if s != hidden and s in positions[expert]))
        owners.append((expert, witnesses))

    def read(expert, slot):
        return {i: role in positions[expert][slot]["roles"]
                for i, (required_slot, role) in enumerate(query["requirements"])
                if scope == "either_publication" or slot == required_slot}

    return owners, read


def run(fixture, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((fixture / "manifest.json").read_bytes())
    inputs = {}
    for name in ("sources", "queries", "groups"):
        path = fixture / f"{name}.json"
        inputs[name] = digest(path)
        if inputs[name] != manifest["files"][f"{name}.json"]:
            raise ValueError(f"Changed fixture: {name}")
    sources, queries, groups = [json.loads((fixture / f"{n}.json").read_bytes())
                                for n in ("sources", "queries", "groups")]
    components = {g["id"]: g["component"] for g in groups}
    selected = [q for q in queries if q["split"] == "test" and q["track"] == "flip"
                and q["view"] in VIEWS]
    results = {}
    for scope in SCOPES:
        results[scope] = {}
        for top in TOP:
            rows = []
            for query in selected:
                owners, lookup = candidates(sources, query, query["view"], scope)
                arms = {method: resolve_constraints(owners, lookup, top=top, strategy=method)
                        for method in METHODS}
                reference = arms["exhaustive"]["ranking"]
                if any(arm["ranking"] != reference for arm in arms.values()):
                    raise AssertionError("Changed selected expert or canonical citation")
                if scope == "specified_publication" and query["view"] != "present" and reference:
                    raise AssertionError("Published-source requirement was bypassed")
                rows.append(dict(view=query["view"], component=components[query["group"]],
                                 returned=len(reference), arms=arms))
            report = {}
            for view in (*VIEWS, "all"):
                subset = [r for r in rows if r["view"] == view or view == "all"]
                groups = [r["component"] for r in subset]
                report[view] = dict(
                    cases=len(subset), nonempty=sum(r["returned"] > 0 for r in subset),
                    methods={method: {
                        "mean_reads": mean(r["arms"][method]["reads"] for r in subset),
                        "mean_checks": mean(r["arms"][method]["checks"] for r in subset),
                        "paired_reads_saved_vs_short_circuit": clustered_paired_interval(
                            [r["arms"]["short_circuit"]["reads"] - r["arms"][method]["reads"] for r in subset],
                            groups) | {"sampling_unit": "publisher-linked component"},
                    } for method in METHODS},
                )
            results[scope][str(top)] = report
    write_json(output, dict(
        inputs=inputs, script_sha256=digest(Path(__file__)),
        certificate_sha256=digest(Path(inspect.getfile(resolve_constraints))),
        test_cases=len(selected), components=len({components[q["group"]] for q in selected}), report=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.fixture, args.output)
