"""Replay span-keyed proof and stopping on held-out PLOS contributor edits."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from expert_search import constraint_verification, requirement_packets
from expert_search.constraint_verification import verify_request
from expert_search.requirement_packets import clauses
from experiments.credit.constraint_certificate import SCOPES, VIEWS, candidates


STRATEGIES = ("exhaustive", "short_circuit", "shortest_first", "refutation_first")


def render(query, scope, layout):
    required = []
    for slot, role in query["requirements"]:
        source = f"publication {slot + 1}"
        if scope == "either_publication":
            source = f"either publication (cite {source} first if both support it)"
        required.append(f"The candidate must have the CRediT role [{role}] in {source}.")
    text = "\n".join((required[0], "Optional: a reference.", *required[1:]))
    first, _, *residual = clauses(text)
    return text, (first if layout == "trusted_anchor" else None), (first, *residual)


def ledger(owners, spans, observed):
    states = {}
    for expert, inventory in owners:
        for span in spans:
            witnesses = inventory[span]
            seen = [observed[expert, source].get(span) for source in witnesses
                    if (expert, source) in observed]
            if True in seen:
                state = "supported"
            elif len(seen) < len(witnesses):
                state = "unassessed"
            elif not witnesses or None in seen:
                state = "unavailable"
            else:
                state = "unsupported"
            states[expert, span] = state
    return states


def compare(sources, query, scope, layout, top, *, partial_declaration=False):
    text, role, spans = render(query, scope, layout)
    owners, lookup = candidates(sources, query, query["view"], scope)
    inventories = [(expert, dict(zip(spans, witness_lists, strict=True)))
                   for expert, witness_lists in owners]

    def evaluate(text, role, strategy):
        observed = {}

        def read(expert, source):
            observed[expert, source] = {
                spans[i]: value for i, value in lookup(expert, source).items()
            }
            if partial_declaration and source == 1 and spans[-1] in observed[expert, source]:
                observed[expert, source][spans[-1]] = None
            return observed[expert, source]

        result = verify_request(text, role, inventories, read, top=top, strategy=strategy)
        return result, ledger(inventories, spans, observed)

    reference, full_ledger = evaluate(text, role, "exhaustive")
    if scope == "specified_publication" and query["view"] != "present" and reference["ranking"]:
        raise AssertionError("Recommended without a required publication")
    arms = {}
    for strategy in STRATEGIES:
        result, partial = (reference, full_ledger) if strategy == "exhaustive" else evaluate(text, role, strategy)
        if result["ranking"] != reference["ranking"]:
            raise AssertionError("Changed ordered experts or verbatim citations")
        if result["packet"] != reference["packet"]:
            raise AssertionError("Changed original condition inventory")
        withheld = Counter()
        for key, status in partial.items():
            if status == "unassessed":
                withheld[full_ledger[key]] += 1
            elif status != full_ledger[key]:
                raise AssertionError("Reported a premature condition judgment")
        arms[strategy] = Counter(
            requests=1, reads=result["reads"], checks=result["checks"],
            returned=len(result["ranking"]),
            exact_ranked_citations=1,
            exact_candidate_ledger=partial == full_ledger,
            **{f"unassessed_{status}": withheld[status]
               for status in ("supported", "unsupported", "unavailable")},
        )

    if role is not None:
        # Model the unsafe shortcut, keeping the first span and its source mapping.
        first_only, _ = evaluate(role.quote(text), role, "shortest_first")
        arms["role_only_ablation"] = Counter(
            requests=1, reads=first_only["reads"], checks=first_only["checks"],
            returned=len(first_only["ranking"]),
            recommendations_lacking_full_proof=sum(
                any(full_ledger[expert, span] != "supported" for span in spans)
                for expert, _ in first_only["ranking"]
            ),
        )
    return reference["ranking"], arms


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def run(fixture):
    manifest = json.loads((fixture / "manifest.json").read_bytes())
    paths = [fixture / f"{name}.json" for name in ("sources", "queries", "groups")]
    inputs = {path.name: digest(path) for path in paths}
    if any(checksum != manifest["files"][name] for name, checksum in inputs.items()):
        raise ValueError("Changed fixture")
    sources, queries, groups = [json.loads(path.read_bytes()) for path in paths]
    components = {row["id"]: row["component"] for row in groups}
    selected = [query for query in queries if query["split"] == "test"
                and query["track"] == "flip" and query["view"] in VIEWS]
    report = {}
    for scope in SCOPES:
        for top in (1, 3):
            per_view = defaultdict(lambda: defaultdict(Counter))
            for query in selected:
                settings = [(query["view"], False)]
                if query["view"] == "present":
                    settings.append(("present_masked_declaration", True))
                for view, partial in settings:
                    ranked, anchored = compare(sources, query, scope, "trusted_anchor", top,
                                               partial_declaration=partial)
                    without_anchor, unanchored = compare(sources, query, scope, "no_anchor", top,
                                                         partial_declaration=partial)
                    if without_anchor != ranked or any(anchored[arm] != unanchored[arm] for arm in STRATEGIES):
                        raise AssertionError("Anchor changed the proof inventory, work, or result")
                    for arm, counts in anchored.items():
                        per_view[view][arm].update(counts)
                        if not partial:
                            per_view["all_original_views"][arm].update(counts)
            report[f"{scope}/top_{top}"] = per_view
    return dict(inputs=inputs,
                script_sha256=digest(__file__),
                verifier_sha256=digest(constraint_verification.__file__),
                packet_builder_sha256=digest(requirement_packets.__file__),
                queries=len(selected),
                paired_anchor_replays_per_scope_and_cutoff=len(selected) + sum(
                    query["view"] == "present" for query in selected
                ),
                edit_groups=len({q["group"] for q in selected}),
                components=len({components[q["group"]] for q in selected}),
                report=report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as output:
        json.dump(run(args.fixture), output, indent=2, sort_keys=True)
        output.write("\n")
