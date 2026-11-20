"""Audit registry-shaped coverage and offline role sharing on frozen development requests."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

NAMES = re.compile(r"[\u3001/|]|\band\b", flags=re.I)
CN_STEMS = frozenset(("CN", "ZL", "label:\u7533\u8bf7\u53f7", "label:\u4e13\u5229\u53f7"))
MODERN_CN = re.compile(r"(?:19|20)[0-9]{2}[12389][0-9]{7}(?:\.[0-9X])?")
REGISTRY_CODE = re.compile(r"^(CN|ZL|US|EP|WO)[ -]*([0-9][0-9A-Z.]*)$", flags=re.I)
FIELD_PREFIX = re.compile(r"^([^0-9\s:]+)[\s:]*([0-9][0-9A-Z.]+)$")
PROJECT_IDENTIFIER = re.compile(r"^(.+?(?:no\.?|number|\u53f7))[\s:]+([a-z0-9][a-z0-9._/-]+)$")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bindings(text, kind, spans):
    results = set()
    for start, end in spans:
        token = unicodedata.normalize("NFKC", text[start:end]).strip().upper()
        if kind == "patent":
            suffix = unicodedata.normalize("NFKC", text[end:end + 2]).upper()
            if suffix == ".X" and "." not in token:
                token += suffix
            marker = REGISTRY_CODE.fullmatch(token)
            if marker:
                results.add(("patent", marker[1], marker[2]))
                continue
            marker = FIELD_PREFIX.fullmatch(token)
        else:
            marker = PROJECT_IDENTIFIER.fullmatch(token.lower())
        if marker:
            results.add((kind, "label:" + marker[1], marker[2]))
    return sorted(results)


def patent_names(text, spans, project):
    options = set()
    for start, end in spans:
        for roster in project.INVENTORS.finditer(text[start:end]):
            options.add(tuple(sorted({project.name_key(name) for name in NAMES.split(roster[1]) if project.name_key(name)})))
    return sorted(options)


def prepare(dataset, projection_path, output):
    from experiments.retrieval import enterprise_project_role_rescue as project

    manifest_path = dataset / "manifest.json"
    manifest = read(manifest_path)
    projection = read(projection_path)
    if projection["manifest_sha256"] != sha(manifest_path):
        raise ValueError("Source cue and raw dataset manifests differ")
    people = project.profile_subset(dataset, "experts", manifest)
    corpus = project.profile_subset(dataset, "corpus", manifest)
    if corpus.keys() != projection["flags"].keys():
        raise ValueError("Sparse or changed source inventory")
    selected = {}
    for source, record in corpus.items():
        item = projection["flags"][source]
        kind = item["field"]
        if kind not in {"patent", "project"}:
            continue
        text = record["text"]
        clauses = [(m.start(), m.end()) for m in project.BREAKS.finditer(text)]
        clause_records = []
        for start, end in clauses:
            ids = [(left, right) for left, right in item["locators"] if start <= left < right <= end]
            if not ids:
                continue
            keys = bindings(text, kind, ids)
            if len(keys) == 1:
                row = {"span": [start, end], "key": keys[0]}
                if kind == "patent":
                    row["rosters"] = patent_names(text, [(start, end)], project)
                clause_records.append(row)
        selected[source] = {
            "kind": kind, "owner": record["expert_id"],
            "located": bool(item["locators"]), "local_named": bool(item["named"]),
            "typed_keys": bindings(text, kind, item["locators"]),
            "clause_records": clause_records,
            "rosters": patent_names(text, clauses, project) if kind == "patent" else [],

        }
    output.write_text(json.dumps({
        "manifest_sha256": sha(manifest_path),
        "source_sha256": projection["source_sha256"],
        "owners": {key: project.name_key(value["name"] or "") for key, value in people.items()},
        "sources": selected,
    }, sort_keys=True))


def locator_class(key):
    kind, system, code = key
    if kind == "project":
        return "project_label"
    if system in CN_STEMS and MODERN_CN.fullmatch(code.upper()):
        return "modern_cn_application_shape"
    if system in {"CN", "ZL", "US", "EP", "WO"}:
        return "other_" + system.lower() + "_form"
    return "patent_label_other_form"


def summarize(projection, trace, corpus_path):
    if sha(corpus_path) != projection["source_sha256"]:
        raise ValueError("The search replay and source projection must share a corpus")
    owners, sources = projection["owners"], projection["sources"]
    name_counts = Counter(value for value in owners.values() if value)
    id_groups = defaultdict(dict)
    for source, row in sources.items():
        if row["kind"] != "patent":
            continue
        for clause in row["clause_records"]:
            group = id_groups[tuple(clause["key"])]
            group.setdefault(source, []).extend(clause["rosters"])

    statuses = defaultdict(Counter)
    keys_by_owner = defaultdict(lambda: defaultdict(set))
    clause_keys_by_owner = defaultdict(lambda: defaultdict(set))
    cross_named = set()
    cross_named_owner = set()
    for entries in id_groups.values():
        if len({sources[source]["owner"] for source in entries}) < 2:
            continue
        role_lists = {tuple(names) for rosters in entries.values() for names in rosters}
        counts = statuses["same_identifier_patent_groups"]
        counts.update(groups=1, rows=len(entries), with_roster=bool(role_lists), multiple_rosters=len(role_lists) > 1)
        if len(role_lists) != 1:
            continue
        names, = role_lists
        for source, own_rosters in entries.items():
            owner = sources[source]["owner"]
            owner_name = owners[owner]
            if not own_rosters and owner_name and name_counts[owner_name] == 1 and owner_name in names:
                cross_named.add(source)
                cross_named_owner.add(owner)

    status_by_owner = defaultdict(set)
    for source, row in sources.items():
        kind, owner = row["kind"], row["owner"]
        if owner not in owners:
            raise ValueError("Unknown profile owner")
        counts = statuses[kind]
        counts.update(
            sources=1, any_locator=row["located"], one_typed_key=len(row["typed_keys"]) == 1,
            multiple_typed_keys=len(row["typed_keys"]) > 1, local_named=row["local_named"],
            singleton_and_local_named=len(row["typed_keys"]) == 1 and row["local_named"],
            new_named_from_other_profile=source in cross_named, raw_roster_field=bool(row["rosters"]),
            sources_with_typed_clause=bool(row["clause_records"]),
            typed_clauses=len(row["clause_records"]),
            named_patent_clauses=sum(
                1 for clause in row["clause_records"]
                if any(owners[owner] in roster for roster in clause.get("rosters", []))
            ),
        )
        if row["located"]:
            status_by_owner[owner].add(f"{kind}_locator")
        if row["local_named"]:
            status_by_owner[owner].add(f"{kind}_local_named")
        if len(row["typed_keys"]) == 1:
            keys_by_owner[owner][kind].add(tuple(row["typed_keys"][0]))
        for clause in row["clause_records"]:
            clause_keys_by_owner[owner][kind].add(tuple(clause["key"]))
    for owner, registry in keys_by_owner.items():
        for kind, keys in registry.items():
            status_by_owner[owner].add(f"{kind}_one_or_more_typed")
            if len(keys) >= 2:
                status_by_owner[owner].add(f"{kind}_two_or_more_typed")
    for registry in clause_keys_by_owner.values():
        for key in registry.get("patent", ()):
            statuses["patent_clause_owner_identifier_pairs_by_form"][locator_class(key)] += 1
    for owner, registry in clause_keys_by_owner.items():
        for kind, keys in registry.items():
            status_by_owner[owner].add(f"{kind}_clause_one_or_more_typed")
            if len(keys) >= 2:
                status_by_owner[owner].add(f"{kind}_clause_two_or_more_typed")
        modern = {key for key in registry.get("patent", ()) if locator_class(key) == "modern_cn_application_shape"}
        if modern:
            status_by_owner[owner].add("patent_modern_cn_application_shape")
        if len(modern) >= 2:
            status_by_owner[owner].add("patent_two_modern_cn_application_shapes")
    for owner in cross_named_owner:
        status_by_owner[owner].add("patent_cross_named")

    scope_names = (
        "patent_locator", "patent_one_or_more_typed", "patent_two_or_more_typed", "patent_local_named",
        "patent_cross_named", "project_locator", "project_one_or_more_typed",
        "project_two_or_more_typed", "project_local_named",
        "patent_clause_one_or_more_typed", "patent_clause_two_or_more_typed",
        "project_clause_one_or_more_typed", "project_clause_two_or_more_typed",
        "patent_modern_cn_application_shape", "patent_two_modern_cn_application_shapes",
    )
    for owner_details in status_by_owner.values():
        statuses["inventory_owner_flags"].update(owner_details)
    statuses["inventory_owner_flags"]["owners_in_manifest"] = len(owners)
    statuses["inventory_owner_flags"]["owners_with_some_role_or_locator_flag"] = len(status_by_owner)

    def record(bucket, candidates):
        if len(candidates) != len(set(candidates)) or set(candidates) - owners.keys():
            raise ValueError("Invalid owner list")
        statuses[bucket].update(requests=1, candidate_slots=len(candidates))
        for name in scope_names:
            matches = sum(name in status_by_owner.get(owner, ()) for owner in candidates)
            statuses[bucket].update({f"{name}_slots": matches, f"{name}_requests": bool(matches)})

    observed = trace["observations"]
    if len(observed) != 80 or len({x["id"] for x in observed}) != 80:
        raise ValueError("Expected the pinned development request trace")
    for item in observed:
        result = item["arms"]["hybrid_prefixes"]
        issued = [row["expert"] for row in result["issued"]]
        record("all_shortlists", result["shortlist"])
        if issued:
            record("issued_slots", issued)
        else:
            record("abstained_shortlists", result["shortlist"])
        if len(result["plan"]) != len(result["shortlist"]):
            raise ValueError("Excerpt plan and shortlist do not align")
        plan_evidence = [view for spans in result["plan"] for view in spans]
        eligible = [(source, start, end) for source, start, end in plan_evidence if source in sources]
        statuses["planned_excerpts"].update(total=len(plan_evidence), project_or_patent=len(eligible))
        reached_in_excerpt = set()
        for source, start, end in eligible:
            row = sources[source]
            local_clause = any(start <= a < b <= end for a, b in (c["span"] for c in row["clause_records"]))
            if local_clause:
                reached_in_excerpt.add(row["kind"])
            statuses["planned_excerpts"].update({
                f"{row['kind']}_typed_clause_inside_excerpt": local_clause,
                f"{row['kind']}_source_has_locator": row["located"],
                f"{row['kind']}_source_has_named": row["local_named"],
                f"{row['kind']}_source_has_one_typed": len(row["typed_keys"]) == 1,
            })
        statuses["planned_excerpts"].update({f"{kind}_requests_with_typed_clause_in_excerpt": 1 for kind in reached_in_excerpt})
    return {scope: dict(sorted(counts.items())) for scope, counts in sorted(statuses.items())}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("prepare")
    for name in ("dataset", "cues", "output"):
        extract.add_argument(name, type=Path)
    audit = commands.add_parser("audit")
    for name in ("projection", "trace", "corpus", "output"):
        audit.add_argument(name, type=Path)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.cues, args.output)
    else:
        result = summarize(read(args.projection), read(args.trace), args.corpus)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
