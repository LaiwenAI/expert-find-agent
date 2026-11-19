"""Audit known exposure and freeze a local enterprise panel without using labels."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from math import ceil, log, log1p
from pathlib import Path
import re
import unicodedata

from openpyxl import load_workbook

from experiments.bilingual.check_validation_containment import grams
from experiments.prepare_fresh_method_data import normalized_hash


SEED = "enterprise-blind-panel-v1:20261119"
SUFFIX = re.compile(
    r"(?:[\s,.]*(?<![a-z])(?:limited|ltd|llc|inc|corporation|corp)\.?|"
    r"\u80a1\u4efd\u6709\u9650\u516c\u53f8|\u6709\u9650\u8d23\u4efb\u516c\u53f8|"
    r"\u6709\u9650\u516c\u53f8)$"
)
EXACT_ID = re.compile(r"demand-[0-9]+\Z")
ID_FIELDS = {"id", "query_id", "parent_id", "demand_id", "parent_demand_id",
             "source_demand_id", "parent_ids", "selected_parent_ids", "demand_ids"}


def digest_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def normalize(text):
    return unicodedata.normalize("NFKC", text).strip().casefold()


def issuer_key(name):
    normalized = " ".join(normalize(name).split())
    return re.sub(r"[\W_]+", "", SUFFIX.sub("", normalized)).strip()


def exposure_ids(path):
    def collect(value, active=False):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from collect(child, key in ID_FIELDS)
        elif isinstance(value, list):
            for child in value:
                yield from collect(child, active)
        elif active and isinstance(value, str) and EXACT_ID.fullmatch(value):
            yield value

    rows = read_rows(path) if path.suffix == ".jsonl" else json.loads(path.read_bytes())
    return set(collect(rows))


def workbook_rows(workbook, wanted):
    keys = {"title": "\u547d\u9898\u540d\u79f0", "body": "\u547d\u9898\u5185\u5bb9\u53ca\u7b54\u9898\u8981\u6c42",
            "company": "\u547d\u9898\u4f01\u4e1a", "url": "\u8be6\u60c5\u94fe\u63a5"}
    book = load_workbook(workbook, read_only=True, data_only=True)
    records = {}
    try:
        rows = iter(book.active.values)
        header = next(rows)
        positions = {key: header.index(field) for key, field in keys.items()}
        for index, row in enumerate(rows, 2):
            identifier = f"demand-{index}"
            if identifier not in wanted:
                continue
            title, body = (row[positions[key]] for key in ("title", "body"))
            if title and body:
                records[identifier] = dict(
                    id=identifier, zh=f"{title}\n{body}", source_row=index,
                    company=str(row[positions["company"]] or ""),
                    url=str(row[positions["url"]] or ""), split="prior", group_id=None,
                )
    finally:
        book.close()
    if records.keys() != wanted:
        raise ValueError("Prior demand references do not match complete workbook rows")
    return records


def screen(rows, declared, prior_hashes):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Expected disjoint unique input identifiers")
    sealed = {row["id"] for row in rows if row["split"] == "sealed"}
    parent = {identifier: identifier for identifier in by_id}

    def root(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left, right):
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact = {}
    exposure = set(by_id) - sealed | declared
    for row in rows:
        key = row["id"]
        body = row["zh"].partition("\n")[2]
        full_key, body_key = normalized_hash(row["zh"]), normalized_hash(body)
        if prior_hashes & {full_key, body_key}:
            exposure.add(key)
        resources = dict(component=row["group_id"], issuer=issuer_key(row["company"]),
                         body=body_key if body else "", full=full_key, url=normalize(str(row["url"] or "")))
        for kind, value in resources.items():
            if value:
                resource = kind, value
                union(key, exact.setdefault(resource, key))

    sizes = {}
    postings = defaultdict(list)
    for identifier in sorted(sealed):
        tokens = grams(normalize(by_id[identifier]["zh"]))
        sizes[identifier] = len(tokens)
        for token in tokens:
            postings[token].append(identifier)

    near_exposure = near_sealed = 0
    for row in rows:
        identifier = row["id"]
        tokens = grams(normalize(row["zh"]))
        matches = Counter(candidate for token in tokens for candidate in postings.get(token, ()))
        for candidate, overlap in matches.items():
            if identifier in sealed and identifier >= candidate:
                continue
            smaller = min(len(tokens), sizes[candidate])
            union_size = len(tokens) + sizes[candidate] - overlap
            # High symmetric containment catches a short request copied into a longer one.
            if overlap >= 64 and (overlap * 5 >= smaller * 4 or overlap * 20 >= union_size * 13):
                union(identifier, candidate)
                if identifier in sealed:
                    near_sealed += 1
                else:
                    near_exposure += 1
    blocked = {root(identifier) for identifier in exposure if identifier in by_id}
    eligible = defaultdict(list)
    for identifier in sorted(sealed):
        row = by_id[identifier]
        if root(identifier) not in blocked and issuer_key(row["company"]) and sizes[identifier] >= 64:
            eligible[root(identifier)].append(identifier)

    selected = []
    for members in eligible.values():
        leader = min(members, key=lambda key: sha256(f"{SEED}:{key}".encode()).digest())
        row = by_id[leader]
        selected.append(dict(id=leader, workbook_row=row["source_row"],
                             text_sha256=sha256(row["zh"].encode()).hexdigest(),
                             component_sha256=sha256("\n".join(sorted(members)).encode()).hexdigest(),
                             component_size=len(members)))
    selected.sort(key=lambda row: sha256(f"{SEED}:panel:{row['id']}".encode()).digest())
    report = dict(
        source_sealed_rows=len(sealed), original_sealed_groups=len({by_id[key]["group_id"] for key in sealed}),
        directly_exposed_sealed_rows=len(exposure & sealed),
        sealed_rows_linked_to_exposure=sum(root(key) in blocked for key in sealed),
        lexical_links_to_other_splits=near_exposure, lexical_links_within_sealed=near_sealed,
        eligible_rows=sum(map(len, eligible.values())), selected_groups=len(selected),
    )
    return selected, report


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    workbook_hash = digest_file(args.workbook)
    protocol = json.loads((args.cohort / "protocol.json").read_bytes())
    if protocol["inputs_sha256"].get(str(args.workbook)) != workbook_hash:
        raise ValueError("Workbook differs from the cohort protocol")
    cohort_file = args.cohort / "queries-zh.jsonl"
    if protocol["outputs_sha256"]["queries"] != digest_file(cohort_file):
        raise ValueError("Changed original-language cohort")
    legacy_file = args.legacy / "queries.jsonl"
    legacy_manifest = json.loads((args.legacy / "manifest.json").read_bytes())
    if legacy_manifest["files_sha256"]["queries.jsonl"] != digest_file(legacy_file):
        raise ValueError("Changed legacy enterprise evaluation cohort")
    cohort = read_rows(cohort_file)
    cohort_ids = {row["id"] for row in cohort}
    registry = json.loads(args.registry.read_bytes())
    known = {row["id"] for row in read_rows(legacy_file)} | set(registry["demand_ids"])
    declared = set().union(*(exposure_ids(path) for path in args.exposure))
    # Programmatic comparisons stay local. Publish only pooled diagnostics.
    old = workbook_rows(args.workbook, (known | declared) - cohort_ids)
    rows = cohort + list(old.values())
    selected, result = screen(rows, declared | known, set(registry["query_hashes"]))
    plan = dict(seed=SEED, source_sha256=workbook_hash, records=selected)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "sealed-index.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    n = len(selected)
    result.update(
        input_sha256=dict(workbook=workbook_hash, cohort=digest_file(cohort_file),
                          legacy=digest_file(legacy_file), registry=digest_file(args.registry),
                          exposure_manifests=[digest_file(path) for path in args.exposure]),
        known_prior_workbook_rows=len(old), registered_recent_parent_ids=len(declared),
        script_sha256=digest_file(Path(__file__)), sealed_index_sha256=digest_file(path),
        zero_event_upper_95_if_independent=1 - .05 ** (1 / n) if n else None,
        groups_for_example_five_percent_gate=ceil(log(.05) / log1p(-.05)),
        source_texts_exported=0, model_queries_run=0, human_labels=0,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "cohort", "legacy", "registry", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--exposure", type=Path, action="append", required=True)
    run(parser.parse_args())
