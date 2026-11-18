"""Capture a separate publisher inventory for a frozen wording replication."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode

from experiments.credit.counterfactual_role_edits import article_url, matched_groups, write_json
from experiments.credit.cross_source_identity import load
from experiments.credit.plos_role_inventory import declared_roles, read_url


QUERY = (
    'title:("speech recognition" OR "sentiment analysis" OR "knowledge graph" OR '
    '"machine reading comprehension" OR "question answering") '
    'AND journal:"PLOS ONE" AND publication_date:[2021-01-01T00:00:00Z TO 2026-01-01T00:00:00Z]'
)
URL = "https://api.plos.org/search?" + urlencode(dict(q=QUERY, rows=60, fl="id", wt="json",
                                                     sort="publication_date desc,id asc"))


def capture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    catalog = folder / "search.json"
    if not catalog.exists():
        body = read_url(URL)
        ids = [row["id"] for row in json.loads(body)["response"]["docs"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Invalid new PLOS search capture")
        catalog.write_bytes(body)
    rows = json.loads(catalog.read_bytes())["response"]["docs"]
    report_path = folder / "capture-report.json"
    previous = json.loads(report_path.read_bytes()) if report_path.exists() else {}

    def fetch(row):
        doi = row["id"]
        suffix = doi.removeprefix("10.1371/journal.pone.")
        if not suffix.isdecimal():
            raise ValueError(f"Unexpected DOI in capture: {doi}")
        path = folder / (suffix + ".xml")
        if previous.get(doi) == "different_license":
            return doi, "different_license"
        if not path.exists():
            raw = read_url(article_url(doi))
            try:
                declared_roles(raw, doi)
            except ValueError as error:
                if str(error) != f"Unexpected source license: {doi}":
                    raise
                return doi, "different_license"
            path.write_bytes(raw)
        return doi, "captured"

    with ThreadPoolExecutor(max_workers=4) as workers:
        report = dict(workers.map(fetch, rows))
    write_json(report_path, report)
    print("Capture", dict(Counter(report.values())), flush=True)


def build(folder, previous, manifest_path, output):
    if output.exists():
        raise FileExistsError(output)
    catalog = (folder / "search.json").read_bytes()
    catalog_ids = {r["id"] for r in json.loads(catalog)["response"]["docs"]}
    capture_results = json.loads((folder / "capture-report.json").read_bytes())
    if catalog_ids != capture_results.keys() or set(capture_results.values()) - {"captured", "different_license"}:
        raise ValueError("Incomplete publisher capture report")
    ids = {doi for doi, status in capture_results.items() if status == "captured"}
    old = json.loads(manifest_path.read_bytes())["input_sources"]
    documents = load([*previous, folder])
    for row in old:
        if documents[row["doi"]]["source_sha256"] != row["source_sha256"]:
            raise ValueError("Changed prior source inventory")
    prior = {row["doi"] for row in old}
    people = {a["person"] for d in prior for a in documents[d]["authors"] if a["person"]}
    names = {a["name"] for d in prior for a in documents[d]["authors"] if a["name"]}
    sources, groups, excluded = [], [], Counter()
    for doi in sorted(ids):
        source = documents[doi]
        roles = [a["roles"] for a in source["authors"]]
        if doi in prior:
            excluded["shared_doi"] += 1
            continue
        if people.intersection(a["person"] for a in source["authors"]):
            excluded["shared_published_identifier"] += 1
            continue
        if names.intersection(a["name"] for a in source["authors"]):
            excluded["shared_normalized_name"] += 1
            continue
        if len(roles) < 2 or not all(roles):
            excluded["incomplete_declarations"] += 1
            continue
        matches = matched_groups(doi, roles, cap=1)
        if not matches:
            excluded["no_matched_edits"] += 1
            continue
        for row in matches:
            row["split"] = "test"
        groups.extend(matches)
        source = {key: value for key, value in source.items() if key != "authors"}
        sources.append(dict(source, roles_by_author=roles, split="test"))
    if not sources:
        raise ValueError("No eligible replication source")
    output.mkdir(parents=True)
    pairs = output / "pairs.jsonl"
    with pairs.open("x") as stream:
        for row in groups:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    write_json(output / "manifest.json", dict(
        version="plos-credit-external-v1", catalog_sha256=sha256(catalog).hexdigest(),
        catalog_url=URL, prior_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        pairs_sha256=sha256(pairs.read_bytes()).hexdigest(), sources=sources, excluded=excluded,
        capture_exclusions=sorted(catalog_ids - ids),
        inventory=[{k: documents[doi][k] for k in ("doi", "url", "source_sha256")} for doi in sorted(ids)],
    ))
    write_json(output / "source-audit.json", dict(
        allowed=[row["doi"] for row in sources],
        fixtures={"transfer": sha256((output / "manifest.json").read_bytes()).hexdigest()},
    ))
    print("Eligible articles", len(sources), "groups", len(groups), dict(excluded), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    harvest = subs.add_parser("capture")
    harvest.add_argument("folder", type=Path)
    fixture = subs.add_parser("build")
    for name in ("folder", "prior_manifest", "output"):
        fixture.add_argument(name, type=Path)
    fixture.add_argument("previous", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.folder)
    else:
        build(args.folder, args.previous, args.prior_manifest, args.output)
