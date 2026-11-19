"""Recover candidate DOIs for frozen citations using exact public title matches."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import urlencode, urlsplit

from experiments.retrieval import enterprise_doi_role_availability as archive
from experiments.retrieval.check_joint_review import check, load
from experiments.retrieval.enterprise_routed_review import digest, write


CROSSREF = "https://api.crossref.org/works"
CONTROL_TITLE = "The use of the Internet and social media by young people"
TAG = re.compile(r"<[^>]+>")


def title_key(title):
    return archive.case_name(unescape(TAG.sub("", title)))


def cited_papers(packet, sidecar, raw_path):
    forms = load(packet / "private-mapping.jsonl")
    expected = {s["id"]: row["expert_id"] for row in forms.values() for s in row["sources"] if s["id"].startswith("paper:")}
    metadata = sidecar / "papers.jsonl"
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    if digest(metadata) != manifest["metadata_sha256"] or digest(raw_path) != manifest["source_sha256"]:
        raise ValueError("Changed publisher projection lineage")
    with metadata.open(encoding="utf-8") as stream:
        rows = [row for line in stream if (row := json.loads(line))["source_id"] in expected]
    if len(rows) != len(expected) or any(row["expert_id"] != expected[row["source_id"]] for row in rows):
        raise ValueError("Incomplete or inconsistent citation join")
    by_line = {row["raw_line"]: row for row in rows}
    with raw_path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if number not in by_line:
                continue
            row = json.loads(line)
            side = by_line[number]
            raw_hash = sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if raw_hash != side["raw_record_sha256"]:
                raise ValueError("Changed cited record")
            title = row.get("title", "").strip()
            start = side["publication"]["start"]
            yield side["source_id"], title, int(start[:4]) if start else None, side["doi"]


def years(record):
    return {record[x]["date-parts"][0][0] for x in ("issued", "published", "published-print", "published-online")
            if x in record and record[x].get("date-parts") and record[x]["date-parts"][0]}


def inspect(paper):
    source, title, year, stored = paper
    query = urlencode({"query.title": title, "rows": 8})
    response, status = archive.get(f"{CROSSREF}?{query}")
    row = dict(source=source, query_status=status, stored_doi=stored, local_publication_year_present=year is not None)
    if status != "ok":
        return row
    result = json.loads(response)["message"]["items"]
    matching = {}
    for article in result:
        doi = article.get("DOI", "").casefold()
        if not archive.DOI.fullmatch(doi):
            continue
        if title_key(title) not in {title_key(t) for t in article.get("title", [])}:
            continue
        if year is not None and year not in years(article):
            continue
        matching[doi] = article
    row["exact_title_year_dois"] = sorted(matching)
    if len(matching) == 1:
        doi, article = next(iter(matching.items()))
        row.update(
            repaired_doi=doi,
            license_hosts=sorted({host for item in article.get("license", []) if
                                  (host := urlsplit(item.get("URL", "")).hostname)}),
            crossref_has_author_roles=any(author.get("role") for author in article.get("author", [])),
        )
    return row


def run(packet, sidecar, raw, output):
    check(packet, packet / "blind-review.jsonl")
    output.mkdir(mode=0o700)
    papers = list(cited_papers(packet, sidecar, raw))
    if not papers or any(not title_key(row[1]) for row in papers):
        raise ValueError("Missing citation titles")
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(inspect, papers))
    repaired = {row["repaired_doi"] for row in outcomes if row.get("repaired_doi")}
    reports = archive.lookup(output, sorted(repaired | {archive.CONTROL}))
    counts = Counter(row["query_status"] for row in outcomes)
    for row in outcomes:
        doi = row.get("repaired_doi")
        if not doi:
            counts["multiple_exact_dois"] += len(row.get("exact_title_year_dois", [])) > 1
            continue
        old = row["stored_doi"]
        counts["unique_exact_title_doi"] += 1
        counts["matched_existing_doi"] += bool(old and old.casefold() == doi)
        counts["changed_existing_doi"] += bool(old and old.casefold() != doi)
        counts["filled_missing_doi"] += not old
        counts["matched_without_local_year"] += not row["local_publication_year_present"]
        recovered = reports[doi]
        counts["papers_with_pmcid"] += bool(recovered["pmcid"])
        counts["papers_with_jats"] += recovered["xml_status"] == "ok"
        counts["papers_with_contribution_prose"] += recovered.get("contribution_prose", False)
        counts["papers_with_structured_roles"] += recovered.get("structured_roles", False)
        counts["papers_with_crossref_author_roles"] += row["crossref_has_author_roles"]
        counts["papers_with_creative_commons_license_link"] += "creativecommons.org" in row["license_hosts"]
    write(output / "doi-candidates-private.json", outcomes)
    write(output / "summary.json", dict(
        paper_views=len(papers), counts=dict(counts), unique_recovered_dois=len(repaired),
        positive_control_jats=reports[archive.CONTROL]["xml_status"] == "ok",
        capture_wall_seconds=perf_counter()-start, case_manifest_sha256=digest(packet / "manifest.json"),
        code_sha256=digest(Path(__file__)), archive_code_sha256=digest(Path(archive.__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("packet", "sidecar", "raw", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
