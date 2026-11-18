"""Compile a small, publisher-declared contributor-role pilot from open JATS."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET


QUERY = ('title:("neural machine translation" OR "information retrieval" OR "recommendation system") '
         'AND journal:"PLOS ONE" AND publication_date:[2022-01-01T00:00:00Z TO 2026-01-01T00:00:00Z]')
SEARCH = "https://api.plos.org/search?" + urlencode(dict(
    q=QUERY, rows=100, fl="id,title", sort="publication_date desc,id asc",
))
ROLES = ("Conceptualization", "Data curation", "Formal analysis", "Funding acquisition",
         "Investigation", "Methodology", "Project administration", "Resources", "Software",
         "Supervision", "Validation", "Visualization", "Writing – original draft", "Writing – review & editing")
LICENSE = "https://creativecommons.org/licenses/by/4.0/"


def read_url(url):
    with urlopen(url, timeout=35) as response:
        return response.read()


def declared_roles(xml, doi):
    document = ET.fromstring(xml)
    metadata = document.find("./front/article-meta")
    if metadata is None or metadata.findtext("article-id[@pub-id-type='doi']") != doi:
        raise ValueError(f"Unexpected article metadata: {doi}")
    license_info = metadata.find("./permissions/license")
    href = None if license_info is None else license_info.get("{http://www.w3.org/1999/xlink}href")
    if href not in (LICENSE, LICENSE.replace("https://", "http://")):
        raise ValueError(f"Unexpected source license: {doi}")
    membership = []
    for author in metadata.findall("./contrib-group/contrib[@contrib-type='author']"):
        roles = tuple(sorted({role.text.strip() for role in author.findall("role") if role.text}))
        if set(roles) - set(ROLES):
            raise ValueError(f"Unrecognized declared role: {doi}")
        membership.append(roles)
    return membership


def fetch(record, source_cache):
    doi = record["id"]
    suffix = doi.removeprefix("10.1371/journal.pone.")
    if not suffix.isdecimal():
        raise ValueError(f"Unexpected PLOS ONE DOI: {doi}")
    url = "https://journals.plos.org/plosone/article/file?" + urlencode(dict(id=doi, type="manuscript"))
    raw = (read_url(url) if source_cache is None else
           (source_cache / f"pone.{suffix}.xml").read_bytes())
    roles = declared_roles(raw, doi)
    rows = []
    if roles and all(roles):
        for role in ROLES:
            positives = [i for i, labels in enumerate(roles, 1) if role in labels]
            if positives and len(positives) < len(roles):
                rows.append({
                    "doi": doi, "publication_title": record["title"], "requested_role": role,
                    "matching_author_ordinals": positives,
                    "other_author_ordinals": [i for i, labels in enumerate(roles, 1) if role not in labels],
                    "source_field": "front/article-meta/contrib-group/contrib[@contrib-type='author']/role",
                })
    meta = {"doi": doi, "url": url, "source_sha256": sha256(raw).hexdigest(),
            "license": LICENSE, "attribution": f"https://doi.org/{doi}",
            "authors": len(roles), "authors_with_declared_role": sum(map(bool, roles)),
            "contrastive_queries": len(rows)}
    return suffix, raw, meta, rows


def main(directory, source_cache):
    if directory.exists():
        raise FileExistsError(directory)
    inventory = read_url(SEARCH) if source_cache is None else (source_cache / "search.json").read_bytes()
    response = json.loads(inventory)["response"]
    records = response["docs"]
    if response["numFound"] != len(records) or len({r["id"] for r in records}) != len(records):
        raise ValueError("Incomplete or duplicate PLOS inventory")
    with ThreadPoolExecutor(max_workers=2) as pool:
        articles = list(pool.map(lambda row: fetch(row, source_cache), records))
    directory.mkdir(parents=True, exist_ok=False)
    if source_cache is None:
        (directory / "search.json").write_bytes(inventory)
    rows, sources = [], []
    for suffix, raw, metadata, queries in articles:
        if source_cache is None:
            (directory / f"pone.{suffix}.xml").write_bytes(raw)
        rows.extend(queries)
        sources.append(metadata)
    outcomes = directory / "queries.json"
    outcomes.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n")
    manifest = {
        "search_url": SEARCH, "search_sha256": sha256(inventory).hexdigest(),
        "sources": sources, "papers": len(sources),
        "papers_with_complete_declared_roles": sum(
            row["authors"] > 0 and row["authors"] == row["authors_with_declared_role"] for row in sources),
        "papers_with_role_contrasts": len({row["doi"] for row in rows}),
        "queries": len(rows),
        "positive_author_labels": sum(len(row["matching_author_ordinals"]) for row in rows),
        "same_paper_role_absences": sum(len(row["other_author_ordinals"]) for row in rows),
        "rows_sha256": sha256(outcomes.read_bytes()).hexdigest(),
        "scope": ("A narrow PLOS ONE feasibility snapshot. The authoritative target is publisher-declared "
                  "contributorship in the cited JATS record. Absence is a missing declaration for that "
                  "role, not proof about the author's broader ability. This is not enterprise retrieval or human adjudication."),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "search_url")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args()
    main(args.output, args.source_cache)
