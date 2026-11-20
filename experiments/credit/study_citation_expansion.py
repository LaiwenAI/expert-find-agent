"""Stress-test source identification with externally cited eLife records."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from experiments.credit.study_citation_anchor import evaluate, load_data, node_tokens, normalized_doi
from experiments.followir.core17_reranking import digest, write_json


def external_targets(catalog, bibliography):
    eligible = bibliography.keys() - catalog.keys()
    return sorted(
        (doi for doi in eligible if re.fullmatch(r"10\.7554/elife\.\d{5}", doi)),
        key=lambda doi: sha256(("elife-citation-expansion-v1:" + doi).encode()).digest(),
    )


def revision_files(catalog, bibliography, count):
    return {
        f"elife-{doi.rsplit('.', 1)[-1]}-v1.xml": doi
        for doi in external_targets(catalog, bibliography)[:count]
    }


def augment(catalog, directory, files, receipts):
    augmented = catalog.copy()
    exclusions = Counter()
    for filename, doi in files.items():
        expected = receipts[filename]
        if expected is None:
            exclusions["not_found"] += 1
            continue
        payload = (directory / filename).read_bytes()
        if sha256(payload).hexdigest() != expected:
            raise ValueError(f"Unexpected revision of {filename}")
        article = ET.fromstring(payload)
        if article.get("article-type") != "research-article":
            exclusions["article_type"] += 1
            continue
        meta = article.find("front/article-meta")
        if normalized_doi(meta.findtext("article-id[@pub-id-type='doi']", "")) != doi:
            raise ValueError(f"Unexpected article DOI in {filename}")
        title = node_tokens(meta.find("title-group/article-title"))
        surname = node_tokens(meta.find("contrib-group/contrib[@contrib-type='author']/name/surname"))
        years = {
            date.findtext("year", "")
            for date in meta.findall("pub-date[@publication-format='electronic']")
            if date.get("date-type") in {"pub", "publication"}
        }
        year = next(iter(years)) if len(years) == 1 else ""
        if not title or not surname or not re.fullmatch(r"\d{4}", year):
            exclusions["metadata"] += 1
            continue
        augmented[doi] = (title, surname, year)
    return augmented, exclusions


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("xml", type=Path)
    action = cli.add_subparsers(dest="action", required=True)
    action.add_parser("list")
    run = action.add_parser("run")
    run.add_argument("added_xml", type=Path)
    run.add_argument("receipts", type=Path)
    run.add_argument("output", type=Path)
    args = cli.parse_args()
    catalog, bibliography = load_data(args.manifest, args.xml)
    files = revision_files(catalog, bibliography, count=192)
    if args.action == "list":
        print("\n".join(files))
        return
    receipts = json.loads(args.receipts.read_bytes())
    if set(receipts) != set(files):
        raise ValueError("Download receipts do not match the prescribed sample")
    augmented, exclusions = augment(catalog, args.added_xml, files, receipts)
    # Keep the original evaluation as a separate panel. Assess the recall
    # change on genuinely external citations, excluding all original targets.
    external = {doi: value for doi, value in bibliography.items() if doi not in catalog}
    result = evaluate(augmented, external)
    old_result = evaluate(catalog, external)
    added_targets = set(augmented) - set(catalog)
    metadata_agreement = Counter()
    for doi in added_targets:
        _, catalog_surname, catalog_year = augmented[doi]
        _, cited_surname, cited_year = bibliography[doi]
        surname_agrees = catalog_surname == cited_surname
        year_agrees = catalog_year == cited_year
        metadata_agreement.update(
            first_surname=int(surname_agrees), year=int(year_agrees),
            first_surname_and_year=int(surname_agrees and year_agrees),
        )
    write_json(args.output, dict(
        manifest_sha256=digest(args.manifest), receipts_sha256=digest(args.receipts),
        script_sha256=digest(Path(__file__)), sampled=len(files), exclusions=exclusions,
        added_articles=len(added_targets), catalog_articles=len(augmented),
        tested_external_citations=len(external), metadata_agreement=metadata_agreement,
        before_missing=old_result["outcomes"]["bibliography_missing"],
        after_present=result["outcomes"]["bibliography_present"],
        after_missing=result["outcomes"]["bibliography_missing"],
    ))


if __name__ == "__main__":
    main()
