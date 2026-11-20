"""Evaluate incomplete source anchors against publisher bibliography DOIs."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from experiments.credit.study_anchor_lookup import Index, tokens
from experiments.credit.study_anchor_omission import omission_matches, unique_available
from experiments.followir.core17_reranking import digest, write_json


DOI_TAG = "pub-id[@pub-id-type='doi']"
FIRST_SURNAME = "person-group[@person-group-type='author']/name/surname"


def node_tokens(node):
    return () if node is None else tokens(" ".join(node.itertext()))


def normalized_doi(value):
    candidate = value.strip().casefold()
    return candidate if re.fullmatch(r"10\.\d{4,9}/\S+", candidate) else ""


def load_data(manifest_path, directory):
    manifest = json.loads(manifest_path.read_bytes())
    catalog, bibliography = {}, {}
    for entry in sorted(manifest["source_inventory"], key=lambda row: row["filename"]):
        if entry["exclusion"] is not None:
            continue
        raw = (directory / entry["filename"]).read_bytes()
        if sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Invalid source checksum for {entry['filename']}")
        article = ET.fromstring(raw)
        meta = article.find("front/article-meta")
        doi = normalized_doi(meta.findtext("article-id[@pub-id-type='doi']", ""))
        if not doi or doi in catalog:
            raise ValueError("Invalid or repeated catalog DOI")
        title = node_tokens(meta.find("title-group/article-title"))
        surname = node_tokens(meta.find("contrib-group/contrib[@contrib-type='author']/name/surname"))
        year = meta.findtext("pub-date[@date-type='publication']/year", "")
        if not title or not surname or not re.fullmatch(r"\d{4}", year):
            raise ValueError(f"Missing article metadata for {doi}")
        catalog[doi] = (title, surname, year)

        for citation in article.findall("back/ref-list/ref/element-citation"):
            target = normalized_doi(citation.findtext(DOI_TAG, ""))
            if not target or target in bibliography or target == doi:
                continue
            clue = node_tokens(citation.find("article-title"))
            quoted_author = node_tokens(citation.find(FIRST_SURNAME))
            quoted_year = citation.findtext("year", "").strip()
            if len(clue) >= 6 and quoted_author and re.fullmatch(r"\d{4}", quoted_year):
                bibliography[target] = (clue, quoted_author, quoted_year)
    return catalog, bibliography


def views(doi, title, candidate_views):
    seed = int.from_bytes(sha256(doi.encode()).digest()[:8], "big")
    if "full" in candidate_views:
        yield "full", title
    if "short" in candidate_views:
        offset = seed % (len(title) - 5)
        yield "short", title[offset:offset + 3]
    if "one_omission" in candidate_views:
        offset = seed % (len(title) - 5)
        span = title[offset:offset + 6]
        cut = 1 + seed % 4
        yield "one_omission", span[:cut] + span[cut + 1:]


def evaluate(catalog, bibliography):
    index = Index({doi: (value[0], 0) for doi, value in catalog.items()})
    outcomes = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

    def score(doi, citation, panel, candidate_views, hidden=None):
        title, surname, year = citation
        for style, clue in views(doi, title, candidate_views):
            candidates = index.matches(clue)
            if style == "one_omission":
                candidates, _ = omission_matches(index, clue, candidates)
            named = {item for item in candidates if catalog[item][1] == surname}
            dated = {item for item in named if catalog[item][2] == year}
            for method, matches in (("title", candidates), ("title_surname", named),
                                    ("title_surname_year", dated)):
                chosen = unique_available(matches, hidden, visible_only=hidden is not None)
                outcomes[panel][style][method].update(
                    cases=1, correct=int(chosen == doi), unresolved=int(chosen is None),
                    wrong=int(chosen is not None and chosen != doi),
                )

    for doi, citation in sorted(bibliography.items()):
        panel = "bibliography_present" if doi in catalog else "bibliography_missing"
        score(doi, citation, panel, ("full", "short", "one_omission"))
    for doi, citation in sorted(catalog.items()):
        if len(citation[0]) >= 6:
            score(doi, citation, "metadata_present", ("short", "one_omission"))
            score(doi, citation, "metadata_removed", ("short", "one_omission"), hidden=doi)

    return dict(catalog_articles=len(catalog), distinct_cited_dois=len(bibliography),
                exact_doi_matches=len(bibliography.keys() & catalog.keys()), outcomes=outcomes)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("xml", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    catalog, bibliography = load_data(args.manifest, args.xml)
    write_json(args.output, dict(manifest_sha256=digest(args.manifest),
                                script_sha256=digest(Path(__file__)),
                                **evaluate(catalog, bibliography)))
