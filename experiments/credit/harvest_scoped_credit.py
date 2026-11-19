"""Capture publisher records with repeated exact ORCID links for evidence evaluation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

from experiments.credit.counterfactual_role_edits import article_url
from experiments.credit.cross_source_identity import normalized_name, orcid_id, xml_sources
from experiments.credit.plos_role_inventory import declared_roles, read_url


SEED_QUERY = (
    'journal:"PLOS ONE" AND title:("traffic prediction" OR "image segmentation" OR "federated learning") '
    'AND publication_date:[2026-01-01T00:00:00Z TO 2026-09-30T23:59:59Z]'
)
FOLLOWUP_RANGE = "2020-01-01T00:00:00Z TO 2026-09-30T23:59:59Z"
MAX_ANCHORS = 20
MAX_FOLLOWUP = 5
DOI_PREFIX = "10.1371/journal.pone."


def search(query, rows):
    return "https://api.plos.org/search?" + urlencode(dict(
        q=query, rows=rows, fl="id,author", wt="json", sort="publication_date asc,id asc",
    ))


def meta(raw):
    node = ET.fromstring(raw).find("./front/article-meta")
    if node is None:
        raise ValueError("Missing publisher article metadata")
    return node


def joined(authors):
    for author in authors:
        ids = {ident for elem in author.findall("contrib-id[@contrib-id-type='orcid']")
               if (ident := orcid_id(elem.text or ""))}
        if len(ids) == 1:
            given = author.findtext("name/given-names") or ""
            surname = author.findtext("name/surname") or ""
            if given and surname:
                yield next(iter(ids)), " ".join((given, surname)), normalized_name(author)


def authors(node):
    return node.findall("./contrib-group/contrib[@contrib-type='author']")


def prior(paths):
    dois, ids, names = set(), set(), set()
    for path in paths:
        for file in xml_sources(path):
            doc = meta(file.read_bytes())
            doi = doc.findtext("article-id[@pub-id-type='doi']")
            if doi:
                dois.add(doi)
            all_authors = authors(doc)
            names.update(value for a in all_authors if (value := normalized_name(a)))
            for a in all_authors:
                for item in a.findall("contrib-id[@contrib-id-type='orcid']"):
                    ident = orcid_id(item.text or "")
                    if ident:
                        ids.add(ident)
    return dois, ids, names


def is_article(doi):
    return doi.startswith(DOI_PREFIX) and doi[len(DOI_PREFIX):].isdecimal()


def get_xml(folder, doi):
    suffix = doi.removeprefix(DOI_PREFIX)
    if not is_article(doi):
        raise ValueError("Require publisher PLOS ONE DOI")
    path = folder / f"pone.{suffix}.xml"
    if path.exists():
        return path.read_bytes()
    raw = read_url(article_url(doi))
    declared_roles(raw, doi)
    with path.open("xb") as stream:
        stream.write(raw)
    return raw


def candidate_xml(folder, doi):
    try:
        return get_xml(folder, doi), None
    except (ValueError, ET.ParseError) as error:
        return None, str(error)


def linked_capture(folder, old_folders, output):
    if output.exists():
        raise FileExistsError(output)
    old_dois, old_ids, old_names = prior(old_folders)
    if not old_dois or not old_ids:
        raise ValueError("Require a prior-exposure inventory")
    seed_file = folder / "seed-search.json"
    if not seed_file.exists():
        seed_file.write_bytes(read_url(search(SEED_QUERY, 36)))
    initial = json.loads(seed_file.read_bytes())["response"]["docs"]
    seed_dois = sorted({row["id"] for row in initial if is_article(row["id"])} - old_dois)
    selected, scans, rejections = {}, [], {}
    with ThreadPoolExecutor(max_workers=2) as threads:
        seed_xml = list(threads.map(lambda doi:get_xml(folder,doi),seed_dois))
        for doi, raw in zip(seed_dois,seed_xml,strict=True):
            for ident, spelling, name in joined(authors(meta(raw))):
                if ident not in old_ids and name not in old_names:
                    selected.setdefault(ident, (spelling,doi))
        for ident,(spelling, seed) in sorted(selected.items(), key=lambda item:sha256(item[0].encode()).digest())[:MAX_ANCHORS]:
            escaped = spelling.replace("\\","\\\\").replace('"','\\"')
            query = (f'journal:"PLOS ONE" AND author:"{escaped}" '
                     f'AND publication_date:[{FOLLOWUP_RANGE}]')
            name = sha256(query.encode()).hexdigest()[:20]
            path = folder / f"author-search-{name}.json"
            if not path.exists():
                path.write_bytes(read_url(search(query,24)))
            response = path.read_bytes()
            records = json.loads(response)["response"]["docs"]
            # Follow only the bounded publisher result list. Retain links only after JATS validation.
            candidates = list(dict.fromkeys(row["id"] for row in records
                                            if is_article(row["id"]) and row["id"] not in old_dois and row["id"] != seed))[:MAX_FOLLOWUP]
            linked = [seed]
            fetched = threads.map(lambda doi:candidate_xml(folder,doi),candidates)
            for doi,(raw,error) in zip(candidates,fetched,strict=True):
                if raw is None:
                    rejections[doi] = error
                    continue
                if ident in {item[0] for item in joined(authors(meta(raw)))}:
                    linked.append(doi)
            scans.append(dict(seed=seed, candidate_count=len(candidates), linked=linked,
                              search_sha256=sha256(response).hexdigest()))
    retained = sorted(set(seed_dois) | {doi for scan in scans for doi in scan["linked"]})
    report = dict(archive=dict(old_articles=len(old_dois),old_orcids=len(old_ids)),
                  seed_search_sha256=sha256(seed_file.read_bytes()).hexdigest(), seed_dois=seed_dois,
                  scans=scans, rejected_followups=rejections, source_digests={doi:sha256(get_xml(folder,doi)).hexdigest() for doi in retained})
    with output.open("x") as stream:
        json.dump(report,stream,indent=2)
        stream.write("\n")
    print(dict(articles=len(retained),anchors_checked=len(scans),
               linked_anchors=sum(len(row["linked"])>1 for row in scans)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("prior_xml_directories", type=Path, nargs="+")
    args = parser.parse_args()
    linked_capture(args.folder,args.prior_xml_directories,args.output)
