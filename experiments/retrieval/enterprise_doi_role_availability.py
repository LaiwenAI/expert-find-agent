"""Audit DOI-to-contribution evidence on the blinded development outputs."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import re
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET

from experiments.credit.plos_role_inventory import ROLES
from experiments.retrieval.audit_enterprise_person_evidence import ROLE_HEADER, normalized
from experiments.retrieval.check_joint_review import check, load
from experiments.retrieval.enterprise_routed_review import digest, write


CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUROPE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
CONTROL = "10.1371/journal.pone.0241819"
DOI = re.compile(r"10\.\d{4,9}/[^\s]+", re.IGNORECASE)
FULLTEXT_ID = re.compile(r"PMC\d+")
HAN = re.compile(r"[\u3400-\u9fff]")
CONTRIBUTION_TITLE = re.compile(r"(?:credit (?:authorship )?contribution statement|author contributions?)", re.IGNORECASE)
LICENSE_ATTR = "{http://www.w3.org/1999/xlink}href"


def join(packet, sidecar):
    check(packet, packet / "blind-review.jsonl")
    forms = load(packet / "blind-review.jsonl")
    mapping = load(packet / "private-mapping.jsonl")
    metadata = sidecar / "papers.jsonl"
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    if digest(metadata) != manifest["metadata_sha256"]:
        raise ValueError("Changed paper metadata sidecar")
    by_owner = defaultdict(list)
    owners = {row["expert_id"] for row in mapping.values()}
    with metadata.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["expert_id"] in owners:
                by_owner[row["expert_id"]].append(row)
    for identity, item in mapping.items():
        owner = item["expert_id"]
        cited = {s["id"] for s in item["sources"]}
        rows = by_owner[owner]
        direct = {r["source_id"] for r in rows} & cited

        def dois(records):
            return sorted({r["doi"].casefold() for r in records if r["doi"] and DOI.fullmatch(r["doi"])})

        yield dict(case=identity, candidate=forms[identity]["candidate"]["name"],
                   cited_papers=len(direct),
                   direct_dois=dois(r for r in rows if r["source_id"] in direct),
                   owner_dois=dois(rows))


def get(url):
    for _ in range(2):
        try:
            with urlopen(url, timeout=18) as stream:
                return stream.read(), "ok"
        except HTTPError as error:
            status = f"http_{error.code}"
            if error.code not in (429, 500, 502, 503, 504):
                return None, status
        except (URLError, TimeoutError):
            status = "transport_failure"
    return None, status


def case_name(text):
    return "".join(c for c in normalized(text) if c.isalnum())


def inspect_jats(raw, doi):
    root = ET.fromstring(raw)
    meta = root.find("./front/article-meta")
    if meta is None:
        raise ValueError("Missing JATS article metadata")
    identifiers = {s.text.strip().casefold() for s in meta.findall("article-id[@pub-id-type='doi']") if s.text}
    if identifiers != {doi}:
        raise ValueError("JATS DOI does not exactly match the requested article")
    license_node = meta.find("./permissions/license")
    license_url = "" if license_node is None else license_node.get(LICENSE_ATTR, "")
    role_keys = {case_name(role) for role in ROLES}
    authors = []
    for author in meta.findall("./contrib-group/contrib[@contrib-type='author']"):
        names = set()
        for name in author.findall("./name") + author.findall("./name-alternatives/name"):
            family = name.findtext("surname", "").strip()
            given = name.findtext("given-names", "").strip()
            if family and given:
                names.update((case_name(family + given), case_name(given + family)))
        for spelling in author.findall("./string-name") + author.findall("./name-alternatives/string-name"):
            full = case_name("".join(spelling.itertext()))
            if full:
                names.add(full)
        roles = {case_name("".join(role.itertext())) for role in author.findall("role")}
        have_roles = bool(roles & role_keys or any("credit.niso.org/contributor-roles/" in
                          role.get("vocab-term-identifier", "") for role in author.findall("role")))
        authors.append(dict(names=sorted(names), structured_roles=have_roles))
    prose = meta.findall("./author-notes/fn[@fn-type='con']")
    prose += [s for s in root.iter("sec") if CONTRIBUTION_TITLE.fullmatch(s.findtext("title", "").strip())]
    prose += [p for p in root.findall("./back//p") if ROLE_HEADER.match(" ".join(p.itertext()).strip())]
    return dict(authors=authors, contribution_prose=bool(prose),
                structured_roles=any(a["structured_roles"] for a in authors),
                license_url=license_url)


def lookup(output, dois):
    response, status = get(CONVERTER + "?" + urlencode(dict(ids=",".join(dois), idtype="doi", format="json")))
    if status != "ok":
        raise RuntimeError(f"PMC conversion failed: {status}")
    links = {}
    for record in json.loads(response)["records"]:
        doi = record.get("requested-id", record.get("doi", "")).casefold()
        if doi not in dois or doi in links:
            raise ValueError("PMC identifier response does not match the requested DOI inventory")
        pmc = record.get("pmcid")
        if pmc and (not FULLTEXT_ID.fullmatch(pmc) or record.get("doi", "").casefold() != doi):
            raise ValueError("Ambiguous PMC identifier mapping")
        links[doi] = pmc
    if set(links) != set(dois) or not links[CONTROL]:
        raise ValueError("Incomplete conversion response or failed positive control")
    raw_dir = output / "xml"
    raw_dir.mkdir(mode=0o700)
    reports = {doi: dict(pmcid=pmc, xml_status="not_mapped") for doi, pmc in links.items()}

    def fetch(pair):
        doi, pmc = pair
        raw, status = get(f"{EUROPE}/{pmc}/fullTextXML")
        row = dict(pmcid=pmc, xml_status=status)
        if status == "ok":
            row.update(inspect_jats(raw, doi), xml_sha256=sha256(raw).hexdigest())
            path = raw_dir / f"{pmc}.xml"
            path.write_bytes(raw)
            path.chmod(0o600)
        return doi, row

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports.update(pool.map(fetch, ((doi, pmc) for doi, pmc in links.items() if pmc)))
    if reports[CONTROL]["xml_status"] != "ok":
        raise ValueError("Known archive article could not be fetched and validated")
    write(output / "doi-lookup-private.json", reports)
    return reports


def coverage(cases, reports, key, pinyin):
    rows = [case for case in cases if case[key]]
    counts = defaultdict(int)
    all_dois = {doi for case in cases for doi in case[key]}
    for doi in all_dois:
        record = reports[doi]
        counts["dois_with_pmcid"] += bool(record["pmcid"])
        counts["dois_with_jats"] += record["xml_status"] == "ok"
        counts["dois_with_structured_roles"] += record.get("structured_roles", False)
        counts["dois_with_contribution_prose"] += record.get("contribution_prose", False)
        counts["xml_http_or_transport_failures"] += record["xml_status"] not in ("ok", "not_mapped")
    for case in rows:
        articles = [reports[doi] for doi in case[key]]
        identities = [a for a in articles if a["xml_status"] == "ok"]
        candidate = case_name(case["candidate"])
        roman = case_name("".join(pinyin(case["candidate"]))) if HAN.search(case["candidate"]) else ""
        matches = []
        for article in identities:
            exact = [author for author in article["authors"] if candidate in author["names"]]
            soft = [author for author in article["authors"] if roman and roman in author["names"]]
            if len(exact) == 1:
                matches.append((True, exact[0]["structured_roles"]))
            elif len(soft) == 1:
                matches.append((False, soft[0]["structured_roles"]))
        counts["cases_with_pmcid"] += any(a["pmcid"] for a in articles)
        counts["cases_with_jats"] += bool(identities)
        counts["cases_with_exact_name"] += any(exact for exact, _ in matches)
        counts["cases_with_exact_name_and_role"] += any(exact and role for exact, role in matches)
        counts["cases_with_phonetic_candidate"] += any(not exact for exact, _ in matches)
        counts["cases_with_phonetic_candidate_and_role"] += any(not exact and role for exact, role in matches)
    return dict(cases_with_doi=len(rows), unique_dois=len(all_dois), **counts)


def run(packet, sidecar, output):
    from pypinyin import __version__, lazy_pinyin

    output.mkdir(mode=0o700)
    cases = list(join(packet, sidecar))
    all_dois = sorted({v for case in cases for key in ("direct_dois", "owner_dois") for v in case[key]} | {CONTROL})
    if len(all_dois) > 150:
        raise ValueError("Unexpected development DOI panel size")
    start = perf_counter()
    reports = lookup(output, all_dois)
    write(output / "summary.json", dict(
        cases=len(cases), cases_citing_catalog_paper=sum(bool(c["cited_papers"]) for c in cases),
        cited_paper_occurrences=sum(c["cited_papers"] for c in cases),
        cases_with_han_names=sum(bool(HAN.search(c["candidate"])) for c in cases),
        direct=coverage(cases, reports, "direct_dois", lazy_pinyin),
        owner_pool=coverage(cases, reports, "owner_dois", lazy_pinyin),
        positive_control_jats=reports[CONTROL]["xml_status"] == "ok",
        positive_control_contains_role_or_prose=reports[CONTROL]["structured_roles"] or reports[CONTROL]["contribution_prose"],
        capture_wall_seconds=perf_counter()-start, case_manifest_sha256=digest(packet / "manifest.json"),
        code_sha256=digest(Path(__file__)), phonetic_package_version=__version__,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("packet", "sidecar", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
