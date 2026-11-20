"""Build a dual-view author contribution panel from frozen engineering JATS."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree as ET

from experiments.credit.plos_role_inventory import ROLES


JOURNALS = {"sensors": "1424-8220", "frontiers": "2296-9144"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
AUTHOR_LABEL = re.compile(r"(?:^|(?<=\.))\s*([^\W\d_][\w.\-]{0,20})\s*:\s*", re.UNICODE)
END_NOTE = re.compile(r"\.\s+All authors (?:have read|contributed to)", re.I)


def normalize(text):
    separated = text.replace("&", " and ").replace("–", " ").replace("—", " ")
    ascii_text = unicodedata.normalize("NFKD", separated).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))


ROLE_LOOKUP = {normalize(name): name for name in ROLES}
ROLE_LOOKUP["writing original draft preparation"] = ROLE_LOOKUP["writing original draft"]


def download(url):
    return subprocess.check_output(["curl", "-fsSL", "--retry", "2", "--retry-delay", "1",
                                    "--max-time", "25", "--max-filesize", "12000000", url])


def identifiers(metadata):
    values = {row.get("pub-id-type"): (row.text or "").strip() for row in metadata.findall("article-id")}
    pmc = values.get("pmcid") or values.get("pmc") or values.get("pmcaid") or ""
    return pmc if pmc.startswith("PMC") else "PMC" + pmc, values.get("doi", "")


def snapshot(out, size=None, frozen=None):
    out.mkdir(parents=True, exist_ok=False)
    xml = out / "xml"
    xml.mkdir()
    if frozen is None:
        selected = {}
        for key, issn in JOURNALS.items():
            query = f"ISSN:{issn} AND PUB_YEAR:2025 AND OPEN_ACCESS:Y"
            params = urlencode(dict(query=query, format="json", pageSize=100))
            items = json.loads(download(f"{EPMC}?{params}"))["resultList"]["result"]
            index = {row["pmcid"]: row["doi"].casefold() for row in items if row.get("pmcid") and row.get("doi")}
            ids = sorted(index, key=lambda pmc: sha256(f"engineering-credit-v1:{pmc}".encode()).digest())[:size]
            selected[key] = {pmc: index[pmc] for pmc in ids}
    else:
        selected = frozen["selected"]
    recover = set(frozen["records"]) if frozen is not None else set().union(*selected.values())
    planned = []
    for journal, by_id in selected.items():
        ids = [pmc for pmc in by_id if pmc in recover]
        planned.extend((journal, ids[i:i+12]) for i in range(0, len(ids), 12))

    def retrieve(unit):
        key, ids = unit
        query = urlencode(dict(db="pmc", id=",".join(pmc.removeprefix("PMC") for pmc in ids), retmode="xml"))
        batch = ET.fromstring(download(f"{FETCH}?{query}"))
        result = {}
        for article in batch.iter("article"):
            metadata = article.find("./front/article-meta")
            if metadata is None:
                continue
            pmc, doi = identifiers(metadata)
            if pmc not in ids or doi.casefold() != selected[key][pmc]:
                continue
            data = ET.tostring(article, encoding="utf-8")
            (xml / f"{pmc}.xml").write_bytes(data)
            result[pmc] = dict(doi=doi, journal=key, sha256=sha256(data).hexdigest())
        return result

    with ThreadPoolExecutor(max_workers=2) as workers:
        records = {pmc: record for batch in workers.map(retrieve, planned) for pmc, record in batch.items()}
    if frozen is not None:
        if records != frozen["records"]:
            raise ValueError("Could not restore the frozen XML projections")
        records = frozen["records"]
    (out / "sources.json").write_text(json.dumps(dict(selected=selected, records=records), indent=2) + "\n")


def author_aliases(author):
    given = normalize(author.findtext("./name/given-names", "")).split()
    family = normalize(author.findtext("./name/surname", "")).split()
    if not given or not family:
        raise ValueError("missing_person_name")
    first, last = given[0], family[-1]
    bases = {first[0] + last[0], first[:2] + last[0],
             "".join(word[0] for word in given) + last[0],
             first[0] + "".join(word[0] for word in family),
             "".join(word[0] for word in given + family)}
    return " ".join(given + family), bases


def blocks(text, form):
    text = END_NOTE.split(text, maxsplit=1)[0].strip()
    if form == "by_author":
        headings = list(AUTHOR_LABEL.finditer(text))
        if not headings or headings[0].start() != 0:
            raise ValueError("unrecognized_author_layout")
        return [("author", match[1], text[match.end():next_start].strip(" .;"),
                 text[match.start():next_start].strip())
                for match, next_start in zip(headings, [m.start() for m in headings[1:]] + [len(text)], strict=True)]
    rows = []
    for clause in text.split(";"):
        if not clause.strip():
            continue
        role, separator, owners = clause.strip().partition(",")
        if not separator:
            raise ValueError("unrecognized_role_layout")
        rows.append(("role", role, owners.strip(" ."), clause.strip()))
    return rows


def parse_statement(paragraph, aliases, form):
    groups = []
    visible = defaultdict(set)
    resolved = defaultdict(set)

    def owner(identifier):
        found = aliases.get("".join(normalize(identifier).split()), set())
        if len(found) != 1:
            raise ValueError("unresolved_initials")
        index = next(iter(found))
        resolved[index].add(identifier.strip())
        return index

    for kind, head, rest, raw in blocks(paragraph, form):
        if kind == "author":
            roles = {ROLE_LOOKUP.get(normalize(label)) for label in rest.split(",")}
            holders = {owner(head)}
        else:
            roles = {ROLE_LOOKUP.get(normalize(head))}
            holders = {owner(label) for label in re.split(r",|\band\b", rest, flags=re.I) if label.strip()}
        if None in roles or not holders:
            raise ValueError("unsupported_statement_role")
        groups.append(dict(text=raw, owners=sorted(holders), roles=sorted(roles)))
        for person in holders:
            visible[person].update(roles)
    return groups, visible, resolved


def article_panel(raw, record):
    article = ET.fromstring(raw)
    meta = article.find("./front/article-meta")
    license_links = {urlparse(value).path for lic in meta.findall(".//license") for node in lic.iter()
                     for key, value in node.attrib.items()
                     if key.endswith("href") and urlparse(value).hostname == "creativecommons.org"}
    if not any(re.fullmatch(r"/licenses/by/[\d.]+/?", url) for url in license_links):
        raise ValueError("no_cc_by_license")
    authors = meta.findall("./contrib-group/contrib[@contrib-type='author']")
    if len(authors) < 2:
        raise ValueError("fewer_than_two_authors")
    tags, identities, aliases = {}, [], defaultdict(set)
    for i, author in enumerate(authors):
        name, possible = author_aliases(author)
        identities.append(name)
        for label in possible:
            aliases[label].add(i)
        tags[i] = {ROLE_LOOKUP.get(normalize(node.get("vocab-term") or node.text or ""))
                   for node in author.findall("role") if "credit" in " ".join(node.attrib.values()).casefold()}
        if None in tags[i] or not tags[i]:
            raise ValueError("incomplete_or_unknown_structured_roles")
    headings = [node for node in article.iter() if normalize(node.findtext("title", "")) == "author contributions"]
    paragraphs = [" ".join("".join(p.itertext()).split()) for node in headings for p in node.findall("p")]
    if len(paragraphs) != 1:
        raise ValueError("unrecognized_declaration_container")
    form = "by_author" if record["journal"] == "frontiers" else "by_role"
    groups, visible, resolved = parse_statement(paragraphs[0], aliases, form)
    if set(visible) != set(tags) or any(tags[i] != visible[i] for i in tags):
        raise ValueError("tag_statement_disagreement_or_omission")
    return dict(identities=identities, roles={i: sorted(labels) for i, labels in tags.items()},
                aliases={i: sorted(values)[0] for i, values in resolved.items()}, blocks=groups, layout=form)


def build(source, output, audit):
    inventory = json.loads((source / "sources.json").read_bytes())
    accepted, excluded = {}, {}
    for pmc, record in inventory["records"].items():
        raw = (source / "xml" / f"{pmc}.xml").read_bytes()
        if sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Source checksum mismatch")
        try:
            accepted[pmc] = record | article_panel(raw, record)
        except ValueError as exc:
            excluded[pmc] = str(exc)
    train_identities = {name for doc in accepted.values() if doc["journal"] == "sensors" for name in doc["identities"]}
    for pmc in list(accepted):
        if accepted[pmc]["journal"] == "frontiers" and train_identities.intersection(accepted[pmc]["identities"]):
            excluded[pmc] = "author_name_overlap"
            del accepted[pmc]
    for path, data in ((output, accepted), (audit, dict(inventory=inventory, included=sorted(accepted), excluded=excluded))):
        with path.open("x") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print("Accepted:", dict(Counter(row["journal"] for row in accepted.values())), flush=True)
    print("Excluded:", dict(Counter(excluded.values())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fetch", type=int)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.fetch and args.restore:
        parser.error("Choose fresh intake or a frozen restore")
    if args.restore:
        snapshot(args.directory, frozen=json.loads(args.restore.read_bytes())["inventory"])
    elif args.fetch:
        snapshot(args.directory, args.fetch)
    if args.output and args.audit:
        build(args.directory, args.output, args.audit)
