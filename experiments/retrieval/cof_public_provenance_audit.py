"""Audit public bibliographic and contribution evidence for blinded CoF citations."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from time import monotonic, sleep
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from experiments.credit.cross_source_identity import orcid_id


CROSSREF = "https://api.crossref.org/works"
PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PANELS = ("kdd", "scirepeval")


def canonical(text):
    text = unicodedata.normalize("NFKC", unescape(re.sub(r"<[^>]+>", "", text))).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def person(given, family):
    first, last = canonical(given or ""), canonical(family or "")
    return (first + " " + last) if first and last and len(first.split()[0]) >= 2 else None


def byline(record):
    results = defaultdict(set)
    for author in record.get("author", []):
        name = person(author.get("given"), author.get("family"))
        if name:
            identifier = orcid_id(author.get("ORCID") or "")
            results[name].update([identifier] if identifier else [])
    return {n: sorted(ids) for n, ids in results.items()}


def exact_match(title, items):
    normalized = canonical(title)
    matches = {x["DOI"].lower(): x for x in items if x.get("DOI") and
               any(canonical(t) == normalized for t in x.get("title", []))}
    status = "exact" if len(matches) == 1 else "ambiguous" if matches else "no_exact_title"
    if status != "exact":
        return dict(status=status)
    doi, metadata = next(iter(matches.items()))
    return dict(status=status, doi=doi, byline=byline(metadata))


def read_packet(root):
    panels = {}
    for panel in PANELS:
        folder = root / panel
        path = folder / "review.jsonl"
        manifest = json.loads((folder / "manifest.json").read_bytes())
        contents = path.read_bytes()
        if sha256(contents).hexdigest() != manifest["form_sha256"]:
            raise ValueError("The blinded packet changed")
        panels[panel] = [json.loads(line) for line in contents.splitlines()]
    return panels


def get(url):
    headers = {"User-Agent": "ExpertSearchResearch/0.1 (public scholarly provenance audit)"}
    request = Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urlopen(request, timeout=12) as response:
                payload = response.read(12_000_001)
            if len(payload) > 12_000_000:
                raise ValueError("Unexpectedly large response")
            return payload
        except (HTTPError, URLError, TimeoutError) as error:
            code = getattr(error, "code", None)
            if code is not None and code not in (429, 500, 502, 503, 504):
                return None
            if attempt < 2:
                retry = getattr(error, "headers", None)
                delay = retry.get("Retry-After", "") if retry else ""
                sleep(max(2 ** attempt, min(60, int(delay))) if delay.isdecimal() else 2 ** attempt)
    return None


def titles_for(row):
    return (*row["profile_titles"], row["citation"].splitlines()[0])


def capture(panels, destination):
    entries = {}
    if destination.exists():
        for line in destination.read_text().splitlines():
            row = json.loads(line)
            if row["result"]["status"] != "request_failed":
                entries[row["title"]] = row["result"]
    titles = sorted({title for rows in panels.values() for row in rows for title in titles_for(row)})
    last_call = 0.
    with destination.open("a") as output:
        for i, title in enumerate([t for t in titles if t not in entries], 1):
            sleep(max(0., 1.15 - (monotonic() - last_call)))
            last_call = monotonic()
            url = CROSSREF + "?" + urlencode({"query.title": title, "rows": 5, "select": "DOI,title,author"})
            raw = get(url)
            outcome = (exact_match(title, json.loads(raw)["message"]["items"]) if raw else
                       {"status": "request_failed"})
            entries[title] = outcome
            output.write(json.dumps(dict(title=title, result=outcome), ensure_ascii=False) + "\n")
            output.flush()
            if i % 30 == 0:
                print(f"Completed {i} title requests.", flush=True)
    return {t: entries[t] for t in titles}


def identity_for(anchors):
    usable = {r["doi"]: r["byline"] for r in anchors if r["status"] == "exact" and r["byline"]}
    if len(usable) < 2:
        return "fewer_than_two_anchors", None, None
    names = set.intersection(*(set(a) for a in usable.values()))
    if not names:
        return "no_common_name", None, None
    if len(names) != 1:
        return "multiple_common_names", None, None
    name, = names
    sets = [set(authorlist[name]) for authorlist in usable.values() if authorlist[name]]
    ids = set.union(*sets) if sets else set()
    if len(ids) > 1:
        return "inconsistent_orcids", None, None
    confirmed = next(iter(ids)) if ids and len(sets) >= 2 else None
    return ("common_name_and_orcid" if confirmed else "common_name_only"), name, confirmed


def pmc_lookup(dois):
    results = {}
    for i in range(0, len(dois), 12):
        chunk = dois[i:i+12]
        query = " OR ".join('DOI:"' + d + '"' for d in chunk)
        url = PMC + "/search?" + urlencode(dict(query=query, format="json", pageSize=100))
        raw = get(url)
        if raw is None:
            results.update({d: "request_failed" for d in chunk})
            continue
        hits = json.loads(raw)["resultList"]["result"]
        available = defaultdict(set)
        for record in hits:
            doi = record.get("doi", "").lower()
            if doi in chunk and record.get("isOpenAccess") == "Y" and record.get("pmcid"):
                available[doi].add(record["pmcid"])
        results.update({d: next(iter(available[d])) if len(available[d]) == 1 else None for d in chunk})
    return results


def xml_roles(raw):
    root = ET.fromstring(raw)
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    node = root.find("./front/article-meta")
    if node is None:
        return {"statement": False, "authors": []}
    roles = []
    for author in node.findall("./contrib-group/contrib[@contrib-type='author']"):
        name = person(author.findtext("./name/given-names"), author.findtext("./name/surname"))
        ids = [key for x in author.findall("./contrib-id[@contrib-id-type='orcid']")
               if (key := orcid_id(x.text or ""))]
        labels = [" ".join(tag.itertext()).strip() for tag in author.findall("role")]
        if any(labels):
            roles.append(dict(name=name, orcids=ids, labels=[label for label in labels if label]))
    notes = node.findall("./author-notes/fn[@fn-type='con']")
    sections = [s for s in root.findall(".//sec") if "author contribution" in canonical(s.findtext("title") or "")]
    return dict(statement=bool(roles or notes or sections), authors=roles)


def capture_statements(available, pending):
    results = json.loads(pending.read_bytes()) if pending.exists() else {}
    for doi, pmcid in available.items():
        if not pmcid or pmcid == "request_failed" or doi in results:
            continue
        raw = get(f"{PMC}/{pmcid}/fullTextXML")
        if raw:
            results[doi] = {"pmcid": pmcid, "xml_sha256": sha256(raw).hexdigest(), **xml_roles(raw)}
    pending.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return results


def evaluate(rows, titles, available, statements):
    counts = Counter()
    candidates = {}
    for row in rows:
        anchors = [titles[t] for t in row["profile_titles"]]
        ident, name, identifier = identity_for(anchors)
        if row["candidate"] in candidates and candidates[row["candidate"]] != ident:
            raise ValueError("Changed identity anchors on a candidate")
        candidates[row["candidate"]] = ident
        counts[f"case_identity_{ident}"] += 1
        cited = titles[row["citation"].splitlines()[0]]
        counts[f"citation_{cited['status']}"] += 1
        if cited["status"] != "exact":
            continue
        doi = cited["doi"]
        counts["common_name_in_cited_byline"] += int(name is not None and name in cited["byline"])
        counts["common_orcid_in_cited_byline"] += int(
            identifier is not None and any(identifier in ids for ids in cited["byline"].values()))
        counts["europe_pmc_request_failed"] += int(available[doi] == "request_failed")
        counts["europe_pmc_open_xml_registered"] += int(bool(available[doi]) and available[doi] != "request_failed")
        if doi in statements:
            data = statements[doi]
            counts["europe_pmc_xml_read"] += 1
            counts["with_any_contribution_statement"] += int(data["statement"])
            counts["with_candidate_structured_role"] += int(any(
                (name is not None and author["name"] == name)
                or (identifier is not None and identifier in author["orcids"]) for author in data["authors"]))
    return dict(citation_cases=len(rows), candidate_slots=len(candidates),
                candidate_identities=dict(Counter(candidates.values())), counts=dict(counts))


def run(packet, cache, output):
    if output.exists():
        raise ValueError("Choose a fresh aggregate output")
    panels = read_packet(packet)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    records = capture(panels, cache / "crossref.jsonl")
    cited = {row["citation"].splitlines()[0] for rows in panels.values() for row in rows}
    dois = sorted({records[t]["doi"] for t in cited if records[t]["status"] == "exact"})
    available = pmc_lookup(dois)
    statements = capture_statements(available, cache / "publisher-roles.json")
    (cache / "pmc.json").write_text(json.dumps(available, indent=2) + "\n")
    summary = dict(
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        unique_title_queries=len(records), lookup_statuses=dict(Counter(r["status"] for r in records.values())),
        distinct_cited_dois=len(dois), distinct_publisher_xml=len(statements),
        results={name: evaluate(rows, records, available, statements) for name, rows in panels.items()},
    )
    output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "cache", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
