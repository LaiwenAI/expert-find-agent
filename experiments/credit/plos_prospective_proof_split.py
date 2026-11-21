"""Discover and freeze independent PLOS proof groups from public metadata and JATS."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from experiments.credit import harvest_scoped_credit as source

QUERY = ('JOURNAL:"PLoS One" AND FIRST_PDATE:[2025-01-01 TO 2025-12-31] '
         'AND AUTHORID_TYPE:ORCID AND (TITLE_ABS:"machine learning" OR TITLE_ABS:"optimization")')
METADATA = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
PAGE_SIZE = 150
MAX_PAGES = 12
MAX_PROPOSALS = 96
WORKERS = 4
SHA_ELIFE = (
    "ee6715866d53d973a437d13c679f8ef2efa8a995d721782843c5b1944859f2a3",
    "8ab52c04301c4b3583ed294d0d62f2aacd652a6d44bd51f1a47286b0d2516fa2",
)


def digest(blob):
    return sha256(blob).hexdigest()


def stable(*things):
    return digest(json.dumps(things, sort_keys=True).encode())


def get(url):
    req = Request(url, headers={"User-Agent": "ResearchBenchmark/1.0 (public contributor metadata)"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=16) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (429, 502, 503, 504) or attempt == 2:
                raise
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
        sleep(2 ** attempt)


def metadata():
    cursor = "*"
    pages, owners, seen = [], defaultdict(set), set()
    total = None
    for _ in range(MAX_PAGES):
        params = urlencode(dict(query=QUERY, format="json", resultType="core", pageSize=PAGE_SIZE, cursorMark=cursor))
        raw = get(METADATA + params)
        response = json.loads(raw)
        if total is None:
            total = response["hitCount"]
        elif total != response["hitCount"]:
            raise ValueError("Search inventory changed while paging")
        rows = response["resultList"]["result"]
        pages.append(dict(sha256=digest(raw), records=len(rows)))
        for row in rows:
            doi = row.get("doi", "").casefold()
            if not source.is_article(doi):
                continue
            seen.add(doi)
            for author in row.get("authorList", {}).get("author", ()):
                identity = author.get("authorId", {})
                if identity.get("type") == "ORCID" and (key := source.orcid_id(identity.get("value", ""))):
                    owners[key].add(doi)
        if not rows or sum(page["records"] for page in pages) >= total:
            break
        following = response.get("nextCursorMark")
        if not following or following == cursor:
            raise ValueError("Europe PMC cursor stopped before the inventory was exhausted")
        cursor = following
    if sum(page["records"] for page in pages) != total:
        raise ValueError("Query exceeds the bounded metadata inventory")
    return owners, seen, dict(query=QUERY, pages=pages, total=total, plos_dois=len(seen))


def load_elife(paths):
    raw = [path.read_bytes() for path in paths]
    if tuple(map(digest, raw)) != SHA_ELIFE:
        raise ValueError("Changed eLife identity exclusion inputs")
    return {item["person"] for data in raw for doc in json.loads(data)["sources"]
            for item in doc["authors"] if item["person"]}


def elife_id(orcid):
    return digest(f"elife-role-search:{orcid}".encode())[:24]


def proposals(records, old_dois, old_ids, elife_ids):
    proposed = set()
    for person, docs in records.items():
        if person in old_ids or elife_id(person) in elife_ids:
            continue
        for pair in combinations(sorted(docs - old_dois), 2):
            proposed.add((pair, person))
    picked, locked_dois, locked_people = [], set(), set()
    for pair, person in sorted(proposed, key=lambda row: stable("plos-bulk-candidate-v1", *row)):
        if person in locked_people or locked_dois.intersection(pair):
            continue
        picked.append((pair, person))
        locked_people.add(person)
        locked_dois.update(pair)
        if len(picked) >= MAX_PROPOSALS:
            break
    return picked, len(proposed)


def document(doi):
    try:
        raw = get(source.article_url(doi))
        metadata = source.meta(raw)
        membership = source.declared_roles(raw, doi)
        rows = source.authors(metadata)
        roles, people = {}, set()
        title = metadata.find("title-group/article-title")
        abstract = metadata.find("abstract")
        if title is None or abstract is None or len(" ".join(abstract.itertext()).split()) < 20:
            return doi, None, "no_usable_abstract"
        for row, labels in zip(rows, membership, strict=True):
            identifiers = {item for node in row.findall("contrib-id[@contrib-id-type='orcid']")
                           if (item := source.orcid_id(node.text or ""))}
            if len(identifiers) > 1 or people.intersection(identifiers):
                return doi, None, "ambiguous_publisher_identity"
            for person in identifiers:
                people.add(person)
                if labels:
                    roles[person] = set(labels)
        return doi, (raw, roles, people), None
    except HTTPError as error:
        return doi, None, f"http_{error.code}"
    except (URLError, TimeoutError):
        return doi, None, "transport_error"
    except (ValueError, ET.ParseError):
        return doi, None, "invalid_or_unsupported_jats"


def contrasts(pair, left, right):
    supported = {binding for person in left.keys() & right.keys()
                 for binding in product(left[person], right[person])}
    left_role = set().union(*left.values())
    right_role = set().union(*right.values())
    choices = {(gold, rival) for gold in supported for rival in
               ({(label, gold[1]) for label in left_role} | {(gold[0], label) for label in right_role})
               if rival not in supported}
    return sorted(choices, key=lambda binding: stable("plos-bulk-role-edit-v1", pair, binding))[:3]


def confirm(picks, archive, old_ids, elife_ids):
    downloads = sorted({doc for pair, _ in picks for doc in pair})
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(document, downloads))
    parsed = {doi: record for doi, record, failure in results if failure is None}
    rejected = Counter(failure for _, _, failure in results if failure is not None)
    selected, hashes, occupied = [], {}, set(old_ids)
    eligibility = Counter()
    for pair, person in picks:
        if any(doi not in parsed for doi in pair):
            eligibility["missing_xml"] += 1
            continue
        left, right = (parsed[doi] for doi in pair)
        if person not in left[1].keys() & right[1].keys():
            eligibility["unconfirmed_index_link"] += 1
            continue
        ids = left[2] | right[2]
        if occupied.intersection(ids) or {elife_id(identity) for identity in ids} & elife_ids:
            eligibility["overlapping_identity"] += 1
            continue
        cases = contrasts(pair, left[1], right[1])
        if not cases:
            eligibility["no_hard_role_edits"] += 1
            continue
        selected.append(dict(papers=pair, contrasts=cases))
        for doi, record in zip(pair, (left, right), strict=True):
            xml = record[0]
            (archive / f"pone.{doi.rsplit('.', 1)[-1]}.xml").write_bytes(xml)
            hashes[doi] = digest(xml)
        occupied.update(ids)
    return selected, hashes, dict(requested=len(downloads), parsed=len(parsed), failures=rejected,
                                  pair_exclusions=eligibility)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("elife_confirmation", "elife_extended", "archive", "manifest"):
        p.add_argument(name, type=Path)
    p.add_argument("prior_folders", type=Path, nargs="+")
    args = p.parse_args()
    if args.manifest.exists() or args.archive.exists():
        raise ValueError("Require new output paths")
    old_dois, old_ids, _ = source.prior(args.prior_folders)
    elife_ids = load_elife((args.elife_confirmation, args.elife_extended))
    owners, _, inventory = metadata()
    picks, available = proposals(owners, old_dois, old_ids, elife_ids)
    args.archive.mkdir()
    print(f"{inventory['total']} metadata records, {available} eligible pairs, screening {len(picks)} pairs", flush=True)
    groups, hashes, diagnostic = confirm(picks, args.archive, old_ids, elife_ids)
    if len(groups) < 2:
        raise ValueError(f"Fewer than two independent publisher groups: {diagnostic}")
    ordered = sorted(groups, key=lambda row: stable("plos-bulk-split-v1", row["papers"]))
    split = len(ordered) // 2
    report = dict(
        inventory=inventory, proposals=available, attempted=len(picks), verified=len(groups),
        source_sha256=hashes, old_papers=len(old_dois), old_recorded_authors=len(old_ids),
        calibration=dict(pairs=ordered[:split]), test=dict(pairs=ordered[split:]),
        diagnostics=diagnostic, elife_sha256=SHA_ELIFE,
    )
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Confirmed {len(groups)} groups; metadata balances {split} calibration / {len(groups)-split} test", flush=True)


if __name__ == "__main__":
    main()
