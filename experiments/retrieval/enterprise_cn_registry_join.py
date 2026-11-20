"""Join private development-catalog keys to downloaded CN patent archives offline."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

CN_APPLICATION = re.compile(r"((?:19|20)[0-9]{2}[12389][0-9]{7}|[0-9]{8})(?:\.[0-9X])?")
CN_PUBLICATION = re.compile(r"([0-9]{7}|[0-9]{9})([A-Z][0-9]?)?")
APP_LABEL = "label:\u7533\u8bf7\u53f7"
PATENT_LABEL = "label:\u4e13\u5229\u53f7"
PUB_LABEL = "label:\u516c\u5f00\u53f7"
AUTH_LABEL = "label:\u6388\u6743\u53f7"
HAN = re.compile(r"[\u3400-\u9fff]")

ARCHIVES = {
    "cn-authority.zip": "098e9038c1c4718d68cf03c13d9f836d2d0608b93670d24257b25effc1697602",
    "cnpub_basedat.zip": "ef17c73963d7aadb60683479adffdcb2aa4e5aaa1f5b143ecea1050323bdbbab",
    "cnpub_inventor.zip": "8fee86480919b1a40c0ed63f2f3898fdbbc541ffe3b5f97706db2b673abf369c",
}
ROR_MD5 = "0769f596ca9e8ebf87224a1ad93a7722"
CONTROL_APPLICATION = "85100001"
CONTROL_PUBLICATION = "104745007"


def build_transliterations(projection):
    from pypinyin import Style, lazy_pinyin

    table = {}
    for owner, name in projection["owners"].items():
        if re.fullmatch(r"[\u3400-\u9fff]{2,4}", name):
            alias = "".join(lazy_pinyin(name, style=Style.NORMAL))
            if re.fullmatch(r"[a-z]+", alias):
                table[owner] = [alias]
    return table


def name_key(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def read(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def registry_key(raw):
    entity, namespace, token = raw
    if entity != "patent":
        return None
    token = token.upper()
    if namespace in {"CN", "ZL", APP_LABEL, PATENT_LABEL} and (app := CN_APPLICATION.fullmatch(token)):
        return "application", app[1], ""
    if namespace in {"CN", PUB_LABEL, AUTH_LABEL} and (doc := CN_PUBLICATION.fullmatch(token)):
        return "publication", doc[1], doc[2] or ""
    return None


def archive_catalog(path, applications, publications):
    app_matches, pub_matches = defaultdict(set), defaultdict(set)
    applicant_by_record = defaultdict(set)
    with ZipFile(path) as bundle, bundle.open("cnpub_basedat.asc") as stream:
        lines = io.TextIOWrapper(stream, encoding="utf-8-sig")
        for line in lines:
            # Character columns in the archive's data dictionary are one-based.
            original = line[150:170].strip()
            found = CN_APPLICATION.fullmatch(original)
            app = found[1] if found else None
            publication = line[178:198].strip()
            if app not in applications and publication not in publications:
                continue
            printed_date = line[198:206].strip()
            date = printed_date[4:] + printed_date[:4] if re.fullmatch(r"[0-9]{8}", printed_date) else ""
            record = (line[286:298].strip(), publication, date)
            applicant_by_record[record].add(line[206:286].strip())
            if app in applications:
                app_matches[app].add(record)
            if publication in publications and date:
                pub_matches[publication, date].add(record)
    pools = (app_matches, pub_matches)
    ambiguous = tuple(sum(len(rows) != 1 for rows in pool.values()) for pool in pools)
    return (
        {key: next(iter(rows)) for key, rows in app_matches.items() if len(rows) == 1},
        {key: next(iter(rows)) for key, rows in pub_matches.items() if len(rows) == 1},
        ambiguous,
        applicant_by_record,
    )


def archive_rosters(path, projects):
    records = defaultdict(list)
    by_key = {row[:2] for row in projects}
    with ZipFile(path) as bundle, bundle.open("cnpub_inventor.asc") as stream:
        for line in io.TextIOWrapper(stream, encoding="utf-8-sig"):
            stem = line[28:40].strip()
            publication = line[:20].strip()
            if (stem, publication) in by_key:
                printed = line[20:28].strip()
                date = printed[4:] + printed[:4] if re.fullmatch(r"[0-9]{8}", printed) else ""
                records[stem, publication, date].append((
                    int(line[40:48]), int(line[128:136]), line[48:128].strip(),
                ))
    rosters = {}
    for record, entries in records.items():
        ordered = sorted(entries, key=lambda row: row[1])
        if all(row[:2] == (len(ordered), pos) and row[2]
               for pos, row in enumerate(ordered, start=1)):
            rosters[record] = tuple(row[2] for row in ordered)
    return rosters


def authority_publications(path, requested):
    encoded = {code.encode(): code for code in requested}
    matches = defaultdict(set)
    with ZipFile(path) as bundle:
        members = [name for name in bundle.namelist() if name.endswith((".txt", ".csv"))]
        if len(members) != 2:
            raise ValueError("Unexpected authority-file members")
        for member in members:
            with bundle.open(member) as stream:
                for line in stream:
                    offices = line.split(b",", 4)
                    if len(offices) < 4 or offices[1] not in encoded:
                        continue
                    office, number, kind, date = offices[:4]
                    if office.removeprefix(b"\xef\xbb\xbf") != b"CN":
                        raise ValueError("Foreign authority in CN file")
                    date = date.strip().removesuffix(b"<CRLF>")
                    if len(date) != 8 or not date.isdigit():
                        raise ValueError("Missing publication date")
                    matches[encoded[number]].add((kind.decode(), date.decode()))
    return matches


def prepare_institutions(dataset, projection):
    manifest = dataset / "manifest.json"
    source = dataset / "experts.jsonl"
    if file_sha(manifest) != projection["manifest_sha256"]:
        raise ValueError("Changed profile manifest")
    if file_sha(source) != read(manifest)["files_sha256"][source.name]:
        raise ValueError("Changed expert catalog")
    with source.open() as stream:
        table = {row["id"]: row["institution"] for row in map(json.loads, stream)
                 if row["split"] == "development"}
    if table.keys() != projection["owners"].keys():
        raise ValueError("Changed owner inventory")
    return {"manifest_sha256": projection["manifest_sha256"], "institutions": table}


def ror_aliases(path, names):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "md5").hexdigest()
    if digest != ROR_MD5:
        raise ValueError("Unexpected ROR release checksum")
    aliases = defaultdict(set)
    with ZipFile(path) as bundle:
        member, = [name for name in bundle.namelist() if name.endswith("-ror-data.json")]
        with bundle.open(member) as stream:
            records = json.load(stream)
    for record in records:
        if not any(location.get("geonames_details", {}).get("country_code") == "CN"
                   for location in record["locations"]):
            continue
        for name in record["names"]:
            key = name_key(name["value"])
            if key in names and name["types"] != ["acronym"]:
                aliases[key].add(record["id"])
    return {key: next(iter(ids)) for key, ids in aliases.items() if len(ids) == 1}


def run(projection, trace, corpus_path, folder, transliterations, institutions=None):
    if file_sha(corpus_path) != projection["source_sha256"]:
        raise ValueError("Changed source inventory")
    for name, expected in ARCHIVES.items():
        if file_sha(folder / name) != expected:
            raise ValueError(f"Changed archive: {name}")
    inputs = defaultdict(set)
    for row in projection["sources"].values():
        if row["owner"] not in projection["owners"]:
            raise ValueError("Unknown candidate")
        for clause in row["clause_records"]:
            key = registry_key(clause["key"])
            if key:
                inputs[row["owner"]].add(key)

    apps = {code for keys in inputs.values() for route, code, _ in keys if route == "application"}
    pubs = {code for keys in inputs.values() for route, code, _ in keys if route == "publication"}
    catalog, publication_catalog, conflicts, applicants = archive_catalog(
        folder / "cnpub_basedat.zip", apps | {CONTROL_APPLICATION}, pubs,
    )
    rosters = archive_rosters(
        folder / "cnpub_inventor.zip", list(catalog.values()) + list(publication_catalog.values()),
    )
    published = authority_publications(folder / "cn-authority.zip", pubs | {CONTROL_PUBLICATION})
    control = catalog[CONTROL_APPLICATION]
    if control[1] != "1000039" or rosters[control] != ("Hu Guohua", "Yang Zhigui"):
        raise AssertionError("Historical catalog or role positive control failed")
    if not any(kind == "B" for kind, _ in published[CONTROL_PUBLICATION]):
        raise AssertionError("Authority-file positive control failed")

    organizations = {}
    aliases = {}
    if institutions is not None:
        if institutions["manifest_sha256"] != projection["manifest_sha256"]:
            raise ValueError("Institution projection is from a different dataset")
        organizations = institutions["institutions"]
        if organizations.keys() != projection["owners"].keys():
            raise ValueError("Institution projection covers different owners")
        observed = {name_key(value) for value in organizations.values()}
        observed.update(name_key(name) for values in applicants.values() for name in values)
        aliases = ror_aliases(folder / "ror.zip", observed - {""})

    name_count = Counter(projection["owners"].values())
    alias_counts = Counter(alias for aliases in transliterations.values() for alias in set(aliases))
    candidate_outcomes = defaultdict(Counter)
    outcomes = (
        "application_keys", "application_filing_year_through_2012", "application_catalog_hit",
        "complete_archived_roster", "roster_contains_han_name", "exact_candidate_name_on_roster",
        "exact_name_unique_in_profile_catalog", "transliteration_candidate_lead",
        "unique_transliteration_candidate_lead", "publication_keys", "publication_hit",
        "one_edition", "multiple_editions", "explicit_kind_hit", "publication_archive_edition_match",
        "application_with_complete_archived_roster", "publication_with_complete_archived_roster",
        "archived_roster_and_applicant_present", "both_organizations_resolved", "same_ror_organization",
        "same_organization_and_pinyin",
    )
    totals = Counter(dict.fromkeys(outcomes, 0))
    totals["application_catalog_conflicts"], totals["publication_catalog_conflicts"] = conflicts
    for owner, keys in inputs.items():
        signals = candidate_outcomes[owner]
        key_name = projection["owners"][owner]
        latin_variants = set(transliterations.get(owner, ()))
        for route, code, kind in keys:
            signals[route + "_keys"] += 1
            if route == "publication":
                editions = {entry for entry in published[code] if not kind or entry[0] == kind}
                signals.update(publication_hit=bool(editions), one_edition=len(editions) == 1,
                               multiple_editions=len(editions) > 1,
                               explicit_kind_hit=bool(editions) and bool(kind))
                if len(editions) != 1:
                    continue
                (_, edition_date), = editions
                match = publication_catalog.get((code, edition_date))
                signals["publication_archive_edition_match"] += bool(match)
            else:
                # Legacy numbers use a two-digit filing year.
                short_year = int(code[:2])
                year = int(code[:4]) if len(code) == 12 else (1900 if short_year >= 85 else 2000) + short_year
                signals["application_filing_year_through_2012"] += 1985 <= year <= 2012
                match = catalog.get(code)
                signals["application_catalog_hit"] += bool(match)
            roster = rosters.get(match)
            if not roster:
                continue
            signals["complete_archived_roster"] += 1
            signals[route + "_with_complete_archived_roster"] += 1
            normalized = {name_key(name) for name in roster}
            signals["roster_contains_han_name"] += any(HAN.search(name) for name in roster)
            if key_name and key_name in normalized:
                signals["exact_candidate_name_on_roster"] += 1
                signals["exact_name_unique_in_profile_catalog"] += name_count[key_name] == 1
            if institutions is not None:
                options = applicants[match] - {""}
                signals["archived_roster_and_applicant_present"] += bool(options)
                if len(options) == 1:
                    applicant, = options
                    listed = aliases.get(name_key(applicant))
                    profile = aliases.get(name_key(organizations[owner]))
                    if listed and profile:
                        signals["both_organizations_resolved"] += 1
                        signals["same_ror_organization"] += listed == profile
                        signals["same_organization_and_pinyin"] += listed == profile and bool(latin_variants & normalized)
            if latin_variants & normalized:
                signals["transliteration_candidate_lead"] += 1
                signals["unique_transliteration_candidate_lead"] += any(
                    alias_counts[alias] == 1 for alias in latin_variants & normalized
                )
        totals.update(signals)

    panels = defaultdict(Counter)
    observations = trace["observations"]
    if len(observations) != 80 or len({case["id"] for case in observations}) != 80:
        raise ValueError("Changed first-query development sample")
    for case in observations:
        arm = case["arms"]["hybrid_prefixes"]
        groups = [("all_shortlists", arm["shortlist"])]
        if arm["issued"]:
            groups.append(("issued_candidates", [record["expert"] for record in arm["issued"]]))
        else:
            groups.append(("abstained_shortlists", arm["shortlist"]))
        for label, candidates in groups:
            stats = panels[label]
            stats.update(requests=1, candidate_slots=len(candidates))
            for key in outcomes:
                count = sum(candidate_outcomes[owner][key] > 0 for owner in candidates)
                stats.update({key + "_slots": count, key + "_requests": bool(count)})

    totals.update(
        unique_application_keys=len(apps), unique_publication_numbers=len(pubs),
        unique_application_catalog_hits=len(apps & catalog.keys()),
        unique_publication_number_hits=sum(bool(published[number]) for number in pubs),
    )
    return {
        "archive_sha256": ARCHIVES,
        "ror_archive_md5": ROR_MD5 if institutions is not None else None,
        "positive_controls": {"historical_roster": True, "current_publication_kind": True},
        "owner_key_pairs": dict(totals),
        "panels": dict(panels),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("transliterate")
    extract.add_argument("projection", type=Path)
    extract.add_argument("output", type=Path)
    org = commands.add_parser("institutions")
    for argument in ("projection", "dataset", "output"):
        org.add_argument(argument, type=Path)
    audit = commands.add_parser("audit")
    for argument in ("projection", "trace", "corpus", "archives", "output"):
        audit.add_argument(argument, type=Path)
    audit.add_argument("--transliterations", type=Path)
    audit.add_argument("--institutions", type=Path)
    args = parser.parse_args()
    projection = read(args.projection)
    if args.command == "transliterate":
        result = build_transliterations(projection)
    elif args.command == "institutions":
        result = prepare_institutions(args.dataset, projection)
    else:
        result = run(
            projection, read(args.trace), args.corpus, args.archives,
            read(args.transliterations) if args.transliterations else {},
            read(args.institutions) if args.institutions else None,
        )
    args.output.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n")
