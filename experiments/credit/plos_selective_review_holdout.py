"""Freeze publisher- and identity-disjoint PLOS proof groups from pinned worktree snapshots."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from experiments.credit.cross_source_identity import orcid_id, xml_sources
from experiments.credit.engineering_cross_source_panel import owner_key
from experiments.credit.freeze_conjunction_calibration import freeze
from experiments.credit.linked_witness_dispatch import stable
from experiments.credit.plos_role_inventory import declared_roles
from experiments.followir.core17_reranking import digest, write_json


def authors(raw, doi, metadata):
    roles = declared_roles(raw, doi)
    views = {}
    ids = set()
    for i, (author, role_set) in enumerate(zip(
        metadata.findall("contrib-group/contrib[@contrib-type='author']"), roles, strict=True
    )):
        valid = {key for node in author.findall("contrib-id[@contrib-id-type='orcid']")
                 if (key := orcid_id(node.text or ""))}
        if not valid:
            continue
        if len(valid) != 1:
            raise ValueError("conflicting_orcid")
        key = owner_key(doi, i, author)
        if key in ids:
            raise ValueError("duplicate_orcid")
        ids.add(key)
        if not role_set:
            raise ValueError("missing_roles")
        alias = f"Author {i+1}"
        original = [node.text.strip() for node in author.findall("role") if node.text]
        views[key] = dict(alias=alias, text=f"{alias}: " + ", ".join(original),
                          roles=list(role_set), engineering_key=key)
    return views


def build(folders, excluded, permitted=None):
    docs, notes, seen, skipped = {}, {}, {}, Counter()
    conflicts = set()
    for folder in folders:
        for candidate in xml_sources(folder):
            xml = candidate.read_bytes()
            try:
                meta = ET.fromstring(xml).find("./front/article-meta")
                doi = meta.findtext("article-id[@pub-id-type='doi']") if meta is not None else None
                if not doi or not doi.startswith("10.1371/journal.pone."):
                    skipped["other_publisher"] += 1
                    continue
                checksum = sha256(xml).hexdigest()
                if permitted is not None and permitted.get(doi) != checksum:
                    continue
                if doi in conflicts:
                    continue
                previous = seen.setdefault(doi, checksum)
                if previous != checksum:
                    skipped["alternate_xml_version"] += 1
                    conflicts.add(doi)
                    docs.pop(doi, None)
                    notes.pop(doi, None)
                    continue
                if doi in docs:
                    continue
                parsed = authors(xml, doi, meta)
            except (ValueError, ET.ParseError) as error:
                key = str(error).split(":", 1)[0]
                skipped[key] += 1
                continue
            if not parsed or set(parsed) & excluded:
                skipped["no_linkable_author_or_prior_identity"] += 1
                continue
            notes[doi] = parsed
            docs[doi] = ({who: set(row["roles"]) for who, row in parsed.items()},
                         dict(filename=candidate.name, sha256=checksum))
    return docs, notes, skipped


def freeze_role_edits(records):
    person_sources = defaultdict(list)
    for doi, (owners, _) in records.items():
        for owner in owners:
            person_sources[owner].append(doi)
    possible = {pair for items in person_sources.values() for pair in combinations(sorted(items), 2)}
    people, keep, cases = set(), set(), []
    for pair in sorted(possible, key=lambda row: stable("transfer-edit", row)):
        a, b = (records[doi][0] for doi in pair)
        union = a.keys() | b.keys()
        if people.intersection(union) or keep.intersection(pair):
            continue
        among = a.keys() & b.keys()
        valid = {(left, right) for person in among for left in a[person] for right in b[person]}
        present = tuple(set.union(*article.values()) for article in (a, b))
        options = []
        for left, right in valid:
            negative = {(other, right) for other in present[0]} | {(left, other) for other in present[1]}
            for changed in sorted(negative - valid):
                options.append(((left, right), changed))
        if options:
            chosen = sorted(options, key=lambda item: stable(pair, item))[:3]
            cases.append(dict(papers=pair, contrasts=chosen))
            people.update(union)
            keep.update(pair)
        if len(cases) == 40:
            break
    return cases, {doi: records[doi][1] for doi in sorted(keep)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old_notes", "old_link_panel", "manifest", "notes"):
        parser.add_argument(name, type=Path)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--role-edits", action="store_true")
    args = parser.parse_args()
    existing = json.loads(args.old_notes.read_bytes())
    links = json.loads(args.old_link_panel.read_bytes())
    excluded = {row["engineering_key"] for article in existing.values() for row in article.values()}
    excluded.update(person for article in links["articles"].values() for person in article["roles"])
    permitted = None
    if args.restore:
        keeper = json.loads(args.restore.read_bytes())
        permitted = {doi: row["sha256"] for doi, row in keeper["panels"]["plos"]["articles"].items()}
    records, notes, excluded_counts = build(args.folders, excluded, permitted)
    use_edits = args.role_edits or bool(args.restore and keeper.get("contrasts") == "role_edits")
    if use_edits:
        pairs, sources = freeze_role_edits(records)
    else:
        pairs, sources, _ = freeze(records, 40)
    panel = {"panels": {"plos": dict(pairs=pairs, articles=sources)}, "contrasts": "role_edits" if use_edits else "swaps"}
    if args.restore:
        if sources.keys() != permitted.keys():
            raise ValueError("Could not restore every source")
        panel = keeper
    else:
        panel.update(
            scanned_folders=[p.name for p in args.folders], papers=len(records), filtered=dict(excluded_counts),
            prior_proof_notes=digest(args.old_notes), prior_engineering_links=digest(args.old_link_panel),
        )
    if not pairs:
        raise ValueError("No source-disjoint PLOS pairs")
    write_json(args.manifest, panel)
    write_json(args.notes, {doi: notes[doi] for doi in sorted(sources)})
    print(dict(articles=len(records), linked_groups=len(pairs),
               request_families=sum(len(x.get("contrasts", x.get("bindings", ()))) for x in pairs)))


if __name__ == "__main__":
    main()
