"""Freeze a new eLife article and author holdout for joined postings."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.request import urlopen

from experiments.credit.elife_paired_role_holdout import ARTICLE_TREE
from experiments.credit.elife_prompt_holdout import REVISION, UPSTREAM, build, inventory, parse, projection, sample_families
from experiments.followir.core17_reranking import digest, write_json


SALT = "elife-joint-postings-holdout-v1"
SIZE = 1200


def sample(tree, previous):
    if tree["truncated"] or tree["sha"] != ARTICLE_TREE or previous["revision"] != REVISION:
        raise ValueError("Expected the pinned tree and compatible preceding manifest")
    old = {int(record["filename"].split("-")[1]) for record in previous["source_inventory"]}
    latest = {}
    for entry in tree["tree"]:
        match = re.fullmatch(r"elife-(\d+)-v(\d+)\.xml", entry["path"])
        if not match or entry["type"] != "blob":
            continue
        article, version = map(int, match.groups())
        if 84000 <= article < 114000 and article not in old:
            if article not in latest or version > latest[article][0]:
                latest[article] = version, entry
    selected = sorted(latest, key=lambda key: sha256(f"{SALT}:{key}".encode()).digest())[:SIZE]
    if len(selected) != SIZE:
        raise ValueError("Insufficient unexposed articles")
    return [latest[key][1] for key in selected]


def read_object(path, expected):
    raw = path.read_bytes()
    blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != expected:
        raise ValueError(f"Changed publisher Git object: {path.name}")
    return raw


def fetch(xml, entries, workers):
    xml.mkdir(parents=True, exist_ok=True)
    def download(entry):
        path = xml / entry["path"]
        if not path.exists():
            url = f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{REVISION}/articles/{path.name}"
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=35) as response:
                        raw = response.read()
                    break
                except URLError:
                    if attempt == 2:
                        raise
                    time.sleep(1 << attempt)
            path.write_bytes(raw)
        read_object(path, entry["sha"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(download, entries), 1):
            if index % 200 == 0:
                print(f"Verified {index}/{len(entries)} pinned XML objects.", flush=True)


def freeze(args):
    tree = json.loads(args.tree.read_bytes())
    previous = json.loads(args.previous.read_bytes())
    selected = sample(tree, previous)
    old_entries = [dict(path=row["filename"], sha=row["git_blob"]) for row in previous["source_inventory"]]
    if args.fetch:
        fetch(args.previous_xml, old_entries, args.workers)
        fetch(args.new_xml, selected, args.workers)
    for entry in old_entries:
        read_object(args.previous_xml / entry["path"], entry["sha"])
    _, earlier_digests = inventory(args.prior_folders)
    if (len(earlier_digests) != previous["prior_documents"] or
            sha256("\n".join(earlier_digests).encode()).hexdigest() != previous["prior_inventory_sha256"]):
        raise ValueError("Incomplete prior author exposure inventory")
    exposure, all_hashes = inventory([*args.prior_folders, args.previous_xml])
    kept, records = [], []
    for entry in selected:
        name = entry["path"]
        raw = read_object(args.new_xml / name, entry["sha"])
        row, reason = parse(raw, name)
        if row is not None:
            if row["doi"].casefold() in exposure["doi"] or row["title"].casefold() in exposure["title"]:
                reason = "prior_doi_or_title"
            elif row["author_ids"] & exposure["orcid"] or row["author_names"] & exposure["name"]:
                reason = "prior_author"
            if reason is None:
                kept.append(row)
        records.append(dict(filename=name, git_blob=entry["sha"], sha256=sha256(raw).hexdigest(), exclusion=reason))
    write_json(args.projection, projection(kept))
    manifest = dict(upstream=UPSTREAM, revision=REVISION, publisher_article_tree=ARTICLE_TREE, sampling_salt=SALT,
                    source_inventory=records, previous_manifest_sha256=digest(args.previous),
                    prior_documents=len(all_hashes), prior_inventory_sha256=sha256("\n".join(all_hashes).encode()).hexdigest(),
                    projection_sha256=digest(args.projection))
    _, cases, census = build(args.projection, manifest)
    chosen = sample_families(cases)
    write_json(args.manifest, manifest | dict(panel=census,
                                             exclusions=dict(Counter(row["exclusion"] for row in records if row["exclusion"])),
                                             chosen_families=sorted({row["family"] for row in chosen})))
    print(json.dumps(census), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tree", "previous", "previous_xml", "new_xml", "projection", "manifest"):
        parser.add_argument(name, type=Path)
    parser.add_argument("prior_folders", nargs="+", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    freeze(parser.parse_args())
