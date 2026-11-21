"""Freeze an enlarged eLife role holdout without prior titles or author identities."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from experiments.credit.elife_joint_postings_holdout import fetch, read_object
from experiments.credit.elife_paired_role_holdout import ARTICLE_TREE
from experiments.credit.elife_prompt_holdout import (
    REVISION, UPSTREAM, build, inventory, parse, projection, sample_families,
)
from experiments.followir.core17_reranking import digest, write_json

SALT = "elife-extended-role-holdout-v1"
SIZE = 2400
CONFIRMATION_SALT = "elife-source-guard-confirmation-v1"
CONFIRMATION_SIZE = 2700
CONFIRMATION_MODERN = 365


def inventory_digest(hashes):
    return sha256("\n".join(hashes).encode()).hexdigest()


def require_prior(manifest, hashes):
    if (manifest["prior_documents"], manifest["prior_inventory_sha256"]) != (
        len(hashes), inventory_digest(hashes)
    ):
        raise ValueError("The prior exposure inventory is incomplete")


def article_number(name):
    return int(name.split("-")[1])


def sample(tree, *manifests, confirmation=False, additional_used=()):
    if tree["truncated"] or tree["sha"] != ARTICLE_TREE:
        raise ValueError("Expected the archived publisher article tree")
    used = {article_number(row["filename"]) for manifest in manifests
            for row in manifest["source_inventory"]}
    used.update(additional_used)
    versions = {}
    lower = 65000 if confirmation else 84000
    for entry in tree["tree"]:
        match = re.fullmatch(r"elife-(\d+)-v(\d+)\.xml", entry["path"])
        if not match or entry["type"] != "blob":
            continue
        article, version = map(int, match.groups())
        if lower <= article < 114000 and article not in used:
            if article not in versions or version > versions[article][0]:
                versions[article] = version, entry
    salt = CONFIRMATION_SALT if confirmation else SALT
    ordered = sorted(versions, key=lambda key: sha256(f"{salt}:{key}".encode()).digest())
    if confirmation:
        modern = [key for key in ordered if key >= 84000]
        earlier = [key for key in ordered if key < 84000]
        if len(modern) != CONFIRMATION_MODERN:
            raise ValueError("The number of remaining modern article IDs changed")
        selected = modern + earlier[:CONFIRMATION_SIZE - CONFIRMATION_MODERN]
        required = CONFIRMATION_SIZE
    else:
        selected = ordered[:SIZE]
        required = SIZE
    if len(selected) != required:
        raise ValueError("Fewer fresh records than the fixed sample size")
    return [versions[key][1] for key in selected]


def freeze(args):
    if args.manifest.exists() or args.projection.exists():
        raise ValueError("Require a fresh manifest and projection")
    previous = json.loads(args.previous.read_bytes())
    latest = json.loads(args.latest.read_bytes())
    if any(row["revision"] != REVISION for row in (previous, latest)):
        raise ValueError("Wrong publisher snapshot")
    if latest["previous_manifest_sha256"] != digest(args.previous):
        raise ValueError("The past prospective manifests do not form the archived chain")

    _, hashes = inventory(args.prior_folders)
    require_prior(previous, hashes)
    groups = [(previous, args.previous_xml), (latest, args.latest_xml)]
    exposure_folders = list(args.prior_folders)
    metadata = {}
    confirmation = args.extended is not None
    if confirmation:
        extended = json.loads(args.extended.read_bytes())
        if (extended["revision"] != REVISION or extended["sampling_salt"] != SALT
                or extended["previous_manifest_sha256"] != digest(args.previous)
                or extended["latest_manifest_sha256"] != digest(args.latest)):
            raise ValueError("Expected the earlier extended holdout from the same exposure chain")
        groups.append((extended, args.extended_xml))
        metadata["extended_manifest_sha256"] = digest(args.extended)
    early_ids = {article_number(path.name) for folder in exposure_folders for path in folder.rglob("elife-*.xml")
                 if re.fullmatch(r"elife-(\d+)-v(\d+)\.xml", path.name)}
    selected = sample(json.loads(args.tree.read_bytes()), *(manifest for manifest, _ in groups),
                      confirmation=confirmation, additional_used=early_ids if confirmation else ())
    saved = [dict(path=row["filename"], sha=row["git_blob"])
             for manifest, _ in groups for row in manifest["source_inventory"]]
    if args.fetch:
        fetch(args.new_xml, selected, args.workers)
        for manifest, folder in groups:
            entries = [dict(path=row["filename"], sha=row["git_blob"]) for row in manifest["source_inventory"]]
            fetch(folder, entries, args.workers)
    if len({row["path"] for row in saved}) != len(saved):
        raise ValueError("Prior source cohorts overlap at the article-version level")
    for manifest, folder in groups:
        require_prior(manifest, hashes)
        for row in manifest["source_inventory"]:
            raw = read_object(folder / row["filename"], row["git_blob"])
            if sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("A prior article does not match its archived hash")
        exposure_folders.append(folder)
        exposure, hashes = inventory(exposure_folders)

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
            else:
                kept.append(row)
        records.append(dict(filename=name, git_blob=entry["sha"], sha256=sha256(raw).hexdigest(), exclusion=reason))
    write_json(args.projection, projection(kept))
    manifest = dict(
        upstream=UPSTREAM, revision=REVISION, publisher_article_tree=ARTICLE_TREE,
        sampling_salt=CONFIRMATION_SALT if confirmation else SALT, source_inventory=records,
        previous_manifest_sha256=digest(args.previous), latest_manifest_sha256=digest(args.latest),
        prior_documents=len(hashes), prior_inventory_sha256=inventory_digest(hashes),
        projection_sha256=digest(args.projection), **metadata,
    )
    _, cases, census = build(args.projection, manifest)
    families = sample_families(cases)
    write_json(args.manifest, manifest | dict(panel=census,
                                             chosen_families=sorted({row["family"] for row in families}),
                                             exclusions=dict(Counter(r["exclusion"] for r in records if r["exclusion"]))))
    print(json.dumps(census), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("tree", "previous", "latest", "previous_xml", "latest_xml", "new_xml", "projection", "manifest"):
        cli.add_argument(key, type=Path)
    cli.add_argument("prior_folders", nargs="+", type=Path)
    cli.add_argument("--extended", type=Path)
    cli.add_argument("--extended-xml", type=Path)
    cli.add_argument("--fetch", action="store_true")
    cli.add_argument("--workers", type=int, default=24)
    args = cli.parse_args()
    if (args.extended is None) != (args.extended_xml is None):
        cli.error("--extended and --extended-xml must be supplied together")
    freeze(args)
