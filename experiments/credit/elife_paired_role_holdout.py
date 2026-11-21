"""Freeze a prospective eLife cohort by document hash before any search."""

import argparse
from collections import Counter
from hashlib import sha1, sha256
import json
from pathlib import Path
import re

from experiments.credit.elife_prompt_holdout import (
    REVISION, UPSTREAM, build, inventory, parse, projection, sample_families,
)
from experiments.followir.core17_reranking import digest, write_json


SALT = "expert-paired-role-prospective-v2"
SIZE = 960
ARTICLE_TREE = "3c6add96e3de305c65b9289f8bb4cb417d01dde8"


def choose(tree):
    if tree["truncated"] or tree["sha"] != ARTICLE_TREE:
        raise ValueError("Expected the complete pinned publisher article tree")
    latest = {}
    for file in tree["tree"]:
        match = re.fullmatch(r"elife-(\d+)-v(\d+)\.xml", file["path"])
        if not match or not 84000 <= int(match[1]) < 114000 or file["type"] != "blob":
            continue
        code, version = map(int, match.groups())
        if code not in latest or version > latest[code][0]:
            latest[code] = version, file
    selected = sorted(latest, key=lambda code: sha256(f"{SALT}:{code}".encode()).digest())[:SIZE]
    if len(selected) != SIZE:
        raise ValueError("Incomplete sampling frame")
    return [latest[code][1] for code in selected]


def freeze(tree, xml, output, manifest, prior_folders):
    if output.exists() or manifest.exists():
        raise ValueError("Require two fresh output paths")
    tree_content = json.loads(tree.read_bytes())
    selected = choose(tree_content)
    prior, exposure = inventory(prior_folders)
    rows, records = [], []
    for file in selected:
        name = file["path"]
        raw = (xml / name).read_bytes()
        git_id = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_id != file["sha"]:
            raise ValueError(f"Publisher object changed: {name}")
        parsed, reason = parse(raw, name)
        if parsed:
            if parsed["doi"].casefold() in prior["doi"] or parsed["title"].casefold() in prior["title"]:
                reason = "prior_doi_or_title"
            elif parsed["author_ids"] & prior["orcid"] or parsed["author_names"] & prior["name"]:
                reason = "prior_author"
        records.append(dict(filename=name, git_blob=git_id, sha256=sha256(raw).hexdigest(), exclusion=reason))
        if reason is None:
            rows.append(parsed)
    write_json(output, projection(rows))
    frozen = dict(upstream=UPSTREAM, revision=REVISION, publisher_article_tree=tree_content["sha"],
                  sampling_salt=SALT, source_inventory=records, prior_documents=len(exposure),
                  prior_inventory_sha256=sha256("\n".join(exposure).encode()).hexdigest(),
                  projection_sha256=digest(output))
    _, cases, census = build(output, frozen)
    pairs = sample_families(cases)
    write_json(manifest, frozen | dict(panel=census,
                                      exclusions=dict(Counter(r["exclusion"] for r in records if r["exclusion"])),
                                      chosen_families=sorted({r["family"] for r in pairs})))
    print(census, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tree", "xml", "output", "manifest"):
        parser.add_argument(name, type=Path)
    parser.add_argument("prior_folders", nargs="+", type=Path)
    args = parser.parse_args()
    freeze(args.tree, args.xml, args.output, args.manifest, args.prior_folders)
