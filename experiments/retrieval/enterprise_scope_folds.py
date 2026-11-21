"""Freeze company and lexical-neighbor folds for development-only evaluation."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path

from experiments.retrieval.enterprise_routed_review import digest, rows, write

FOLDS = 5
SEED = "enterprise-scope-cross-fit-v1"


def partitions(metadata):
    from experiments.freeze_enterprise_panel import grams, issuer_key, normalize
    from experiments.prepare_fresh_method_data import normalized_hash

    parents = {key: key for key in metadata}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    by_issuer = {key: issuer_key(row["company"]) for key, row in metadata.items()}
    if not all(by_issuer.values()):
        raise ValueError("Cannot guarantee issuer separation without company metadata")
    texts = {key: row["zh"] for key, row in metadata.items()}
    shingles = {key: grams(normalize(text)) for key, text in texts.items()}
    bodies = {key: normalized_hash(text.partition("\n")[2]) for key, text in texts.items()}
    for left, right in combinations(sorted(metadata), 2):
        a, b = shingles[left], shingles[right]
        common = len(a & b)
        near = common >= 64 and (common * 5 >= min(len(a), len(b)) * 4
                                  or common * 20 >= (len(a) + len(b) - common) * 13)
        if by_issuer[left] == by_issuer[right] or bodies[left] == bodies[right] or near:
            x, y = root(left), root(right)
            parents[max(x, y)] = min(x, y)
    groups = defaultdict(list)
    for key in sorted(metadata):
        groups[root(key)].append(key)

    folds = [[] for _ in range(FOLDS)]
    order = sorted(groups.values(), key=lambda keys: (
        -len(keys), sha256(f"{SEED}:{','.join(keys)}".encode()).digest(),
    ))
    for group in order:
        smallest = min(folds, key=lambda entries: sum(len(item) for item in entries))
        smallest.append(group)
    return folds


def freeze(args):
    from experiments.freeze_enterprise_panel import workbook_rows

    development = rows(args.development)
    ids = {row["id"] for row in development}
    source = {row["id"]: row for row in rows(args.metadata) if row["id"] in ids}
    if len(development) != len(ids) or source.keys() != ids:
        raise ValueError("Development identifiers do not match metadata")
    workbook_hash = digest(args.workbook)
    metadata = workbook_rows(args.workbook, ids)
    for row in development:
        key = row["id"]
        ancestor = source[key]
        if (ancestor["split"] != "development" or row["split"] != "development"
                or row["text"] != ancestor["text"] or row["text"] != metadata[key]["zh"]
                or ancestor["provenance"]["file_sha256"] != workbook_hash
                or ancestor["provenance"]["row"] != metadata[key]["source_row"]):
            raise ValueError("Original request and issuer provenance do not match")

    folds = partitions(metadata)
    write(args.output, dict(development_sha256=digest(args.development),
                            workbook_sha256=workbook_hash, folds=folds))
    print("Requests by fold:", [sum(map(len, groups)) for groups in folds])
    print("Issuer/lexical groups by fold:", list(map(len, folds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "metadata", "workbook", "output"):
        parser.add_argument(field, type=Path)
    freeze(parser.parse_args())
