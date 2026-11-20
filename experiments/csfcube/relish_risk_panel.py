"""Freeze disjoint judged RELISH query-by-example panels."""

import argparse
from hashlib import md5, sha256
import json
from pathlib import Path


def file_digest(path, algorithm):
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            algorithm.update(part)
    return algorithm.hexdigest()


def key(*parts):
    return sha256(":".join(map(str, parts)).encode()).digest()


def normalize_title(title):
    return " ".join(title.casefold().split())


def write_json(path, rows):
    with path.open("x", encoding="utf-8") as out:
        json.dump(rows, out, ensure_ascii=False)
        out.write("\n")


FILES = {
    "abstracts-relish.jsonl": "aa27b3ffa5ae7cbbeae9283e5a5dd327",
    "relish-evaluation_splits.json": "e5d723df874b36296432b474f020d813",
    "test-pid2anns-relish.json": "8ef048c700c661eb155999808c092308",
}
SIZES = (("held", "test", 100), ("fit", "dev", 65), ("risk", "dev", 55))
POOL_SIZE = 24


def freeze(root, output, manifest):
    if output.exists() or manifest.exists():
        raise ValueError("Use fresh panel and manifest")
    for name, md5_digest in FILES.items():
        if file_digest(root / name, md5()) != md5_digest:
            raise ValueError(f"Unexpected RELISH file: {name}")
    splits = json.loads((root / "relish-evaluation_splits.json").read_bytes())
    labels = json.loads((root / "test-pid2anns-relish.json").read_bytes())
    seeds = {name: sorted(set(splits[name]) & labels.keys(), key=lambda q: key("relish-routing", q))[:400]
             for name in ("dev", "test")}
    selected = set(seeds["dev"]) | set(seeds["test"])
    wanted = selected | {str(doc) for q in selected for doc in labels[q]["cands"]}
    papers = {}
    with (root / "abstracts-relish.jsonl").open() as stream:
        for article in map(json.loads, stream):
            pid = str(article["paper_id"])
            if pid in wanted and article["title"].strip() and article["abstract"]:
                papers[pid] = (article["title"], article["abstract"])
    used, titles, panel, inventory = set(), set(), {}, {}
    for phase, split, size in SIZES:
        panel[phase] = []
        for q in seeds[split]:
            if q in used or q not in papers or normalize_title(papers[q][0]) in titles:
                continue
            qtitle = normalize_title(papers[q][0])
            known = dict(zip(map(str, labels[q]["cands"]), labels[q]["relevance_adju"], strict=True))
            offered, seen = {}, {qtitle}
            for doc in sorted(known, key=lambda p: key("relish-candidates", q, p)):
                if doc in used or doc == q or doc not in papers:
                    continue
                title = normalize_title(papers[doc][0])
                if title in titles or title in seen:
                    continue
                seen.add(title)
                offered[doc] = known[doc]
                if len(offered) == POOL_SIZE:
                    break
            if len(offered) != POOL_SIZE:
                continue
            panel[phase].append(dict(query_id=q, facet="relevance", grades=offered))
            used.update([q, *offered])
            titles.update(seen)
            inventory.update((doc, papers[doc]) for doc in (q, *offered))
            if len(panel[phase]) == size:
                break
        if len(panel[phase]) != size:
            raise ValueError(f"Insufficient independent {phase} requests")
    write_json(output, dict(panels=panel, documents=inventory))
    write_json(manifest, dict(panel_sha256=file_digest(output, sha256()), upstream_md5=FILES, membership=panel))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "output", "manifest"):
        parser.add_argument(field, type=Path)
    freeze(**vars(parser.parse_args()))
