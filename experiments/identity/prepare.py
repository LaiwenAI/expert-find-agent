"""Project public S2AND files into a benchmark with pre-existing author profiles."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata


def text(value):
    value = unicodedata.normalize("NFKD", str(value or "").casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def key(value):
    return int.from_bytes(sha256(value.encode()).digest()[:8], "big")


def project(signature, paper, number):
    info = signature["author_info"]
    name = text(" ".join(info[k] or "" for k in ("first", "middle", "last", "suffix")))
    venue = text(paper["venue"] or paper["journal_name"])
    return dict(
        position=number, paper=str(signature["paper_id"]), year=paper["year"] or 0,
        name=name, first=text(info["first"]), middle=text(info["middle"]),
        affiliations=sorted({parsed for a in info["affiliations"] or [] if (parsed := text(a))}),
        coauthors=sorted({parsed for a in paper["authors"] if a["position"] != info["position"]
                         if (parsed := text(a["author_name"]))}),
        title=sorted({word for word in text(paper["title"]).split() if len(word) > 2}),
        venue=venue,
    )


def panel(folder, name):
    files = {kind: folder / f"{name}_{kind}.json" for kind in ("papers", "clusters", "signatures")}
    raw = {kind: path.read_bytes() for kind, path in files.items()}
    digest = {kind: sha256(data).hexdigest() for kind, data in raw.items()}
    papers, clusters, signatures = (json.loads(raw[field]) for field in ("papers", "clusters", "signatures"))
    blocks = defaultdict(list)
    stats = Counter()
    for cluster in clusters.values():
        refs = [signatures[s] for s in cluster["signature_ids"]]
        block = {ref["author_info"]["block"] for ref in refs}
        if len(block) != 1 or not next(iter(block)):
            stats["excluded_cross_block_or_missing_clusters"] += 1
            continue
        name_block = next(iter(block))
        refs.sort(key=lambda row: (papers[str(row["paper_id"])]["year"] or 0, key(row["signature_id"])))
        records = [project(ref, papers[str(ref["paper_id"])], j) for j, ref in enumerate(refs)]
        absent = key(name + ":" + str(cluster["cluster_id"])) % 5 == 0
        count = min(2, max(1, len(records) // 2)) if not absent else 0
        # At most two later queries per author, keeping the earliest anchors.
        blocks[name_block].append(dict(profile=records[:count], queries=records[-min(2, len(records)-count):]
                                       if len(records)>count else []))
    stats["blocks"] = len(blocks)
    return dict(sha256=digest, stats=dict(stats), blocks=dict(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ("arnetminer", "qian", "zbmath")
    panels = {name: panel(args.datasets / name, name) for name in names}
    args.output.write_text(json.dumps(panels, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
