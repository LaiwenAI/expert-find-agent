"""Build a source- and owner-bound test from verified engineering JATS."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from experiments.credit.engineering_contribution_panel import article_panel, identifiers, snapshot


ORCID = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[0-9X]")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def valid_orcid(value):
    if not ORCID.fullmatch(value):
        return False
    digits = value.replace("-", "")
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


def owner_key(pmc, slot, author):
    ids = {match.group() for node in author.findall("contrib-id[@contrib-id-type='orcid']")
           for match in ORCID.finditer(node.text or "") if valid_orcid(match.group())}
    if len(ids) > 1:
        raise ValueError("Conflicting ORCIDs on one author")
    token = f"orcid:{next(iter(ids))}" if ids else f"local:{pmc}:{slot}"
    return sha256(f"engineering-source-audit-v1:{token}".encode()).hexdigest()


def text(node):
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def load_record(directory, pmc, record):
    raw = (directory / "xml" / f"{pmc}.xml").read_bytes()
    if sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Publisher source hash changed")
    article = ET.fromstring(raw)
    meta = article.find("./front/article-meta")
    if identifiers(meta) != (pmc, record["doi"]):
        raise ValueError("Publisher identifiers changed")
    declaration = article_panel(raw, record)
    authors = meta.findall("./contrib-group/contrib[@contrib-type='author']")
    keys = [owner_key(pmc, i, author) for i, author in enumerate(authors)]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated owner in one source")
    roles = {}
    for i in range(len(keys)):
        # Obtain model-visible membership from the contribution blocks.
        visible = {role for block in declaration["blocks"] if i in block["owners"] for role in block["roles"]}
        if visible != set(declaration["roles"][i]):
            raise ValueError("Visible and structured roles disagree")
        roles[keys[i]] = sorted(visible)
    abstract = SENTENCE.split(text(meta.find("abstract")))
    if len(abstract) < 3:
        raise ValueError("Need non-overlapping query and index text")
    half = max(1, len(abstract) // 2)
    return dict(
        doi=record["doi"], sha256=record["sha256"], journal=record["journal"],
        indexed=text(meta.find("./title-group/article-title")) + "\n" + " ".join(abstract[:half]),
        clue=" ".join(abstract[half:]), roles=roles,
    ), keys


def gold(articles, pair, roles):
    first, second = (articles[p]["roles"] for p in pair)
    a, b = roles
    return sorted(owner for owner in first.keys() & second.keys() if a in first[owner] and b in second[owner])


def requests(articles, links):
    rows = []
    for pair, owner in sorted(set(links)):
        left, right = [articles[p]["roles"] for p in pair]
        first, second = set(left[owner]), set(right[owner])
        complements = []
        for a in sorted(first):
            for b in sorted(second - {a}):
                binding = (a, b)
                swap = (b, a)
                if not any(b in roles for roles in left.values()) or not any(a in roles for roles in right.values()):
                    continue
                if gold(articles, pair, swap):
                    continue
                priority = -int(a not in second) - int(b not in first)
                stable = sha256(repr((pair, binding)).encode()).hexdigest()
                complements.append((priority, stable, binding, swap))
        for _, key, present, swapped in sorted(complements)[:5]:
            rows.extend(
                dict(family=key, papers=pair, roles=roles, variant=variant, gold=gold(articles, pair, roles))
                for variant, roles in (("present", present), ("swap", swapped))
            )
    return rows


def build(seed, linked, panel, manifest):
    source = json.loads((seed / "audit-confirmed.json").read_bytes())
    extension = json.loads((linked / "manifest.json").read_bytes())
    articles, ordinal, excluded = {}, {}, {}
    records = {key: source["inventory"]["records"][key] for key in source["included"]}
    records.update({key: row | {"journal": "sensors"} for key, row in extension["records"].items()})
    for key, record in sorted(records.items()):
        try:
            origin = linked if key in extension["records"] else seed
            articles[key], ordinal[key] = load_record(origin, key, record)
        except ValueError as exc:
            if str(exc) != "Need non-overlapping query and index text":
                raise
            excluded[key] = str(exc)
    links = []
    kept = []
    for row in extension["edges"]:
        a, b = row["seed_pmcid"], row["linked_pmcid"]
        if a not in articles or b not in articles:
            continue
        owner = ordinal[a][row["seed_author_ordinal"]]
        if owner != ordinal[b][row["linked_author_ordinal"]]:
            raise ValueError("ORCID link did not verify in both publications")
        links.append(((a, b), owner))
        kept.append(row)
    bench = requests(articles, links)
    if not bench:
        raise ValueError("No admitted scope-swap families")
    encoded = (json.dumps(dict(articles=articles, queries=bench), ensure_ascii=False) + "\n").encode()
    with panel.open("xb") as stream:
        stream.write(encoded)
    with manifest.open("x") as stream:
        json.dump(dict(
            panel_sha256=sha256(encoded).hexdigest(), records=records, excluded=excluded, links=kept,
            families=[{k:v for k,v in row.items() if k != "gold"} for row in bench],
        ), stream, indent=2)
        stream.write("\n")
    print(dict(articles=len(articles), unique_links=len(kept), variants=dict(Counter(r["variant"] for r in bench))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("seed", "linked", "panel", "manifest"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--restore-linked", type=Path)
    args = parser.parse_args()
    if args.restore_linked:
        raw = args.restore_linked.read_bytes()
        link_manifest = json.loads(raw)
        snapshot(args.linked, frozen=dict(
            selected={"sensors": link_manifest["selected_sources"]},
            records={key: row | {"journal": "sensors"} for key, row in link_manifest["records"].items()},
        ))
        (args.linked / "manifest.json").write_bytes(raw)
    build(args.seed, args.linked, args.panel, args.manifest)
