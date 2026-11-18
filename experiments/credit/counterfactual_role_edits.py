"""Build matched requirement edits over publisher-declared author evidence."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from urllib.parse import urlencode

from experiments.credit.plos_role_inventory import LICENSE, ROLES, declared_roles, read_url


VERSION = "plos-credit-edits-v1"
SOURCE_FIELD = "front/article-meta/contrib-group/contrib[@contrib-type='author']/role"
PATTERNS = ("10:11", "10:00", "01:11", "01:00")
POSITIVE = "declared"
NEGATIVE = "not_declared"
UNKNOWN = "unavailable"
CAP_PER_STRATUM = 4
CATALOG_SHA256 = "40f802d95aec45535c6b0427a87cb212ce7edbd355ab6030495358dd2be6acd7"
PILOT_SHA256 = "4cd80c70967b2b8234589af40b615ff262a57a6249bb4fcd2ff9147bf77eea43"

# Authored task phrases for the controlled rendering, keyed by the public taxonomy.
TASKS = dict(zip(ROLES, (
    "formulating the initial research idea",
    "organizing and maintaining the underlying data",
    "applying statistical or other analytical methods",
    "obtaining financial support for the project",
    "carrying out the research and collecting evidence",
    "designing the study procedures",
    "coordinating the work and managing its execution",
    "providing materials or equipment for the work",
    "developing the computer programs used in the study",
    "overseeing the research team",
    "checking whether the findings could be reproduced",
    "preparing the figures or graphical presentations",
    "preparing the first manuscript",
    "critically revising the manuscript",
), strict=True))


def fingerprint(*parts):
    return sha256(json.dumps([VERSION, *parts], ensure_ascii=True).encode()).hexdigest()


def split_for(doi):
    bucket = int(fingerprint("split", doi)[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def label(visible_roles, required):
    if visible_roles is None:
        return UNKNOWN
    return POSITIVE if set(required).issubset(visible_roles) else NEGATIVE


def render(author, required, style):
    first, second = required
    if style == "canonical":
        return f"Does the statement credit Author {author} with both {first} and {second}?"
    if style == "descriptive":
        return f"Did Author {author} both contribute to {TASKS[first]} and take part in {TASKS[second]}?"
    raise ValueError(f"Unknown rendering: {style}")


def matched_groups(doi, authors, cap=CAP_PER_STRATUM):
    available = sorted(set().union(*authors))
    groups = defaultdict(list)
    for anchor, before, after in permutations(available, 3):
        by_transition = defaultdict(list)
        for author, roles in enumerate(authors, 1):
            if anchor in roles:
                key = f"{int(before in roles)}{int(after in roles)}"
                by_transition[key].append(author)
        for pattern in PATTERNS:
            flip, stable = pattern.split(":")
            if by_transition[flip] and by_transition[stable]:
                changed, unchanged = (
                    min(by_transition[key], key=lambda a: fingerprint(doi, anchor, before, after, key, a))
                    for key in (flip, stable)
                )
                item_id = fingerprint(doi, anchor, before, after, changed, unchanged)
                groups[pattern].append((item_id, [anchor, before], [anchor, after], changed, unchanged))
    result = []
    for pattern in PATTERNS:
        selected = sorted(groups[pattern])[:cap]
        for key, old, new, flip, stable in selected:
            result.append(dict(
                id=key, doi=doi, split=split_for(doi), before=old, after=new,
                changed_author=flip, stable_author=stable,
                changed_labels=[label(authors[flip - 1], required) for required in (old, new)],
                stable_labels=[label(authors[stable - 1], required) for required in (old, new)],
            ))
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def article_url(doi):
    return "https://journals.plos.org/plosone/article/file?" + urlencode(dict(id=doi, type="manuscript"))


def capture(snapshot, catalog_path, pilot_path):
    catalog, pilot = catalog_path.read_bytes(), pilot_path.read_bytes()
    if sha256(catalog).hexdigest() != CATALOG_SHA256 or sha256(pilot).hexdigest() != PILOT_SHA256:
        raise ValueError("Unexpected search or pilot snapshot")
    excluded = {row["doi"] for row in json.loads(pilot)["sources"]}
    records = [row["id"] for row in json.loads(catalog)["response"]["docs"] if row["id"] not in excluded]
    selected = sorted(records, key=lambda doi: sha256(f"role-edits-v1:{doi}".encode()).digest())[:60]
    snapshot.mkdir(parents=True, exist_ok=True)

    def fetch(doi):
        target = snapshot / f"{doi.rsplit('.', 1)[-1]}.xml"
        try:
            raw = target.read_bytes() if target.exists() else read_url(article_url(doi))
            roles = declared_roles(raw, doi)
            if not target.exists():
                target.write_bytes(raw)
            return doi, len(roles), sum(map(bool, roles)), ""
        except (OSError, ValueError, SyntaxError) as error:
            return doi, 0, 0, f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, selected))
    write_json(snapshot / "selection.json", dict(catalog_sha256=CATALOG_SHA256, articles=rows))


def build(snapshot, output):
    selection = json.loads((snapshot / "selection.json").read_bytes())
    sources, excluded, groups = [], [], []
    for doi, declared_count, complete_count, error in selection["articles"]:
        if error or declared_count <= 1 or declared_count != complete_count:
            excluded.append({"doi": doi, "reason": "unavailable" if error else "incomplete_author_declarations"})
            continue
        suffix = doi.removeprefix("10.1371/journal.pone.")
        if not suffix.isdecimal():
            raise ValueError("Unexpected PLOS ONE DOI")
        raw = (snapshot / f"{suffix}.xml").read_bytes()
        authors = declared_roles(raw, doi)
        if len(authors) != declared_count or not all(authors):
            raise ValueError(f"Role inventory changed for {doi}")
        items = matched_groups(doi, authors)
        if not items:
            excluded.append({"doi": doi, "reason": "no_matched_groups"})
            continue
        groups.extend(items)
        sources.append(dict(
            doi=doi, source_sha256=sha256(raw).hexdigest(), source_field=SOURCE_FIELD, license=LICENSE,
            attribution=f"https://doi.org/{doi}",
            xml_url=article_url(doi),
            split=split_for(doi), roles_by_author=authors,
        ))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "pairs.jsonl").open("w") as file:
        for group in sorted(groups, key=lambda row: (row["split"], row["doi"], row["id"])):
            file.write(json.dumps(group, sort_keys=True, ensure_ascii=True) + "\n")
    manifest = dict(
        version=VERSION, catalog_sha256=selection["catalog_sha256"], sources=sources, excluded=excluded,
        split_groups=dict(Counter(group["split"] for group in groups)),
        split_sources=dict(Counter(source["split"] for source in sources)),
        pairs_sha256=sha256((output / "pairs.jsonl").read_bytes()).hexdigest(),
    )
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sources", "excluded")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture-from", type=Path, nargs=2, metavar=("CATALOG", "PILOT_MANIFEST"))
    args = parser.parse_args()
    if args.capture_from:
        capture(args.snapshot, *args.capture_from)
    build(args.snapshot, args.output)
